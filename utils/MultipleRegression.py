import numpy as np
from scipy import stats

class MultipleLinearRegression:

    def __init__(self, method ="ols", alpha = 1.0, confidence_level=0.95):

        self.method = method.lower()
        self.alpha = alpha
        self.confidence_level = confidence_level
        
        self.coefficients = None
        self.intercept = None
        self.standard_errors = None
        self.t_stats = None
        self.p_values = None
        self.r_squared = None
        self.adj_r_squared = None
        self.f_stat = None
        self.f_p_value = None
        self.feature_names = None
        

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None):

        """
        Fit the multiple linear regression model using Ordinary Least Squares (OLS).

        The regression coefficients are estimated by minimizing the sum of
        squared residuals using the least-squares solution based on SVD.

        Parameters
        ----------
        X : np.ndarray
            Independent variables (predictors).
        y : np.ndarray
            Dependent variable (target).
        feature_names : list, optional
            Names of the independent variables.

        Returns
        -------
        self : MultipleLinearRegression
            The fitted regression model.
        """

        # Convert input data to NumPy arrays and ensure numerical compatibility
        X = np.asarray(X, dtype=float)

        # Reshape one-dimensional input into a two-dimensional feature matrix
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Ensure the target variable is represented as a one-dimensional array
        y = np.asarray(y, dtype=float).reshape(-1)


        n_samples, n_features = X.shape # Extract the number of observations and predictor variables

        # Assign user-provided feature names or generate default names
        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else [f"X{i+1}" for i in range(n_features)]
        )

        # Add a column of ones to X for the intercept term (beta_0)
        X_design = np.column_stack((np.ones(n_samples), X))

        # --------------------------------
        # 1. OLS
        # --------------------------------

        # Estimate regression coefficients using the least-squares solution.
        # NumPy's lstsq uses a numerically stable SVD-based approach.
        if self.method == "ols":
            beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)

        # --------------------------------
        # 2. Ridge regression
        # --------------------------------

        elif self.method == "ridge":
            I = np.eye(n_features + 1)
            I[0, 0] = 0

            beta = np.linalg.solve(X_design.T @ X_design + self.alpha * I, X_design.T @ y)


        # Separate the intercept from the predictor coefficients
        self.intercept = beta[0]
        self.coefficients = beta[1:]

        # Calculate fitted values and residuals
        y_pred = X_design @ beta
        residuals = y - y_pred

        # Calculate the residual and model degrees of freedom
        # The residual degrees of freedom account for all estimated parameters, including the intercept.
        df_resid = n_samples - (n_features + 1)
        df_model = n_features

        # Calculate the residual sum of squares (RSS),
        # total sum of squares (TSS), and explained sum of squares (ESS)
        rss = np.sum(residuals**2)
        tss = np.sum((y - np.mean(y)) ** 2)
        ess = tss - rss

        # Calculate the coefficient of determination (R-squared),
        # which measures the proportion of variance explained by the model.
        self.r_squared = 1 - (rss / tss)
        self.adj_r_squared = 1 - (1 - self.r_squared) * (n_samples - 1) / df_resid

        # Estimate the residual variance using the mean squared error (MSE)
        mse = rss / df_resid

        # Calculate the covariance matrix of the estimated coefficients.
        # The pseudoinverse is used to handle singular or nearly singular design matrices more robustly.
        cov_matrix = mse * np.linalg.pinv(X_design.T @ X_design)

        # Extract standard errors from the diagonal of the covariance matrix
        self.standard_errors = np.sqrt(np.diag(cov_matrix))

        # Calculate t-statistics for testing whether each coefficient is statistically different from zero.
        self.t_stats = beta / self.standard_errors
        self.p_values = 2 * stats.t.sf(np.abs(self.t_stats), df=df_resid)

        # Overall F-statistic and its p-value
        self.f_stat = (ess / df_model) / mse
        self.f_p_value = stats.f.sf(self.f_stat, dfn=df_model, dfd=df_resid)

        # Confidence intervals
        alpha = 1 - self.confidence_level
        # Calculate the critical t-value based on the specified
        t_critical = stats.t.ppf(
            1 - alpha / 2,
            df=df_resid
        )

        # Calculate the lower and upper confidence bounds
        # for each estimated regression coefficient, including the intercept.
        self.ci_lower = beta - t_critical * self.standard_errors
        self.ci_upper = beta + t_critical * self.standard_errors

        self.confidence_intervals = np.column_stack(
            (self.ci_lower, self.ci_upper)
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the fitted regression model.

        Predictions are calculated as a linear combination of the
        input features and their estimated coefficients, plus the intercept.

        Parameters
        ----------
        X : np.ndarray
            Independent variables for which predictions are required.

        Returns
        -------
        np.ndarray
            Predicted values of the dependent variable.
        """

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.intercept + X @ self.coefficients

    def summary(self):
        """
        Generate predictions using the fitted regression model.

        Predictions are calculated as a linear combination of the
        input features and their estimated coefficients, plus the intercept.

        Parameters
        ----------
        X : np.ndarray
            Independent variables for which predictions are required.

        Returns
        -------
        np.ndarray
            Predicted values of the dependent variable.
        """
        # Display overall goodness-of-fit and model significance statistics
        print(f"R-squared:          {self.r_squared:.4f}")
        print(f"Adjusted R-squared: {self.adj_r_squared:.4f}")
        print(f"F-statistic:        {self.f_stat:.4f}") 
        print(f"P-value:            {self.f_p_value:.5f}")

        # Combine the intercept and predictor coefficients into one array
        all_betas = np.concatenate(([self.intercept], self.coefficients))

        # Add the intercept to the list of predictor names
        terms = ["Intercept"] + self.feature_names

        result = []

        # Iterate through the regression terms and their corresponding statistical measures to construct the model summary.
        for term, b, se, t, p, ci_lower, ci_upper in zip(terms, all_betas, self.standard_errors, self.t_stats, self.p_values, self.ci_lower, self.ci_upper):

            # Store the statistical results for the current regression term
            result.append({
                "Term": term,
                "Coef": b,
                "Std err": se,
                "T-statistics": t,
                "P-Value": round(p, 5),
                "Conf interval low": round(ci_lower, 2),
                "Conf interval upper": round(ci_upper, 2)
            })

        return result


    def calculate_vif(X):
        """
        Calculate Variance Inflation Factors (VIF) to check multicolinearity

        Parameters
        ----------
        X : numpy.ndarray
            2D array containing independent variables.
            Rows represent observations and columns represent variables.

        Returns
        -------
        numpy.ndarray
            VIF value for each independent variable.
        """

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        n, p = X.shape

        if n <= p:
            raise ValueError("Number of observations must exceed number of variables.")

        vif_values = np.zeros(p)

        for j in range(p):

            # Select target variable
            y = X[:, j]

            # Select all remaining variables
            X_other = np.delete(X, j, axis=1)

            # Add intercept
            X_other = np.column_stack([
                np.ones(n),
                X_other
            ])

            # Regress variable j on all other variables
            beta, _, _, _ = np.linalg.lstsq(X_other, y)

            # Calculate predictions
            y_pred = X_other @ beta

            # Calculate R-squared
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            if ss_tot == 0:
                vif_values[j] = np.inf
                continue

            r_squared = 1 - ss_res / ss_tot

            # Calculate VIF
            if np.isclose(1 - r_squared, 0):
                vif_values[j] = np.inf
            else:
                vif_values[j] = 1 / (1 - r_squared)

        return vif_values