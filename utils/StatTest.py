import numpy as np
from scipy import stats


class WelchT:

    """
    Statistical utility class for performing Welch's independent two-sample
    t-test and calculating confidence intervals for the difference between
    two independent group means.

    Welch's t-test is an extension of the independent two-sample t-test
    that does not assume equal population variances between the two groups.
    """

    @staticmethod
    def welch_confidence_interval(group1, group2, confidence=0.95):

        """
        Calculate the confidence interval for the difference between
        two independent group means using Welch's t-distribution.

        The method uses Welch-Satterthwaite degrees of freedom to account
        for potentially unequal variances and sample sizes.

        Parameters
        ----------
        group1 : array-like
            Observations belonging to the first independent group.
        group2 : array-like
            Observations belonging to the second independent group.
        confidence : float, default=0.95
            Desired confidence level for the interval.

        Returns
        -------
        tuple of float
            Lower and upper bounds of the confidence interval
            for the difference in population means (group1 - group2).
        """

        # Determine the sample size of each independent group
        n1, n2 = len(group1), len(group2)

        # Ensure that both groups contain enough observations to estimate their sample variances.
        if n1 < 2 or n2 < 2:
            raise ValueError(
                "Each group must contain at least two observations."
            )

        # Calculate the observed difference between the two sample means
        mean_diff = np.mean(group1) - np.mean(group2)

        # Calculate the unbiased sample variance for each group.
        # ddof=1 applies Bessel's correction by using n-1 in the denominator.
        var1 = group1.var(ddof=1)
        var2 = group2.var(ddof=1)

        # Calculate the standard error of the difference between means.
        se = np.sqrt(var1/n1 + var2/n2)

        # Handle the special case where both groups have zero variance.
        if se == 0:
            return mean_diff, mean_diff

        # Calculate the approximate degrees of freedom using the Welch-Satterthwaite equation.
        # This adjustment accounts for unequal group variances and potentially unequal sample sizes.
        df = (var1/n1 + var2/n2)**2 / (
            (var1/n1)**2/(n1-1) +
            (var2/n2)**2/(n2-1)
        )
        
        alpha = 1 - confidence

        # Obtain the two-tailed critical t-value using the Welch-Satterthwaite degrees of freedom.
        critical_value = stats.t.ppf(1 - alpha/2, df)

        # Calculate the lower and upper confidence interval bounds
        lower = mean_diff - critical_value * se
        upper = mean_diff + critical_value * se

        return lower, upper

    
    @staticmethod
    def welch_t_test(features, param, df, confidence=0.95):

        """
        Perform Welch's independent two-sample t-test for multiple
        binary features against a specified numerical variable.

        For each feature, observations are divided into two independent
        groups according to whether the feature is present (1) or absent (0).

        The method calculates descriptive statistics, the difference
        between group means, Welch's t-statistic, the corresponding
        p-value, and the confidence interval for the mean difference.

        Parameters
        ----------
        features : list
            Names of the binary feature columns to be tested.
        param : str
            Name of the numerical dependent variable being compared.
        df : pandas.DataFrame
            Dataset containing the binary features and dependent variable.
        confidence : float, default=0.95
            Confidence level used to calculate the confidence intervals.

        Returns
        -------
        list of dict
            A list of dictionaries containing the descriptive statistics
            and inferential test results for each eligible feature.
        """

        results = []
        for feature in features:
            # Separating groups
            with_feature = df[df[feature] == 1][param]
            without_feature = df[df[feature] == 0][param]

            # Skip combinations with insufficient observations
            if len(with_feature) < 2 or len(without_feature) < 2:
                continue

            # Calculate the confidence interval for the difference between the two group means using Welch's method.
            ci_low, ci_high = WelchT.welch_confidence_interval(with_feature, without_feature, confidence=confidence)
            
            # Perform Welch T-test
            t_stat, p_value = stats.ttest_ind(
                with_feature,
                without_feature,
                equal_var = False # This indicate that we are doing Welch and not Student t-test
            )

            # Calculate the observed difference between the group means.
            difference = np.mean(with_feature) - np.mean(without_feature)

            # Store the descriptive statistics and inferential results for the current feature comparison.
            results.append({
                "Feature": feature,
                "N with feature": len(with_feature),
                "N without feature": len(without_feature),
                "Mean with feature": round(np.mean(with_feature), 2),
                "Mean without feature": round(np.mean(without_feature), 2),
                "Difference": round(difference, 2),
                "T-statistic": t_stat,
                "P-value": p_value,
                "Conf Interval low": round(ci_low, 2),
                "Conf Interval high": round(ci_high, 2)
            })
        

        return results
