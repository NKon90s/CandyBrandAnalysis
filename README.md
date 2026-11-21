# Candy Product Exploratory Data Analysis

## Project overview

This project uses exploratory data analysis (EDA) to support a data-driven product-development decision for the Purchase Group's private-label candy range. The business question is which type of candy, and which ingredient combination, is the most promising direction for a new product.

The analysis uses FiveThirtyEight's **Candy Power Ranking** dataset. The notebook examines product characteristics, their co-occurrence, their relationship with consumer preference, and whether particular ingredient combinations appear especially promising.

**Main recommendation in the notebook:** consider developing a chocolate-based candy containing peanuts and/or almonds, then validate the concept with additional consumer and commercial research.

This is an exploratory recommendation based on the available dataset. The analysis does not establish sales potential, profitability, or a causal effect of ingredients on purchasing behaviour.

## Dataset

- **Source:** [FiveThirtyEight – Candy Power Ranking](https://github.com/fivethirtyeight/data/tree/master/candy-power-ranking)
- **Data file:** `data/candy-data.csv`
- **Observations:** 85 candy products
- **Outcome of interest:** `winpercent`, the overall win percentage from approximately 269,000 consumer matchups.
- **License:** The source dataset is published under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. Please retain attribution to FiveThirtyEight when redistributing the dataset or derived work.

### Variables

| Variable | Description |
|---|---|
| `competitorname` | Candy product name |
| `chocolate` | Contains chocolate |
| `fruity` | Fruit-flavoured |
| `caramel` | Contains caramel |
| `peanutyalmondy` | Contains peanuts, peanut butter, or almonds |
| `nougat` | Contains nougat |
| `crispedricewafer` | Contains crisped rice, wafers, or a cookie component |
| `hard` | Is a hard candy |
| `bar` | Is a candy bar |
| `pluribus` | Is one of many candies in a bag or box |
| `sugarpercent` | Sugar percentile relative to the dataset |
| `pricepercent` | Unit-price percentile relative to the dataset |
| `winpercent` | Overall win percentage from the matchups |

The product-characteristic columns are binary indicators (`1` = yes, `0` = no). The three percentile / preference variables are numeric.

## Analysis workflow

The notebook is organised around the following stages:

1. **Load and inspect the data**
   - Load the CSV into pandas.
   - Review data types, completeness, and product uniqueness.
   - Inspect the frequency of product characteristics.

2. **Explore relationships between product characteristics**
   - Calculate and visualise a correlation matrix.
   - Calculate conditional probabilities for pairs of characteristics.
   - Calculate lift to assess whether characteristics co-occur more or less often than expected under independence.
   - Check selected co-occurrence calculations manually.

3. **Relate product characteristics to consumer preference**
   - Examine correlations between `sugarpercent`, `pricepercent`, and `winpercent`.
   - Use Welch's independent-samples t-test to compare mean `winpercent` for products with and without each binary characteristic.
   - Report mean differences, p-values, and confidence intervals.

4. **Create product categories with hierarchical clustering**
   - Use eight binary product characteristics; exclude `pluribus` because packaging format is not the focus of the flavour / product categorisation.
   - Calculate Jaccard distances.
   - Compare average, single, and complete linkage across cluster counts from 2 to 8 using silhouette scores.
   - Assign descriptive names to the selected clusters and compare their average `winpercent`, `sugarpercent`, and `pricepercent`.

5. **Explore ingredient combinations**
   - Within the chocolate-based cluster, calculate the number of products and average `winpercent` for combinations of selected ingredients.
   - Use a bubble chart to compare observed average preference with the number of products represented.

6. **Fit a multiple linear regression**
   - Model `winpercent` using selected ingredient indicators and eligible pairwise interaction terms.
   - Review coefficient estimates, confidence intervals, p-values, and model-level statistics.
   - Assess multicollinearity using variance inflation factors (VIF).

## Key findings reported in the notebook

- Chocolate-containing products have a higher average `winpercent` than products without chocolate. The notebook reports a difference of **18.78 percentage points**.
- Wafer / crisped-rice and peanut / almond characteristics also show positive differences in the Welch tests. Fruity, hard-candy, and `pluribus` characteristics show negative differences.
- The chocolate-based cluster has the highest average `winpercent` among the eight categories identified in the notebook.
- The observed **chocolate + nuts** combination has an average `winpercent` of **68.50% across 12 products**.
- The chocolate + caramel + crisped-rice combination has a higher observed average (**69.38%**) but is represented by only **3 products**, so the notebook treats it as less reliable.
- In the regression, the chocolate × peanut/almond interaction has a positive coefficient of **+39.35** (reported p-value approximately **0.001**). This is an interaction coefficient, not a predicted `winpercent` and not a standalone 39.35-percentage-point uplift.
- The notebook also identifies multicollinearity among some predictors and interaction terms, which limits how confidently individual regression coefficients can be separated and interpreted.

## Interpretation and limitations

Keep the following points in mind when interpreting the results:

- **Small sample:** the dataset contains 85 products. Some characteristics and combinations are represented by relatively few observations.
- **Preference proxy:** `winpercent` reflects relative performance in the dataset's matchups. It is not actual sales, market share, repeat purchase, or willingness to pay.
- **Observational analysis:** associations do not demonstrate that an ingredient causes higher consumer preference.
- **Multiple comparisons:** the notebook tests several characteristics. Individual p-values should be interpreted in the context of the number of comparisons performed.
- **Clustering choices:** the selected linkage and number of clusters are analytical choices. Some resulting clusters may be small and should not automatically be treated as robust market segments.
- **Regression multicollinearity:** interaction terms overlap with their component variables. High VIF values indicate that some coefficients may be unstable or difficult to interpret independently. A statistically significant coefficient does not remove this concern.
- **Commercial validation is still needed:** pricing, costs, margins, manufacturing feasibility, shelf performance, competitive positioning, and consumer testing are outside the scope of this dataset.

The recommendation should therefore be treated as a **shortlist for further product development and validation**, rather than a final commercial decision.

## Repository structure

A suggested project layout is:

```text
Candy_brand_analysis/
├── README.md
├── notebooks/
│   └── EDA.ipynb
├── data/
│   └── candy-data.csv
└── utils/
    ├── StatTest.py
    ├── ClusterAnalysis.py
    └── MultipleRegression.py
```

The notebook imports custom helper classes from the `utils` package. Ensure these modules are present in the project in addition to the notebook and dataset.

## Requirements

The notebook uses Python and the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `requests`
- `scipy` (used by the clustering / statistical helper functionality, depending on the supplied helper modules)
- `plotly`

It also imports project-specific modules from `utils`:

- `utils.StatTest`
- `utils.ClusterAnalysis`
- `utils.MultipleRegression`

The custom `utils` modules are project files, not packages installed by pip. They must be available in the repository with the expected class and method names.

## Setup and running the notebook

1. Clone or download this project.
2. Ensure the notebook, `utils/` directory, and its helper modules are in place.
3. Create and activate a Python virtual environment (recommended).
4. Install the required libraries:

   ```bash
   python -m pip install numpy pandas matplotlib seaborn requests scipy plotly jupyter
   ```

5. Start Jupyter:

   ```bash
   jupyter notebook
   ```

6. Open `EDA.ipynb` and run the cells from top to bottom.

### Dataset download

The notebook creates a `data/` directory and downloads `candy-data.csv` from the FiveThirtyEight GitHub repository if the file is not already present. If the CSV is already in `data/`, the download is skipped.

Run the notebook from the project directory so its relative paths resolve as intended. The notebook currently determines its data directory relative to the working directory / notebook path setup; if you change the repository layout or execution directory, check the `BASE_DIR` and `DATA_DIR` path logic first.

## Tools and visualisations

The notebook uses:

- pandas and NumPy for data manipulation and calculations;
- Matplotlib and Seaborn for static charts;
- Plotly for the ingredient-combination bubble chart;
- custom helper classes in `utils/` for Welch's t-test, hierarchical clustering, and multiple linear regression.

## Attribution

Dataset: FiveThirtyEight, [Candy Power Ranking](https://github.com/fivethirtyeight/data/tree/master/candy-power-ranking), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

This project is an independent exploratory analysis for product-development decision support. The analysis and recommendation should not be interpreted as a statement or endorsement by FiveThirtyEight.
