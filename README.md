# Titanic Survival: EDA + Baseline Model

A quick, self-contained data science project analyzing the Titanic dataset:
exploratory data analysis, feature engineering, and a baseline logistic
regression model predicting passenger survival.

## Results

- **Accuracy:** ~79% on a held-out test set
- Key survival drivers: sex, passenger class, and family size (see
  `outputs/feature_importance.png`)

## Project Structure

```
.
├── data/
│   └── titanic.csv          # raw dataset
├── notebooks/
│   └── analysis.py          # EDA + model pipeline
├── outputs/                 # generated plots (created on run)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python notebooks/analysis.py
```

This prints EDA summaries and model metrics to the console, and saves plots
to `outputs/`:

- `survival_overview.png` — survival counts and rate by class
- `survival_by_sex_age.png` — survival by sex and age distribution
- `confusion_matrix.png` — model performance
- `feature_importance.png` — logistic regression coefficients

## Data

[Titanic dataset](https://github.com/datasciencedojo/datasets), a classic
benchmark dataset with passenger demographics, ticket info, and survival
outcomes from the 1912 sinking.

## Next Steps

- Try tree-based models (Random Forest, XGBoost) for comparison
- Cross-validation and hyperparameter tuning
- More feature engineering (titles extracted from names, cabin deck)
