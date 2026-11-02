# Credit Default Prediction Using Machine Learning

**Can machine learning help identify customers likely to miss their next credit card payment?**

I explored this using 30,000 historical credit card records from Taiwan. I started with logistic regression, tried regularisation to see whether a simpler model would work just as well, and then tested a random forest after finding that my initial model missed quite a lot of defaults.

**[Read the full investigation (Jupyter notebook)](credit_default_analysis.ipynb)**

## What I found

| Model | Test ROC–AUC | Test average precision |
|---|---:|---:|
| Majority-class baseline | 0.500 | 0.221 |
| L1 logistic regression | 0.758 | 0.526 |
| **Random forest** | **0.775** | **0.553** |

The forest came out ahead, although the difference wasn't huge. When both models flagged the highest-risk **20% of test customers**, the forest identified **676 actual defaults** compared with **649** for logistic regression, while generating **27 fewer false alarms**.

![ROC-AUC and average precision across the three models](figures/model_comparison.png)

![Defaults caught and false alarms at a fixed review capacity](figures/review_outcomes.png)

A separate finding was that stronger L1 regularisation reduced the logistic model from **62 to 36 non-zero encoded coefficients**, with almost no change in cross-validated ROC–AUC. I found the trade-off between simplicity, default detection and false alarms more interesting than any one score.

## How I approached it

- Explored repayment history, credit limits and the balance of default outcomes.
- Checked the data and handled undocumented education category codes.
- Used one-hot encoding, feature scaling and pipelines to avoid leakage during cross-validation.
- Compared a simple baseline with logistic regression, tuned L1/L2 regularisation, and evaluated a random forest.
- Investigated classification thresholds and an illustrative rule of reviewing the highest-risk 20% of customers.

**Tools:** Python, pandas, NumPy, scikit-learn, Matplotlib, seaborn and `ucimlrepo`.

## Run the notebook

The easiest way is to open [`credit_default_analysis.ipynb`](credit_default_analysis.ipynb) in **Google Colab** and choose **Runtime → Run all**. The notebook downloads the public dataset automatically, so internet access is required.

For a local Jupyter environment, install dependencies with:

```bash
pip install -r requirements.txt
```

Then run the notebook cells in order. Random seeds are specified for the split and models, but package versions may cause small numerical differences.

## Limitations

This is a **portfolio investigation, not a real lending model**. The data are from **Taiwan in 2005**, so the findings shouldn't be generalised to modern borrowers without new validation. The 20% review capacity is illustrative; a real choice would require the actual costs of missed defaults and false alarms, along with fairness and calibration checks. Also, I examined the test set after the initial logistic model, so the later model comparison is an additional held-out check rather than a completely blind final evaluation.

## Data source

Yeh, I.-C. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. [Dataset and citation](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) · [DOI](https://doi.org/10.24432/C55S3H).

The dataset is downloaded from UCI, not included in this repository.
