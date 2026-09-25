# Heart Disease Classification

A binary classification project that predicts whether a patient has heart disease from 13 clinical attributes. It covers the full workflow in one Jupyter notebook: exploratory data analysis, baseline models, hyperparameter tuning, and evaluation with cross-validated metrics.

**Best result:** a tuned Logistic Regression model with **88.5% test accuracy** (ROC AUC 0.88) and **84.5% 5-fold cross-validated accuracy** (recall 0.92).

## Problem

> Given clinical parameters about a patient, can we predict whether or not they have heart disease?

The proof-of-concept target was set at 95% accuracy before starting. The final model reaches 88.5% on the test split, so it falls short of that goal. The Limitations section covers what could close the gap.

## Dataset

- **Source:** Cleveland heart disease data from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease). A version is also on [Kaggle](https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset).
- **Size:** 303 patients, 13 features, 1 target column.
- **Missing values:** none.
- **Class balance:** 165 with heart disease (target = 1), 138 without (target = 0). Close enough to balanced that accuracy is a fair headline metric.

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic) |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting ECG result (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels (0-3) coloured by fluoroscopy |
| `thal` | Thalassemia result |
| `target` | Heart disease (1 = yes, 0 = no) |

## Exploratory data analysis

- **Sex vs. target:** men make up most of the positive cases in raw counts (93 vs. 72), but the rate tells the opposite story. 72 of 96 women (about 75%) have heart disease, compared with 93 of 207 men (about 45%). The overall rate is about 54%.
- **Chest pain type:** patients with type 0 (typical angina) are mostly negative (104 of 143). Types 1, 2 and 3 are mostly positive, with type 2 the largest positive group (69 of 87).
- **Age vs. max heart rate:** scatter plot shows max heart rate trending down with age, and patients with heart disease tend to reach higher max heart rates.
- **Age distribution:** roughly normal, skewed towards older patients (mean age about 54).
- **Correlation heatmap:** `exang` shows a negative correlation with the target, which a crosstab confirms: exercise-induced angina is more common among patients labelled negative in this dataset.

## Modelling

Data was split 80/20 into train and test sets (`random_state` fixed with `np.random.seed(42)`), giving 242 training and 61 test samples.

### Baseline models

Three models were trained with default settings:

| Model | Test accuracy |
|---|---|
| Logistic Regression | 88.52% |
| Random Forest | 83.61% |
| XGBoost | 81.97% |

XGBoost had the weakest baseline, so it was dropped from tuning.

### Hyperparameter tuning

| Model | Method | Best parameters | Test accuracy |
|---|---|---|---|
| Logistic Regression | RandomizedSearchCV (20 iterations, 5-fold) | `C=0.2336`, `solver='liblinear'` | 88.52% |
| Random Forest | RandomizedSearchCV (20 iterations, 5-fold) | `n_estimators=210`, `max_depth=3`, `min_samples_split=4`, `min_samples_leaf=19` | 86.89% |
| Logistic Regression | GridSearchCV (30 values of `C`, 5-fold) | `C=0.2043`, `solver='liblinear'` | 88.52% |

Tuning improved Random Forest from 83.6% to 86.9%, but it still trailed Logistic Regression. Tuning did not change Logistic Regression's test accuracy. The GridSearchCV Logistic Regression model was used as the final model.

## Evaluation

### Test set (61 samples)

Confusion matrix:

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 25 | 4 |
| **Actual 1** | 3 | 29 |

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (no disease) | 0.89 | 0.86 | 0.88 | 29 |
| 1 (disease) | 0.88 | 0.91 | 0.89 | 32 |
| **Accuracy** | | | **0.89** | 61 |

ROC AUC: **0.88** (computed from predicted labels).

### 5-fold cross-validation (full dataset)

A single 61-sample test split can be lucky or unlucky, so the final model (`LogisticRegression(C=0.2043, solver='liblinear')`) was also scored with 5-fold cross-validation:

| Metric | Mean score |
|---|---|
| Accuracy | 0.845 |
| Precision | 0.821 |
| Recall | 0.921 |
| F1-score | 0.867 |

High recall matters here: the model catches about 92% of patients who have heart disease, at the cost of some false positives.

### Feature importance

Logistic Regression coefficients from the final model:

- **Largest positive coefficients:** `cp` (0.66), `slope` (0.45), `restecg` (0.31)
- **Largest negative coefficients:** `sex` (-0.86), `thal` (-0.68), `ca` (-0.64), `exang` (-0.60), `oldpeak` (-0.57)
- **Near zero:** `age`, `chol`, `trestbps`, `fbs`, `thalach`

## Results summary

- Logistic Regression beat Random Forest and XGBoost at every stage.
- Final model: 88.5% test accuracy, 0.88 ROC AUC, 84.5% cross-validated accuracy, 0.92 cross-validated recall.
- The 95% accuracy target was not reached.

## Limitations and next steps

- **Small dataset.** 303 rows means results can move a few points depending on the split. The cross-validated scores are the more reliable estimate.
- **No feature scaling.** The baseline Logistic Regression raised a convergence warning. Adding `StandardScaler` in a `Pipeline` should help convergence and make the coefficients comparable as feature importances (right now they reflect each feature's units as well as its effect).
- **Categorical features treated as numbers.** `cp`, `thal`, `slope` and `restecg` are categories. One-hot encoding them may improve results.
- **ROC from hard labels.** The ROC curve uses predicted classes, not probabilities. Using `predict_proba` would give a full curve and a more informative AUC.
- **Wider model search.** Try SVM and KNN as extra comparisons.

## Future updates

- **Tuned XGBoost classifier.** XGBoost was dropped after a weak default baseline (81.97%), but it was never tuned. The next update will fit an `XGBClassifier` with a proper hyperparameter search (for example `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`) and cross-validation, to push accuracy as high as it will go and see whether it can beat Logistic Regression or get closer to the 95% target.

## How to run

**Requirements:** Python 3.12 (the version used), Jupyter, and:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

**Steps:**

```bash
git clone https://github.com/eva-protoype/<repo-name>.git
cd heart_eda
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

Place the dataset at `sklearn-data/heart-disease.csv` (the path the notebook reads from), then:

```bash
jupyter notebook
```

Open the notebook and run all cells.

## Project structure

```
.
├── main-heart-disease.ipynb
├── sklearn-data/
│   └── heart-disease.csv
└── README.md
```

## Tech stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, Jupyter
