"""How to apply penalised logistic regression, explainable boosting machine, and xgboost models
to predict a binary outcome.

This script contains minimal code for applying these types of models to the data,
including the finding of best hyperparameteres.

We first split data into train (80%), validation (20%) and test (20%) sets.
We fit models with different hyperparameters on train set, evaluate them on validation set using the AUC metric,
then pick the hyperparameter combination that had the highest validation set AUC.

If the validation set is small, or has a small number of individuals with an outcome.
then this is not the most rigorous way of finding hyperparameters.
In that case, the validation set scores may not be representative,
as they are driven by the smaller number of individuals with the outcome.
A more rigorous way is to use an inner cross-validation loop: 
- split out the test set
- split the remaining data into four folds (as an example)
- at each iteration, one fold serves as the validation set and the remaining as the train set
  we train the model with a hyperparameter combination on each of the four training sets,
  and evaluate it on the corresponding validation set.
  For each hyperparameter combination, we thus have four models we trained and four validation set scores.
  We compute the average validation set score, and perhaps its standard deviation.
  We select the combination that had highest average validation set score, 
However, this can be time consuming in practice. 
For simplicity, we keep a single train-validation split.
"""
from constants import PROJECT_ROOT
from sklearn.linear_model import LogisticRegression   # for logistic regression
from sklearn.model_selection import train_test_split  # to split data into train-validation-test sets
from sklearn.metrics import roc_auc_score             # to evaluate how the models perform
from sklearn.preprocessing import StandardScaler      # to standardise values of input data to zero mean and unit standard deviation
from interpret.glassbox import ExplainableBoostingClassifier  # for explainable boosting machine (EBM)
from interpret import show         # to visualise shape functions of EBM
from xgboost import XGBClassifier  # for XGBoost
import optuna        # for hyperparameter tuning
import pandas as pd  # for working with data tables
import numpy as np


# ---- PREPARE DATA ----

# Load data
#data_path = PROJECT_ROOT / 'data'
#df = pd.read_csv(data_path / 'nhanes.csv')

# Helper function to create a dummy dataset
def dummy_data(n=1000, bias=-6, random_state=42,
               w_plt=3., w_age=4.5, w_sex=1., p_noise=30):
    
    # Dummy predictor variables
    rng = np.random.default_rng(seed=random_state)
    #fit = rng.lognormal(mean=1.2, sigma=1.75, size=n).clip(0, 400)
    ind_male = rng.binomial(n=1, p=0.4, size=n)
    platelets = rng.gamma(shape=1.2, scale=170, size=n).clip(50, 800).round(1)
    age = rng.normal(loc=70, scale=10, size=n).clip(18, 100).round(1)

    # Transform the predictor variables by their respective functions
    f_age = -1 * ((age - 85) / 10)**2
    f_plt = ((platelets - 400) / 100)**2
    f_male = w_sex * ind_male

    # scale to [0, 1] * weight
    f_age = w_age * (f_age - f_age.min()) / (f_age.max() - f_age.min())
    f_plt = w_plt * (f_plt - f_plt.min()) / (f_plt.max() - f_plt.min())
    
    # Get additive predictor
    m = bias + f_age + f_plt + f_male
    prob = 1 / (1 + np.exp(-m))
    crc = rng.binomial(n=1, p=prob, size=n)

    # Collect to dataframe
    df = pd.DataFrame({'cancer': crc, 
                       'age': age, 
                       'sex_male': ind_male, 
                       'platelets': platelets})
    print(f"{n} observations, {crc.sum()} cancers, {crc.mean()*100}% cancers")
    shapes = pd.DataFrame({'f_age': f_age, 'f_male': f_male, 'f_platelets': f_plt})

    # Add noise variables
    x_noise = rng.uniform(-1, 1, size=(n, p_noise))
    x_noise = pd.DataFrame(x_noise)
    x_noise.columns = ['noise' + str(i) for i, c in enumerate(x_noise.columns)]
    df = pd.concat(objs=[df, x_noise], axis=1)

    return df, shapes


# Create simulated data
#  The data will have three variables that are associated with the cancer outcome: age, sex, platelets
#  it will also have 100 variables that have no association, these are called "noise1", "noise2", etc
#  When the noise variables are included, the models are more likely to overfit, as they will try to use
#  the random patterns that appear in sampled noise variables to classify patients, even thouhgh these patterns
#  are not generalisable
df, __ = dummy_data(n=1000, p_noise=100, bias=-6, w_age=5., w_plt=4.8)
print(df.shape[0], "rows in data")
print(df.shape[1], "columns in data")
print(df.cancer.sum(), "num individuals with cancer")

# Divide the data into an outcome variable (y) and predictor variables (x)
x = df.drop(labels=['cancer'], axis=1)
y = df.cancer

# Split data into "model selection (sel) and test sets, setting aside 20% as the test set
# We use a stratified split, so that the proportion of patients with cancer remains constant
x_sel, x_test, y_sel, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

# Split model selection set further into train and validation (val) sets, so that val set forms 20% of the total data
x_train, x_val, y_train, y_val = train_test_split(x_sel, y_sel, stratify=y_sel, test_size=0.25, random_state=42)

# Review number of observations in the datasets, and number of people with cancer ine ach
print(y_train.shape[0], y_val.shape[0], y_test.shape[0])
print(y_train.sum(), y_val.sum(), y_test.sum())

# Scale the input data (only relevant for logistic regression, as it helps the optimisation algorithm to converge faster)
scaler = StandardScaler()
scaler.fit(x_train)
x_train_tf = scaler.transform(x_train)
x_val_tf = scaler.transform(x_val)

scaler_sel = StandardScaler()
scaler_sel.fit(x_sel)
x_sel_tf = scaler_sel.transform(x_sel)
x_test_tf = scaler_sel.transform(x_test)


# ---- LOGISTIC REGRESSION ----

# Fit a regular logistic model on the model selection set and see how it performs on test set
# In that case we don't need the train + val split, as there's no hyperparameters to tune
# so our model selection set IS the train set
clf = LogisticRegression(penalty=None)
clf.fit(x_sel_tf, y_sel)

pred_sel = clf.predict_proba(x_sel_tf)[:, 1]
pred_test = clf.predict_proba(x_test_tf)[:, 1]

auc_sel = roc_auc_score(y_sel, pred_sel)
auc_test = roc_auc_score(y_test, pred_test)
print("auc train+val:", auc_sel)
print("auc test", auc_test)

# Look at regression coefficients
coef = clf.coef_
coef = pd.DataFrame(coef, columns=x_train.columns)
coef = coef.transpose().reset_index()
coef.columns = ['variable', 'coef']
coef['coef_abs'] = coef.coef.abs()
coef = coef.sort_values(by=['coef_abs'], ascending=False).reset_index(drop=True)
print(coef)
print("Top 10 highest coefficients:")
print(coef.iloc[0:10])
print((coef.coef == 0).sum(), "of", coef.shape[0], "coefficients are zero")


# ---- PENALISED LOGISTIC REGRESSION (LASSO) ----

# Fit penalised logistic regression with Lasso penalty and default penalty strength value of 1
clf = LogisticRegression(penalty="l1", solver="saga", C=1, max_iter=10000)
clf.fit(x_sel_tf, y_sel)

pred_sel = clf.predict_proba(x_sel_tf)[:, 1]
pred_test = clf.predict_proba(x_test_tf)[:, 1]

auc_sel = roc_auc_score(y_sel, pred_sel)
auc_test = roc_auc_score(y_test, pred_test)
print("auc train+val:", auc_sel)
print("auc test", auc_test)

# Look at regression coefficients
coef = clf.coef_
coef = pd.DataFrame(coef, columns=x_train.columns)
coef = coef.transpose().reset_index()
coef.columns = ['variable', 'coef']
coef['coef_abs'] = coef.coef.abs()
coef = coef.sort_values(by=['coef_abs'], ascending=False).reset_index(drop=True)
print(coef)
print("Top 10 highest coefficients:")
print(coef.iloc[0:10])
print((coef.coef == 0).sum(), "of", coef.shape[0], "coefficients are zero")

# Now, let's tune penalty strength
# Try different values for penalty strength lambda (note in sklearn, the parameter "C" is inverse of lambda)
result = pd.DataFrame()
lambdas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
for lam in lambdas: 
    print("penalty strength", lam)

    clf = LogisticRegression(penalty='l1', C=1/lam, max_iter=10000, solver='saga')
    clf.fit(x_train_tf, y_train)

    pred_train = clf.predict_proba(x_train_tf)[:, 1]
    pred_val = clf.predict_proba(x_val_tf)[:, 1]

    auc_train = roc_auc_score(y_train, pred_train)
    auc_val = roc_auc_score(y_val, pred_val)
    print("auc train:", auc_train, "auc val", auc_val)

    result_lam = {'lam': lam, "auc_train": auc_train, "auc_val": auc_val}
    result_lam = pd.DataFrame(result_lam, index=[0])
    result = pd.concat(objs=[result, result_lam], axis=0)

# Look at the results
# In this case, the penalties perform similarly, although lambda=0.1 achieves a slightly higher validation set score
result = result.reset_index(drop=True)
result

# Refit the model with best lambda on model selection set (train+val), evaluate on test set
lam = result.loc[result.auc_val.idxmax(), 'lam']
clf = LogisticRegression(penalty='l1', C=1/lam, max_iter=10000, solver='saga')
clf.fit(x_sel_tf, y_sel)

pred_sel = clf.predict_proba(x_sel_tf)[:, 1]
pred_test = clf.predict_proba(x_test_tf)[:, 1]

auc_sel = roc_auc_score(y_sel, pred_sel)
auc_test = roc_auc_score(y_test, pred_test)
print("auc train+val:", auc_sel)
print("auc test", auc_test)

# Look at regression coefficients
coef = clf.coef_
coef = pd.DataFrame(coef, columns=x_train.columns)
coef = coef.transpose().reset_index()
coef.columns = ['variable', 'coef']
coef['coef_abs'] = coef.coef.abs()
coef = coef.sort_values(by=['coef_abs'], ascending=False).reset_index(drop=True)
print(coef)
print("Top 10 highest coefficients:")
print(coef.iloc[0:10])
print((coef.coef == 0).sum(), "of", coef.shape[0], "coefficients are zero")

 
# ---- PENALISED LOGISTIC REGRESSION: ELASTIC NET ----

# Fit elastic net with penalty strength 1 and alpha 0.5 (both lasso and ridge penalties get a weight of 0.5)
clf = LogisticRegression(penalty="elasticnet", solver="saga", C=1, l1_ratio=0.5, max_iter=10000)
clf.fit(x_sel_tf, y_sel)

pred_sel = clf.predict_proba(x_sel_tf)[:, 1]
pred_test = clf.predict_proba(x_test_tf)[:, 1]

auc_sel = roc_auc_score(y_sel, pred_sel)
auc_test = roc_auc_score(y_test, pred_test)
print("auc train+val:", auc_sel)
print("auc test", auc_test)

# Look at regression coefficients
coef = clf.coef_
coef = pd.DataFrame(coef, columns=x_train.columns)
coef = coef.transpose().reset_index()
coef.columns = ['variable', 'coef']
coef['coef_abs'] = coef.coef.abs()
coef = coef.sort_values(by=['coef_abs'], ascending=False).reset_index(drop=True)
print(coef)
print("Top 10 highest coefficients:")
print(coef.iloc[0:10])
print((coef.coef == 0).sum(), "of", coef.shape[0], "coefficients are zero")

# Let's tune parameters of elastic net using sequential model-based optimisation
def objective(trial):
    C = trial.suggest_float("C", 1e-5, 1e5, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
    clf = LogisticRegression(penalty='elasticnet', C=C, l1_ratio=l1_ratio, max_iter=10000, solver='saga')
    clf.fit(x_train_tf, y_train)
    pred_val = clf.predict_proba(x_val_tf)[:, 1]
    return roc_auc_score(y_val, pred_val)  # validation AUC
 
# Run the hyperparameter search
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100)
print("Best hyperparameters:", study.best_params)
print("Best penalty strength lambda (1/C):", 1 / study.best_params["C"])
print("Best validation AUC:", study.best_value)
 
# Refit elastic net with the best hyperparameters on model selection set (train+val), evaluate on test set
# The test set is scaled with the scaler fitted on the model selection set, to match the data the model is fitted on
clf = LogisticRegression(penalty='elasticnet', **study.best_params, max_iter=10000, solver='saga')
clf.fit(x_sel_tf, y_sel)
 
pred_sel = clf.predict_proba(x_sel_tf)[:, 1]
pred_test = clf.predict_proba(x_test_tf)[:, 1]
 
auc_sel = roc_auc_score(y_sel, pred_sel)
auc_test = roc_auc_score(y_test, pred_test)
print(f"auc sel: {auc_sel}, auc test: {auc_test}")
 
# Look at regression coefficients (some may be exactly zero due to the L1 part of the penalty)
coef = clf.coef_
coef = pd.DataFrame(coef, columns=x_train.columns)
coef = coef.transpose().reset_index()
coef.columns = ['variable', 'coef']
coef['coef_abs'] = coef.coef.abs()
coef = coef.sort_values(by=['coef_abs'], ascending=False).reset_index(drop=True)
print(coef)
print("Top 10 highest coefficients:")
print(coef.iloc[0:10])
print((coef.coef == 0).sum(), "of", coef.shape[0], "coefficients are zero")


# ---- EXPLAINABLE BOOSTING MACHINE ----

# Fit EBM with default settings on train+val data
# EBM does not require data to be scaled beforehand (i.e. it does not improve optimisation speed)
clf = ExplainableBoostingClassifier()
clf.fit(x_sel, y_sel)

pred_sel = clf.predict_proba(x_sel)[:, 1]
pred_test = clf.predict_proba(x_test)[:, 1]

auc_sel = roc_auc_score(y_sel, pred_sel)
auc_test = roc_auc_score(y_test, pred_test)
print("auc train+val:", auc_sel)
print("auc test", auc_test)

# Visualise shape functions
show(clf.explain_global())

# Look at feature importances
coef = clf.term_importances().reshape(1, -1)
coef = pd.DataFrame(coef, columns=clf.term_names_)
coef = coef.transpose().reset_index()
coef.columns = ['variable', 'importance']
coef = coef.sort_values(by=['importance'], ascending=False).reset_index(drop=True)
print(coef)
print("Top 10 most important variables:")
print(coef.iloc[0:10])


# Use sequential model optimisation to find best hyperparameters for EBM
def objective(trial):
    params = {
        "max_bins": trial.suggest_categorical("max_bins", [128, 1024]),
        "max_interaction_bins": trial.suggest_categorical("max_interaction_bins", [16, 64]),
        "interactions": trial.suggest_categorical("interactions", [0, 10]),
        "reg_alpha": trial.suggest_categorical("reg_alpha", [0, 0.01, 0.1, 1, 10]),
        "reg_lambda": trial.suggest_categorical("reg_lambda", [0, 0.01, 0.1, 1, 10]),
        "max_leaves": trial.suggest_categorical("max_leaves", [2, 3]),
        "learning_rate": trial.suggest_float("learning_rate", 0.0025, 0.2, log=True),
        "validation_size": trial.suggest_categorical("validation_size", [0.15, 0.3]),
    }
    clf = ExplainableBoostingClassifier(**params)
    clf.fit(x_train, y_train)
    pred_val = clf.predict_proba(x_val)[:, 1]
    return roc_auc_score(y_val, pred_val)  # validation AUC
 
 
# Run the hyperparameter search
# EBM is slower to fit than XGBoost, so fewer trials are used; increase n_trials if time allows
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50)
best_params = study.best_params
print("Best hyperparameters:", best_params)
print("Best validation AUC:", study.best_value)
 
# Refit EBM with the best hyperparameters on train + val set
clf = ExplainableBoostingClassifier(**best_params)
clf.fit(x_sel, y_sel)
 
# Make predictions on training+validation and test data
pred_sel = clf.predict_proba(x_sel)[:, 1]
pred_test = clf.predict_proba(x_test)[:, 1]
 
# Evaluate
auc_sel = roc_auc_score(y_sel, pred_sel)
auc_test = roc_auc_score(y_test, pred_test)
print(f"auc sel: {auc_sel}, auc test: {auc_test}")
 
# Visualise shape functions of the tuned model
show(clf.explain_global())

# Look at feature importances
coef = clf.term_importances().reshape(1, -1)
coef = pd.DataFrame(coef, columns=clf.term_names_)
coef = coef.transpose().reset_index()
coef.columns = ['variable', 'importance']
coef = coef.sort_values(by=['importance'], ascending=False).reset_index(drop=True)
print(coef)
print("Top 10 most important variables:")
print(coef.iloc[0:10])
print((coef.importance == 0).sum(), "of", coef.shape[0], "coefficients are zero")


# ---- XGBoost ----

# Fit XGBoost with default settings on train+val data
# Like EBM, XGBoost does not require data to be scaled beforehand
clf = XGBClassifier()
clf.fit(x_sel, y_sel)
 
pred_sel = clf.predict_proba(x_sel)[:, 1]
pred_test = clf.predict_proba(x_test)[:, 1]
 
auc_sel = roc_auc_score(y_sel, pred_sel)
auc_test = roc_auc_score(y_test, pred_test)
print("auc train+val:", auc_sel)
print("auc test", auc_test)

# Look at feature importances
coef = clf.feature_importances_.reshape(1, -1)
coef = pd.DataFrame(coef, columns=x_train.columns)
coef = coef.transpose().reset_index()
coef.columns = ['variable', 'importance']
coef = coef.sort_values(by=['importance'], ascending=False).reset_index(drop=True)
print(coef)
print("Top 10 most important variables:")
print(coef.iloc[0:10])

 
# Use sequential model optimisation to find best hyperparameters for XGBoost
def objective_xgb(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1e3, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1e3, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 10, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
    }
    clf = XGBClassifier(**params, n_estimators=2000, eval_metric="auc", early_stopping_rounds=50)
    clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    trial.set_user_attr("n_estimators", clf.best_iteration + 1)  # store the best number of trees
    return clf.best_score  # validation AUC at the best iteration
 
 
# Run the hyperparameter search
study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(objective_xgb, n_trials=100)  # XGBoost runs quite fast so we use a higher number of trials here
best_n_estimators = study_xgb.best_trial.user_attrs["n_estimators"]
print("Best hyperparameters:", study_xgb.best_params)
print("Best number of trees:", best_n_estimators)
print("Best validation AUC:", study_xgb.best_value)
 
# Refit XGBoost with the best hyperparameters and number of trees on train + val set
# (no early stopping here, as there is no separate validation set left: the number of trees found above is used)
clf = XGBClassifier(**study_xgb.best_params, n_estimators=best_n_estimators)
clf.fit(x_sel, y_sel)
 
# Make predictions on training+validation and test data
pred_sel = clf.predict_proba(x_sel)[:, 1]
pred_test = clf.predict_proba(x_test)[:, 1]
 
# Evaluate
auc_sel = roc_auc_score(y_sel, pred_sel)
auc_test = roc_auc_score(y_test, pred_test)
print(f"auc sel: {auc_sel}, auc test: {auc_test}")

# Look at feature importances
coef = clf.feature_importances_.reshape(1, -1)
coef = pd.DataFrame(coef, columns=x_train.columns)
coef = coef.transpose().reset_index()
coef.columns = ['variable', 'importance']
coef = coef.sort_values(by=['importance'], ascending=False).reset_index(drop=True)
print(coef)
print("Top 10 most important variables:")
print(coef.iloc[0:10])
print((coef.importance == 0).sum(), "of", coef.shape[0], "coefficients are zero")
