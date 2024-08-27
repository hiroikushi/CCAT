# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import math
import shap
shap.initjs()
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, precision_score, recall_score, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import pickle

# %%
# Load the dataset
dataset = pd.read_csv('./dataset3_drug_abc.csv', encoding='utf8')


# %%
# AYA group
dataset_aya = dataset[(dataset['Age'] >= 15) & (dataset['Age'] < 40)]
dataset_aya.reset_index(drop=True, inplace=True)
print(len(dataset_aya))

# %%
report_info = dataset_aya['drug_eva'] + dataset_aya['drug_evb']
print(sum(report_info == 0), sum(report_info > 0))
print(sum(report_info == 0)/len(report_info), sum(report_info > 0)/len(report_info))

# %%
# preprocess the dataset
sns.set_palette("Set2", 6)
case_info = dataset_aya.iloc[:, :130]
is_nan_df_sub = case_info.isna()

report_info = dataset_aya['drug_eva'] + dataset_aya['drug_evb']
drug_class = report_info > 0
    

# multiple cancer
case_info = case_info.filter(regex='^(?!Double).*$', axis=1)

# family history
case_info['family'] = 0
for i in range(len(case_info)):
    for j in range(1,21):
        if is_nan_df_sub.loc[i, f'Family history-{j}-cancer type(name)']:
            break
        else:
            case_info.loc[i, 'family'] = j
case_info = case_info.filter(regex='^(?!Family).*$', axis=1)

# metastasis
case_info['meta_site'] = 0
case_info['lymph'] = 0
case_info['lung'] = 0
case_info['liver'] = 0
case_info['bone'] = 0
case_info['brain'] = 0
case_info['pleura'] = 0
case_info['peritoneum'] = 0
case_info['kidney'] = 0
case_info['adrenal'] = 0
case_info['muscle'] = 0
case_info['soft'] = 0
case_info['ovary'] = 0
for i in range(len(case_info)):
    for j in range(1,16):
        if is_nan_df_sub.loc[i, f'Metastatic-{j}-Metastatic site at registration(name)']:
            break
        else:
            case_info.loc[i, 'meta_site'] = j
        
        if case_info.loc[i, f'Metastatic-{j}-Metastatic site at registration(name)'] == 'Lymph nodes / lymph vessels':
            case_info.loc[i, 'lymph'] = 1
        elif case_info.loc[i, f'Metastatic-{j}-Metastatic site at registration(name)'] == 'Lung':
            case_info.loc[i, 'lung'] = 1
        elif case_info.loc[i, f'Metastatic-{j}-Metastatic site at registration(name)'] == 'Liver':
            case_info.loc[i, 'liver'] = 1
        elif case_info.loc[i, f'Metastatic-{j}-Metastatic site at registration(name)'] == 'Bone':
            case_info.loc[i, 'bone'] = 1
        elif case_info.loc[i, f'Metastatic-{j}-Metastatic site at registration(name)'] == 'Brain':
            case_info.loc[i, 'brain'] = 1
        elif case_info.loc[i, f'Metastatic-{j}-Metastatic site at registration(name)'] == 'Pleura':
            case_info.loc[i, 'pleura'] = 1
        elif case_info.loc[i, f'Metastatic-{j}-Metastatic site at registration(name)'] == 'Peritoneum':
            case_info.loc[i, 'peritoneum'] = 1
        elif case_info.loc[i, f'Metastatic-{j}-Metastatic site at registration(name)'] == 'Kidney':
            case_info.loc[i, 'kidney'] = 1
        elif case_info.loc[i, f'Metastatic-{j}-Metastatic site at registration(name)'] == 'Adrenal glands':
            case_info.loc[i, 'adrenal'] = 1
        elif case_info.loc[i, f'Metastatic-{j}-Metastatic site at registration(name)'] == 'Muscle':
            case_info.loc[i, 'muscle'] = 1
        elif case_info.loc[i, f'Metastatic-{j}-Metastatic site at registration(name)'] == 'Soft tissue':
            case_info.loc[i, 'soft'] = 1
        elif case_info.loc[i, f'Metastatic-{j}-Metastatic site at registration(name)'] == 'Ovary / fallopian tube':
            case_info.loc[i, 'ovary'] = 1
case_info = case_info.filter(regex='^(?!Metastatic).*$', axis=1)



case_info['smoking'] = case_info['Number of years of smoking']*case_info['Number of cigarettes per day']//20

case_info['Gender(name)'] = case_info['Gender(name)'].replace({'Woman': 0, 'Man': 1})

case_info['Registration date'] = pd.to_datetime(case_info['Registration date'])
case_info['Diagnosis date'] = pd.to_datetime(case_info['Diagnosis date'])
case_info['Reg-Diag'] = (case_info['Registration date'] - case_info['Diagnosis date']).dt.days
case_info['Specimen collection date (tumor tissue)'] = pd.to_datetime(case_info['Specimen collection date (tumor tissue)'])
case_info['Reg-Spe'] = (case_info['Registration date'] - case_info['Specimen collection date (tumor tissue)']).dt.days


# Separate numerical and categorical features
num_features = ['Reg-Diag', 'Spe-Diag', 'Reg-Spe', 'Age', 'Number of years of smoking', 'Number of cigarettes per day', 'smoking', 'family', 'meta_site', 'Lung-positive rate', 'Tumor cell content']
bi_features = ['lymph', 'lung', 'liver', 'bone', 'brain', 'pleura', 'peritoneum', 'kidney', 'adrenal', 'muscle', 'soft', 'ovary', 'Gender(name)']
cat_features = [col for col in case_info.columns if col not in num_features and col not in bi_features]

# Convert categorical features to one-hot encoding
case_info = pd.get_dummies(case_info, columns=cat_features)



# %%
# Split the data into training and testing sets
X_trainval, X_test, y_trainval, y_test = train_test_split(case_info, drug_class, test_size=0.2, random_state=1941)
X_trainval.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_trainval.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)


# %%
# model training with cross-validation
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1941)
plt.figure(figsize=(6, 6))

y_pred_combined = np.zeros(len(y_test))

for i, (train, val) in enumerate(skf.split(X_trainval, y_trainval)):
    X_train = X_trainval.iloc[train]
    y_train = y_trainval.iloc[train]
    X_val = X_trainval.iloc[val]
    y_val = y_trainval.iloc[val]

    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]


    # Define the objective function for Optuna to optimize
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'boosting_type': 'gbtree',
            'eval_metric': 'auc',
            'verbosity': 0,
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'eta': trial.suggest_loguniform('eta', 1e-8, 1.0),
            'max_depth': trial.suggest_int('max_depth', 4, 9),
            'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-8, 1.0),
            'subsample': trial.suggest_loguniform('subsample', 1e-8, 1.0),
            'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 1e-8, 1.0),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
            #'enable_categorical': True,
            'random_state': 1941,
            'scale_pos_weight': scale_pos_weight
        }

        # Train the model with the current set of hyperparameters
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        # Predict the test set and calculate the AUC score
        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        return auc

    # Run the optimization with Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # Print the best set of hyperparameters found by Optuna
    print('Fold:', i)
    print('Best trial:', study.best_trial.params)
    with open('./result_aya_cv.txt', mode='a') as f:
        f.write(f'Fold: {i}\n')
        f.write(f'Best trial: {study.best_trial.params}\n')

    # Train the final model with the best set of hyperparameters
    best_params = study.best_trial.params
    final_model = xgb.XGBClassifier(**best_params, random_state=1941, verbosity=0, scale_pos_weight=scale_pos_weight, objective='binary:logistic', boosting_type='gbtree', eval_metric='auc')
    final_model.fit(X_trainval, y_trainval)

    # Save the final model
    pickle.dump(final_model, open(f'./xgb_aya_cv_{i}.pkl', 'wb'))

    # Predict the test set and calculate the AUC score
    y_pred = final_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print('AUC:', auc)
    with open('./result_aya_cv.txt', mode='a') as f:
        f.write(f'AUC: {auc}\n')

    y_pred_combined += y_pred

    # Plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label=f'Fold {i} (AUC = {auc:.3f})')

    # calculate the best threshold and corresponding recall, precision, and specificity
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    print('Best Threshold=%f' % (best_thresh))
    y_pred = np.where(y_pred > best_thresh, 1, 0)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    print('Recall=%.3f, Precision=%.3f, Specificity=%.3f' % (recall, precision, specificity))
    print('-'*50)
    with open('./result_aya_cv.txt', mode='a') as f:
        f.write(f'Best Threshold={best_thresh}\n')
        f.write(f'Recall={recall}, Precision={precision}, Specificity={specificity}\n')
        f.write('-'*50+'\n\n')



y_pred_combined /= 5
auc = roc_auc_score(y_test, y_pred_combined)
print('Combined AUC:', auc)
with open('./result_aya_cv.txt', mode='a') as f:
    f.write(f'Combined AUC: {auc}\n')

fpr, tpr, thresholds = roc_curve(y_test, y_pred_combined)
plt.plot(fpr, tpr, label=f'Combined (AUC = {auc:.3f})')

J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f' % (best_thresh))
y_pred_combined = np.where(y_pred_combined > best_thresh, 1, 0)
recall = recall_score(y_test, y_pred_combined)
precision = precision_score(y_test, y_pred_combined)
specificity = recall_score(y_test, y_pred_combined, pos_label=0)
print('Recall=%.3f, Precision=%.3f, Specificity=%.3f' % (recall, precision, specificity))
print('-'*50)
with open('./result_aya_cv.txt', mode='a') as f:
    f.write(f'Best Threshold={best_thresh}\n')
    f.write(f'Recall={recall}, Precision={precision}, Specificity={specificity}\n')
    f.write('-'*50+'\n\n')



plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig(f'./xgb_roc_aya_cv.pdf', dpi=300, bbox_inches='tight')
plt.show()


