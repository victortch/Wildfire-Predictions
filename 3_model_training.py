import sqlite3
import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, log_loss, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import json
from bayes_opt import BayesianOptimization
from joblib import dump
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score



os.chdir("C:\\Users\\Victor\\Documents\\GeorgiaTech\\Data and Visual Analytics\\Project")
conn = sqlite3.connect('FPA_FOD_20170508.sqlite')



query = """ 
WITH GeorgiaFires AS (
    SELECT 
		DATE(f.DISCOVERY_DATE, '+0 days') as fire_date,
        m.nearest_city as city,
        m.county_name as county
    FROM Fires f
    JOIN LatLon_City_Mapping m
    ON f.LATITUDE = m.LATITUDE AND f.LONGITUDE = m.LONGITUDE
    WHERE m.state_name = 'Georgia'
)

, FireOccurrences AS (
    SELECT
        w.time as date,
        w.city,
        w.county,
        CASE WHEN g.fire_date IS NOT NULL THEN 1 ELSE 0 END as fire,
        w.* -- Fetch all weather metrics
    FROM AggregatedWeatherData w
    LEFT JOIN GeorgiaFires g
    ON w.time = g.fire_date AND w.city = g.city AND w.county = g.county
    WHERE w.state = 'Georgia'
)


-- Now, balance the dataset
, MajoritySample AS (
    SELECT * 
    FROM FireOccurrences
    WHERE fire = 0
    ORDER BY RANDOM()
    LIMIT (SELECT COUNT(*) FROM FireOccurrences WHERE fire = 1) * 2
)

SELECT * FROM MajoritySample
UNION ALL
SELECT * FROM FireOccurrences WHERE fire = 1;
    """

data = pd.read_sql(query, conn)



X = data.iloc[:, 10:176]
X = X.drop(columns=['sunrise_daily', 'sunset_daily'])
patterns_to_drop = ["soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm", "soil_moisture_100_to_255cm"]
columns_to_drop = [col for col in data.columns if any(col.startswith(pattern) for pattern in patterns_to_drop)]
X.drop(columns=columns_to_drop, inplace = True)

y = data['fire']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#XGBoost model

def xgb_evaluate(max_depth, gamma, colsample_bytree, learning_rate, n_estimators, reg_alpha, reg_lambda):
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        max_depth=int(max_depth),
        gamma=gamma,
        colsample_bytree=colsample_bytree,
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42
    )
    
    # Use cross_val_score to compute the ROC AUC for each fold
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    
    # Return the average ROC AUC score across all folds
    return scores.mean()

optimizer = BayesianOptimization(
    f=xgb_evaluate,
    pbounds={
        'max_depth': (3, 15),
        'gamma': (0, 1),
        'colsample_bytree': (0.5, 1),
        'learning_rate': (0.01, 0.2),
        'n_estimators': (100, 1000),
        'reg_alpha': (0, 1),
        'reg_lambda': (0, 1)
    },
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points=5, n_iter=50)

xgb_hyperparameters = optimizer.max['params']
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    max_depth=int(xgb_hyperparameters['max_depth']),
    gamma=xgb_hyperparameters['gamma'],
    colsample_bytree=xgb_hyperparameters['colsample_bytree'],
    learning_rate=xgb_hyperparameters['learning_rate'],
    n_estimators=int(xgb_hyperparameters['n_estimators']),
    reg_alpha=xgb_hyperparameters['reg_alpha'],
    reg_lambda=xgb_hyperparameters['reg_lambda'],
    random_state=42
)

xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict_proba(X_test)[:, 1]
xgb_roc_auc = roc_auc_score(y_test, xgb_predictions)

#Random forest model


# Function for Bayesian Optimization of Random Forest
def rf_crossval(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    """ 
    Function to compute cross-validated ROC AUC score for Random Forest.
    All parameters should be cast to integer values before model fitting.
    """
    n_estimators = int(n_estimators)
    max_depth = int(max_depth) if max_depth > 0 else None
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42 # for reproducibility
    )
    cv_scores = cross_val_score(rf, X_train, y_train, cv=3, scoring='roc_auc')
    return np.mean(cv_scores)

# Hyperparameter tuning using Bayesian Optimization
rf_bo = BayesianOptimization(
    rf_crossval,
    {
        'n_estimators': (100, 500),
        'max_depth': (0, 50),  # 0 will be treated as None (i.e., no limit)
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 4)
    },
    random_state=42
)
rf_bo.maximize(init_points=15, n_iter=25)

# Best parameters
rf_hyperparameters = rf_bo.max['params']
rf_hyperparameters['n_estimators'] = int(rf_hyperparameters['n_estimators'])
rf_hyperparameters['max_depth'] = int(rf_hyperparameters['max_depth']) if rf_hyperparameters['max_depth'] > 0 else None
rf_hyperparameters['min_samples_split'] = int(rf_hyperparameters['min_samples_split'])
rf_hyperparameters['min_samples_leaf'] = int(rf_hyperparameters['min_samples_leaf'])

# Train the Random Forest model with best parameters
rf_model = RandomForestClassifier(**rf_hyperparameters, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and ROC AUC Score for Random Forest
rf_predictions = rf_model.predict_proba(X_test)[:, 1]  # get probabilities for the positive class
rf_roc_auc = roc_auc_score(y_test, rf_predictions)


# Compare ROC AUC scores and select the best model
print(f'XGBoost ROC AUC Score: {xgb_roc_auc}')
print(f'Random Forest ROC AUC Score: {rf_roc_auc}')

if xgb_roc_auc > rf_roc_auc:
    best_model = xgb_model
    probabilities = xgb_predictions
    best_hyperparameters = xgb_hyperparameters
    best_auc = xgb_roc_auc
else:
    best_model = rf_model
    probabilities = rf_predictions
    best_hyperparameters = rf_hyperparameters
    best_auc = rf_roc_auc
    
#Chose classification threshold


thresholds = np.linspace(0, 1, 80)
best_threshold = 0.5
best_f1 = 0


for threshold in thresholds:
    predictions = (probabilities > threshold).astype(int)
    score = f1_score(y_test, predictions)
    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

print(f"Optimal Threshold: {best_threshold}")


# Test set evaluation
probabilities = best_model.predict_proba(X_test)[:, 1]
predictions = (probabilities > best_threshold).astype(int)


#Confusion Matrices

def plot_combined_cm(cm, cm_baseline, title, ax):
    # Combine the values from cm and cm_baseline
    combined_cm = np.array([f"{val}\n({baseline_val})" for val, baseline_val in zip(cm.flatten(), cm_baseline.flatten())]).reshape(cm.shape)
    
    sns.heatmap(cm, annot=combined_cm, fmt="", cmap="Blues", xticklabels=['No Fire', 'Fire'], yticklabels=['No Fire', 'Fire'], ax=ax, annot_kws={"size": 15})
    ax.set_ylabel('Actual', fontsize=16)
    ax.set_xlabel('Predicted', fontsize=16)
    ax.set_title(title, fontsize=17)
    ax.tick_params(axis='both', which='major', labelsize=15)




    
def precision_recall_f1(cm):
    precision, recall = cm[1,1]/predicted_positive, cm[1,1]/actual_positive
    f1 = (2*precision*recall)/(precision+recall)
    return precision, recall, f1
    
#Get actual cm
cm = confusion_matrix(y_test, predictions)

#Calculate baseline cm
cm_baseline = cm.copy()
total = y_test.count()
predicted_positive = predictions.sum()
predicted_negative = total - predicted_positive
actual_positive = y_test.sum()
actual_negative = total - actual_positive
cm_baseline[0,0] = int(((actual_negative/total)*(predicted_negative/total))*total)
cm_baseline[1,1] = int(((actual_positive/total)*(predicted_positive/total))*total)
cm_baseline[0,1] = int(((actual_negative/total)*(predicted_positive/total))*total)
cm_baseline[1,0] = int(((actual_positive/total)*(predicted_negative/total))*total)


fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the size as needed
plot_combined_cm(cm, cm_baseline, 'Confusion Matrix - Model (with Baseline Comparison)', ax)

#Compare metrics
precision, recall, f1 = precision_recall_f1(cm)
baseline_precision, baseline_recall, baseline_f1 = precision_recall_f1(cm_baseline)




metrics = ['Precision', 'Recall', 'F1 Score']
model_scores = [precision, recall, f1]
baseline_scores = [baseline_precision, baseline_recall, baseline_f1]

x = range(len(metrics))

plt.figure()
model_bars = plt.bar(x, model_scores, width=0.4, label='Model', align='center')
baseline_bars = plt.bar(x, baseline_scores, width=0.4, label='Baseline', align='edge')

plt.xticks(x, metrics)
plt.ylabel('Scores')
plt.title('Model vs Baseline')
plt.legend()

# Adding numeric labels to bars
def add_labels(bars):
    for bar in bars:
        yval = bar.get_height()-0.06
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

add_labels(model_bars)
add_labels(baseline_bars)

plt.show()



#ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#Precision-Recall Curve
positive_class_proportion = np.mean(y_train)
baseline_probabilities = np.full_like(y_test, fill_value=positive_class_proportion, dtype=float)

precision, recall, _ = precision_recall_curve(y_test, probabilities)
average_precision = average_precision_score(y_test, probabilities)
baseline_precision, baseline_recall, _ = precision_recall_curve(y_test, baseline_probabilities)
baseline_average_precision = average_precision_score(y_test, baseline_probabilities)

plt.figure()
plt.step(recall, precision, where='post', label='Model AP = %0.2f' % average_precision, color='blue')
plt.step(baseline_recall, baseline_precision, where='post', label='Baseline AP = %0.2f' % baseline_average_precision, color='red')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")
plt.show()



# Feature importances
feature_importances = best_model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)


top_features_df = importance_df.head(15)

plt.figure(figsize=(12, 8))
sns.barplot(data=top_features_df, x='Importance', y='Feature')
plt.title('Top 15 Most Important Features', fontsize=18)
plt.xlabel('Importance', fontsize=17)
plt.ylabel('Feature', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()



#Export relevant files

best_model.save_model("best_model.xgb")
dump(best_model, 'best_model.joblib')

best_threshold_json = json.dumps(best_threshold)
cm_json = json.dumps(cm.tolist())  # Convert np.array to list first
importance_df_json = importance_df.to_json(orient="split")

with open("best_threshold.json", "w") as file:
    file.write(best_threshold_json)

with open("cm.json", "w") as file:
    file.write(cm_json)

with open("importance_df.json", "w") as file:
    file.write(importance_df_json)
    
with open("best_hyperparameters.json", "w") as file:
    json.dump(best_hyperparameters, file)

loss = log_loss(y_test, probabilities)
baseline_loss = log_loss(y_test, baseline_probabilities)

print(f"Log Loss of the best model: {loss:.4f}")
print(f"AUC of the best model: {best_auc:.4f}")

