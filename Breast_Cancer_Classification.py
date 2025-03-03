# -*- coding: utf-8 -*-
"""
Breast Cancer Classification - Data Analysis and Machine Learning Pipeline
"""

# ==============================
#  1. Import Required Libraries
# ==============================
from ucimlrepo import fetch_ucirepo, list_available_datasets
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, confusion_matrix, accuracy_score, 
                             classification_report, roc_auc_score, precision_score, 
                             recall_score, ConfusionMatrixDisplay)



# ==============================
#  2. Load Dataset from UCI
# ==============================
def load_data():
    """Fetch and preprocess the UCI breast cancer dataset."""
    # Fetch dataset
    breast_cancer = fetch_ucirepo(id=17)

    # Extract features and target
    X = breast_cancer.data.features
    y = breast_cancer.data.targets.rename(columns={"Diagnosis": "diagnosis"})  # Rename target column

    # Convert target column to binary format (M = 1, B = 0)
    y['diagnosis'] = y['diagnosis'].map({'M': 1, 'B': 0})

    # Convert numpy arrays to DataFrame, including feature names
    df = pd.DataFrame(X, columns=breast_cancer.feature_names)

    # Adding the target column to the DataFrame
    df['diagnosis'] = y

    # Remove special characters from column names and standardize format
    df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)

    # Replace numbers at the end of column names
    df.columns = df.columns.str.replace(r'1$', '_mean', regex=True)
    df.columns = df.columns.str.replace(r'2$', '_se', regex=True)
    df.columns = df.columns.str.replace(r'3$', '_worst', regex=True)

    return df



# ==============================
#  3. Exploratory Data Analysis (EDA)
# ==============================
def perform_eda(df):
    """Perform concise exploratory data analysis."""
    print("\nDataset Overview:")
    print(df.describe())

    diagnosis_colors = ['#68b377', '#d66970']
    
    # Distribution of Benign and Malignant Cases
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.pie(df['diagnosis'].value_counts(), autopct='%1.1f%%', labels=['Benign', 'Malignant'], colors=diagnosis_colors)
    
    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x='diagnosis', palette=diagnosis_colors)
    
    plt.tight_layout()
    plt.show()
    
    # Summary Statistics by Diagnosis
    print("\nAggregated Statistics by Diagnosis:")
    print(df.groupby("diagnosis").agg(["sum", "mean", "max", "min"]))

    # Scatterplot Matrix - Feature Relationships
    plt.figure(figsize=[15, 15])
    
    feature_pairs = [
        ('concavity_mean', 'radius_mean'), ('concavity_mean', 'area_mean'),
        ('concavity_mean', 'perimeter_mean'), ('concavity_mean', 'compactness_mean'),
        ('concavity_mean', 'concave_points_mean'), ('radius_mean', 'perimeter_mean'),
        ('radius_mean', 'area_mean'), ('radius_mean', 'concave_points_mean'),
        ('perimeter_mean', 'area_mean'), ('compactness_mean', 'concave_points_mean'),
        ('area_mean', 'concave_points_mean'), ('perimeter_mean', 'concave_points_mean'),
        ('compactness_mean', 'smoothness_mean')
    ]

    for idx, (x_feature, y_feature) in enumerate(feature_pairs):
        plt.subplot(5, 3, idx + 1)
        sns.scatterplot(data=df, x=x_feature, y=y_feature, hue='diagnosis', palette=diagnosis_colors)

    plt.legend(loc='upper left', title='Diagnosis')
    plt.tight_layout()
    plt.show()


# ==============================
#  4. Data Preprocessing
# ==============================
def preprocess_data(df):
    """Separate features and target, standardize features."""
    features = df.drop(columns=['diagnosis'])
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    return features, features_normalized, df['diagnosis']


# ==============================
#  5. Data Splitting
# ==============================
def split_data(X, y, test_size=0.2):
    """Slipt data."""
    return train_test_split(X, y, test_size=test_size, random_state=42)


# ==============================
#  6. Model Training Functions
# ==============================
def train_logistic_regression(X_train, y_train, penalty='l1'):
    """Hyperparameter tuning logistic_regression."""
    param_grid = {'C': np.logspace(-4, 4, 20), 'penalty': [penalty], 'solver': ['saga']}
    grid_search = GridSearchCV(LogisticRegression(max_iter=10000, random_state=42), param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters for Logistic Regression ({penalty.upper()}):", grid_search.best_params_)
    return grid_search.best_estimator_


def train_random_forest(X_train, y_train):
    """Hyperparameter tuning random_forest."""
    param_grid = {
        'n_estimators': np.arange(100, 1001, 100),
        'max_features': ['sqrt', 'log2', None],
        'max_depth': np.arange(5, 51, 5),  # Allow deeper trees  
        'min_samples_leaf': [3, 5, 10]  # Allow smaller leaves for better sensitivity
    }
    rf = GridSearchCV(RandomForestClassifier(min_samples_leaf=10, random_state=10),
                      param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1')
    rf.fit(X_train, y_train)
    print("Best Parameters for Random Forest:", rf.best_params_)
    return rf.best_estimator_


def train_boosted_trees(X_train, y_train):
    """Hyperparameter tuning boosted_trees."""
    param_grid = {
        'n_estimators': np.linspace(50, 500, 10, dtype=int),
        'learning_rate': np.logspace(-2, 0, 8),  # More granularity  
        'max_depth': np.arange(2, 15),  # Allow deeper trees  
        'max_features': ['sqrt', 'log2', None]
    }
    
    gbc = GridSearchCV(GradientBoostingClassifier(min_samples_leaf=10, random_state=1),
                       param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1')
    gbc.fit(X_train, y_train)
    print("Best Parameters for Boosted Trees:", gbc.best_params_)
    return gbc.best_estimator_


def train_bagged_trees(X_train, y_train):
    """Hyperparameter tuning bagged_trees."""
    param_grid = {
        'n_estimators': np.linspace(100, 1000, 10, dtype=int),
        'max_features': [None],  
        'min_samples_leaf': [3, 5, 10]  # Allow smaller nodes
    }

    bagging = GridSearchCV(RandomForestClassifier(random_state=42, bootstrap=True),
                           param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1')

    bagging.fit(X_train, y_train)
    print("Best Parameters for Bagged Trees:", bagging.best_params_)
    return bagging.best_estimator_


# ==============================
#  7. Model Evaluation
# ==============================
def evaluate_model(model, X_test, y_test, title):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test))

    print(f"\n{title} Model Performance")
    print("Confusion Matrix:\n", cm)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("ROC-AUC Score:", roc_auc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot()
    plt.title(f'{title} Confusion Matrix')
    plt.show()


# ==============================
#  8. Main Execution
# ==============================
if __name__ == "__main__":
    """Execute the pipeline."""
    df = load_data()  # Fetch dataset from UCI


    perform_eda(df)

    features, X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    best_modelL1 = train_logistic_regression(X_train, y_train, penalty='l1')
    best_modelL2 = train_logistic_regression(X_train, y_train, penalty='l2')
    best_rf = train_random_forest(X_train, y_train)
    best_boosted = train_boosted_trees(X_train, y_train)
    best_bagged = train_bagged_trees(X_train, y_train)

    evaluate_model(best_modelL1, X_test, y_test, title="Logistic Regression (L1)")
    evaluate_model(best_modelL2, X_test, y_test, title="Logistic Regression (L2)")
    evaluate_model(best_rf, X_test, y_test, title="Random Forest")
    evaluate_model(best_boosted, X_test, y_test, title="Boosted Trees")
    evaluate_model(best_bagged, X_test, y_test, title="Bagged Trees")
    
    
# ==============================
#  8. Results
# ==============================

################################ L1 and L2 features Comparison 

# Ensure features are already defined (Replace with your dataset feature names)
# features = df.drop(columns=['diagnosis'])  # Uncomment if 'features' isn't defined

# Define and fit feature selection models
selector_l1 = SelectFromModel(LogisticRegression(penalty='l1', solver='saga', max_iter=10000, random_state=42))
selector_l1.fit(X_train, y_train)

selector_l2 = SelectFromModel(LogisticRegression(penalty='l2', solver='saga', max_iter=10000, random_state=42))
selector_l2.fit(X_train, y_train)

# Extract selected feature names
l1_features = list(features.columns[selector_l1.get_support()])
l2_features = list(features.columns[selector_l2.get_support()])

# Create DataFrame for comparison
feature_set = set(l1_features + l2_features)  # Get all unique features
df_features = pd.DataFrame({"Feature": list(feature_set)})

# Assign indicators (✓ for selected, empty otherwise)
df_features["L1"] = df_features["Feature"].apply(lambda x: "✓" if x in l1_features else "")
df_features["L2"] = df_features["Feature"].apply(lambda x: "✓" if x in l2_features else "")

# Sort alphabetically for readability
df_features.sort_values("Feature", inplace=True)

# Add Total row
df_total = pd.DataFrame({"Feature": ["Total Features"], "L1": [len(l1_features)], "L2": [len(l2_features)]})
df_features = pd.concat([df_features, df_total])

# Plot table
fig, ax = plt.subplots(figsize=(6, 8))
ax.set_facecolor("green")  # Background color like the reference image
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)

# Create table
table = plt.table(cellText=df_features.values,
                  colLabels=df_features.columns,
                  cellLoc='center',
                  loc='center')

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width([0, 1, 2])  # Adjust column width

# Header styling
for key, cell in table.get_celld().items():
    row, col = key
    if row == 0:  # Header row
        cell.set_fontsize(12)
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('darkgreen')
    elif row == len(df_features) :  # Total row
        cell.set_fontsize(12)
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('darkgreen')

plt.show()



################################ Models Comparison 
def metrics(X,CV_clf):
    y_pred = CV_clf.predict(X)
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0,0]
    fp = cm[0,1]
    fn = cm[1,0]
    tp = cm[1,1]
    Accuracy=(tp+tn)/(tp+tn+fp+fn)
    Recall=tp/(tp+fn)
    Specificity=tn/(tn+fp)
    Precision=tp/(tp+fp)
    F_measure=2*tp/(2*tp+fp+fn)
    print('Accuracy=%.3f'%Accuracy)
    print('Recall%.3f'%Recall) # as the same as sensitivity
    print('Specificity=%.3f'%Specificity)
    print('Precision=%.3f'%Precision)
    print('F-measure=%.3f'%F_measure)
    return Accuracy, Recall, Specificity, Precision, F_measure



# Metrics
LR_l1_metrics = metrics(X_test,best_modelL1)
LR_l2_metrics = metrics(X_test,best_modelL2)
bagged_metrics = metrics(X_test,best_bagged)
boosted_metrics = metrics(X_test,best_boosted)
rf_metrics = metrics(X_test,best_rf)


models_metrics = {'logisticRegressionl1': [round(elem, 3) for elem in LR_l1_metrics], 
                 'loogisticRegressionl2': [round(elem, 3) for elem in LR_l2_metrics],
                 'BaggedTree' : [round(elem, 3) for elem in bagged_metrics],
                 'BoostedTree' : [round(elem, 3) for elem in boosted_metrics],
                 'RandomForest' : [round(elem, 3) for elem in rf_metrics]
                }

index=['Accuracy','Recall','Specificity','Precision', 'F-measure']
df_scores = pd.DataFrame(data = models_metrics, index=index)
ax = df_scores.plot(kind='bar', figsize = (15,6), ylim = (0.90, 1.02), 
                    color = ['gold', 'mediumturquoise', 'darkorange', 'MediumSeaGreen','pink'],
                    rot = 0, title ='Models performance (test scores)',
                    edgecolor = 'grey', alpha = 0.5)
ax.legend(loc='upper center', ncol=5, title="models")
for container in ax.containers:
    ax.bar_label(container)
plt.show()


################################ Feature Importance

boostedImportance = pd.DataFrame({'Feature':features.columns.values, 'Boosted_Tree_Importance':best_boosted.feature_importances_})
boostedImportance_non_zero = boostedImportance[boostedImportance['Boosted_Tree_Importance'] != 0].sort_values(by='Boosted_Tree_Importance' , ascending=False)
boostedImportance_nz_best= boostedImportance[boostedImportance['Boosted_Tree_Importance'] >0.0015].sort_values(by='Boosted_Tree_Importance' , ascending=False)

sns.barplot(x='Boosted_Tree_Importance', y='Feature',data=boostedImportance_nz_best);
plt.title("Boosted Tree - Feature Importance > 0")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()



RFImportance = pd.DataFrame({'Feature':features.columns.values, 'Random_Forest_Importance':best_rf.feature_importances_})
RFImportance_non_zero = RFImportance[RFImportance['Random_Forest_Importance'] != 0].sort_values(by='Random_Forest_Importance' , ascending=False)
RFImportance_nz_best=RFImportance[RFImportance['Random_Forest_Importance'] >0.0015].sort_values(by='Random_Forest_Importance' , ascending=False)

sns.barplot(x='Random_Forest_Importance', y='Feature',data=RFImportance_nz_best);
plt.title("Random Forest - Feature Importance > 0")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

baggedImportance = pd.DataFrame({'Feature':features.columns.values, 'Bagged_Tree_Importance':best_bagged.feature_importances_})
baggedImportance_non_zero = baggedImportance[baggedImportance['Bagged_Tree_Importance'] != 0].sort_values(by='Bagged_Tree_Importance' , ascending=False)
baggedImportance_nz_best = baggedImportance[baggedImportance['Bagged_Tree_Importance'] >0.0015].sort_values(by='Bagged_Tree_Importance' , ascending=False)

sns.barplot(x='Bagged_Tree_Importance', y='Feature',data=baggedImportance_nz_best);
plt.title("Bagged Tree - Feature Importance > 0")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


