import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv('AHMAD ZAKARIYA - ML_Internship_dataset_One.csv')

id_col = 'Legacy_Customer_ID'
text_col = 'Customer_Feedback'
target_col = 'Target'
numeric_cols = ['Age', 'Annual_Income($)', 'Credit_Score', 'CLV_Score', 'Complaint_Count']
categorical_cols = ['Employment_Type', 'Education_Level', 'Region', 'Account_Type', 'Contact_Preference', 'Subscription_Tier']

raw_df = df.dropna(subset=numeric_cols + categorical_cols + [target_col])
X_raw = raw_df.drop(columns=[id_col, text_col, target_col])
y_raw = raw_df[target_col]

ord_enc = OrdinalEncoder()
X_raw[categorical_cols] = ord_enc.fit_transform(X_raw[categorical_cols])

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_raw, y_raw, test_size=0.3, stratify=y_raw, random_state=42
)

clf_raw = RandomForestClassifier(random_state=42)
clf_raw.fit(X_train_r, y_train_r)
y_pred_r = clf_raw.predict(X_test_r)
y_prob_r = clf_raw.predict_proba(X_test_r)[:, 1]

X = df.drop(columns=[id_col, text_col, target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

clf_pre = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
clf_pre.fit(X_train, y_train)
y_pred_p = clf_pre.predict(X_test)
y_prob_p = clf_pre.predict_proba(X_test)[:, 1]

def compute_metrics(y_true, y_pred, y_prob):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_prob)
    }


metrics_raw = compute_metrics(y_test_r, y_pred_r, y_prob_r)
metrics_pre = compute_metrics(y_test, y_pred_p, y_prob_p)

results = pd.DataFrame([metrics_raw, metrics_pre], index=['Raw Data Model', 'Preprocessed Model'])
print(results)
