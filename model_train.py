# model_train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("your_dataset.csv")  # Change filename

# Replace missing values
data.replace('?', 'Unknown', inplace=True)

# Use only selected columns
selected_columns = [
    'age', 'gender', 'native-country', 'occupation',
    'marital-status', 'workclass', 'education',
    'hours-per-week', 'income'
]
data = data[selected_columns]

# Label Encoding for categorical features
label_encoders = {}
for column in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Separate features and target
X = data.drop('income', axis=1)
y = data['income']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and encoders
with open('salary_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
