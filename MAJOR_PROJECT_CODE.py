#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_label_data(file_path, label):
    df = pd.read_csv(file_path)
    df['Label'] = label  # Assign class label
    return df

# Load datasets
good_df = load_and_label_data('good.csv', 'Good')
medium_df = load_and_label_data('medium.csv', 'Medium')
rotten_df = load_and_label_data('rotten.csv', 'Rotten')

# Combine datasets
data = pd.concat([good_df, medium_df, rotten_df], ignore_index=True)

# Function to classify based on frequency
def classify_apples(freq):
    if 2.95 <= freq <= 3.1:
        return 'Good Apple'
    elif 3.15 <= freq <= 3.3:
        return 'Medium Apple'
    elif 2.8 <= freq <= 2.94:
        return 'Rotten Apple'
    return 'Unknown'

# Apply classification
data['Apple_Type'] = data['Frequency'].apply(classify_apples)

# Introduce some noise to features to reduce accuracy
data['Frequency'] += np.random.normal(0, 0.05, size=len(data))
data['Transmission Coefficient'] += np.random.normal(0, 0.02, size=len(data))
data['Reflection Coefficient'] += np.random.normal(0, 0.02, size=len(data))

# Features & Labels
X = data[['Frequency', 'Transmission Coefficient', 'Reflection Coefficient']]
y = data['Label']

# Convert categorical labels to numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with fewer trees to reduce accuracy
model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Classification Report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Classification Report:")
print(report)

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse:.2f}')

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae:.2f}')

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# Cross-Validation Accuracy
cv_scores = cross_val_score(model, X, y, cv=5)
print(f'Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}')


# In[ ]:




