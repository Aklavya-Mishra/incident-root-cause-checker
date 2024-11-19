import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# DATA GENERATION

# Generate synthetic event data
data = pd.read_csv('data.csv')

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert the 'time' column to datetime
df['time'] = pd.to_datetime(df['time'])

# Fill NaN values in 'description' and 'notes' with empty strings
df['description'] = df['description'].fillna('')
df['notes'] = df['notes'].fillna('')

# Check for NaN values in the target variable
if df['is_root_cause'].isnull().any():
    print("NaN values found in 'is_root_cause'. Handling them now.")
    # Drop rows with NaN in 'is_root_cause'
    df = df.dropna(subset=['is_root_cause'])

#---------------------------------------------#

# FEATURE ENGINEERING

# Create new features based on time
df['hour'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.dayofweek

# Create features for length of description and notes
df['description_length'] = df['description'].apply(len)
df['notes_length'] = df['notes'].apply(len)

# Time difference feature
df['time_diff'] = df['time'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()

# TF-IDF vectorization for 'description' and 'notes'
vectorizer = TfidfVectorizer()
combined_text = df['description'] + ' ' + df['notes']
description_notes_tfidf = vectorizer.fit_transform(combined_text)

# One-hot encode 'entity_name' and 'component'
encoder = OneHotEncoder(sparse_output=False)
encoded_entities = encoder.fit_transform(df[['entity_name', 'component']])

# Concatenate all features into a single dataset
X = np.hstack([df[['time_diff', 'hour', 'day_of_week', 'description_length', 'notes_length']].values,
                 description_notes_tfidf.toarray(),
                 encoded_entities])

# Labels for training
y = df['is_root_cause'].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

#---------------------------------------------#

# MODEL TRAINING

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': [None, 'balanced']
}

# Initialize Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1_weighted')

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Print best parameters and best score
print("Best parameters: ", grid_search.best_params_)
print("F1 score: ", grid_search.best_score_)

# Train the model using the best parameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

#---------------------------------------------#

# MODEL EVALUATION

# Evaluate the model on the test set
accuracy = best_rf.score(X_test, y_test)
print(f"Model accuracy on the test set: {accuracy:.2f}")

# Predict labels for the test set
y_pred = best_rf.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

#---------------------------------------------#

# PREDICTIONS

# Sample new event data
new_event_data = pd.read_csv('test.csv')

# Convert new event data to DataFrame
new_df = pd.DataFrame(new_event_data)
new_df['time'] = pd.to_datetime(new_df['time'])

# Fill NaN values in new event data
new_df['description'] = new_df['description'].fillna('')
new_df['notes'] = new_df['notes'].fillna('')

# Feature engineering for the new data
new_df['hour'] = new_df['time'].dt.hour
new_df['day_of_week'] = new_df['time'].dt.dayofweek
new_df['description_length'] = new_df['description'].apply(len)
new_df['notes_length'] = new_df['notes'].apply(len)
new_df['time_diff'] = new_df['time'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()

# Transform the new data using the same vectorizer and encoder
new_combined_text = new_df['description'] + ' ' + new_df['notes']
new_description_notes_tfidf = vectorizer.transform(new_combined_text)
new_encoded_entities = encoder.transform(new_df[['entity_name', 'component']])

# Concatenate all features for new data
new_X = np.hstack([new_df[['time_diff', 'hour', 'day_of_week', 'description_length', 'notes_length']].values,
                       new_description_notes_tfidf.toarray(),
                       new_encoded_entities])

# Normalize the new features
new_X = scaler.transform(new_X)

# Predict using the trained model
new_predictions = best_rf.predict(new_X)

# Convert predictions to integers (1 and 0)
new_predictions = new_predictions.astype(int)

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame(new_predictions, columns=['predicted_is_root_cause'])

# Combine the original new_df with predictions_df
result_df = pd.concat([new_df.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)

# Save the result to a new CSV file
result_df.to_csv('predicted_test_results.csv', index=False)
print("Predicted data has been saved to 'predicted_test_results.csv'.")
