# Root Cause Prediction Using Machine Learning

This project provides a comprehensive pipeline for predicting root causes of critical events in a system using machine learning. The pipeline processes event logs, extracts meaningful features, and builds a predictive model to classify events as root causes or not. 

## Features
- **Synthetic Data Generation:** Generates or uses existing data in CSV format for training and testing.
- **Data Preprocessing:** Handles missing values, time-based feature extraction, text vectorization, and one-hot encoding.
- **Feature Engineering:** Extracts new features like `hour`, `day_of_week`, `description_length`, `notes_length`, and `time_diff` from event data.
- **Model Training:** Implements a Random Forest Classifier with hyperparameter tuning using GridSearchCV.
- **Prediction:** Predicts root causes for new event data and saves the results in a CSV file.
- **Evaluation:** Includes model performance metrics such as F1 score and classification report.

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/root-cause-prediction.git
   cd root-cause-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare your data files:
   - Training Data: Save your training data in `data.csv` (sample provided in the repository).
   - Test Data: Save your new event data in `test.csv` for predictions.

## Usage

1. **Preprocessing and Training**
Run the main script to preprocess data, extract features, train the Random Forest model, and evaluate it:

```bash
python main.py
```

This script performs:
  - Data preprocessing (handling missing values, feature engineering, and vectorization).
  - Model training with hyperparameter tuning using GridSearchCV.
  - Evaluation on a test set with accuracy and F1 score.

2. **Prediction for New Events**
After training, the script will load new event data from `test.csv`, preprocess it, and predict whether each event is a root cause. The results are saved to `predicted_test_results.csv`.

## Input and Output
**Input**
Training Data (`data.csv`) - Sample file present in the repo
New Event Data (`test.csv`) - Sample file present in the repo

**Output**
Training Logs:

Best hyperparameters for the Random Forest model.
Evaluation metrics: Accuracy, F1 score, and classification report.
Prediction Results (`predicted_test_results.csv`)


## File Structure
```bash
root-cause-prediction/
├── README.md                  # Project documentation
├── main.py                    # Main script for preprocessing, training, and prediction
├── data.csv                   # Training data (replace with your own data)
├── test.csv                   # New event data for prediction
├── predicted_test_results.csv # Output file with predictions
├── requirements.txt           # Python dependencies
```

## How It Works
1. **Data Preprocessing:**

	- Converts the `time` column into datetime format.
	- Creates additional features like `hour`, `day_of_week`, `description_length`, and `notes_length`.
	- Handles missing values in `description` and `notes` by filling them with empty strings.

2. **Feature Engineering:**

	- Uses TF-IDF Vectorization for textual columns (`description` and `notes`).
	- Applies One-Hot Encoding for categorical columns (`entity_name` and `component`).
	- Standardizes numerical features using `StandardScaler`.

3. **Model Training:**

	- Splits the dataset into training and testing sets.
	- Performs hyperparameter tuning with GridSearchCV to find the best configuration for the Random Forest model.
	- Evaluates the model on a test set using accuracy and F1 score.

4. **Prediction:**

	- Processes new event data using the same feature engineering pipeline.
	- Predicts root causes (`is_root_cause`) using the trained model.
	- Saves predictions in `predicted_test_results.csv`.

## Requirements
- Python 3.8+
- Libraries:
	- pandas
	- numpy
	- scikit-learn

Install all required libraries:
```bash
pip install -r requirements.txt
```

## Example Usage
1. Train the Model:
	```bash
	python main.py
	``` 
2. View Results: Open `predicted_test_results.csv` to see predictions for new event data.

## Contribution
Feel free to fork the repository and create pull requests to improve the pipeline or add new features.
