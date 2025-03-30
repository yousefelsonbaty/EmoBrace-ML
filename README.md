# Emotion Recognition from Physiological Signals

## Overview

This project demonstrates emotion recognition using physiological signals like temperature, electrocardiogram (ECG), and galvanic skin response (GSR). It involves data preprocessing, model comparison, hyperparameter tuning, and deployment.

## Dataset

The project uses a dataset named `sensor_data.csv`, containing physiological sensor readings and corresponding emotion labels. The dataset should have the following columns:

- Timestamp
- Temperature
- ECG
- GSR
- Emotion

## Steps

1. **Data Preprocessing:**
   - Loads the dataset from `sensor_data.csv`.
   - Drops the `Timestamp` column.
   - Encodes emotion labels using Label Encoding.
   - Standardizes physiological features (Temperature, ECG, GSR) using StandardScaler.
   - Splits the data into training and testing sets.

2. **Model Comparison and Automatic Selection:**
   - Compares five classification models: Random Forest, SVM, KNN, Logistic Regression, and XGBoost.
   - Evaluates each model using accuracy, precision, recall, and F1-score.
   - Selects the best model based on the F1-score.

3. **Hyperparameter Tuning of the Best Model:**
   - Performs hyperparameter tuning on the selected best model using GridSearchCV.
   - Optimizes the model's performance by finding the best hyperparameter combination.

4. **Final Model Training and Deployment:**
   - Trains the final model with the best hyperparameters on the entire training set.
   - Evaluates the final model on the test set.
   - Saves the final model and scaler using `joblib` for deployment.

## Dependencies

- Python 3.x
- pandas
- scikit-learn
- TensorFlow/Keras
- seaborn
- matplotlib
- joblib

## Usage

1. Install the required dependencies.
2. Place the `sensor_data.csv` file in the same directory as the code.
3. Execute the code in a Jupyter Notebook or Python environment.
4. The trained model will be saved as `final_emotion_model.pkl` and the scaler as `scaler.pkl`.

## Deployment

The saved model and scaler can be used to predict emotions from new physiological data. Load the model and scaler using `joblib.load()` and preprocess the new data using the same steps as in the training pipeline before making predictions.
