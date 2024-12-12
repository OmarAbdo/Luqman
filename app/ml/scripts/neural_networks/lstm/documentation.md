# LSTM Model Explanation and Configuration

This document combines an in-depth explanation of the **LSTM Model** architecture, hyperparameters, and the supporting **feature engineering** pipeline that prepares data for the model.

---

## Table of Contents

1. [LSTM Model Explanation](#lstm-model-explanation)
   - [Model Architecture](#lstm-model-architecture)
   - [Key Components](#key-components)
   - [Common Questions about LSTMs](#common-questions-about-lstms)
   - [Good Practices for LSTMs](#good-practices-for-lstms)
2. [Feature Engineering Module](#feature-engineering-module)
   - [Data Preparation Classes](#data-preparation-classes)
   - [Feature Engineering and Scaling](#feature-engineering-and-scaling)
   - [Sequence Preparation](#sequence-preparation)
   - [Pipeline Execution](#pipeline-execution)
3. [Hyperparameter Configuration](#hyperparameter-configuration)
   - [Key Hyperparameters](#key-hyperparameters)
   - [Evaluation Metrics](#evaluation-metrics)
   
---

## LSTM Model Explanation

### LSTM Model Architecture

The **LSTM (Long Short-Term Memory)** model is designed to capture temporal patterns in time-series data. Below is the architecture used in this project:

```python
self.model = Sequential()
self.model.add(Input(shape=(self.X_train.shape[1], self.X_train.shape[2])))
self.model.add(LSTM(50, activation="relu", return_sequences=True))
self.model.add(Dropout(0.2))
self.model.add(LSTM(50, activation="relu"))
self.model.add(Dropout(0.2))
self.model.add(Dense(1))
self.model.compile(optimizer="adam", loss="mse")
```

### Key Components

#### Input Layer
- **Shape:** `(sequence_length, num_features)`
- Defines the 3D tensor input for the LSTM model.

#### LSTM Layers
- **First LSTM Layer:** Captures low-level temporal patterns with `return_sequences=True`.
- **Second LSTM Layer:** Extracts higher-level representations from the sequence data.

#### Dropout Layers
- Prevent overfitting by randomly disabling neurons during training.

#### Dense Layer
- Outputs a single value for regression tasks (e.g., predicting stock prices).

#### Optimizer and Loss Function
- **Optimizer:** Adam for adaptive learning rate.
- **Loss Function:** Mean Squared Error (MSE), suitable for regression.

### Common Questions about LSTMs

1. **Why use multiple LSTM layers?**
   - Multiple layers allow the model to learn both low-level and high-level temporal dependencies.

2. **What is the role of Dropout?**
   - Prevents overfitting by introducing noise during training.

3. **Why use Adam optimizer?**
   - Combines momentum and adaptive learning rates for faster convergence.

4. **What does `return_sequences` do?**
   - Ensures the LSTM outputs the full sequence, necessary for stacking layers.

### Good Practices for LSTMs

- **Sequence Length:** Adjust to capture meaningful patterns (e.g., two trading days).
- **Batch Size:** Balance computational efficiency with generalization.
- **Regularization:** Use Dropout and L2 penalties to reduce overfitting.

---

## Feature Engineering Module

### Data Preparation Classes

#### **DataLoader**
- **Purpose:** Load raw data from CSV files.
- **Methods:**
  - `load_data()`: Returns the data as a Pandas DataFrame.

#### **DataCleaner**
- **Purpose:** Handle missing values and outliers.
- **Methods:**
  - `handle_missing_values()`: Imputes missing values based on column type.
  - `handle_outliers()`: Clips outliers using the IQR method.

### Feature Engineering and Scaling

#### **FeatureEngineer**
- **Purpose:** Transform raw features (e.g., datetime, boolean, categorical).
- **Methods:**
  - `handle_datetime()`: Extracts and normalizes components from datetime fields.
  - `convert_boolean()`: Converts boolean columns to binary (0/1).
  - `one_hot_encode()`: Applies one-hot encoding to categorical fields.

#### **DataScaler**
- **Purpose:** Scale and normalize features.
- **Methods:**
  - `apply_log_scaling()`: Applies log scaling to skewed data (e.g., volume).
  - `standardize_columns()`: Scales data using MinMax or Standard Scaler.

### Sequence Preparation

#### **SequencePreparer**
- **Purpose:** Prepare data sequences for LSTM.
- **Methods:**
  - `prepare_sequences()`: Converts data into sequences of inputs (X) and targets (y).

### Pipeline Execution

The pipeline integrates the above components:

1. Load raw data using `DataLoader`.
2. Clean and preprocess data with `DataCleaner`.
3. Engineer features using `FeatureEngineer`.
4. Split data chronologically with `DataSplitter`.
5. Scale data with `DataScaler`.
6. Prepare sequences with `SequencePreparer`.

---

## Hyperparameter Configuration

### Key Hyperparameters

#### Sequence Length
- **Value:** `780`
- **Reasoning:** Covers approximately two trading days.

#### Batch Size
- **Value:** `64`
- **Reasoning:** Balances computational efficiency and gradient stability.

#### Number of Epochs
- **Value:** `50` (with early stopping).
- **Reasoning:** Ensures sufficient training while avoiding overfitting.

#### Optimizer and Learning Rate
- **Optimizer:** Adam.
- **Learning Rate:** `0.0005` (adjustable).

### Evaluation Metrics

#### Primary Metrics
- **Mean Squared Error (MSE):** Measures squared error between predicted and actual values.
- **Directional Accuracy:** Percentage of correct directional predictions (up/down).

#### Additional Metrics
- **Mean Absolute Error (MAE):** Measures average absolute differences.
- **Root Mean Squared Error (RMSE):** Square root of MSE, penalizing larger errors.
- **Mean Absolute Percentage Error (MAPE):** Error as a percentage of actual values.

---

## Conclusion

This document outlines the LSTM model architecture, the supporting feature engineering pipeline, and the hyperparameter configuration. The modular pipeline ensures reproducibility, scalability, and flexibility for experimentation.

### Next Steps

- Fine-tune hyperparameters.
- Explore additional features.
- Evaluate model performance on unseen data.

Feel free to modify the pipeline or model configurations to suit your specific use case!
