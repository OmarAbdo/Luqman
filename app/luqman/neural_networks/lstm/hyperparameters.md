# **LSTM Model Hyperparameters and Configuration Documentation**

## **Table of Contents**

1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
   - [DataLoader](#dataloader)
   - [DataCleaner](#datacleaner)
   - [DataScaler](#datascaler)
   - [FeatureEngineer](#featureengineer)
   - [DataStandardizer](#datastandardizer)
   - [LSTMDataPreparer](#lstmdatapreparer)
3. [Model Training](#model-training)
   - [LSTMModelTrainer](#lstmmodeltrainer)
4. [Hyperparameter Configuration](#hyperparameter-configuration)
   - [Sequence Length](#sequence-length)
   - [Batch Size](#batch-size)
   - [Number of Epochs](#number-of-epochs)
   - [Model Architecture](#model-architecture)
   - [Regularization Techniques](#regularization-techniques)
   - [Optimizer and Learning Rate](#optimizer-and-learning-rate)
   - [Data Splitting Logic](#data-splitting-logic)
5. [Feature Emphasis Strategies](#feature-emphasis-strategies)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Conclusion](#conclusion)

---

## **Introduction**

This document provides a detailed overview of the configurations and hyperparameters used in developing an LSTM-based time series forecasting model for stock prices. It includes the reasoning behind each decision, aiming to create a comprehensive reference for understanding and further improving the model.

---

## **Data Preparation**

### **DataLoader**

- **Purpose:** Load raw stock price data from CSV files into a DataFrame.
- **Key Functions:**
  - **Reading Data:** Utilizes `pd.read_csv` with `parse_dates` to correctly parse the `timestamp` column.
  - **Data Integrity:** Ensures that data is loaded without corruption or loss.
- **Reasoning:**
  - Accurate loading of time series data is critical for analysis and modeling.

### **DataCleaner**

- **Purpose:** Clean and preprocess the raw data to prepare it for scaling and feature engineering.
- **Key Functions:**
  - **Identifying Column Types:** Segregates columns into numeric, boolean, categorical, and datetime types.
  - **Handling Missing Values:**
    - **Numeric Columns:** Imputes missing values with the mean.
    - **Categorical Columns:** Fills missing values with a placeholder like "No Divergence" or "None."
    - **Boolean Columns:** Fills missing values with `False`.
  - **Outlier Treatment:**
    - Applies the Interquartile Range (IQR) method to detect and clip outliers in numeric columns.
- **Reasoning:**
  - **Data Integrity:** Ensures that the dataset is free of inconsistencies and errors that could affect model training.
  - **Model Performance:** Cleaning data helps improve the model's ability to learn meaningful patterns.

### **DataScaler**

- **Purpose:** Scale and normalize features to ensure that they contribute equally to the model training process.
- **Key Functions:**
  - **Log Scaling:**
    - **Volume Column:** Applies logarithmic scaling to handle skewness and reduce the impact of extreme values.
  - **Standardization:**
    - **Min-Max Scaling:** Applied to features like `RSI`, `MA_10`, `MA_50` to scale them to a [0,1] range.
    - **Standard Scaling:** Applied to features like `MACD`, `EMA_12`, `EMA_26` to center them around zero with unit variance.
  - **Saving Scalers:**
    - Stores the fitted scaler objects for each feature to allow inverse transformation after prediction.
- **Reasoning:**
  - **Consistent Scale:** Features on similar scales improve the efficiency and convergence of gradient-based optimization algorithms.
  - **Preservation of Distribution:** Appropriate scaling methods are chosen to maintain the underlying distribution characteristics of each feature.

### **FeatureEngineer**

- **Purpose:** Transform and engineer features to enhance the model's ability to learn from the data.
- **Key Functions:**
  - **Encoding Categorical Variables:**
    - Converts categorical features into numerical representations using one-hot encoding.
  - **Boolean to Binary:**
    - Transforms boolean features into binary (0 or 1) format.
  - **Datetime Features:**
    - Extracts components like year, month, day, hour, and minute.
    - Normalizes datetime components and applies one-hot encoding to features like `DayOfWeek`.
- **Reasoning:**
  - **Model Compatibility:** Neural networks require numerical input; encoding ensures all features are in the appropriate format.
  - **Feature Enrichment:** Extracting and encoding datetime components can help the model capture temporal patterns.

### **DataStandardizer**

- **Purpose:** Integrate all data preprocessing steps into a cohesive pipeline.
- **Key Functions:**
  - **Data Integration:** Combines loading, cleaning, scaling, and feature engineering processes.
  - **Data Saving:** Outputs the standardized dataset and associated metadata for model training.
- **Reasoning:**
  - **Consistency:** Ensures that data preprocessing is applied uniformly across different runs.
  - **Reproducibility:** Facilitates consistent results and simplifies troubleshooting.

### **LSTMDataPreparer**

- **Purpose:** Prepare the data sequences required for training the LSTM model.
- **Key Functions:**
  - **Loading Data:** Reads the standardized data and timestamps.
  - **Sampling Data:**
    - **Modification:** Switched from random sampling to sequential selection to maintain chronological order.
    - **Sample Rate:** Set to `1.0` to use the entire dataset.
  - **Sequence Preparation:**
    - Creates input sequences (`X`) and corresponding targets (`y`) based on the specified `sequence_length`.
  - **Saving Prepared Data:**
    - Stores the sequences and timestamps in `.npy` files for efficient loading during training.
- **Reasoning:**
  - **Chronological Integrity:** Preserving the time order is crucial for time series forecasting to prevent data leakage.
  - **Model Requirements:** LSTMs require sequential input data to learn temporal dependencies.

---

## **Model Training**

### **LSTMModelTrainer**

- **Purpose:** Train, evaluate, and make predictions with the LSTM model.
- **Key Functions:**
  - **Data Loading:**
    - Loads preprocessed sequences and timestamps.
  - **Data Sorting and Splitting:**
    - **Chronological Sorting:** Ensures data is ordered by time.
    - **Time-Based Splitting:** Splits data into training and testing sets without shuffling.
  - **Missing Value Handling:**
    - Replaces any remaining NaNs in the dataset with the mean value.
  - **Model Building:**
    - Constructs the LSTM model architecture based on specified hyperparameters.
  - **Model Training:**
    - Trains the model using the training data.
    - Incorporates callbacks for early stopping and learning rate reduction.
  - **Evaluation:**
    - Evaluates model performance on the test set.
  - **Prediction and Visualization:**
    - Makes predictions on the test data.
    - Inverse transforms the predictions to the original scale.
    - Plots actual vs. predicted values over time.
- **Reasoning:**
  - **Avoiding Data Leakage:** Time-based splitting prevents the model from learning from future data.
  - **Model Performance:** Early stopping and learning rate adjustments help prevent overfitting and improve convergence.

---

## **Hyperparameter Configuration**

### **Sequence Length**

- **Value:** `780`
- **Reasoning:**
  - **Coverage:** At a 5-minute interval, 780 time steps cover approximately two trading days.
  - **Temporal Dependencies:** A longer sequence length allows the model to capture longer-term patterns and dependencies in the data.
- **Considerations:**
  - **Computational Load:** Longer sequences increase the computational requirements and memory usage.
  - **Experimentation:** It's advisable to experiment with different sequence lengths (e.g., 390 for one trading day or 1560 for four trading days) to find the optimal balance between model performance and resource utilization.

### **Batch Size**

- **Value:** `64`
- **Reasoning:**
  - **Computational Efficiency:** Larger batch sizes can better utilize GPU resources, leading to faster computation per epoch.
  - **Stable Gradient Updates:** Provides more accurate and stable gradient estimates during training, which can aid convergence.
  - **Sequence Length Consideration:** Given the large size of each input sample due to the long sequence length, a larger batch size helps in efficient memory utilization.
- **Considerations:**
  - **Generalization:** Smaller batch sizes (e.g., 16 or 32) can sometimes improve model generalization due to the introduction of noise in gradient updates.
  - **Resource Constraints:** The batch size should be adjusted based on the available hardware to prevent memory errors.

### **Number of Epochs**

- **Value:** `50` (with early stopping)
- **Reasoning:**
  - **Sufficient Training Time:** Allows the model to learn complex patterns in the data.
  - **Early Stopping:** Monitors validation loss to prevent overfitting by stopping training when performance stops improving.
- **Considerations:**
  - **Training Time:** More epochs increase training time; early stopping mitigates unnecessary computation.
  - **Monitoring:** It's important to monitor both training and validation losses to detect overfitting or underfitting.

### **Model Architecture**

- **Layers and Configuration:**
  1. **Input Layer:**
     - Shape: `(sequence_length, num_features)`
  2. **First LSTM Layer:**
     - Units: `128`
     - Activation: `'tanh'`
     - Return Sequences: `True`
     - Regularization: `kernel_regularizer=l2(0.001)`
  3. **Dropout Layer:**
     - Rate: `0.2`
  4. **Second LSTM Layer:**
     - Units: `64`
     - Activation: `'tanh'`
     - Return Sequences: `False`
  5. **Dropout Layer:**
     - Rate: `0.2`
  6. **Dense Output Layer:**
     - Units: `1`
- **Reasoning:**
  - **Unidirectional LSTMs:**
    - Suitable for time series forecasting as they process data in a forward direction, respecting temporal causality.
  - **Layer Sizes:**
    - A higher number of units in the first layer captures complex patterns, while the second layer refines these patterns.
  - **Activation Function:**
    - `'tanh'` is commonly used in LSTMs to handle the vanishing gradient problem.
  - **Regularization:**
    - L2 regularization and dropout layers help prevent overfitting by penalizing large weights and randomly dropping units.

### **Regularization Techniques**

- **Dropout Layers:**
  - **Rate:** `0.2`
  - **Purpose:** Prevents overfitting by randomly setting a fraction of input units to 0 at each update during training.
- **L2 Regularization:**
  - **Factor:** `0.001`
  - **Purpose:** Adds a penalty term to the loss function to discourage large weights, promoting simpler models.

### **Optimizer and Learning Rate**

- **Optimizer:** `Adam`
- **Learning Rate:** `0.0005`
- **Reasoning:**
  - **Adam Optimizer:**
    - An adaptive learning rate optimizer that combines the advantages of two other extensions of stochastic gradient descent: AdaGrad and RMSProp.
  - **Lower Learning Rate:**
    - A smaller learning rate allows the model to make fine-grained updates to the weights, leading to potentially better convergence.
- **Callbacks:**
  - **EarlyStopping:**
    - **Monitor:** `'val_loss'`
    - **Patience:** `10`
    - **Restore Best Weights:** `True`
    - **Purpose:** Stops training when the validation loss doesn't improve for a specified number of epochs.
  - **ReduceLROnPlateau:**
    - **Monitor:** `'val_loss'`
    - **Factor:** `0.5`
    - **Patience:** `5`
    - **Min LR:** `1e-6`
    - **Purpose:** Reduces the learning rate when a metric has stopped improving, helping the model to converge to a better minimum.

### **Data Splitting Logic**

- **Method:** Time-based splitting without shuffling.
- **Training Set:** First 80% of the data.
- **Testing Set:** Last 20% of the data.
- **Reasoning:**
  - **Temporal Integrity:** Ensures that the model is trained on past data and tested on future data, mirroring real-world forecasting scenarios.
  - **Avoiding Data Leakage:** Prevents the model from gaining information about future data during training.

---

## **Feature Emphasis Strategies**

### **Objective**

- To have the model pay more attention to specific features believed to be significant predictors, such as divergence indicators.

### **Methods**

1. **Feature Scaling Adjustment:**
   - **Action:** Multiply divergence features by a scaling factor (e.g., `3`) to increase their numerical importance.
   - **Reasoning:** Larger numerical values can cause the model to assign more weight to these features during training.

2. **Separate Input Pathways:**
   - **Action:** Create separate input layers and pathways in the model for divergence features and other features.
   - **Reasoning:** Allows the model to learn specialized representations for important features.

3. **Attention Mechanisms:**
   - **Action:** Implement attention layers to let the model learn which features and time steps are most important.
   - **Reasoning:** Provides a dynamic way for the model to focus on relevant parts of the input data.

4. **Feature Duplication:**
   - **Action:** Duplicate important features in the dataset.
   - **Reasoning:** Increases the feature's presence in the data, potentially increasing its influence on the model.

### **Implementation Plan**

- **Start with Feature Scaling Adjustment:**
  - Simple to implement and observe its impact on model performance.
- **Monitor Model Performance:**
  - Check for overfitting or any adverse effects.
- **Iterate and Experiment:**
  - If necessary, experiment with more advanced methods like attention mechanisms.

---

## **Evaluation Metrics**

### **Primary Metrics**

1. **Mean Squared Error (MSE):**
   - **Usage:** Used as the loss function during training.
   - **Interpretation:** Measures the average squared difference between the predicted and actual values.

2. **Directional Accuracy:**
   - **Usage:** Calculates the percentage of times the model correctly predicts the direction of price movement (up or down).
   - **Interpretation:** Useful for understanding the model's ability to capture trends.

### **Additional Metrics**

1. **Mean Absolute Error (MAE):**
   - **Usage:** Measures the average absolute difference between predicted and actual values.
   - **Interpretation:** Provides a more interpretable error metric in the same units as the target variable.

2. **Root Mean Squared Error (RMSE):**
   - **Usage:** Square root of MSE.
   - **Interpretation:** Gives error in the same units as the target variable, penalizing larger errors more heavily.

3. **Mean Absolute Percentage Error (MAPE):**
   - **Usage:** Expresses the error as a percentage of the actual values.
   - **Interpretation:** Useful for understanding the error relative to the size of the actual values.

### **Reasoning**

- **Comprehensive Evaluation:** Using multiple metrics provides a better understanding of model performance from different perspectives.
- **Business Relevance:** Metrics like MAE and MAPE are more interpretable in a financial context.

---

## **Conclusion**

This document provides a comprehensive overview of the configurations and hyperparameters used in developing the LSTM model for stock price forecasting. Each parameter and decision is accompanied by the reasoning behind it, ensuring transparency and facilitating future adjustments.

### **Key Takeaways**

- **Batch Size:** A larger batch size of 64 is used to balance computational efficiency and stable training, given the long sequence length and model complexity.
- **Model Architecture:** Unidirectional LSTMs are chosen to respect temporal causality, with layers and units configured to capture complex patterns without overfitting.
- **Data Handling:** Time-based data splitting and sequential data preparation are critical to prevent data leakage and maintain the integrity of the time series.
- **Feature Emphasis:** Strategies are implemented to enhance the model's focus on important features, starting with feature scaling adjustments.
- **Evaluation:** Multiple metrics are used to evaluate the model comprehensively, ensuring that it performs well in both error minimization and trend prediction.

### **Next Steps**

- **Model Training and Evaluation:**
  - Train the model with the current configurations.
  - Evaluate performance using the outlined metrics.

- **Hyperparameter Tuning:**
  - Experiment with different batch sizes, sequence lengths, and learning rates based on observed performance.

- **Feature Engineering:**
  - Further explore feature emphasis strategies and consider incorporating additional relevant features.

- **Monitoring and Iteration:**
  - Continuously monitor training and validation metrics.
  - Iterate on the model configuration to improve performance.

---

**Note:** The choices made in this configuration are based on the current understanding and assumptions about the data and the problem. It's important to remain flexible and adjust the configurations as new insights are gained through experimentation and analysis.

---
