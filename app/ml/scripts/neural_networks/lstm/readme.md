# LSTM Model Explanation

This README provides a detailed explanation of the **LSTM (Long Short-Term Memory)** model used in the code, specifically focusing on the choices and architecture implemented, the meaning behind each component, and answers to various questions related to LSTM networks.

## LSTM Model Architecture Overview
The method `build_model()` in the code constructs an LSTM model using **Keras**. Below is an in-depth explanation of each part of the model:

### Code Block Breakdown

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

### Detailed Explanation
1. **Model Initialization**:
   ```python
   self.model = Sequential()
   ```
   - Initializes a **Sequential model**, which is a linear stack of layers where each layer has exactly one input tensor and one output tensor.
   - This is ideal when the model architecture has one input and one output, with each layer connected in a linear sequence.

2. **Input Layer**:
   ```python
   self.model.add(Input(shape=(self.X_train.shape[1], self.X_train.shape[2])))
   ```
   - Adds an **Input layer** specifying the shape of the input data.
   - `shape=(self.X_train.shape[1], self.X_train.shape[2])`:
     - `self.X_train.shape[1]`: The **sequence length** (number of time steps).
     - `self.X_train.shape[2]`: The **number of features** per time step.
   - This defines the expected input shape for the LSTM, which is a **3D tensor** of shape `(batch_size, sequence_length, features)`.

3. **First LSTM Layer**:
   ```python
   self.model.add(LSTM(50, activation="relu", return_sequences=True))
   ```
   - Adds an **LSTM layer** with **50 units (neurons)**.
   - `activation="relu"`: Uses the **ReLU (Rectified Linear Unit)** activation function, which helps the model learn complex patterns effectively.
   - `return_sequences=True`: Tells the LSTM to return the entire output sequence, which is required as another LSTM layer follows this one.

4. **First Dropout Layer**:
   ```python
   self.model.add(Dropout(0.2))
   ```
   - Adds a **Dropout layer** with a rate of **0.2** (20%).
   - Dropout helps prevent **overfitting** by randomly setting 20% of the input units to zero during training.

5. **Second LSTM Layer**:
   ```python
   self.model.add(LSTM(50, activation="relu"))
   ```
   - Adds a second **LSTM layer** with **50 units** and uses the **ReLU** activation function.
   - `return_sequences=False` (default value) means only the final output of this LSTM layer is returned, not the entire sequence.

6. **Second Dropout Layer**:
   ```python
   self.model.add(Dropout(0.2))
   ```
   - Adds another **Dropout layer** with a rate of **0.2** to help prevent overfitting.

7. **Dense Layer**:
   ```python
   self.model.add(Dense(1))
   ```
   - Adds a **Dense (fully connected) layer** with **1 unit** to generate the final output.
   - This is ideal for regression problems like predicting stock prices.

8. **Compiling the Model**:
   ```python
   self.model.compile(optimizer="adam", loss="mse")
   ```
   - **Compile**s the model to prepare it for training.
   - `optimizer="adam"`: Uses the **Adam optimizer**, which is adaptive and helps with faster and more stable convergence.
   - `loss="mse"`: Uses **Mean Squared Error (MSE)** as the loss function, which is commonly used for regression tasks.

## Common Questions about LSTMs

### 1. **Bidirectional LSTMs**
The provided code does **not** use a bidirectional LSTM. To make a model bidirectional, you would need to use the **`Bidirectional`** wrapper in Keras:
```python
from tensorflow.keras.layers import Bidirectional
self.model.add(Bidirectional(LSTM(50, activation="relu")))
```
Bidirectional LSTMs read sequences both forward and backward, helping the model capture dependencies that a unidirectional LSTM might miss.

### 2. **Number of Units (Neurons)**
- When specifying **50 neurons** in an LSTM layer, it means that there are **50 units** (also called cells) within that LSTM layer.
- These units work in parallel, each with their own set of **weights**, capturing different aspects of the time-series data.
- In the provided model, there are **two hidden layers**, both with **50 units** each.

### 3. **Multiple LSTM Layers**
- Adding a **second LSTM layer** increases the **depth** of the model, allowing it to learn more abstract, higher-level features from the sequence data.
- It does **not** make the model bidirectional or double the number of layers; instead, it adds depth to the representation.

### 4. **Activation Functions**
- **ReLU**: Outputs the input if positive, otherwise returns **0**. It’s computationally efficient and reduces vanishing gradient problems.
- **Other Activation Functions**:
  - **Sigmoid**: Outputs values between **0 and 1** but suffers from **vanishing gradients**.
  - **Tanh**: Outputs between **-1 and 1**; better than sigmoid for centered data.
  - **Leaky ReLU**: Allows a small negative slope, solving the **dying ReLU** problem.
  - **Softmax**: Often used for classification to output probabilities for each class.

### 5. **Dropout**
- **Dropout** randomly sets a fraction of units to **zero** during training. This prevents the model from becoming overly reliant on specific paths, improving **generalization**.
- The dropout rate (`0.2`) is a hyperparameter that needs tuning. A higher rate may lead to **underfitting**, while a lower rate may not prevent overfitting effectively.

### 6. **Stacking LSTM Layers**
- Adding more LSTM layers allows the model to learn more complex features. Having a **third LSTM layer** could further improve performance, but it could also lead to **overfitting** if not controlled properly.

### 7. **Dense Layer Explanation**
- A **Dense layer** is a **fully connected layer** where every neuron from the previous layer is connected to every neuron in the current layer.
- It is used to combine the learned features from previous layers to produce the final prediction.
- The term "Dense" refers to the full connectivity between layers, and it is often used in **output layers** for regression or classification.

### 8. **Optimizer: Adam**
- **Adam (Adaptive Moment Estimation)** is a popular optimizer that combines the benefits of **SGD**, **RMSProp**, and **momentum**.
- **Adam** adapts the learning rate for each parameter, making it efficient and faster to converge compared to simpler optimizers like **SGD**.
- **Other Optimizers**:
  - **SGD**: Simple, but may converge slowly.
  - **RMSProp**: Suitable for non-stationary objectives.
  - **AdaGrad**: Adapts learning rates for each parameter, especially beneficial for sparse data.

### 9. **Loss Function: Mean Squared Error (MSE)**
- **MSE** measures the average squared difference between predicted and actual values.
- It is ideal for regression problems where you want to minimize the prediction error.
- **Other Loss Functions**:
  - **Mean Absolute Error (MAE)**: Measures average absolute differences; less sensitive to outliers compared to MSE.
  - **Huber Loss**: A combination of **MSE** and **MAE**, robust to outliers.

## Good Practices for LSTMs

### Unequal Neurons in Different Layers
- Having **different numbers of neurons** in different layers is a good practice as it enables **hierarchical learning**.
- **Deeper layers** often have fewer neurons to reduce model complexity, improve **generalization**, and avoid **overfitting**.
- The provided model has **two LSTM layers** with **50 units** each. You could experiment with different numbers of neurons in each layer to see if performance improves.

### Hyperparameters of LSTMs
- **Network Architecture**:
  - Number of LSTM Layers: **2**.
  - Number of Units per LSTM Layer: **50**.
  - Activation Function: **ReLU**.
  - Return Sequences: `True` for the first LSTM, `False` for the second.
  - Dropout Rate: **0.2**.
- **Training Process**:
  - Optimizer: **Adam**.
  - Learning Rate: Default **0.001** for Adam.
  - Loss Function: **MSE**.
  - Batch Size: **32**.
  - Number of Epochs: **50**.

### Summary
- The LSTM model provided is well-designed for **time-series forecasting** tasks such as **stock price prediction**.
- The **two LSTM layers** work together to extract both detailed and abstract temporal relationships, while **dropout layers** help mitigate overfitting.
- The model’s architecture, hyperparameters, and training process can be fine-tuned to further improve its performance.

Feel free to experiment with different activation functions, optimizers, and hyperparameter settings to see how they affect model performance!

