# Documentation for Feature Engineering Module

## Overview

The **Feature Engineering** module contains classes that prepare and transform raw data into a format suitable for machine learning models. This process involves cleaning, splitting, scaling, engineering features, and saving processed data. Below is a detailed documentation of each class in the `feature_engineering` folder.

---

## 1. `DataLoader`

### Purpose
Loads raw data from a specified file into a Pandas DataFrame.

### Methods

#### `__init__(self, input_file: str)`
- **Parameters**:
  - `input_file`: Path to the CSV file containing raw data.

#### `load_data(self) -> pd.DataFrame`
- **Description**: Loads the CSV file and returns it as a Pandas DataFrame.
- **Raises**:
  - `FileNotFoundError` if the file does not exist.

### Example Usage
```python
from feature_engineering.data_loader import DataLoader

loader = DataLoader(input_file="path/to/your/data.csv")
data = loader.load_data()
```

---

## 2. `DataCleaner`

### Purpose
Cleans the data by handling missing values and outliers.

### Methods

#### `identify_columns(self, data: pd.DataFrame)`
- **Description**: Identifies columns by data type (numeric, categorical, boolean, datetime).
- **Returns**: Tuple containing lists of column names.

#### `handle_missing_values(self, data: pd.DataFrame, numeric_columns: list, boolean_columns: list, categorical_columns: list) -> pd.DataFrame`
- **Description**: Imputes missing values for numeric, boolean, and categorical columns.

#### `handle_outliers(self, data: pd.DataFrame, numeric_columns: list) -> pd.DataFrame`
- **Description**: Clips outliers in numeric columns using the IQR method.

### Example Usage
```python
from feature_engineering.data_cleaner import DataCleaner

cleaner = DataCleaner()
columns = cleaner.identify_columns(data)
cleaned_data = cleaner.handle_missing_values(data, *columns)
cleaned_data = cleaner.handle_outliers(cleaned_data, columns[0])
```

---

## 3. `FeatureEngineer`

### Purpose
Transforms raw data by engineering additional features, such as datetime components and one-hot encoding categorical columns.

### Methods

#### `convert_boolean(self, data: pd.DataFrame, boolean_columns: list) -> pd.DataFrame`
- **Description**: Converts boolean columns to binary format (0/1).

#### `one_hot_encode(self, data: pd.DataFrame, categorical_columns: list) -> pd.DataFrame`
- **Description**: Applies one-hot encoding to categorical columns.

#### `handle_datetime(self, data: pd.DataFrame, datetime_column: str) -> pd.DataFrame`
- **Description**: Extracts components from a datetime column (year, month, day, etc.) and normalizes them.

### Example Usage
```python
from feature_engineering.feature_engineer import FeatureEngineer

engineer = FeatureEngineer()
data = engineer.handle_datetime(data, "timestamp")
data = engineer.convert_boolean(data, boolean_columns)
data = engineer.one_hot_encode(data, categorical_columns)
```

---

## 4. `DataScaler`

### Purpose
Scales numeric columns using MinMaxScaler, StandardScaler, or log scaling.

### Methods

#### `__init__(self, scaler_directory: str)`
- **Parameters**:
  - `scaler_directory`: Directory to save scaler files.

#### `apply_log_scaling(self, data: pd.DataFrame, column: str) -> pd.DataFrame`
- **Description**: Applies logarithmic scaling to a column.

#### `standardize_columns(self, data: pd.DataFrame, columns: list, scaler_type: str = "minmax") -> pd.DataFrame`
- **Description**: Scales specified columns using the given scaler type (`minmax` or `standard`).

#### `save_scalers_info(self)`
- **Description**: Saves scaler metadata to a JSON file.

### Example Usage
```python
from feature_engineering.data_scaler import DataScaler

scaler = DataScaler(scaler_directory="path/to/scalers")
data = scaler.apply_log_scaling(data, "volume")
data = scaler.standardize_columns(data, numeric_columns, scaler_type="minmax")
scaler.save_scalers_info()
```

---

## 5. `DataSplitter`

### Purpose
Splits data into training and testing sets in chronological order to maintain temporal integrity.

### Methods

#### `__init__(self, test_size: float = 0.2)`
- **Parameters**:
  - `test_size`: Fraction of data to use as the test set (default is 20%).

#### `split_data(self, data: pd.DataFrame) -> tuple`
- **Description**: Splits data into training and testing sets.

### Example Usage
```python
from feature_engineering.data_splitter import DataSplitter

splitter = DataSplitter(test_size=0.2)
train_data, test_data = splitter.split_data(data)
```

---

## 6. `SequencePreparer`

### Purpose
Converts data into sequences for time series modeling.

### Methods

#### `__init__(self, sequence_length: int)`
- **Parameters**:
  - `sequence_length`: Length of each sequence (number of time steps).

#### `prepare_sequences(self, data: pd.DataFrame, target_column: str) -> tuple`
- **Description**: Prepares sequences of features and corresponding targets.

### Example Usage
```python
from feature_engineering.sequence_preparer import SequencePreparer

preparer = SequencePreparer(sequence_length=60)
X, y = preparer.prepare_sequences(data, target_column="close")
```

---

## 7. `DataSaver`

### Purpose
Saves processed data and metadata to specified directories.

### Methods

#### `__init__(self, output_directory: str)`
- **Parameters**:
  - `output_directory`: Directory to save processed data.

#### `save_data(self, X_train, X_test, y_train, y_test, feature_names, timestamps_train, timestamps_test)`
- **Description**: Saves training/testing data and metadata.

### Example Usage
```python
from feature_engineering.data_saver import DataSaver

saver = DataSaver(output_directory="path/to/output")
saver.save_data(X_train, X_test, y_train, y_test, feature_names, timestamps_train, timestamps_test)
```

---

## 8. `DataPlotter`

### Purpose
Visualizes data and model predictions.

### Methods

#### `plot_feature(self, data: pd.DataFrame, feature_name: str)`
- **Description**: Plots a feature over time.

#### `plot_predictions(self, timestamps, actual, predicted)`
- **Description**: Plots actual vs predicted values.

### Example Usage
```python
from feature_engineering.data_plotter import DataPlotter

plotter = DataPlotter()
plotter.plot_feature(data, feature_name="close")
plotter.plot_predictions(timestamps, actual, predicted)
```

---

## 9. `MainPipeline`

### Purpose
Orchestrates the execution of all feature engineering steps in the correct order.

### Methods

#### `run(self)`
- **Description**: Executes the pipeline: loads data, cleans it, scales it, splits it, prepares sequences, and saves processed data.

### Example Usage
```python
from feature_engineering.main_pipeline import MainPipeline

pipeline = MainPipeline(
    input_file="path/to/your/data.csv",
    output_directory="path/to/output",
    scaler_directory="path/to/scalers",
    test_size=0.2,
    sequence_length=60
)
pipeline.run()
