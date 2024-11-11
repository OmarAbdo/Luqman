# import itertools
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (
#     LSTM,
#     Dense,
#     Dropout,
#     Input,
#     Bidirectional,
#     BatchNormalization,
# )
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.optimizers import Adam
# from lstm_trainer_class import LSTMModelTrainer


# class LSTMHypParameterSearch:
#     def __init__(self, ticker, epochs=10, batch_size=32):
#         self.ticker = ticker
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.best_accuracy = -np.inf
#         self.best_config = None
#         self.combinations = self.generate_combinations()

#     def generate_combinations(self):
#         # Define hyperparameter grid
#         bidirectional_options = [True, False]
#         lstm_layer_options = [1, 2]
#         neurons_options = [[200], [100], [200, 100], [100, 50]]
#         activation_options = ["relu", "tanh"]
#         l2_regularizer_options = [True, False]
#         dropout_rate_options = [0.1, 0.2, 0.3]
#         batch_normalization_options = [True, False]

#         # Generate all possible combinations of hyperparameters
#         return list(
#             itertools.product(
#                 bidirectional_options,
#                 lstm_layer_options,
#                 neurons_options,
#                 activation_options,
#                 l2_regularizer_options,
#                 dropout_rate_options,
#                 batch_normalization_options,
#             )
#         )

#     def build_model(self, config, X_train_shape):
#         (
#             bidirectional,
#             lstm_layers,
#             neurons,
#             activation,
#             l2_regularizer,
#             dropout_rate,
#             batch_normalization,
#         ) = config

#         model = Sequential()
#         model.add(Input(shape=(X_train_shape[1], X_train_shape[2])))

#         for i in range(lstm_layers):
#             lstm_layer = LSTM(
#                 neurons[i],
#                 activation=activation,
#                 return_sequences=(i < lstm_layers - 1),
#                 kernel_regularizer=l2(0.001) if l2_regularizer else None,
#             )
#             if bidirectional:
#                 model.add(Bidirectional(lstm_layer))
#             else:
#                 model.add(lstm_layer)

#             if batch_normalization:
#                 model.add(BatchNormalization())
#             model.add(Dropout(dropout_rate))

#         model.add(Dense(1))
#         model.compile(optimizer=Adam(learning_rate=0.0001), loss="mse")
#         return model

#     def run_search(self):
#         # Loop through each combination
#         for config in self.combinations:
#             print(f"Trying configuration: {config}")
#             try:
#                 # Prepare the LSTMModelTrainer with the current configuration
#                 trainer = LSTMModelTrainer(
#                     self.ticker, sample_fraction=1.0, external_model=None
#                 )

#                 # Build and assign the model with the given configuration
#                 trainer.model = self.build_model(config, trainer.X_train.shape)

#                 # Train the model
#                 trainer.train_model(epochs=self.epochs, batch_size=self.batch_size)

#                 # Evaluate the model
#                 _, directional_accuracy = trainer.calculate_accuracies()
#                 print(f"Directional Accuracy: {directional_accuracy}%")

#                 # Update best configuration if current is better
#                 if directional_accuracy > self.best_accuracy:
#                     self.best_accuracy = directional_accuracy
#                     self.best_config = config

#             except Exception as e:
#                 print(f"Error with configuration {config}: {e}")

#         print(
#             f"Best Configuration: {self.best_config} with Directional Accuracy: {self.best_accuracy}%"
#         )


# if __name__ == "__main__":
#     ticker = os.getenv("TICKER")
#     search = LSTMHypParameterSearch(ticker)
#     search.run_search()
