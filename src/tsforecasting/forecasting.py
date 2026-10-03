from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure

import tensorflow as tf


def compile_and_fit(
    model: tf.keras.Model,
    window: tf.data.Dataset,
    patience: int = 2,
    epochs: int = 20,
) -> tf.keras.callbacks.History:

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min'
    )

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

    history = model.fit(
        window.train,
        epochs=epochs,
        validation_data=window.val,
        callbacks=[early_stopping]
    )
    return history


class WindowGenerator():

    def __init__(
        self,
        input_width: int,
        label_width: int,
        shift: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label_columns: List[str] = None,
    ) -> None:

        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }

        # Work out the column indices.
        self.column_indices = {name: i for i,
                               name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.label_start = self.total_window_size - self.label_width

        # Work out input indices
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        # Work out label indices
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    def split_window(
        self,
        features: tf.Tensor
    ):
        # Split features into inputs and labels
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(
        self,
        data: pd.DataFrame
    ) -> tf.data.Dataset:

        # Prepare data
        data = np.array(data, dtype=np.float32)

        # Creates a dataset of sliding windows
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,
        )

        # Apply transformation
        ds = ds.map(self.split_window)

        return ds

    def plot(
        self,
        plot_col: str,
        model: tf.keras.Model = None,
        max_subplots: int = 3,
    ) -> matplotlib.figure.Figure:
        """
        Plots the specified column of the dataset as a line plot with optional labels and predictions.

        Parameters
        ----------
        plot_col : str
            The name of the column to plot.
        model : tf.keras.Model, optional
            A Keras model to generate predictions, if provided. Defaults to None.
        max_subplots : int, optional
            The maximum number of subplots to display, defaults to 3.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure containing the plot(s).
        """

        # Get example inputs and labels
        inputs, labels = self.example

        plt.figure(figsize=(12, 7))

        # Get index of the column to plot
        plot_col_index = self.column_indices[plot_col]

        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f"{plot_col} [normed]")

            # Plot the inputs. They will appear as a continuous blue line with dots.
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(
                    plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            # Plot the labels or actual values. They will appear as green squares.
            plt.scatter(
                self.label_indices,
                labels[n, :, label_col_index],
                edgecolors="k",
                label="Labels",
                c="#2ca02c",
                s=64
            )

            if model is not None:
                predictions = model(inputs)

                # Plot the predictions. They will appear as red crosses.
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, label_col_index],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=64
                )

            if n == 0:
                plt.legend()

        plt.xlabel("Time [h]")

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


class Baseline(tf.keras.Model):

    def __init__(self, label_index: int = None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs: tf.Tensor):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


class Linear(tf.keras.Model):

    def __init__(self, output_units: int = 1):
        super().__init__()
        self.dense = tf.keras.layers.Dense(units=output_units)

    def call(self, inputs: tf.Tensor):
        return self.dense(inputs)


class MultiLayerDense(tf.keras.Model):

    def __init__(self, output_units: int = 1):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(units=64, activation="relu")
        self.dense3 = tf.keras.layers.Dense(units=output_units)

    def call(self, inputs: tf.Tensor):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)


class MultiStepMultiLayerDense(tf.keras.Model):

    def __init__(self, output_units: int = 1):
        super().__init__()
        # Shape: (time, features) => (time*features)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=output_units)
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        self.reshape = tf.keras.layers.Reshape([1, -1])

    def call(self, inputs: tf.Tensor):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.reshape(x)


class ConvNet(tf.keras.Model):

    def __init__(self, kernel_size: int, output_units: int = 1):
        super().__init__()
        self.conv = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=kernel_size,
            activation='relu',
        )
        self.dense1 = tf.keras.layers.Dense(units=32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=output_units)

    def call(self, inputs: tf.Tensor):
        x = self.conv(inputs)
        x = self.dense1(x)
        return self.dense2(x)


class RecurrentLSTM(tf.keras.Model):

    def __init__(self, output_units: int = 1):
        super().__init__()
        # Shape [batch, time, features] => [batch, time, lstm_units]
        self.lstm = tf.keras.layers.LSTM(32, return_sequences=True)
        # Shape => [batch, time, features]
        self.dense = tf.keras.layers.Dense(
            units=output_units,
            kernel_initializer=tf.initializers.zeros() 
        )

    def call(self, inputs: tf.Tensor):
        x = self.lstm(inputs)
        return self.dense(x)


class ResidualWrapper(tf.keras.Model):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs: tf.Tensor, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)

        # The prediction for each time step is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta

class MultiStepLastBaseline(tf.keras.Model):
    
    def __init__(self, out_steps: int, label_index: int = None):
        super().__init__()
        self.out_steps = out_steps
        self.label_index = label_index
    
    def call(self, inputs: tf.Tensor):
        if self.label_index is None:
            return tf.tile(inputs[:, -1:, :], [1, self.out_steps, 1])
        result = tf.tile(inputs[:, -1:, self.label_index], [1, self.out_steps])
        return result[:, :, tf.newaxis]
    
class RepeatBaseline(tf.keras.Model):
    
    def __init__(self, label_index: int = None):
        super().__init__()
        self.label_index = label_index
    
    def call(self, inputs: tf.Tensor):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]
    
class MultiStepLastLinear(tf.keras.Model):
    
    def __init__(self, out_steps: int, output_units: int = 1):
        super().__init__()
        
        # Take the last time-step.
        self.get_last_timestep = tf.keras.layers.Lambda(lambda x: x[:, -1:, :])
        self.dense = tf.keras.layers.Dense(
            out_steps*output_units,
            kernel_initializer=tf.initializers.zeros()
        )
        self.reshape = tf.keras.layers.Reshape([out_steps, output_units])
        
    def call(self, inputs: tf.Tensor):
        # Shape [batch, time, features] => [batch, 1, features]
        x = self.get_last_timestep(inputs)
        # Shape => [batch, 1, out_steps*features]
        x = self.dense(x)
        return self.reshape(x)

class MultiStepLastMultiLayerDense(tf.keras.Model):
    
    def __init__(self, out_steps: int, output_units: int = 1):
        super().__init__()
        
        # Take the last time-step.
        self.get_last_timestep = tf.keras.layers.Lambda(lambda x: x[:, -1:, :])
        self.dense1 = tf.keras.layers.Dense(
            512,
            activation="relu"
        )
        self.dense2 = tf.keras.layers.Dense(
            out_steps*output_units,
            kernel_initializer=tf.initializers.zeros()
        )
        self.reshape = tf.keras.layers.Reshape([out_steps, output_units])
        
    def call(self, inputs: tf.Tensor):
        # Shape [batch, time, features] => [batch, 1, features]
        x = self.get_last_timestep(inputs)
        # Shape => [batch, 1, dense_units]
        x = self.dense1(x)
        # Shape => [batch, 1, out_steps*features]
        x = self.dense2(x)
        # Shape => [batch, out_steps, features]
        return self.reshape(x)
    

class MultiStepConvNet(tf.keras.Model):
    
    def __init__(self, kernel_size: int, out_steps: int, output_units: int = 1):
        super().__init__()
        # Take the last time-steps.
        self.get_last_timesteps = tf.keras.layers.Lambda(lambda x: x[:, -kernel_size:, :])
        self.conv = tf.keras.layers.Conv1D(256, activation="relu", kernel_size=(kernel_size))
        self.dense = tf.keras.layers.Dense(
            out_steps*output_units,
            kernel_initializer=tf.initializers.zeros()
        )
        self.reshape = tf.keras.layers.Reshape([out_steps, output_units])
    
    def call(self, inputs: tf.Tensor):
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        x = self.get_last_timesteps(inputs)
        # Shape => [batch, 1, conv_units]
        x = self.conv(x)
        # Shape => [batch, 1,  out_steps*features]
        x = self.dense(x)
        # Shape => [batch, out_steps, features]
        return self.reshape(x)


class MultiStepLSTM(tf.keras.Model):
    def __init__(self, out_steps: int, output_units: int = 1):
        super().__init__()        
        # Adding more `lstm_units` just overfits more quickly.
        self.lstm = tf.keras.layers.LSTM(32, return_sequences=False)
        self.dense = tf.keras.layers.Dense(
            out_steps*output_units, 
            kernel_initializer=tf.initializers.zeros()
        )
        self.reshape = tf.keras.layers.Reshape([out_steps, output_units])
        
    def call(self, inputs: tf.Tensor):
        # Shape [batch, time, features] => [batch, time, lstm_units]
        x = self.lstm(inputs)
        # Shape => [batch, out_steps*features].
        x = self.dense(x)
        return self.reshape(x)

class FeedBack(tf.keras.Model):
    
    def __init__(self, units, out_steps: int, output_units: int = 1):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(output_units)
        
    def warmup(self, inputs: tf.Tensor):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)
        
        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state
    
    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)
        # Insert the first prediction.
        predictions.append(prediction)
        
        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            
            # Execute one lstm step.
            x, state = self.lstm_cell(
                x,
                states=state,
                training=training
            )
            
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            
            # Add the prediction to the output.
            predictions.append(prediction)
            
        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        
        return predictions









        