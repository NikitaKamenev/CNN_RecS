import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import requests


class ActivationAverageValue(tf.keras.callbacks.Callback):
    def __init__(self, test_data, ratio=1.0, url=None, token=None, clear=False):
        super(ActivationAverageValue, self).__init__()
        self.epochs = []
        self.previous_epoch_activation = []
        self.clear = clear
        val_size = max(1, int(len(test_data) * ratio))
        indices = np.arange(len(test_data))
        np.random.shuffle(indices)
        self.test_data = test_data[indices[:val_size]]
        self.layer_activation = []
        self.url = url
        self.token = token

    def on_epoch_end(self, epoch, logs=None):
        input_data = self.test_data
        layer_outputs = [layer.output for layer in self.model.layers
                         if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D))]
        activation_model = tf.keras.models.Model(self.model.layers[0].input, outputs=layer_outputs)
        activations = activation_model.predict(input_data)
        curr_epoch = epoch + 1
        self.epochs.append(curr_epoch)
        for i, activation in enumerate(activations):
            layer_mean = np.mean(activation)
            if self.url is not None:
                self.update_metrics(curr_epoch, i + 1, layer_mean)
            elif epoch == 0:
                self.layer_activation.insert(i, [layer_mean])
            else:
                self.layer_activation[i].append(layer_mean)

        if self.url is None:
            self.update_plots()

    def update_plots(self):
        num_layers = len(self.layer_activation)
        fig, axes = plt.subplots(num_layers, 1, figsize=(10, num_layers * 5))
        for i in range(num_layers):
            if num_layers == 1:
                ax = axes
            else:
                ax = axes[i]
            ax.plot(self.epochs[:len(self.layer_activation[i])], self.layer_activation[i], label=f'Слой {i + 1}')
            ax.set_title(f'Среднее значение функции активации слоя {i + 1}')
            ax.set_xlabel('Эпоха')
            ax.set_ylabel('Среднее значение функции активации')
            ax.legend()
            ax.grid(True)
            x_ticks = np.arange(1, len(self.epochs) + 2, 1)
            ax.set_xticks(x_ticks)

        if self.clear:
            clear_output(wait=True)
        plt.tight_layout()
        plt.show()

    def update_metrics(self, epoch, layer, value):
        url = f'{self.url}/{self.token}'
        data = {
            'epoch': epoch,
            'layer': layer,
            'value': float(value)
        }
        response = requests.post(url, json=data)
        if response.status_code != 201:
            print(f'Failed to send data for ActivationAverageValue at epoch {epoch}')

