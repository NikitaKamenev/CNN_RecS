import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import requests


class WeightsAverageChange(tf.keras.callbacks.Callback):
    def __init__(self, url=None, token=None, clear=False):
        super(WeightsAverageChange, self).__init__()
        self.epochs = []
        self.previous_epoch_weights = []
        self.weights = []
        self.clear = clear
        self.url = url
        self.token = token

    def on_epoch_end(self, epoch, logs=None):
        layer_weights = [layer.get_weights() for layer in self.model.layers
                         if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D))]
        epoch_mean = []
        curr_epoch = epoch + 1
        for i, weights in enumerate(layer_weights):
            weights_arrays = [tf.convert_to_tensor(w) for w in weights]
            if epoch == 0:
                self.weights.insert(i, [])
                self.previous_epoch_weights.insert(i, weights_arrays)
                continue
            result = [tf.abs(w1 - w2) for w1, w2 in zip(weights_arrays, self.previous_epoch_weights[i])]
            mean = tf.reduce_mean(tf.concat([tf.reshape(res, [-1]) for res in result], axis=0))
            if self.url is not None:
                self.update_metrics(curr_epoch, i + 1, mean)
            else:
                self.weights[i].append(mean)
            self.previous_epoch_weights[i] = weights_arrays
        if epoch == 0:
            return
        if self.url is None:
            self.epochs.append(curr_epoch)
            self.update_plots()

    def update_plots(self):
        num_layers = len(self.weights)
        fig, axes = plt.subplots(num_layers, 1, figsize=(10, num_layers * 5))
        for i in range(num_layers):
            if num_layers == 1:
                ax = axes
            else:
                ax = axes[i]
            ax.plot(self.epochs[:len(self.weights[i])], self.weights[i], label=f'Слой {i + 1}')
            ax.set_title(f'Абсолютная разница весов {i + 1}')
            ax.set_xlabel('Эпоха')
            ax.set_ylabel('Абсолютная разница весов')
            ax.legend()
            ax.grid(True)
            x_ticks = np.arange(2, len(self.epochs) + 2, 1)
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
            print(f'Failed to send data for WeightsAverageChange at epoch {epoch}')



