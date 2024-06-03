import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import requests


class GradientAverageValue(tf.keras.callbacks.Callback):
    def __init__(self, test_data, test_label, ratio=1.0, url=None, token=None, clear=False):
        super(GradientAverageValue, self).__init__()
        self.clear = clear
        val_size = max(1, int(len(test_data) * ratio))
        indices = np.arange(len(test_data))
        np.random.shuffle(indices)
        self.test_data = test_data[indices[:val_size]]
        self.test_label = test_label[indices[:val_size]]
        self.epochs = []
        self.gradients_means = []
        self.url = url
        self.token = token

    def on_epoch_end(self, epoch, logs=None):
        with tf.GradientTape() as tape:
            predictions = self.model(self.test_data, training=True)
            loss = self.model.compute_loss(self.test_data, self.test_label, predictions)
        trainable_weights = [var for var in self.model.trainable_variables if 'kernel' in var.name]
        gradients = tape.gradient(loss, trainable_weights)
        curr_epoch = epoch + 1
        self.epochs.append(curr_epoch)

        for i, grad in enumerate(gradients):
            mean_grad = tf.reduce_mean(grad).numpy()
            if self.url is not None:
                self.update_metrics(curr_epoch, i + 1, mean_grad)
            elif epoch == 0:
                self.gradients_means.insert(i, [mean_grad])
            else:
                self.gradients_means[i].append(mean_grad)

        if self.url is None:
            self.plot_gradients()

    def plot_gradients(self):
        num_layers = len(self.gradients_means)
        fig, axes = plt.subplots(num_layers, 1, figsize=(10, num_layers * 4))

        for i in range(num_layers):
            ax = axes if num_layers == 1 else axes[i]
            ax.plot(self.epochs[:len(self.gradients_means[i])], self.gradients_means[i], label=f'Слой {i + 1}')
            ax.set_title(f'Среднее значение градиента слоя {i + 1}')
            ax.set_xlabel('Эпоха')
            ax.set_ylabel('Среднее значение градиента')
            ax.legend()
            ax.grid(True)
            ax.set_xticks(np.arange(1, len(self.epochs) + 2, 1))

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
            print(f'Failed to send data for GradientAverageValue at epoch {epoch}')