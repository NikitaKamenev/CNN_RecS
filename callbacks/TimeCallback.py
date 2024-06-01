import tensorflow as tf
import time

class TimeCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TimeCallback, self).__init__()
        self.start_time = None
        self.end_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        self.end_time = time.time()
        t = self.end_time - self.start_time
        print(f"Обучение завершено за {t:.2f} сек")