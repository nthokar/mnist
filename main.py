import time

import numpy as np
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras import layers
from keras.layers import Flatten
from keras.utils import to_categorical

# Устанавливаем seed для повторяемости результатов
np.random.seed(42)

# Загружаем данные
(X_train, y_train), (X_test, y_test) = mnist.load_data()
input_shape = (28, 28, 1)

init_time = time.time()
print()

# Нормализация данных
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# Преобразуем метки в категории
Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)

# Создаем последовательную модель
model = Sequential()

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)
print(model.summary())


# Компилируем модель
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Обучаем сеть
batch_size = 100
epochs = 4

start_time = time.time()
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
end_time = time.time()

# Оцениваем качество обучения сети на тестовых данных
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))
print(f"Время инициализации{end_time - init_time}, Время обученияend_time - start_time{end_time - start_time}")