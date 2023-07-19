import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full),(X_test, y_test) = fashion_mnist.load_data()
print(X_train_full.shape)
X_train_full_ = X_train_full / 255.0
X_train, X_test, y_train, y_test = train_test_split(X_train_full_, y_train_full, test_size=1/12, random_state=42)
class_name = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print(X_train.shape)
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28,28]))
model.add(keras.layers.Dense(300,activation="relu"))
model.add(keras.layers.Dense(100,activation="relu"))
model.add(keras.layers.Dense(10,activation="softmax"))
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer= "adam",
              metrics= ["accuracy"]
              )
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
X_new = X_test[:10]
y_pro = model.predict(X_new)
print(y_pro)
y_true = y_test[:10]
print(y_true)
