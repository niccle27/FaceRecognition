import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D,
                                     MaxPooling2D,
                                     Flatten,
                                     Dense,
                                     Dropout)
from sklearn.model_selection import train_test_split

x_test = pd.read_csv("./df_x_test.csv").to_numpy()[:, 1:]\
    .reshape(-1, 50, 50, 3)
x_train = pd.read_csv("./df_x_train.csv").to_numpy()[:, 1:]\
    .reshape(-1, 50, 50, 3)
y_test = pd.read_csv("./df_y_test.csv").to_numpy()[:, 1:]
y_train = pd.read_csv("./df_y_train.csv").to_numpy()[:, 1:]

y_test = np.squeeze(y_test, axis=1)
y_train = np.squeeze(y_train, axis=1)

# Create model
model = Sequential()

model.add(Conv2D(8, kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=(50, 50, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(4, kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=(50, 50, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(4, kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=(50, 50, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

callbacks = [
    tensorflow.keras.callbacks.ModelCheckpoint(
        filepath='face_recognition_model.h5',
        # filepath='mymodel_{epoch}.h5',
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        save_best_only=True,
        monitor='val_loss',
        verbose=1)
]

model.fit(x=x_train,
          y=y_train,
          nb_epoch=100,
          batch_size=256,
          validation_data=(x_test,
                           y_test),
          callbacks=callbacks)
