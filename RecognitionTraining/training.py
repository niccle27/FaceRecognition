import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

y = pd.read_csv("./dataset_X.csv").to_numpy()[:, 1:]
# remove dimension because [[y1],[y2]] -> [y1,y2]
y = np.squeeze(y, axis=1)
y = np.array([i - 1 for i in y])


x = pd.read_csv("./dataset_Y.csv")\
      .to_numpy(dtype='uint8')[:, 1:]\
      .reshape(-1, 50, 50, 3)

print(x.shape)  # (9115, 50, 50, 3)
print(y.shape)  # (9115,)

# Shuffle
index = np.arange(len(x))
np.random.shuffle(index)
x = x[index]
y = y[index]

# Create model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=(50, 50, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=(50, 50, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=(50, 50, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# print("x_train", x_train.shape)  # 7292, 50, 50, 3
# print("x_test", x_test.shape)    # 1823, 50, 50, 3
# print("y_train", y_train.shape)  # 7292,
# print("y_test", y_test.shape)    # 1823,

model.summary()

model.fit(x=x_train, y=y_train, nb_epoch=15, batch_size=256)

# Save entire model to a HDF5 file
model.save('face_recognition_model.h5')
