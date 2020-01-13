import numpy as np
import pandas as pd

x_test = pd.read_csv("./df_x_test.csv").to_numpy()[:, 1:]\
    .reshape(-1, 50, 50, 3)
x_train = pd.read_csv("./df_x_train.csv").to_numpy()[:, 1:]\
    .reshape(-1, 50, 50, 3)
y_test = pd.read_csv("./df_y_test.csv").to_numpy()[:, 1:]
y_train = pd.read_csv("./df_y_train.csv").to_numpy()[:, 1:]

y_test = np.squeeze(y_test, axis=1)
y_train = np.squeeze(y_train, axis=1)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print("x_train", x_train.shape)  # 7292, 50, 50, 3
print("x_test", x_test.shape)    # 1823, 50, 50, 3
print("y_train", y_train.shape)  # 7292,
print("y_test", y_test.shape)    # 1823,
