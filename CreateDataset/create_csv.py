import numpy as np
import cv2
import os
import pandas as pd

n_sample_per_class = 1823
percentage_of_train = 0.8

class_dic = {"Berengere": 0,
             "Clement": 1,
             "Edwin": 2,
             "Pascal": 3,
             "Ronald": 4,
             "Arthur": 5}

test_training_limit = int(n_sample_per_class * percentage_of_train)
df_y_train = pd.DataFrame(columns=[0])
df_y_test = pd.DataFrame(columns=[0])
df_x_train = pd.DataFrame(columns=np.arange(0, 7500))
df_x_test = pd.DataFrame(columns=np.arange(0, 7500))

for dir_class in class_dic.keys():
    dir_class_path = "./Dataset/" + dir_class + "/"
    imlist = []
    for el in sorted(os.listdir(dir_class_path)):
        im_path = dir_class_path + el
        img = cv2.imread(im_path)
        imlist.append(img.flatten())
    df_x_tmp = pd.DataFrame(imlist)
    df_x_tmp = df_x_tmp.sample(n=n_sample_per_class)  # shuffle all
    df_x_tmp_train = df_x_tmp.iloc[:test_training_limit]
    df_x_tmp_test = df_x_tmp.iloc[test_training_limit:]
    yVal = np.empty(n_sample_per_class, dtype='uint8')
    yVal.fill(class_dic[dir_class])
    df_y_tmp_train = pd.DataFrame(yVal[:test_training_limit])
    df_y_tmp_test = pd.DataFrame(yVal[test_training_limit:])
    df_y_train = df_y_train.append(df_y_tmp_train, ignore_index=True)
    df_y_test = df_y_test.append(df_y_tmp_test, ignore_index=True)
    df_x_train = df_x_train.append(df_x_tmp_train, ignore_index=True)
    df_x_test = df_x_test.append(df_x_tmp_test, ignore_index=True)
# df_x.to_csv("dataset_X.csv")
# df_y.to_csv("dataset_Y.csv")
df_x_test.to_csv("df_x_test.csv")
df_x_train.to_csv("df_x_train.csv")
df_y_test.to_csv("df_y_test.csv")
df_y_train.to_csv("df_y_train.csv")
# (50, 50, 3)
