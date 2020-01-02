import numpy as np
import cv2
import os
import pandas as pd

n_sample_per_class = 1823
class_dic = {"Berengere": 1,
             "Clement": 2,
             "Edwin": 3,
             "Pascal": 4,
             "Ronald": 5}

df_x = pd.DataFrame(columns=[0])
df_y = pd.DataFrame(columns=np.arange(0, 7500))
for dir_class in class_dic.keys():
    dir_class_path = "./Dataset/" + dir_class + "/"
    imlist = []
    for el in sorted(os.listdir(dir_class_path)):
        im_path = dir_class_path + el
        img = cv2.imread(im_path)
        imlist.append(img.flatten())
    df_y_tmp = pd.DataFrame(imlist)
    df_y_tmp = df_y_tmp.sample(n=n_sample_per_class)
    xVal = np.empty(n_sample_per_class, dtype='uint8')
    xVal.fill(class_dic[dir_class])
    df_x_tmp = pd.DataFrame(xVal)
    df_x = df_x.append(df_x_tmp, ignore_index=True)
    df_y = df_y.append(df_y_tmp, ignore_index=True)
df_x.to_csv("dataset_X.csv")
df_y.to_csv("dataset_Y.csv")
# (50, 50, 3)
