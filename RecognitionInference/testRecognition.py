

class_dic = {"Berengere": 0,
             "Clement": 1,
             "Edwin": 2,
             "Pascal": 3,
             "Ronald": 4}

class_dic_inverted = dict(map(reversed, class_dic.items()))


import tensorflow as tf
import numpy as np
import cv2

print(class_dic_inverted)
face_recognition_model = tf.keras.models.load_model('face_recognition_model.h5')

face = cv2.imread("./testclem.jpg").reshape(-1, 50, 50, 3)
# print(face)

recognition_prediction = face_recognition_model.predict(face)[0]
print("recognition_prediction:", recognition_prediction)
predicted_class = np.asarray(np.where(recognition_prediction == np.amax(recognition_prediction)))

print(predicted_class)
predicted_class = np.asscalar(predicted_class)
print(predicted_class)

print(class_dic_inverted[predicted_class])
