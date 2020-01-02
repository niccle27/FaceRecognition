# https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/camera.html
import numpy as np
import sys
import tensorflow as tf
import cv2
from utils import label_map_util

cap = cv2.VideoCapture(0)

# Add to python search path
sys.path.append("..")

# Model preparation

# Variables


class_dic = {"Berengere": 0,
             "Clement": 1,
             "Edwin": 2,
             "Pascal": 3,
             "Ronald": 4}

class_dic_inverted = dict(map(reversed, class_dic.items()))


# Path to frozen detection graph.
# This is the actual model that is
# used for the object detection.
PATH_TO_CKPT = './MyInferenceGraph/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './data/object-detection.pbtxt'

NUM_CLASSES = 1

threshold_detection = 0.7


def predictRecognition(face_recognition_model, face):
    face = face.reshape(-1, 50, 50, 3)
    with tf.get_default_graph().as_default():
        predicted_class_id = face_recognition_model.predict(face)
        recognition_prediction = face_recognition_model.predict(face)[0]
        predicted_class_id = np.asarray(np.where(recognition_prediction == np.amax(recognition_prediction)))
        predicted_class_id = np.asscalar(predicted_class_id)
        proba = recognition_prediction[predicted_class_id]
        return predicted_class_id, proba


def detectFace(detection_graph, img):
    image_np_expanded = np.expand_dims(img, axis=0) # [1, 2, 3, 4] -> [[1, 2, 3, 4]]
    face = None
    pt1, pt2 = None, None
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)

            data_boxes = np.c_[boxes, classes, scores]
            data_boxes = [data_boxe for data_boxe in data_boxes if data_boxe[5]>threshold_detection]
            if len(data_boxes) <= 0:
                pass
            else:
                y1, x1, y2, x2, classe, score = data_boxes[0]
                x1 *= image_width
                y1 *= image_height
                x2 *= image_width
                y2 *= image_height
                pt1, pt2 = (int(x1),int(y1)),(int(x2),int(y2))
                face = image_np[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                face = cv2.resize(face,(50,50),cv2.INTER_AREA)
    return face, pt1, pt2


def image_processing(img, predicted_class_name, proba):
    text = predicted_class_name + ":" + str(proba)[:4]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    textBox_size = cv2.getTextSize(text, font, font_scale, 2)
    textbox_width, textBox_height = textBox_size[0]
    color = (255, 0, 0)
    text_offset_x, text_offset_y = (5, 5)
    text_org = tuple(np.add(pt1, (text_offset_x, textBox_height + text_offset_y)))
    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 3)
    cv2.rectangle(img, pt1,
                  tuple(np.add(pt1, (2*text_offset_x+textbox_width,
                                     2*text_offset_y+textBox_height))),
                  color, 3)
    img = cv2.putText(img,
                      text,
                      text_org,
                      font,
                      font_scale,
                      color,
                      thickness,
                      cv2.LINE_AA)
    return img


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    # od_graph_def = tf.compat.v1.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        # with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map

# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

# print(label) = "item {
#                 name: "face"
#                 id: 1
#               }"

categories = label_map_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
# categories = categories[0]
# print(categories) = [{'id': 1, 'name': 'face'}]
category_index = label_map_util.create_category_index(categories)
# print(category_index) = {1: {'id': 1, 'name': 'face'}}

face_recognition_model = tf.keras.models.load_model('face_recognition_model.h5')


while True:
    keyPressed = cv2.waitKey(1) & 0xFF
    if keyPressed == ord('q'):
        break
    ret, image_np = cap.read()
    image_height, image_width, _ = image_np.shape
    face, pt1, pt2 = detectFace(detection_graph, image_np)
    if face is not None:
        predicted_class_id, proba = predictRecognition(face_recognition_model,
                                                       face)
        predicted_class_name = class_dic_inverted[predicted_class_id]
        img_np = image_processing(image_np, predicted_class_name, proba)
    cv2.imshow("FaceRecognition", image_np)
    # print("test")
