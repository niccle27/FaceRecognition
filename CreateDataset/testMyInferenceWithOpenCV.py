# https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/camera.html

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
cap = cv2.VideoCapture(0)

# Add to python search path
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

from utils import label_map_util
from utils import visualization_utils as vis_util


# Model preparation

# Variables

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './MyInferenceGraph/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './data/object-detection.pbtxt'

NUM_CLASSES = 1

threshold_detection = 0.7

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

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# categories = categories[0]
# print(categories) = [{'id': 1, 'name': 'face'}]
category_index = label_map_util.create_category_index(categories)
# print(category_index) = {1: {'id': 1, 'name': 'face'}}


# # ## Helper code
# def load_image_into_numpy_array(image):
#   (im_width, im_height) = image.size
#   return np.array(image.getdata()).reshape(
#     (im_height, im_width, 3)).astype(np.uint8)


# # Detection

with detection_graph.as_default():
  with tf.compat.v1.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      image_height,image_width,_ = image_np.shape

      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0) # [1, 2, 3, 4] -> [[1, 2, 3, 4]]

      # Extract image tensor
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

      boxes = np.squeeze(boxes)
      classes = np.squeeze(classes).astype(np.int32)
      scores = np.squeeze(scores)

      data_boxes = np.c_[boxes, classes, scores]
      data_boxes = [data_boxe for data_boxe in data_boxes if data_boxe[5]>threshold_detection]

      for y1, x1, y2, x2, classe, score in data_boxes:
        x1 *= image_width
        y1 *= image_height
        x2 *= image_width
        y2 *= image_height
        pt1,pt2 = (int(x1),int(y1)),(int(x2),int(y2))
        text = category_index[classe]["name"]+":"+str(score)[:4]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness=2
        textBox_size = cv2.getTextSize(text,font,font_scale,2)
        textbox_width, textBox_height = textBox_size[0]
        line = textBox_size[1]
        color = (255, 0, 0)
        text_offset_x, text_offset_y = (5, 5)
        text_org = tuple(np.add(pt1, (text_offset_x, textBox_height + text_offset_y)))
        cv2.rectangle(image_np, pt1, pt2, (0, 0, 255), 3)
        cv2.rectangle(image_np,pt1,tuple(np.add(pt1,(2*text_offset_x+textbox_width,2*text_offset_y+textBox_height))),color, 3)
        image = cv2.putText(image_np, text ,text_org, font, font_scale, color, thickness, cv2.LINE_AA)

      # Visualization of the results of a detection.

      cv2.imshow('object detection', image_np)
      # cv2.imshow('object detection', cv2.resize(image_np, (800,600)))

      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
