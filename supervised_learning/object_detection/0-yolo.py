#!/usr/bin/env python3
"""
0-yolo.py
"""
from tensorflow import keras as K


class Yolo:
    """
    Yolo class for loading a Darknet Keras model and its associated parameters.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo model with the given parameters.
        """
        self.model = K.models.load_model(model_path)
        self.class_names = self.load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def load_classes(classes_path):
        """
        Loads class names from a file.
        """
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        return classes
