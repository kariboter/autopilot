# -----------------------------------------------raspberry remot gpio importing------------------------------

from gpiozero import PWMLED
from gpiozero.pins.pigpio import PiGPIOFactory
import json
from pid import PID

# -----------------------------------------------keras moduls import-----------------------------------------

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_L2Normalization import L2Normalization
import tensorflow.compat.v1 as tf
tf.enable_eager_execution(tf.ConfigProto(log_device_placement=False))
from tensorflow.keras.layers import Layer

# -----------------------------------------------------------------------------------------------------------
# ------------------------------------importing others library-----------------------------------------------

import numpy as np
from app import main

# -----------------------------------------------------------------------------------------------------------
# --------------------------------------importing computer vision library------------------------------------

import cv2
from Video import Video  # it is a library that take a frame from video stream and convert it into two images.

# -----------------------------------------------------------------------------------------------------------
# --------------------------------------disabling tensorflow warnings and errors-----------------------------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -----------------------------------------------------------------------------------------------------------
# -------------------------------------------initializations-------------------------------------------------

factory = PiGPIOFactory(host='192.168.88.163')
PWM_l = PWMLED(17, pin_factory=factory)
DIR_l = PWMLED(4, pin_factory=factory)
PWM_r = PWMLED(3, pin_factory=factory)
DIR_r = PWMLED(2, pin_factory=factory)


flage = True
pred = []
last_img = []
cropped_frame = []
pid = PID(1, 0, 0)
pid.setPoint(160)
# http://192.168.88.78:9000/stream.mjpg is the link to our stream in local network(it's can be any link to any stream).
vs = Video('http://192.168.88.163:9000/stream.mjpg')

dim_of_frame = (300, 300)  # the dimension of the input tensors of our model(i.e. ssd300).
img_height, img_width = vs.get_frame_shape()  # initialization of frame dimension.
# Initialization of classes that our model can predict
classes = ['background',
           'person']

class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# ---------------------------------------loading keras pre-rained model--------------------------------------

model_path = 'C:/Users/User/Downloads/ssd_keras/model.h5'  # full path to keras model.

# siamese_model_path = 'C:/Users/User/Documents/resnet siamese network/siamese_model.h5'
# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
K.clear_session()  # Clear previous models from memory.
# loading our model with custom layers(i.e. AnchorBoxes, L2Normalization, DecodeDetections, compute_loss).
model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})

# siamese_model = tf.keras.models.load_model(siamese_model_path,
#                                   custom_objects={'L1Dist':L1Dist})
# -----------------------------------------------------------------------------------------------------------
# -------------------------------------------functions initialization----------------------------------------



def motors(feedback):
        if feedback:
            if feedback > 0:
                print(1 - ((feedback * 1) / 100), feedback)
                PWM_r.value = 1 - ((feedback * 1) / 100)
                PWM_l.value = 1

            else:
                PWM_r.value = 1
                PWM_l.value = 1 + ((feedback * 1) / 100)
                print(1 + ((feedback * 1) / 100), feedback)


def pid_result(xmin, xmax):
    result = pid.update(xmax + (xmin / 2))
    if (result < 100) and (result > -100):
        return result

def motors_stop():
    print("stop")
    PWM_r.value = 0
    PWM_l.value = 0

# this function tack the coordinates of the predicted boxs and draw them in the initial frame
def show_output(xmin, ymin, xmax, ymax, frame):
    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                  1, 2)


#  This function tack as input prediction and threshold for prediction of the model and return boxs with they
#  class and confidence.
def model_output(prediction, confidence_threshold, frame):
    # our model(e.i. ssd300) return as output an 3 dimension array (1, 200, 6) were the second dim is the max
    # quantity of predicted object.
    # in this loop we check each prediction.
    result = []
    for i in np.arange(prediction.shape[1]):
        idx = prediction[0, i, 0]  # idx is the index of prediction(e.g. person, aeroplane, dog).
        if idx != 0.:  # if the first number of array is not 0 we continue the loop
            confidence = prediction[0, i, 1]  # confidence is the confidence of prediction.
            if confidence_threshold < confidence:
                box = prediction[0, i, 2:]  # box is the coordinates of the object in input frame
                # (Xmin, Ymin, Xmax, Ymax).
                # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
                xmin = box[0] * (img_width / 2) / dim_of_frame[0]
                ymin = box[1] * img_height / dim_of_frame[1]
                xmax = box[2] * (img_width / 2) / dim_of_frame[0]
                ymax = box[3] * img_height / dim_of_frame[1]
                #  place the output function here
                # cropped_frame = crop_img(int(xmin), int(ymin), int(xmax), int(ymax), frame)
                # cv2.resize(cropped_frame, (200, 200))
                # pred.append(siamese_model.predict([last_img, cropped_frame]))
                percentage = (xmax - xmin) / (img_width / 2)
                show_output(xmin, ymin, xmax, ymax, frame)
                if percentage < 0.8:
                    motors(pid_result(xmin, ymax))
                else:
                    motors_stop()
        # if else we close the loop and continue the main code, because 0 as the first number means that there
        # no more prediction
        else:
            break



def crop_img(xmin, ymin, xmax, ymax, frame):
    img = frame[xmin:xmax, ymin:ymax]
    return img

# -----------------------------------------------------------------------------------------------------------
# --------------------------------------experimental functions initialization--------------------------------

# This is an experimental functions
def exp_model_output(prediction, confidence_threshold):
    prediction_output = [prediction[k][prediction[k, :, 1] > confidence_threshold] for k in range(1)]
    return prediction_output


def exp_read_box(box_prediction, frame):
    for box in box_prediction[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2]
        ymin = box[3]
        xmax = box[4]
        ymax = box[5]
        # draw the prediction on the frame
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                      1, 2)

with open("app/test.json", "r") as file:
    data = json.load(file)
# -----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------main code----------------------------------------------

while True:
    # grab a frames from the video stream and resize it.
    # imgR, imgL, ret = vs.get_splintered_video()
    frame, _,  ret = vs.get_splintered_video()
    resized_frame = cv2.resize(frame, dim_of_frame)
    # frames shape has 3 dimensions (300 pixels width, 300 pixels height, 3 colors) but our model(i.e. ssd300) needs
    # 4 dimension, so we need to add another axis to input frame.
    image = np.expand_dims(resized_frame, axis=0)
    image = image.astype('float')
    # here we add an axis to input frame.


    if ret:
            detections = model.predict(image)
            model_output(detections, 0.5, frame)
            cv2.imshow('detection', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    # else:
    #     motors_stop()
    #     cv2.destroyAllWindows()

cv2.destroyAllWindows()
vs.video_stop()
