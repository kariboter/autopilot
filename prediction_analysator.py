import cv2


class Prediction:

    def __init__(self, frame, img_width, img_height):
        self.dim_of_frame = frame.shape()
        self.img_width = img_width
        self.img_height = img_height

    def exp_model_output(self, prediction, confidence_threshold):
        prediction_output = [prediction[k][prediction[k, :, 1] > confidence_threshold] for k in range(1)]
        return prediction_output

    def exp_read_box(self, box_prediction, classes, frame):
        for box in box_prediction[0]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            xmin = box[2] * self.img_width / self.dim_of_frame[0]
            ymin = box[3] * self.img_height / self.dim_of_frame[1]
            xmax = box[4] * self.img_width / self.dim_of_frame[0]
            ymax = box[5] * self.img_height / self.dim_of_frame[1]
            # draw the prediction on the frame
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                          1, 2)
