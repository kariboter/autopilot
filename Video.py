import cv2
from imutils.video import VideoStream


class Video:

    def __init__(self, url):
        '''
        here we initialize the class:
        width and height are dimensions of the stream
        half_width an arguments for slicing
        '''
        self.vs = VideoStream(src=url).start()
        self.frame = self.vs.read()
        self.height, self.width = 0, 0
        self.get_frame_shape()
        self.half_width = int(self.width / 2)
        self.grabbed = 0
        self.img_r = []
        self.img_l = []

    def video_stop(self):
        self.vs.stop()

    def get_frame_shape(self):
        self.height, self.width, _ = self.frame.shape
        return self.height, self.width

    def get_video(self):
        self.frame = self.vs.read()
        self.grabbed = self.vs.grabbed
        return self.frame, self.grabbed

    def get_splintered_video(self):
        self.get_video()
        self.img_r = self.frame[0:, 0:self.half_width]
        self.img_l = self.frame[0:, self.half_width:self.width]
        return self.img_r, self.img_l, self.grabbed

    def show_video(self):
        while True:
            self.get_splintered_video()
            cv2.imshow("Frame_r", self.img_r)
            cv2.imshow("Frame_l", self.img_l)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        self.video_stop()




