from gpiozero import PWMLED
from gpiozero.pins.pigpio import PiGPIOFactory


class Motors:

    def __init__(self, img_width):
        factory = PiGPIOFactory(host='192.168.88.163')
        self.PWM_l = PWMLED(17, pin_factory=factory)
        self.DIR_l = PWMLED(4, pin_factory=factory)
        self.PWM_r = PWMLED(3, pin_factory=factory)
        self.DIR_r = PWMLED(2, pin_factory=factory)
        self.value_r = 0
        self.value_l = 0
        self.set_value_r = 0
        self.set_value_l = 0
        self.stop_time = 1
        self.div = 100 / 16
        self.xmin = 0
        self.xmax = 0
        self.img_width = img_width
        self.anchor = img_width / 2
        self.div_r = 0
        self.div_l = 0

    def update(self, xmin=0, xmax=0):
        self.xmin = xmin
        self.xmax = xmax

    def error(self):

        return ((self.anchor - (self.xmin + ((self.xmax - self.xmin) / 2))) / self.img_width) * 100

    def run_motors(self):
        print(self.value_r, self.value_l)
        self.PWM_r.value = round(self.value_r) / 100
        self.PWM_l.value = round(self.value_l) / 100

    def soft_run(self):
        self.div_r = (self.set_value_r - self.value_r) / 8
        self.div_l = (self.set_value_l - self.value_l) / 8
        self.value_r = self.value_r + self.div_r
        self.value_l = self.value_l + self.div_l

        self.run_motors()

    def main(self):
        print(self.xmax - self.xmin / (self.img_width))
        if (self.xmax - self.xmin) / (self.img_width) < 0.8:
            if self.error() < 0:
                self.set_value_l = 100
                self.set_value_r = 100 + self.error()
            else:
                self.set_value_l = 100 - self.error()
                self.set_value_r = 100
        else:
            if self.value_r < 20 and self.value_l < 20:
                self.value_l = 0
                self.value_r = 0
            self.set_value_l = 0
            self.set_value_r = 0
        self.soft_run()
