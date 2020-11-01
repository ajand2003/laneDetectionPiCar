
#class dedicated to starting the car and feeding information based on processed images given by imageProcessingAndDetection.py


import logging
import picar
import cv2
import datetime
from imageProcessingAndDetection import startProcessing




class carControl(object):

    __INITIAL_SPEED = 0
   

    def __init__(self):
   

        picar.setup()

        logging.debug('Set up camera')
        self.camera = cv2.VideoCapture(-1)
        self.camera.set(3, 420)
        self.camera.set(4, 340)

        self.pan_servo = picar.Servo.Servo(1)
        self.pan_servo.offset = -30  
        self.pan_servo.write(90)

        self.tilt_servo = picar.Servo.Servo(2)
        self.tilt_servo.offset = 20 
        self.tilt_servo.write(90)

        logging.debug('backwheels check')
        self.back_wheels = picar.back_wheels.Back_Wheels()
        self.back_wheels.speed = 0 

        logging.debug('frontwheels check')
        self.front_wheels = picar.front_wheels.Front_Wheels()
        self.front_wheels.turning_offset = -25 
        self.front_wheels.turn(90) 

        self.lane_follower = imageProcessingAndDetection(self)
 

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        datestr = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.video_orig = self.videoRecorder('../data/tmp/car_video%s.avi' % datestr)
        self.video_lane = self.videoRecorder('../data/tmp/car_video_lane%s.avi' % datestr)

       
        #method that refers back to imageProcessingAndDetection to process image and effectively return a steering angle
    def steer(self, frame):
        logging.debug('steering')
        steeringangle, processedimage = startProcessing(frame)
        if self.car is not None:
            return steeringangle
        else: 
            self.cleanup()


    def videoRecorder(self, path):
        return cv2.VideoWriter(path, self.fourcc, 20.0, (320,240))

    def __enter__(self):
        """ Entering a with statement """
        return self

    def __exit__(self, type, value, traceback):
        
        if traceback is not None:
           
            logging.error('Exiting with statement with exception %s' % traceback)

        self.cleanup()

    def cleanup(self):
        """ Reset the hardware"""
        logging.info('Stopping the car, resetting hardware.')
        self.back_wheels.speed = 0
        self.front_wheels.turn(90)
        self.camera.release()
        self.video_orig.release()
        self.video_lane.release()
        self.video_objs.release()
        cv2.destroyAllWindows()

    def drive(self, speed):
        previousSteeringAngle = 90
        logging.info('Starting to drive at speed %s...' % speed)
        self.back_wheels.speed = speed
        while self.camera.isOpened():
            image = self.camera.read()
            self.video_orig.write(image)
            if(abs(currentSteer-previousSteeringAngle) >= 15):
                previousSteeringAngle = previousSteeringAngle
            else:
                previousSteeringAngle = currentSteer

            currentSteer = steer(image)

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cleanup()
                break


    def follow_lane(self, image):
        image = self.lane_follower.follow_lane(image)
        return image



def show_image(title, frame):
    cv2.imshow(title, frame)


def main():
    with carControl() as car:
        car.drive(25)


