ENABLE_SINRIC_PRO = True
IS_RASPPI = True #False #True

import os
if IS_RASPPI:
    os.nice(-20)
import threading
import time
import subprocess

from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model

from sinric import SinricPro
import asyncio



if IS_RASPPI:
    import RPi.GPIO as GPIO

#======= GLOBAL PARAMETERS ========
PIN_TO_DOORLOCK = 12
PIN_TO_MOVEMENT_SENSOR = 13

FACE_TO_UNLOCK = "alessandro" #"giorgio"
TIME_SEC_HMI_TO_ENTER_IN_SLEEP = 120 #sec

#ENABLE FULL-SCREEN
ENABLE_FULL_SCREEN = True

# resolution of the screen
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 480

# size of the image to predict = input of the CNN
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

PATH_TO_CNN_MODEL = './saved_models/vgg16_2023_01_31_12_14_59_cnn.h5'

PERCENTAGE_OF_RECT_INTERSECTION_ = 50
PREDICTED_PROBABILITY_THRESHOLD = 98.0

FRAME_WAIT_TIME_MS = 200

APP_KEY = ''
APP_SECRET = ''
LOCK_ID = ''


class cTimer:
    def __init__(self):
        self._start_time = None
        self._elapsed_time = 0.0

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            #print(f"Timer is running. Use .stop() to stop it")
            return

        self._start_time = time.perf_counter()

    def getET(self):
        if self._start_time:
            self._elapsed_time = time.perf_counter() - self._start_time
        else:
            self._elapsed_time = 0
        return self._elapsed_time

    def reset(self):
        self._start_time = time.perf_counter()
        self._elapsed_time = 0.0

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            #print(f"Timer is not running. Use .start() to start it")
            return

        self._start_time = None
        self._elapsed_time = 0.0
        print(f"Elapsed time: {self._elapsed_time:0.4f} seconds")


class FP(object):

    def __init__(self, init_val=False):
        self.out = False
        self.in_ = init_val
        self.in_old = init_val

    def update(self, in_val):
        self.in_ = in_val
        if self.in_ == True and self.in_old == False:
            self.out = True
        else:
            self.out = False
        self.in_old = self.in_
        return self.out


class SinRicStateClass:
    def __init__(self, iloop, iface):
        self.loop = iloop
        self.iface = iface
        self.callbacks = {'setLockState': self.lock_state}
        self.client = None
        try:
            self.client = SinricPro(APP_KEY, [LOCK_ID], self.callbacks,
                                    enable_log=False, restore_states=False,
                                    secretKey=APP_SECRET)
        except Exception as e:
            print(e)
            self.client = None
        # To update the lock state on server.
        state = 'LOCKED' #FACE RECOGNITION DISABLED
        if self.iface.get_enable():
            state = 'UNLOCKED' #FACE RECOGNITION ENABLED
        self.client.event_handler.raiseEvent(LOCK_ID, 'setLockState', data={'state': state})
        # client.event_handler.raiseEvent(lockId, 'setLockState',data={'state': 'UNLOCKED'})

    def lock_state(self, device_id, state):
        print(device_id, state, type(state))
        if state == 'lock':
            benable = False
            self.iface.set_enable(benable)
        elif state == 'unlock':
            benable = True
            self.iface.set_enable(benable)
        return True, state

    def run(self, p1, p2):
        print(f"run SinRicStateClass {self.client}")
        while True:
            if self.client is not None:
                self.loop.run_until_complete(self.client.connect())
            else:
                time.sleep(10)


class Rectangle:
    def __init__(self, x, y, w, h):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.xmax = 0
        self.ymax = 0
        self.update(x, y, w, h)

    def update(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.xmax = x + w
        self.ymax = y + h


    def intersection(self, r):
        x = max(self.x, r.x)
        y = max(self.y, r.y)
        w = min(self.xmax, r.xmax) - x
        h = min(self.ymax, r.ymax) - y
        if w < 0 or h < 0:
            return None
        return Rectangle(x, y, w, h)

    def area(self):
        return (self.xmax-self.x)*(self.ymax-self.y)


class FaceRecognition:
    def __init__(self):
        if IS_RASPPI:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(PIN_TO_DOORLOCK, GPIO.OUT)
            GPIO.setup(PIN_TO_MOVEMENT_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            # add rising edge detection on a channel with 200ms of debouncer
            GPIO.add_event_detect(PIN_TO_MOVEMENT_SENSOR, GPIO.RISING, bouncetime=200)

        self.timer_set_hmi_to_sleep = cTimer()
        self.timer_set_hmi_to_sleep.start()

        self.face_recognition_enabled = True
        self.name_window = "Face Lock"
        if ENABLE_FULL_SCREEN:
            cv2.namedWindow(self.name_window, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(self.name_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(self.name_window)
        cv2.moveWindow(self.name_window, 0, 0) #pixels x,y

        #FONTS
        self.fontpath = r"./font/A4SPEED-Bold-DAFONT TTF.ttf"
        self.font1 = ImageFont.truetype(self.fontpath, 32)

        # for face detection
        self.face_cascade = cv2.CascadeClassifier('./haarcascades_config/haarcascade_frontalface_default.xml')

        # resolution of the screen
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT

        # size of the image to predict
        self.image_width = IMAGE_WIDTH
        self.image_height = IMAGE_HEIGHT

        self.fixed_rect = Rectangle(x=int(self.screen_width/2-self.image_width/2),
                                    y=int(self.screen_height/2-self.image_height/2),
                                    w=self.image_width, h=self.image_height)
        self.estimated_face_rect = Rectangle(0, 0, 1, 1)

        #LABELS
        self.FACE_LABEL_FILENAME = r'./labels/face-labels.pickle'

        # load the trained model
        #work fine on PC
        #self.model = load_model('./saved_models/vgg16_2023_01_31_12_14_59_cnn.h5')
        #for raspberry pi 4
        if IS_RASPPI:
            self.model = load_model(PATH_TO_CNN_MODEL, compile=False)
            self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            self.model = load_model(PATH_TO_CNN_MODEL)

        # the labels for the trained model
        with open(self.FACE_LABEL_FILENAME, 'rb') as f:
            og_labels = pickle.load(f)
            self.labels = {key:value for key,value in og_labels.items()}
            print(self.labels)

        self.background_img = cv2.imread(r'./background/backg4.png')
        self.background_img = cv2.resize(self.background_img, (self.screen_width, self.screen_height), interpolation=cv2.INTER_AREA)
        # default webcam
        self.stream = cv2.VideoCapture(0)
        self.Run = True

    def set_enable(self, enable):
        self.face_recognition_enabled = enable
    def get_enable(self):
        return self.face_recognition_enabled

    def run(self, p1, p2):
        while self.Run:
            self.__readMovSensState()
            # Capture frame-by-frame
            (grabbed, frame) = self.stream.read()
            #frame = self.background_img
            #print(frame.shape)
            #print(grabbed)
            # resize to fit the entire screen
            frame = cv2.resize(frame, (self.screen_width, self.screen_height), interpolation=cv2.INTER_AREA)
            # convert color space : used for CNN face recognition
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.background_img[self.fixed_rect.y:self.fixed_rect.ymax, self.fixed_rect.x:self.fixed_rect.xmax] = frame[self.fixed_rect.y:self.fixed_rect.ymax, self.fixed_rect.x:self.fixed_rect.xmax]
            frame = self.background_img
            txt = "DISABLED"
            if self.face_recognition_enabled:
                txt = "ENABLED"
            frame = self.__writeTextToImg(img_=frame, x=20, y=10, txt=txt, font=self.font1, color=(255, 255, 255))
            cv2.rectangle(frame, (self.fixed_rect.x, self.fixed_rect.y),
                          (self.fixed_rect.xmax, self.fixed_rect.ymax), (255, 255, 255), 2)
            # print(frame.shape)
            # try to detect faces in the webcam
            faces = self.face_cascade.detectMultiScale(rgb, scaleFactor=1.3, minNeighbors=5)
            #print(faces)
            #filter on face numbers
            if len(faces) != 1 or not self.face_recognition_enabled:
                cv2.imshow(self.name_window, frame)
                key = cv2.waitKey(FRAME_WAIT_TIME_MS) & 0xFF
                if key == ord("q"):  # Press q to break out of the loop
                    self.Run = False
                    break
                continue
            #filter on face size
            '''
            if faces[0][2] < self.image_width or faces[0][3] < self.image_height:
                cv2.imshow(self.name_window, frame)
                key = cv2.waitKey(FRAME_WAIT_TIME_MS) & 0xFF
                if key == ord("q"):  # Press q to break out of the loop
                    self.Run = False
                    break
                continue
            '''
            # for each faces found
            for (x, y, w, h) in faces:
                roi_rgb = rgb[y:y+h, x:x+w]
                #print((x,y,w,h))

                self.estimated_face_rect.update(x, y, w, h)
                #find intersection
                intersected_rect = self.fixed_rect.intersection(r=self.estimated_face_rect)
                if intersected_rect is None:
                    #print("Place the face inside the Rect")
                    break
                percentage_of_rect_intersection = intersected_rect.area()/self.fixed_rect.area() * 100.0
                #print(f"{percentage_of_rect_intersection:0.1f}")
                if percentage_of_rect_intersection < PERCENTAGE_OF_RECT_INTERSECTION_:
                    #print("Fit the face with the Rect")
                    break
                if percentage_of_rect_intersection >= PERCENTAGE_OF_RECT_INTERSECTION_ and \
                        self.estimated_face_rect.area() > self.fixed_rect.area()*1.2:
                    # to close to the camera
                    #print("Too close to the camera. Please Step Back")
                    break
                # Draw a rectangle around the face
                color = (255, 0, 0) # in BGR
                stroke = 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

                # resize the image
                size = (self.image_width, self.image_height)
                resized_image = cv2.resize(roi_rgb, size)
                image_array = np.array(resized_image, "uint8")
                img = image_array.reshape(1, self.image_width, self.image_height, 3)
                img = img.astype('float32')
                img /= 255

                # predict the image
                predicted_prob = self.model.predict(img)
                #print(predicted_prob)
                # Display the label
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # stroke = 2
                label_id = predicted_prob[0].argmax()
                predicted_probability = predicted_prob[0][label_id]*100.0
                name = self.labels[label_id]
                color = (255, 255, 255)
                #cv2.putText(frame, f'{name}/{predicted_probability:0.1f}%', (x, y-8), font, 1, color, stroke, cv2.LINE_AA)
                #with costum font
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text((x, y-32), f'{name} {predicted_probability:0.0f}', font=self.font1, fill=color)
                frame = np.array(img_pil)

                # filter by name
                if name.lower() != FACE_TO_UNLOCK:
                    #print(f"Access Denied For: {name}")
                    break

                #filter by predicted probability
                if predicted_probability < PREDICTED_PROBABILITY_THRESHOLD:
                    #print("Low Probability")
                    break

                #HERE WE CAN UNLOCK THE DOOR
                self.__unlock_the_door(ontime_sec=3, frame=frame)

            # Show the frame
            cv2.imshow(self.name_window, frame)
            key = cv2.waitKey(FRAME_WAIT_TIME_MS) & 0xFF
            if key == ord("q"):    # Press q to break out of the loop
                self.Run = False
                break #exit the while loop

        # Cleanup
        self.stream.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def __writeTextToImg(self, img_, x, y, txt, font, color):
        img_pil = Image.fromarray(img_)
        draw = ImageDraw.Draw(img_pil)
        draw.text((x, y), txt, font=font, fill=color)
        return np.array(img_pil)

    def __unlock_the_door(self, ontime_sec, frame):
        #ANIMATION
        full_txt = "UNLOCKED"
        txt = ""
        stroke = [2,4]
        id_stroke=0
        for c in full_txt:
            res_img = frame.copy()
            txt += c
            res_img = self.__writeTextToImg(img_=res_img, x=20, y=40, txt=txt, font=self.font1, color=(255, 255, 255))
            cv2.rectangle(res_img, (self.fixed_rect.x, self.fixed_rect.y),
                          (self.fixed_rect.xmax, self.fixed_rect.ymax), (0, 255, 0), stroke[id_stroke % 2])
            id_stroke += 1
            cv2.imshow(self.name_window, res_img)
            cv2.waitKey(100)

        if IS_RASPPI:
            GPIO.output(PIN_TO_DOORLOCK, True)
        time.sleep(ontime_sec)
        if IS_RASPPI:
            GPIO.output(PIN_TO_DOORLOCK, False)
        time.sleep(1)

    def __readMovSensState(self):
        if IS_RASPPI:
            if GPIO.event_detected(PIN_TO_MOVEMENT_SENSOR):
                #rising edge
                #TODO: HMI EXIT FROM SLEEP MODE
                #print("run(xset dpms force on)")
                subprocess.run("xset dpms force on", shell=True)
                self.timer_set_hmi_to_sleep.reset()
        #print(f'ET.{self.timer_set_hmi_to_sleep.getET()}')
        if self.timer_set_hmi_to_sleep.getET() > TIME_SEC_HMI_TO_ENTER_IN_SLEEP:
            #after 2 mins of no presence,
            #put the hmi in sleep mode
            #TODO:HMI ENTER IN SLEEP MODE
            if IS_RASPPI:
                #print("run(xset dpms force off)")
                subprocess.run("xset dpms force off", shell=True)
            self.timer_set_hmi_to_sleep.reset()

if __name__ == "__main__":

    iFR = FaceRecognition()
    loop = None
    if ENABLE_SINRIC_PRO:
        loop = asyncio.get_event_loop()
        iSin = SinRicStateClass(iloop=loop, iface=iFR)
        thSin = threading.Thread(name=f"thread-SinRicPro", target=iSin.run, args=(0, 1,), daemon=True)
        thSin.start()

    iFR.run(0,1)

    #todo run a second thread
    #thFR = threading.Thread(name=f"thread-Face-Rec", target=iFR.run, args=(0, 1,), daemon=True)
    #thFR.start()
    #thFR.join()

    #iSin = SinRicStateClass(iloop=loop, iface=iFR)
    #iSin.run()