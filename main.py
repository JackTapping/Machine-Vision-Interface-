import time

import cv2
import numpy as np
import mediapipe as mp
import pyautogui as pag
import math
import tkinter as tk # Making a GUI
import winsound
frequency = 750
duration = 150


import speech_recognition


class Square:

    def __init__(self, start, end, colour, active, text, inside, activated):
        self.start = start
        self.end = end
        self.color = colour
        self.active = active
        self.text = text
        self.inside = inside
        self.activated = activated

    def draw(self, img):
        cv2.rectangle(img, self.start, self.end, self.color, -1)
        cv2.putText(image, self.text, (self.end[0] + 10, self.end[1]), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)

    def set_color(self):
        if self.active and self.color[1] < 255 and not self.inside:
            self.color[1] = 255
            self.color[2] = 0

        if not self.active and self.color[2] < 255 and not self.inside:
            self.color[2] = 255
            self.color[1] = 0

    def detect_cursor(self, cursor):

        if self.start[0] < cursor[0][0] * video_width < self.end[0]:
            if self.start[1] < cursor[0][1] * video_height < self.end[1]:
                self.inside = True
            else:
                self.inside = False
                self.activated = False
        else:
            self.inside = False
            self.activated = False

        if self.inside and self.active and not self.activated:
            self.color[1] = self.color[1] - 10
            self.color[2] = self.color[2] + 10

            if self.color[1] < 0:
                self.active = False
                self.activated = True

        if self.inside and not self.active and not self.activated:
            self.color[1] = self.color[1] + 10
            self.color[2] = self.color[2] - 10

            if self.color[2] < 0:
                self.active = True
                self.activated = True

    def is_active(self):
        return self.active

    def set_active(self, x):
        self.active = x

def Speech2Text():
    recog = speech_recognition.Recognizer()

    with speech_recognition.Microphone() as mic:

        recog.adjust_for_ambient_noise(mic, duration=0.2)
        winsound.Beep(frequency, duration)
        audio = recog.listen(mic)
        text = recog.recognize_google(audio)
        text = text.lower()

        print(f"Recognized {text}")
        return text

def speech_command(text):

    users_words = text.split()
    if users_words[0] == "open":
        pag.press('winleft')
        text = text.replace('open', '')
        pag.typewrite(text)
        pag.press('enter')
    if users_words[0] == "close":
        option_close.set_active(True)

    if users_words[0] == "find":

        text = text.replace('find', '')
        with pag.hold('ctrlleft'):
            pag.press(['f'])
        pag.typewrite(text)

    if users_words[0] == "paste":
        with pag.hold('ctrlleft'):
            pag.press(['v'])

    if users_words[0] == "copy":
        with pag.hold('ctrlleft'):
            pag.press(['c'])

    if users_words[0] == "cut":
        with pag.hold('ctrlleft'):
            pag.press(['x'])

    if len(users_words) >= 2:
        if users_words[0] == "cut" and users_words[1] == "paragraph":
            pag.click()
            pag.click()
            pag.click()
            with pag.hold('ctrlleft'):
                pag.press(['x'])

    if len(users_words) >= 3:
        if users_words[0] == "insert":
            if users_words[1] == "for" and users_words[2] == "loop":
                pag.typewrite("for x in range(10):")
                pag.press('enter')
                pag.typewrite("    print(x)")

        if users_words[1] == "class" and users_words[2] == "called":
            pag.typewrite("class ")
            pag.typewrite(users_words[3])
            pag.typewrite(":")
            pag.press('enter')
            pag.typewrite("    def __init__(self):")
            pag.press('enter')
            pag.typewrite("    pass")

        if users_words[0] == "new" and users_words[1] == "python" and users_words[2]  == "script":
            pag.press('winleft')
            pag.typewrite("idel")
            pag.press('enter')
            pag.PAUSE=1
            with pag.hold('ctrlleft'):
                pag.press(['n'])
            pag.PAUSE = 0
            pag.typewrite("print(\"Hello, World\")")

#--------- Variables for Media pipe
mpHands = mp.solutions.hands # telling media pipe we will be using a hand dection model
hands = mpHands.Hands(min_tracking_confidence=0.9, min_detection_confidence= 0.9, max_num_hands=2) # settgin up hand destion model
mpDraw = mp.solutions.drawing_utils # settgin up drawin utils so handland moarks can be drawn
# https://www.youtube.com/watch?v=NZde8Xt78Iw&t=1034s for version 1 set up


#--------- Variables for Open CV
cap = cv2.VideoCapture(0)
video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height
screen_width, screen_hight = pag.size()  # getting the size of the useres screen



#------- Other
click_mouse = True
current_state_time = 0
past_state_time =  0
past_state = 0

trigger_hand_pose = False
new_pose_detected = False
running_hand_pose_timer = False
hand_pose_timer_done = False


have_starting_wrist_crood = False
stating_writs_X = 0
starting_wrist_Y = 0
new_wrist_x = 0
new_writs_y = 0
current_move_timer = 0
past_move_timer = 0
right_wrist_x, right_wrist_y, left_wrist_x, left_wrist_y = 0, 0, 0, 0

finger_array = np.zeros((16,2), dtype=int)
one_handed_finger_extended = np.zeros(10, dtype=bool)

current_time = 0
past_time = 0

hand_orientaion = 0

def ColorThumb(hand,c):

 cv2.circle(image, (int(hand[0][0] * video_width),
                    int(hand[0][1] * video_height)), 5, c, cv2.FILLED)
 cv2.circle(image, (int(hand[1][0] * video_width),
                    int(hand[1][1] * video_height)), 5, c, cv2.FILLED)


def ColorIndex(hand, c):
    cv2.circle(image, (int(hand[2][0] * video_width),
                       int(hand[2][1] * video_height)), 5, c, cv2.FILLED)

    cv2.circle(image, (int(hand[3][0] * video_width),
                       int(hand[3][1] * video_height)), 5, c, cv2.FILLED)


def ColorMiddle(hand, c):
    cv2.circle(image, (int(hand[4][0] * video_width),
                       int(hand[4][1] * video_height)), 5, c, cv2.FILLED)

    cv2.circle(image, (int(hand[5][0] * video_width),
                       int(hand[5][1] * video_height)), 5, c, cv2.FILLED)


def ColorRing(hand, c):
    cv2.circle(image, (int(hand[6][0] * video_width),
                       int(hand[6][1] * video_height)), 5, c, cv2.FILLED)

    cv2.circle(image, (int(hand[7][0] * video_width),
                       int(hand[7][1] * video_height)), 5, c, cv2.FILLED)

def ColorLittle(hand, c):
    cv2.circle(image, (int(hand[8][0] * video_width),
                       int(hand[8][1] * video_height)), 5, c, cv2.FILLED)

    cv2.circle(image, (int(hand[9][0] * video_width),
                       int(hand[9][1] * video_height)), 5, c, cv2.FILLED)

def debug_gestuers(state, hand):

    if option_single_hand.is_active():
        if left_hand_orientation == 1 or right_hand_orientation == 1:
            if state == 17 or state == 544:
                ColorLittle(hand,(139,69,19))
                ColorThumb(hand,(139,69,19))
            if state == 19 or state == 608:
                ColorLittle(hand, (245,245,220))
                ColorThumb(hand, (245,245,220))
                ColorIndex(hand,(245,245,220))
            if state == 3 or state == 96:
                ColorThumb(hand, (221,160,221))
                ColorIndex(hand,(221,160,221))
            if state == 7 or state == 224:
                ColorThumb(hand, (255,69,0))
                ColorIndex(hand, (255,69,0))
                ColorMiddle(hand,(255,69,0))
            if state == 2 or state == 64:
                ColorIndex(hand,(255,255,0))
            if state == 6 or state == 192:
                ColorIndex(hand,(0,250,154))
                ColorMiddle(hand,(0,250,154))
            if state == 14 or state == 448:
                ColorIndex(hand, (64,224,208))
                ColorMiddle(hand, (64,224,208))
                ColorRing(hand,(64,224,208) )
            if state == 16 or state == 512:
                ColorLittle(hand, (123,104,238))
            if state == 30 or state == 960:
                ColorLittle(hand, (112,128,144))
                ColorIndex(hand, (112,128,144))
                ColorMiddle(hand, (112,128,144))
                ColorRing(hand, (112,128,144))
            if state == 31 or state == 992:
                ColorLittle(hand, (255, 255, 255))
                ColorIndex(hand, (255, 255, 255))
                ColorMiddle(hand, (255, 255, 255))
                ColorRing(hand, (255, 255, 255))
                ColorThumb(hand, (255, 255, 255))

        if left_hand_orientation == 2 or right_hand_orientation == 2:
            if state == 31 or state == 992:
                ColorLittle(hand, (0, 0, 0))
                ColorIndex(hand, (0, 0, 0))
                ColorMiddle(hand, (0, 0, 0))
                ColorRing(hand, (0, 0, 0))

            if state == 3 or state == 96:
                ColorThumb(hand, (0,255,50))
                ColorIndex(hand,(0,255,50))

            if state == 7 or state == 224:
                ColorThumb(hand, (123,123,123))
                ColorIndex(hand, (123,123,123))
                ColorMiddle(hand,(123,123,123))

        if left_hand_orientation == 0 or right_hand_orientation == 0:
            if state == 30 or state == 960:
                ColorLittle(hand, (0, 0, 255))
                ColorIndex(hand, (0, 0, 255))
                ColorMiddle(hand, (0, 0, 255))
                ColorRing(hand, (0, 0, 255))

    if option_multi_hand.is_active():
        #print("here")
        if left_hand_orientation == 1:
            if state == 17 and option_speech.is_active():
                ColorLittle(hand, (139, 69, 19))
                ColorThumb(hand, (139, 69, 19))

            if state == 19 and option_speech.is_active():
                ColorLittle(hand, (245, 245, 220))
                ColorThumb(hand, (245, 245, 220))
                ColorIndex(hand, (245, 245, 220))

        if right_hand_orientation == 1:

            if state == 97 or state == 96:
                ColorThumb(hand, (221, 160, 221))
                ColorIndex(hand, (221, 160, 221))
            if state == 225 or state == 224:
                ColorThumb(hand, (255, 69, 0))
                ColorIndex(hand, (255, 69, 0))
                ColorMiddle(hand, (255, 69, 0))
            if state == 65 or state == 64:
                ColorIndex(hand, (255, 255, 0))
            if state == 193 or state == 192:
                ColorIndex(hand, (0, 250, 154))
                ColorMiddle(hand, (0, 250, 154))
            if state == 449 or state == 448:
                ColorIndex(hand, (64, 224, 208))
                ColorMiddle(hand, (64, 224, 208))
                ColorRing(hand, (64, 224, 208))
            if state == 513 or state == 512:
                ColorLittle(hand, (123, 104, 238))
            if state == 961 or state == 960:
                ColorLittle(hand, (112, 128, 144))
                ColorIndex(hand, (112, 128, 144))
                ColorMiddle(hand, (112, 128, 144))
                ColorRing(hand, (112, 128, 144))
            if state == 993 or state == 992:
                ColorLittle(hand, (255, 255, 255))
                ColorIndex(hand, (255, 255, 255))
                ColorMiddle(hand, (255, 255, 255))
                ColorRing(hand, (255, 255, 255))

        elif left_hand_orientation == 2 or right_hand_orientation == 2:
            if state == 3:
                ColorThumb(hand, (123, 123, 123))
                ColorIndex(hand, (123, 123, 123))
                ColorMiddle(hand, (123, 123, 123))
            if state == 7:
                ColorThumb(hand, (0, 255, 50))
                ColorIndex(hand, (0, 255, 50))
            if state == 31:
                ColorLittle(hand, (0, 0, 0))
                ColorIndex(hand, (0, 0, 0))
                ColorMiddle(hand, (0, 0, 0))
                ColorRing(hand, (0, 0, 0))

        elif right_hand_orientation == 0:
            if state == 961 or state == 960:
                ColorLittle(hand, (255, 255, 255))
                ColorIndex(hand, (255, 255, 255))
                ColorMiddle(hand, (255, 255, 255))
                ColorRing(hand, (255, 255, 255))
                ColorThumb(hand, (255, 255, 255))

def FPS(past_time):
    current_time = time.time()
    fps = 1 / (current_time - past_time)
    cv2.putText(image, f'{int(fps)}', (0, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)  # Printing the FPS
    past_time = current_time
    return past_time


#~~~~~~~~~~~~~~~~~~~NEW Version Functions~~~~~~~~~~~~~~~~~~~~

# variables used to for contorlling what stage of claibariotn the user is at
calibrated = False
calibrated_hand_in_frame = False
calibrated_wrist_on_dot = False
calibration_done = False
calibrated_timer_running = False
calibration_got_closed = False
calibrated_current_timer = 0
calibrated_past_timer = 0

open_options_menu = False

# Variables for getting the measurements of the users hand
take_open_hand_measurement = True # if true then we are taking a measurement of an open hand

# System takes 10 measurements 5 open-handed and 5 closed, once all values in the arrays are true
# the system will know it has all the measuremtns it needs
open_measurement_taken = np.zeros((2, 5))
closed_measurement_taken = np.zeros((2, 5))

# System takes note of the distance the users hand is at each open-handed measurment
# values are stored here
distance_line_measurements = np.zeros((2, 5))

# Distance of the fingertips from the hand are in open and cloased states are stored here
extended_lenghts = np.zeros((2,5,5))
closed_lenghts = np.zeros((2,5,5))
thumb_to_pinky_lengths = np.zeros((2,5))

# variable is used to determine if the current hand being measured is the left or right hand
current_hand = 2
mouse_down = False

five_second_timer_over = False
triggered_state = 0
triggered_left_state = 0
triggered_right_state = 0
past_left_state = 0
past_right_sate = 0

multi_left_hand_state = 0
right_hand_active = 0
left_hand_active = 0
mouse_mode = 0



entered_fast_mode = False
fast_mode_triggered = False
fast_mouse_timer = 0
fast_mode_timer = 0

def color_full_hand(hand, color):
    cv2.circle(image, (int(hand[0][0] * video_width),
                       int(hand[0][1] * video_height)), 5, color, cv2.FILLED)

    cv2.circle(image, (int(hand[1][0] * video_width),
                       int(hand[1][1] * video_height)), 5, color, cv2.FILLED)

    cv2.circle(image, (int(hand[2][0] * video_width),
                       int(hand[2][1] * video_height)), 5, color, cv2.FILLED)

    cv2.circle(image, (int(hand[3][0] * video_width),
                       int(hand[3][1] * video_height)), 5, color, cv2.FILLED)

    cv2.circle(image, (int(hand[4][0] * video_width),
                       int(hand[4][1] * video_height)), 5, color, cv2.FILLED)

    cv2.circle(image, (int(hand[5][0] * video_width),
                       int(hand[5][1] * video_height)), 5, color, cv2.FILLED)

    cv2.circle(image, (int(hand[6][0] * video_width),
                       int(hand[6][1] * video_height)), 5, color, cv2.FILLED)

    cv2.circle(image, (int(hand[7][0] * video_width),
                       int(hand[7][1] * video_height)), 5, color, cv2.FILLED)

    cv2.circle(image, (int(hand[8][0] * video_width),
                       int(hand[8][1] * video_height)), 5, color, cv2.FILLED)

    cv2.circle(image, (int(hand[9][0] * video_width),
                       int(hand[9][1] * video_height)), 5, color, cv2.FILLED)

    cv2.circle(image, (int(hand[10][0] * video_width),
                       int(hand[10][1] * video_height)), 5, color, cv2.FILLED)


def debug_paint_full_hands(debug, two_hands):
    if debug:
        if len(two_hands) == 0 and hand_order[0] == "Right":
            color_full_hand(right_hand_landmarks, (255, 0, 0))
        if len(two_hands) == 0 and hand_order[0] == "Left":
            color_full_hand(left_hand_landmarks, (0, 0, 255))
        if len(two_hands) == 11:
            color_full_hand(left_hand_landmarks, (0, 255, 0))
            color_full_hand(right_hand_landmarks, (255, 255, 0))


def get_left_hand_orientation(left_landmarks, thumb_to_pinky):
    l2m = math.fabs(left_landmarks[2][0] - left_landmarks[4][0])
    r2m = math.fabs(left_landmarks[6][0] - left_landmarks[4][0])
    i2m = math.fabs(left_landmarks[8][0] - left_landmarks[4][0])

    if (left_landmarks[0][1] + thumb_to_pinky * 8) < left_landmarks[8][1]:
        return 2
    elif (l2m + r2m + i2m) < 0.07:
        return 0
    elif left_landmarks[0][0] > left_landmarks[8][0]:
        return 1
    elif left_landmarks[0][0] < left_landmarks[8][0]:
        return 0


def debug_left_hand_orientation(left_hand_orientation):
    if left_hand_orientation == 0:
        cv2.putText(image, f'Left: Undefined', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
    elif left_hand_orientation == 1:
        cv2.putText(image, f'Left: Inwards', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
    elif left_hand_orientation == 3:
        cv2.putText(image, f'Left: Outwards', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
    elif left_hand_orientation == 2:
        cv2.putText(image, f'Left: upwards', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

def get_right_hand_orientation(right_landmarks, thumb_to_pinky):
    l2m = math.fabs(right_landmarks[2][0] - right_landmarks[4][0])
    r2m = math.fabs(right_landmarks[6][0] - right_landmarks[4][0])
    i2m = math.fabs(right_landmarks[8][0] - right_landmarks[4][0])

    if (right_landmarks[0][1] + thumb_to_pinky * 8) < right_landmarks[8][1]:
        return 2
    elif (l2m + r2m + i2m) < 0.07:
        return 0
    elif right_landmarks[0][0] < right_landmarks[8][0]:
        return 1
    elif right_landmarks[0][0] > right_landmarks[8][0]:
        return 0


def debug_right_hand_orientation(right_hand_orientation):
    if right_hand_orientation == 0:
        cv2.putText(image, f'Right: Undefined', (0, 75), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
    elif right_hand_orientation == 1:
        cv2.putText(image, f'Right: Inwards', (0, 75), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
    elif right_hand_orientation == 3:
        cv2.putText(image, f'Right: Outwards', (0, 75), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
    elif right_hand_orientation == 2:
        cv2.putText(image, f'Right: upwards', (0, 75), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

def debug_draw_lines(c):

    cv2.line(image, (int(hand_one_landmarks[0][0] * video_width), int(hand_one_landmarks[0][1] * video_height)),
             (int(hand_one_landmarks[9][0] * video_width), int(hand_one_landmarks[9][1] * video_height)), c,
             2)

    cv2.line(image, (int(hand_one_landmarks[2][0] * video_width), int(hand_one_landmarks[2][1] * video_height)),
             (int(hand_one_landmarks[10][0] * video_width), int(hand_one_landmarks[10][1] * video_height)), c,
             2)

    cv2.line(image, (int(hand_one_landmarks[4][0] * video_width), int(hand_one_landmarks[4][1] * video_height)),
             (int(hand_one_landmarks[10][0] * video_width), int(hand_one_landmarks[10][1] * video_height)), c,
             2)

    cv2.line(image, (int(hand_one_landmarks[6][0] * video_width), int(hand_one_landmarks[6][1] * video_height)),
             (int(hand_one_landmarks[10][0] * video_width), int(hand_one_landmarks[10][1] * video_height)), c,
             2)

    cv2.line(image, (int(hand_one_landmarks[8][0] * video_width), int(hand_one_landmarks[8][1] * video_height)),
             (int(hand_one_landmarks[10][0] * video_width), int(hand_one_landmarks[10][1] * video_height)), c,
             2)

## chnage back to hand[10] if things gets worse
def get_hypot(hand):
    hypots = [math.hypot(hand[0][0] * video_width - hand[1][0] * video_width,
                         hand[0][1] * video_height - hand[9][1] * video_height),
              math.hypot(hand[2][0] * video_width - hand[10][0] * video_width,
                         hand[2][1] * video_height - hand[10][1] * video_height),
              math.hypot(hand[4][0] * video_width - hand[10][0] * video_width,
                         hand[4][1] * video_height - hand[10][1] * video_height),
              math.hypot(hand[6][0] * video_width - hand[10][0] * video_width,
                         hand[6][1] * video_height - hand[10][1] * video_height),
              math.hypot(hand[8][0] * video_width - hand[10][0] * video_width,
                         hand[8][1] * video_height - hand[10][1] * video_height)]
    return hypots

def are_fingers_extended(closed_measurements, open_measurements, current_measurments,right_hand):

    extended_array = np.zeros((5))
    if math.fabs(closed_measurements[0] - current_measurments[0]) > math.fabs(open_measurements[0] - current_measurments[0] * 1.2):
        extended_array[0] = True

    if math.fabs(closed_measurements[1] - current_measurments[1]) > math.fabs(open_measurements[1] - current_measurments[1] * 1.2):
        extended_array[1] = True

    if math.fabs(closed_measurements[2] - current_measurments[2]) > math.fabs(open_measurements[2] - current_measurments[2] * 1.2):
        extended_array[2] = True

    if math.fabs(closed_measurements[3] - current_measurments[3]) > math.fabs(open_measurements[3] - current_measurments[3] * 1.2):
        extended_array[3] = True

    if math.fabs(closed_measurements[4] - current_measurments[4]) > math.fabs(open_measurements[4] - current_measurments[4] * 1.2):
        extended_array[4] = True

    return extended_array

options_in_box = False
options_box_triggered = False
options_box_number = 0
options_start_timer = False
options_current_timer = 0
options_past_timer = 0

option_general     =  Square((10, 0), (60, 50), [0, 255, 0], True, "General", False, False)
option_debug       =  Square((500, 0), (550, 50), [0, 255, 0], False, "Debug", False, False)
option_single_hand =  Square((10, 100), (60, 150), [0, 255, 0], True, "Single Hand", False, False)
option_left_hand   =  Square((40, 160), (90, 210), [0, 255, 0], True, "Left hand", False, False)
option_right_hand  =  Square((40, 220), (90, 270), [0, 255, 0], False, "Right Hand", False, False)
option_multi_hand  =  Square((10, 300), (60, 350), [0, 255, 0], False, "Multi Hand", False, False)
option_speech      =  Square((10, 400), (60, 450), [0, 255, 0], True, "Speech", False, False)
option_gesture     =  Square((10, 520), (60, 570), [0, 255, 0], True, "Gesture", False, False)
option_done        =  Square((10, 900), (60, 950), [0, 255, 0], False, "Done", False, False)
option_close       =  Square((500, 900), (550, 950), [0, 255, 0], False, "Close", False, False)

option_track_hands =  Square((10, 100), (60, 150), [0, 255, 0], False, "Paint Hands", False, False)
option_extended    =  Square((10, 200), (60, 250), [0, 255, 0], False, "Paint Fingers", False, False)
option_gestures    =  Square((10, 300), (60, 350), [0, 255, 0], False, "Paint Gestures", False, False)
option_orientation =  Square((10, 400), (60, 450), [0, 255, 0], False, "Show Orientation", False, False)
option_FPS         =  Square((10, 500), (60, 550), [0, 255, 0], False, "Show FPS", False, False)


if __name__ == '__main__':

    # If in the TK window before run program option was selected the rest of the system will start running
    while not option_close.is_active():

        # Getting Webcam image data
        success, image = cap.read()

        # Flipping the image so left and right are on the correct side for the user viewing the system
        image = cv2.flip(image, 90)

        if open_options_menu:
            image = cv2.resize(image, (1000, 1000))
            video_width = 1000
            video_height = 1000
        else:
            video_height = 480
            video_width = 640

        # Swapping the pixel color data of the image from blue, green, red to red, green, blue as this is
        # the order Media Pipe needs to use
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Passing the converted image to the Media Pipe hand detection model to get the results
        results = hands.process(image_RGB)

        # If results is not empty then the system detected a hand in which case the rest of the program can run
        if results.multi_hand_landmarks:

            calibrated_hand_in_frame = True

            # Making sure we know what the order the hands came one the screen
            # This is important for assigning the correct landmark data to the
            # correct hand
            hand_order = []
            i = 0
            for hand in results.multi_handedness:
                if i == 0 and hand.classification[0].label == "Right":
                    hand_order.append("Right")
                elif i == 0 and hand.classification[0].label == "Left":
                    hand_order.append("Left")
                elif i == 1 and hand.classification[0].label == "Right":
                    hand_order.append("Right")
                elif i == 1 and hand.classification[0].label == "Left":
                    hand_order.append("Left")
                i = i + 1

            # assigning hand landmark data for one or two hands, loop will run for as many hand as the system detected
            hand_one_landmarks = []
            hand_two_landmarks = []
            i = 0
            for hand_landmarks in results.multi_hand_landmarks:

                if i == 0:
                    hand_one_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP].y])
                    hand_one_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.THUMB_CMC].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.THUMB_CMC].y])

                    hand_one_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y])
                    hand_one_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP].y])

                    hand_one_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].y])
                    hand_one_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].y])

                    hand_one_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_TIP].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_TIP].y])
                    hand_one_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_MCP].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_MCP].y])

                    hand_one_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP].y])
                    hand_one_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.PINKY_MCP].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.PINKY_MCP].y])

                    hand_one_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.WRIST].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.WRIST].y])

                elif len(hand_order) == 2 and i == 1:
                    hand_two_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP].y])
                    hand_two_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.THUMB_CMC].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.THUMB_CMC].y])

                    hand_two_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y])
                    hand_two_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP].y])

                    hand_two_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].y])
                    hand_two_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].y])

                    hand_two_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_TIP].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_TIP].y])
                    hand_two_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_MCP].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_MCP].y])

                    hand_two_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP].y])
                    hand_two_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.PINKY_MCP].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.PINKY_MCP].y])

                    hand_two_landmarks.append([hand_landmarks.landmark[mpHands.HandLandmark.WRIST].x,
                                               hand_landmarks.landmark[mpHands.HandLandmark.WRIST].y])
                i = i + 1

                # Using the order we detected the hands in along with the landmark data we collected
                # and assigning the landmarks form hane_one_landmarks & hand_two_landmarks to
                # the correct either  left_hand_landmarks or right_hand_landmarks to make it easier
                # to know which set of landmark data belongs to which hand.
                left_hand_landmarks = 0
                right_hand_landmarks = 0
                if len(hand_two_landmarks) == 0 and hand_order[0] == "Right":
                    right_hand_landmarks = hand_one_landmarks
                elif len(hand_two_landmarks) == 0 and hand_order[0] == "Left":
                    left_hand_landmarks = hand_one_landmarks
                elif len(hand_two_landmarks) == 11 and hand_order[0] == "Left":
                    left_hand_landmarks = hand_one_landmarks
                    right_hand_landmarks = hand_two_landmarks
                elif len(hand_two_landmarks) == 11 and hand_order[0] == "Right":
                    right_hand_landmarks = hand_one_landmarks
                    left_hand_landmarks = hand_two_landmarks

                # Debug option to show all the landmarks of any hand detected on screen
                debug_paint_full_hands(option_track_hands.is_active(), hand_two_landmarks)

                if not calibrated and not calibration_done:

                    calibrated_current_timer = time.time()
                    current_hand = 2
                    if hand_order[0] == "Left":
                        if not closed_measurement_taken[0][4]:
                            current_hand = 0

                    elif hand_order[0] == "Right":
                        if not closed_measurement_taken[1][4]:
                            current_hand = 1

                    if not calibrated_timer_running:
                        calibrated_past_timer = time.time()
                        calibrated_timer_running = True

                    cv2.line(image, (int(hand_one_landmarks[3][0] * video_width), int(hand_one_landmarks[3][1] * video_height)),(int(hand_one_landmarks[9][0] * video_width),int(hand_one_landmarks[9][1] * video_height)), (0, 0, 0),2)

                    if not five_second_timer_over:
                        cv2.putText(image, f'{round(calibrated_current_timer - calibrated_past_timer)}',(int(video_width / 2), 25), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(image, f'{round(calibrated_current_timer - calibrated_past_timer)}',(int(video_width / 2), 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
                        if round(calibrated_current_timer - calibrated_past_timer) > 2:
                            five_second_timer_over = True
                            calibrated_past_timer = time.time()

                    if calibrated_timer_running and five_second_timer_over and current_hand != 2:
                        cv2.putText(image, f'{round(calibrated_current_timer - calibrated_past_timer)}',(int(video_width / 2), 25), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(image, f'{round(calibrated_current_timer - calibrated_past_timer) }', (int(video_width / 2), 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

                        # Taking the 10 measurements at different time intervals
                        if take_open_hand_measurement:

                            debug_draw_lines((255,255,255))
                            if round(calibrated_current_timer - calibrated_past_timer) > 2 and not open_measurement_taken[current_hand][0]:
                                distance_line_measurements[current_hand][0] = math.hypot(hand_one_landmarks[3][0] * video_width - hand_one_landmarks[9][0] * video_width, hand_one_landmarks[3][1] * video_height - hand_one_landmarks[9][1] * video_height)
                                open_measurement_taken[current_hand][0] = True
                                calibrated_past_timer = time.time()
                                extended_lenghts[current_hand][0] = get_hypot(hand_one_landmarks)
                                take_open_hand_measurement = False

                            elif(round(calibrated_current_timer - calibrated_past_timer) > 2 and not open_measurement_taken[current_hand][1]):
                                distance_line_measurements[current_hand][1] = math.hypot(hand_one_landmarks[3][0] * video_width - hand_one_landmarks[9][0] * video_width, hand_one_landmarks[3][1] * video_height - hand_one_landmarks[9][1] * video_height)
                                open_measurement_taken[current_hand][1] = True
                                calibrated_past_timer = time.time()
                                extended_lenghts[current_hand][1] = get_hypot(hand_one_landmarks)
                                take_open_hand_measurement = False

                            elif(round(calibrated_current_timer - calibrated_past_timer) > 2 and not open_measurement_taken[current_hand][2]):
                                distance_line_measurements[current_hand][2] = math.hypot(hand_one_landmarks[3][0] * video_width - hand_one_landmarks[9][0] * video_width, hand_one_landmarks[3][1] * video_height - hand_one_landmarks[9][1] * video_height)
                                open_measurement_taken[current_hand][2] = True
                                calibrated_past_timer = time.time()
                                extended_lenghts[current_hand][2] = get_hypot(hand_one_landmarks)
                                take_open_hand_measurement = False

                            elif (round(calibrated_current_timer - calibrated_past_timer) > 2 and not open_measurement_taken[current_hand][3]):
                                distance_line_measurements[current_hand][3] = math.hypot(hand_one_landmarks[3][0] * video_width - hand_one_landmarks[9][0] * video_width, hand_one_landmarks[3][1] * video_height - hand_one_landmarks[9][1] * video_height)
                                open_measurement_taken[current_hand][3] = True
                                calibrated_past_timer = time.time()
                                extended_lenghts[current_hand][3] = get_hypot(hand_one_landmarks)
                                take_open_hand_measurement = False

                            elif (round(calibrated_current_timer - calibrated_past_timer) > 2 and not open_measurement_taken[current_hand][4]):
                                distance_line_measurements[current_hand][4] = math.hypot(hand_one_landmarks[3][0] * video_width - hand_one_landmarks[9][0] * video_width, hand_one_landmarks[3][1] * video_height - hand_one_landmarks[9][1] * video_height)
                                open_measurement_taken[current_hand][4] = True
                                calibrated_past_timer = time.time()
                                extended_lenghts[current_hand][4] = get_hypot(hand_one_landmarks)
                                take_open_hand_measurement = False

                        else:
                            debug_draw_lines((255, 0, 0))
                            if round(calibrated_current_timer - calibrated_past_timer) > 2 and not closed_measurement_taken[current_hand][0]:
                                closed_measurement_taken[current_hand][0] = True
                                calibrated_past_timer = time.time()
                                closed_lenghts[current_hand][0] = get_hypot(hand_one_landmarks)
                                thumb_to_pinky_lengths[current_hand][0] = math.fabs(hand_one_landmarks[0][0] - hand_one_landmarks[8][0])
                                take_open_hand_measurement = True

                            elif (round(calibrated_current_timer - calibrated_past_timer) > 2 and not closed_measurement_taken[current_hand][1]):
                                closed_measurement_taken[current_hand][1] = True
                                calibrated_past_timer = time.time()
                                closed_lenghts[current_hand][1] = get_hypot(hand_one_landmarks)
                                thumb_to_pinky_lengths[current_hand][1] = math.fabs(hand_one_landmarks[0][0] - hand_one_landmarks[8][0])
                                take_open_hand_measurement = True

                            elif (round(calibrated_current_timer - calibrated_past_timer) > 2 and not closed_measurement_taken[current_hand][2]):
                                closed_measurement_taken[current_hand][2] = True
                                calibrated_past_timer = time.time()
                                closed_lenghts[current_hand][2] = get_hypot(hand_one_landmarks)
                                thumb_to_pinky_lengths[current_hand][2] = math.fabs(hand_one_landmarks[0][0] - hand_one_landmarks[8][0])
                                take_open_hand_measurement = True

                            elif (round(calibrated_current_timer - calibrated_past_timer) > 2 and not closed_measurement_taken[current_hand][3]):
                                closed_measurement_taken[current_hand][3] = True
                                calibrated_past_timer = time.time()
                                closed_lenghts[current_hand][3] = get_hypot(hand_one_landmarks)
                                thumb_to_pinky_lengths[current_hand][3] = math.fabs(hand_one_landmarks[0][0] - hand_one_landmarks[8][0])
                                take_open_hand_measurement = True

                            elif (round(calibrated_current_timer - calibrated_past_timer) > 2 and not closed_measurement_taken[current_hand][4]):
                                closed_measurement_taken[current_hand][4] = True
                                calibrated_past_timer = time.time()
                                closed_lenghts[current_hand][4] = get_hypot(hand_one_landmarks)
                                thumb_to_pinky_lengths[current_hand][4] = math.fabs(hand_one_landmarks[0][0] - hand_one_landmarks[8][0])
                                take_open_hand_measurement = True

                            if closed_measurement_taken[0][4] and closed_measurement_taken[1][4]:
                                calibrated = True
                                calibration_done = True
                                open_options_menu = True

            if calibration_done and open_options_menu:
                cv2.circle(image, (int(hand_one_landmarks[0][0] * video_width),
                                   int(hand_one_landmarks[0][1] * video_height)), 5, (255, 0, 255), cv2.FILLED)

            if calibration_done and not open_options_menu:
                state = 0
                left_hand_orientation = 0
                left_fingers_extended = np.zeros((5))
                if not left_hand_landmarks == 0 and (option_left_hand.is_active() or option_multi_hand.is_active()):
                    current_measurment = math.hypot(hand_one_landmarks[3][0] * video_width - hand_one_landmarks[9][0] * video_width, hand_one_landmarks[3][1] * video_height - hand_one_landmarks[9][1] * video_height)

                    comparison_of_measurement = math.fabs(current_measurment - distance_line_measurements[0][0])
                    orientation_measurement = distance_line_measurements[0][0]
                    closest_measurement = 1

                    if math.fabs(current_measurment - distance_line_measurements[0][1]) < comparison_of_measurement:
                        comparison_of_measurement = math.fabs(current_measurment - distance_line_measurements[0][1])
                        orientation_measurement = distance_line_measurements[0][1]
                        closest_measurement = 2
                    if math.fabs(current_measurment - distance_line_measurements[0][2]) < comparison_of_measurement:
                        comparison_of_measurement = math.fabs(current_measurment - distance_line_measurements[0][2])
                        orientation_measurement = distance_line_measurements[0][2]
                        closest_measurement = 3
                    if math.fabs(current_measurment - distance_line_measurements[0][3]) < comparison_of_measurement:
                        comparison_of_measurement = math.fabs(current_measurment - distance_line_measurements[0][3])
                        orientation_measurement = distance_line_measurements[0][3]
                        closest_measurement = 4
                    if math.fabs(current_measurment - distance_line_measurements[0][4]) < comparison_of_measurement:
                        comparison_of_measurement = math.fabs(current_measurment - distance_line_measurements[0][4])
                        orientation_measurement = distance_line_measurements[0][4]
                        closest_measurement = 5

                    finger_line_lengths = get_hypot(left_hand_landmarks)
                    left_hand_orientation = get_left_hand_orientation(left_hand_landmarks, thumb_to_pinky_lengths[0][closest_measurement - 1])

                    if option_orientation.is_active():
                        debug_left_hand_orientation(left_hand_orientation)

                    if closest_measurement == 1:
                        left_fingers_extended = are_fingers_extended(closed_lenghts[0][0], extended_lenghts[0][0], finger_line_lengths, False)

                    if closest_measurement == 2:
                        left_fingers_extended = are_fingers_extended(closed_lenghts[0][1], extended_lenghts[0][1], finger_line_lengths, False)

                    if closest_measurement == 3:
                        left_fingers_extended = are_fingers_extended(closed_lenghts[0][2], extended_lenghts[0][2], finger_line_lengths, False)

                    if closest_measurement == 4:
                        left_fingers_extended = are_fingers_extended(closed_lenghts[0][3], extended_lenghts[0][3], finger_line_lengths, False)

                    if closest_measurement == 5:
                        left_fingers_extended = are_fingers_extended(closed_lenghts[0][4], extended_lenghts[0][4], finger_line_lengths, False)

                    # assigning values to left fingers if they are found to be extended
                    if left_fingers_extended[0]:
                        state = state + 1
                    if left_fingers_extended[1]:
                        state = state + 2
                    if left_fingers_extended[2]:
                        state = state + 4
                    if left_fingers_extended[3]:
                        state = state + 8
                    if left_fingers_extended[4]:
                        state = state + 16

                right_hand_orientation = 0
                right_fingers_extended = np.zeros((5))
                if not right_hand_landmarks == 0 and (option_right_hand.is_active() or option_multi_hand.is_active()):
                    current_measurment = math.hypot(hand_one_landmarks[3][0] * video_width - hand_one_landmarks[9][0] * video_width,hand_one_landmarks[3][1] * video_height - hand_one_landmarks[9][1] * video_height)

                    comparison_of_measurement = math.fabs(current_measurment - distance_line_measurements[1][0])
                    orientation_measurement = distance_line_measurements[1][0]
                    closest_measurement = 1

                    if math.fabs(current_measurment - distance_line_measurements[1][1]) < comparison_of_measurement:
                        comparison_of_measurement = math.fabs(current_measurment - distance_line_measurements[1][1])
                        orientation_measurement = distance_line_measurements[1][1]
                        closest_measurement = 2
                    if math.fabs(current_measurment - distance_line_measurements[1][2]) < comparison_of_measurement:
                        comparison_of_measurement = math.fabs(current_measurment - distance_line_measurements[1][2])
                        orientation_measurement = distance_line_measurements[1][2]
                        closest_measurement = 3
                    if math.fabs(current_measurment - distance_line_measurements[1][3]) < comparison_of_measurement:
                        comparison_of_measurement = math.fabs(current_measurment - distance_line_measurements[1][3])
                        orientation_measurement = distance_line_measurements[1][3]
                        closest_measurement = 4
                    if math.fabs(current_measurment - distance_line_measurements[1][4]) < comparison_of_measurement:
                        comparison_of_measurement = math.fabs(current_measurment - distance_line_measurements[1][4])
                        orientation_measurement = distance_line_measurements[1][4]
                        closest_measurement = 5

                    finger_line_lengths = get_hypot(right_hand_landmarks)
                    right_hand_orientation = get_right_hand_orientation(right_hand_landmarks, thumb_to_pinky_lengths[1][closest_measurement - 1])

                    if option_orientation.is_active():
                        debug_right_hand_orientation(right_hand_orientation)

                    if closest_measurement == 1:
                        right_fingers_extended = are_fingers_extended(closed_lenghts[1][0], extended_lenghts[1][0],finger_line_lengths, False)

                    if closest_measurement == 2:
                        right_fingers_extended = are_fingers_extended(closed_lenghts[1][1], extended_lenghts[1][1],finger_line_lengths, False)

                    if closest_measurement == 3:
                        right_fingers_extended = are_fingers_extended(closed_lenghts[1][2], extended_lenghts[1][2],finger_line_lengths, False)

                    if closest_measurement == 4:
                        right_fingers_extended = are_fingers_extended(closed_lenghts[1][3], extended_lenghts[1][3],finger_line_lengths, False)

                    if closest_measurement == 5:
                        right_fingers_extended = are_fingers_extended(closed_lenghts[1][4], extended_lenghts[1][4],finger_line_lengths, False)

                    # assigning values to right fingers if they are found to be extended

                    if right_fingers_extended[0]:
                        state = state + 32
                    if right_fingers_extended[1]:
                        state = state + 64
                    if right_fingers_extended[2]:
                        state = state + 128
                    if right_fingers_extended[3]:
                        state = state + 256
                    if right_fingers_extended[4]:
                        state = state + 512

                #-----------------Variables for desiding on which mouse mode ot use
                mouse_landmarks = 0
                right_hand_active = option_multi_hand.is_active()
                left_hand_active  =  option_multi_hand.is_active()
                mouse_mode = 0

                fast_mode_timer = 0

                if option_left_hand.is_active() or option_multi_hand.is_active():

                    if left_hand_orientation == 1 and state == 1 or multi_left_hand_state == 1:
                        mouse_mode = 1

                    elif left_hand_orientation == 0 and not entered_fast_mode:
                        fast_mouse_timer = time.time()
                        entered_fast_mode = True

                    if(time.time() - fast_mouse_timer) > 1.5 and entered_fast_mode:
                        fast_mode_triggered = True

                    if left_hand_orientation == 0 and fast_mode_triggered:
                        mouse_mode = 2
                    elif left_hand_orientation == 1 or left_hand_orientation == 2:
                        fast_mode_triggered = False
                        entered_fast_mode = False

                    mouse_landmarks = left_hand_landmarks

                elif option_right_hand.is_active():

                    if right_hand_orientation == 1 and state == 32:
                        mouse_mode = 1

                    elif right_hand_orientation == 0 and not entered_fast_mode:
                        fast_mouse_timer = time.time()
                        entered_fast_mode = True

                    if (time.time() - fast_mouse_timer) > 1.5 and entered_fast_mode:
                        fast_mode_triggered = True

                    if right_hand_orientation == 0 and fast_mode_triggered:
                        mouse_mode = 2
                    elif right_hand_orientation == 1 or right_hand_orientation == 2 and state == 0:
                        fast_mode_triggered = False
                        entered_fast_mode = False
                    mouse_landmarks = right_hand_landmarks

                # ______________________Updating Mouse Postion

                if mouse_mode == 1 and mouse_landmarks  != 0:
                    ## multipying x/y index tip with screen size to get mouse position

                    current_move_timer = time.time()
                    if not have_starting_wrist_crood:
                        have_starting_wrist_crood = True
                        starting_wrist_Y = mouse_landmarks[10][1] * screen_hight
                        stating_writs_X = mouse_landmarks[10][0] * screen_width
                        past_move_timer = time.time()

                    if current_move_timer - past_move_timer > 0.01:

                        new_wrist_x = (mouse_landmarks[10][0] * screen_width) - stating_writs_X
                        new_writs_y = (mouse_landmarks[10][1] * screen_hight) - starting_wrist_Y
                        have_starting_wrist_crood = False
                    else:
                        new_wrist_x = 0
                        new_writs_y = 0

                    if 0 < pag.position().y + new_writs_y < screen_hight:
                        if 0 < pag.position().x + new_wrist_x < screen_width:
                            pag.move(new_wrist_x, new_writs_y)

                elif mouse_mode == 2 and mouse_landmarks  != 0:
                    if 0 < mouse_landmarks[10][1] * screen_hight < screen_hight:
                        if 0 < mouse_landmarks[10][0] * screen_width < screen_width:
                            pag.moveTo(mouse_landmarks[10][0] * screen_width, mouse_landmarks[10][1] * screen_hight)
                else:
                    have_starting_wrist_crood = False
                # ___________________________Pose Detection____________________

                if state > 1 and state != 32:
                    new_pose_detected = True
                else:
                    new_pose_detected = False
                    past_state = 0

                if new_pose_detected and not running_hand_pose_timer:
                    past_state_time = time.time()
                    past_state = state
                    running_hand_pose_timer = True

                current_state_time = time.time()
                time_to_pass = 0.2
                if state == 17 or state == 544 or state == 19 or state == 608:
                    time_to_pass = 1

                if running_hand_pose_timer and current_state_time - past_state_time > time_to_pass:
                    running_hand_pose_timer = False
                    hand_pose_timer_done = True

                if hand_pose_timer_done and past_state == state and past_state != triggered_state:
                    trigger_hand_pose = True
                    hand_pose_timer_done = False

                # ___________________________Pose Detection____________________
                if option_gestures.is_active() and (option_left_hand.is_active() or option_multi_hand.is_active()) and left_hand_landmarks != 0:
                    debug_gestuers(state, left_hand_landmarks)
                if option_gestures.is_active() and (option_left_hand.is_active() or option_multi_hand.is_active()) and right_hand_landmarks != 0:
                    debug_gestuers(state, right_hand_landmarks)

                print(f'{left_hand_orientation} ---   {state}')
                if option_single_hand.is_active():
                    if left_hand_orientation == 1 or right_hand_orientation == 1:
                        if state == 3 or state == 96:
                            pag.scroll(50)
                        if state == 7 or state == 224:
                            pag.scroll(-50)
                elif option_multi_hand.is_active():
                    if right_hand_orientation == 1:
                        if state == 97 or state == 96:
                            pag.scroll(50)
                        if state == 225 or state == 224:
                            pag.scroll(-50)

                #---------- Functions that get called if the corect hand pose is deteced
                if trigger_hand_pose and option_gesture.is_active():

                    if option_single_hand.is_active():
                        if left_hand_orientation == 1 or right_hand_orientation == 1:
                            if state == 17 or state == 544 and option_speech.is_active():

                                try:
                                    text = Speech2Text()
                                except BaseException as e:
                                    print(e)
                                else:
                                    pag.typewrite(text)

                            if state == 19 or state == 608 and option_speech.is_active():
                                try:
                                    text = Speech2Text()
                                except BaseException as e:
                                    print(e)
                                else:
                                    speech_command(text)

                            if state == 2 or state == 64:
                                pag.click()
                            if state == 6 or state == 192:
                                pag.click(button='right')
                            if state == 14 or state == 448:
                                pag.click()
                                pag.click()
                            if state == 16 or state == 512:
                                pag.press('backspace')
                            if state == 30 or state == 960:
                                pag.press('enter')
                            if state == 31 or state == 992:
                                pag.mouseDown()

                        elif left_hand_orientation == 2 or right_hand_orientation == 2:
                            if state == 3 or state == 96:
                                with pag.hold('ctrlleft'):
                                    pag.press(['x'])
                            if state == 7 or state == 224:
                                with pag.hold('ctrlleft'):
                                    pag.press(['c'])
                            if state == 31 or state == 992:
                                with pag.hold('ctrlleft'):
                                    pag.press(['v'])

                        elif left_hand_orientation == 0 or right_hand_orientation == 0:
                            if state == 30 or state == 960:
                                pag.mouseUp()

                    elif option_multi_hand.is_active():
                        if left_hand_orientation == 1:
                            if state == 17 and option_speech.is_active():

                                try:
                                    text = Speech2Text()
                                except BaseException as e:
                                    print(e)
                                else:
                                    pag.typewrite(text)

                            if state == 19 and option_speech.is_active():
                                try:
                                    text = Speech2Text()
                                except BaseException as e:
                                    print(e)
                                else:
                                    speech_command(text)

                            if state == 65 or state == 64:
                                pag.click()
                            if state == 193 or state == 192:
                                pag.click(button='right')
                            if state == 449 or state == 448:
                                pag.click()
                                pag.click()
                            if state == 513 or state == 512:
                                pag.press('backspace')
                            if state == 961 or state == 960:
                                pag.press('enter')
                            if state == 993 or state == 992:
                                pag.mouseDown()

                        elif left_hand_orientation == 2 or right_hand_orientation == 2:
                            if state == 3:
                                with pag.hold('ctrlleft'):
                                    pag.press(['x'])
                            if state == 7:
                                with pag.hold('ctrlleft'):
                                    pag.press(['c'])
                            if state == 31:
                                with pag.hold('ctrlleft'):
                                    pag.press(['v'])

                        elif right_hand_orientation == 0:
                            if state == 961 or state == 960:
                                pag.mouseUp()

                    trigger_hand_pose = False
                    triggered_state = past_state


                #-----------------Debug optin to draw circules on extended fingers
                if option_extended.is_active():
                    if left_fingers_extended[0] == True:
                        cv2.circle(image, (int(left_hand_landmarks[0][0] * video_width),
                                           int(left_hand_landmarks[0][1] * video_height)), 5, (0,0,0), cv2.FILLED)
                        cv2.circle(image, (int(left_hand_landmarks[1][0] * video_width),
                                           int(left_hand_landmarks[1][1] * video_height)), 5, (0, 0, 0), cv2.FILLED)
                    if left_fingers_extended[1] == True:
                        cv2.circle(image, (int(left_hand_landmarks[2][0] * video_width),
                                           int(left_hand_landmarks[2][1] * video_height)), 5, (255,0,0), cv2.FILLED)
                        cv2.circle(image, (int(left_hand_landmarks[3][0] * video_width),
                                           int(left_hand_landmarks[3][1] * video_height)), 5, (255, 0, 0), cv2.FILLED)
                    if left_fingers_extended[2] == True:
                        cv2.circle(image, (int(left_hand_landmarks[4][0] * video_width),
                                           int(left_hand_landmarks[4][1] * video_height)), 5, (0,255,0), cv2.FILLED)
                        cv2.circle(image, (int(left_hand_landmarks[5][0] * video_width),
                                           int(left_hand_landmarks[5][1] * video_height)), 5, (0, 255, 0), cv2.FILLED)
                    if left_fingers_extended[3] == True:
                        cv2.circle(image, (int(left_hand_landmarks[6][0] * video_width),
                                           int(left_hand_landmarks[6][1] * video_height)), 5, (0,0,255), cv2.FILLED)
                        cv2.circle(image, (int(left_hand_landmarks[7][0] * video_width),
                                           int(left_hand_landmarks[7][1] * video_height)), 5, (0, 0, 255), cv2.FILLED)
                    if left_fingers_extended[4] == True:
                        cv2.circle(image, (int(left_hand_landmarks[8][0] * video_width),
                                           int(left_hand_landmarks[8][1] * video_height)), 5, (255,255,0), cv2.FILLED)
                        cv2.circle(image, (int(left_hand_landmarks[9][0] * video_width),
                                           int(left_hand_landmarks[9][1] * video_height)), 5, (255, 255, 0), cv2.FILLED)

                    if right_fingers_extended[0] == True:
                        cv2.circle(image, (int(right_hand_landmarks[0][0] * video_width),
                                           int(right_hand_landmarks[0][1] * video_height)), 5, (0,255,255), cv2.FILLED)
                        cv2.circle(image, (int(right_hand_landmarks[1][0] * video_width),
                                           int(right_hand_landmarks[1][1] * video_height)), 5, (0, 255, 255), cv2.FILLED)
                    if right_fingers_extended[1] == True:
                        cv2.circle(image, (int(right_hand_landmarks[2][0] * video_width),
                                           int(right_hand_landmarks[2][1] * video_height)), 5, (255,255,255), cv2.FILLED)
                        cv2.circle(image, (int(right_hand_landmarks[3][0] * video_width),
                                           int(right_hand_landmarks[3][1] * video_height)), 5, (255, 255, 255), cv2.FILLED)
                    if right_fingers_extended[2] == True:
                        cv2.circle(image, (int(right_hand_landmarks[4][0] * video_width),
                                           int(right_hand_landmarks[4][1] * video_height)), 5, (255,0,255), cv2.FILLED)
                        cv2.circle(image, (int(right_hand_landmarks[5][0] * video_width),
                                           int(right_hand_landmarks[5][1] * video_height)), 5, (255, 0, 255), cv2.FILLED)
                    if right_fingers_extended[3] == True:
                        cv2.circle(image, (int(right_hand_landmarks[6][0] * video_width),
                                           int(right_hand_landmarks[6][1] * video_height)), 5, (205,90,106), cv2.FILLED)
                        cv2.circle(image, (int(right_hand_landmarks[7][0] * video_width),
                                           int(right_hand_landmarks[7][1] * video_height)), 5, (205,90,106), cv2.FILLED)
                    if right_fingers_extended[4] == True:
                        cv2.circle(image, (int(right_hand_landmarks[8][0] * video_width),
                                           int(right_hand_landmarks[8][1] * video_height)), 5, (160 ,158,195), cv2.FILLED)
                        cv2.circle(image, (int(right_hand_landmarks[9][0] * video_width),
                                           int(right_hand_landmarks[9][1] * video_height)), 5, (160 ,158,195), cv2.FILLED)

                options_past_timer = time.time()
        elif not open_options_menu and calibrated:
            if not options_start_timer:
                options_past_timer = time.time()
                options_start_timer = True
            if options_start_timer and time.time() - options_past_timer > 5:
                open_options_menu = True
                options_start_timer = False
                option_done.set_active(False)
        else:
            calibrated_timer_running = False
            calibrated_hand_in_frame = False

        if not calibrated and not calibrated_hand_in_frame:
            cv2.putText(image, f'Please place hand in frame', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f'Please place hand in frame', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

        #------------------Optiones Menu-------------------------------------------------
        if calibration_done and open_options_menu:
            cv2.rectangle(image, (0, 0), (1000, 1000), (255, 255, 255), -1)

            option_general.draw(image)
            option_debug.draw(image)

            if option_general.is_active():
                option_single_hand.draw(image)
                option_left_hand.draw(image)
                option_right_hand.draw(image)
                option_multi_hand.draw(image)
                option_speech.draw(image)
                option_gesture.draw(image)

            else:
                option_track_hands.draw(image)
                option_extended.draw(image)
                option_gestures.draw(image)
                option_orientation.draw(image)
                option_FPS.draw(image)

            option_done.draw(image)
            option_close.draw(image)

            cv2.circle(image, (int(hand_one_landmarks[0][0] * video_width),
                               int(hand_one_landmarks[0][1] * video_height)), 5, (255, 0, 255), cv2.FILLED)

            is_single_active = option_single_hand.is_active()
            is_left_hand_active = option_left_hand.is_active()
            is_general_active = option_general.is_active()
            is_debug_active = option_debug.is_active()

            option_general.detect_cursor(hand_one_landmarks)
            option_debug.detect_cursor(hand_one_landmarks)

            if option_general.is_active():
                option_single_hand.detect_cursor(hand_one_landmarks)
                option_left_hand.detect_cursor(hand_one_landmarks)
                option_right_hand.detect_cursor(hand_one_landmarks)
                option_multi_hand.detect_cursor(hand_one_landmarks)
                option_speech.detect_cursor(hand_one_landmarks)
                option_gesture.detect_cursor(hand_one_landmarks)

            else:
                option_track_hands.detect_cursor(hand_one_landmarks)
                option_extended.detect_cursor(hand_one_landmarks)
                option_gestures.detect_cursor(hand_one_landmarks)
                option_orientation.detect_cursor(hand_one_landmarks)
                option_FPS.detect_cursor(hand_one_landmarks)

            option_done.detect_cursor(hand_one_landmarks)
            option_close.detect_cursor(hand_one_landmarks)

            if is_single_active != option_single_hand.is_active():
                if is_single_active:
                    option_multi_hand.set_active(True)
                else:
                    option_multi_hand.set_active(False)

            if is_left_hand_active != option_left_hand.is_active():
                if is_left_hand_active:
                    option_right_hand.set_active(True)
                else:
                    option_right_hand.set_active(False)

            if is_general_active != option_general.is_active():
                if is_general_active:
                    option_debug.set_active(True)
                else:
                    option_debug.set_active(False)

            if is_debug_active != option_debug.is_active():
                if is_debug_active:
                    option_general.set_active(True)
                else:
                    option_general.set_active(False)

            option_general.set_color()
            option_debug.set_color()
            option_single_hand.set_color()
            option_left_hand.set_color()
            option_right_hand.set_color()
            option_multi_hand.set_color()
            option_speech.set_color()
            option_gesture.set_color()
            option_done.set_color()
            option_close.set_color()

            option_track_hands.set_color()
            option_extended.set_color()
            option_gestures.set_color()
            option_orientation.set_color()
            option_FPS.set_color()

            if option_general.is_active():
                option_debug.set_active(False)

            if option_multi_hand.is_active():
                option_single_hand.set_active(False)
                option_right_hand.set_active(False)
                option_left_hand.set_active(False)

            if option_single_hand.is_active():

                if option_right_hand.is_active():
                    option_left_hand.set_active(False)
                elif option_left_hand.is_active():
                    option_right_hand.set_active(False)
                else:
                    option_left_hand.set_active(True)

            if option_done.is_active():
                open_options_menu = False

        if option_FPS.is_active():
            past_time = FPS(past_time)
        cv2.imshow('image', image)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




