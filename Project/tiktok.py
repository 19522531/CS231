import cv2
import numpy as np
import dlib
from math import hypot
# Loading Camera and Nose image and Creating mask
# nose_image = cv2.imread("static/pig_nose.png")
# gray = cv2.cvtColor(nose_image, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
# nose_image[thresh == 255] = 0
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# nose_image = cv2.erode(nose_image, kernel, iterations = 1)
# # _, frame = cap.read()
# # rows, cols, _ = frame.shape
# nose_mask = np.zeros((rows, cols), np.uint8)
# # Loading Face detector
# detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./__pycache__/shape_predictor_68_face_landmarks.dat")
nose_image = cv2.imread("static/pig_nose.png")

def attach_nose(frame, nose_image, predictor):
# while True:
#     _, frame = cap.read()
#     frame = cv2.flip(frame, 1)

    #nose_image = cv2.imread("static/pig_nose.png")
    gray = cv2.cvtColor(nose_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    nose_image[thresh == 255] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    nose_image = cv2.erode(nose_image, kernel, iterations = 1)
    rows, cols, _ = frame.shape
    nose_mask = np.zeros((rows, cols), np.uint8)
    # Loading Face detector
    detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor("./__pycache__/shape_predictor_68_face_landmarks.dat")

    nose_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)
        top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)
        nose_width = int(hypot(left_nose[0] - right_nose[0],
                               left_nose[1] - right_nose[1]) * 2.2)
        nose_height = int(nose_width * 0.77)
        nose_width =int(abs(left_nose[0] - right_nose[0])*2.2)
        # New nose position
        top_left = (int(center_nose[0] - nose_width / 2 ),
                    int(center_nose[1] - nose_height / 2) )
        bottom_right = (int(center_nose[0] + nose_width / 2 ),
                        int(center_nose[1] + nose_height / 2) )
        # Adding the new nose
        # print(top_left,'  ',bottom_right)
        nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_pig_gray, 100, 255, cv2.THRESH_BINARY_INV)
        #cv2.imshow('haha',nose_mask)
        nose_area = frame[top_left[1]: top_left[1] + nose_height,
                    top_left[0]: top_left[0] + nose_width]
        nose_area_no_nose = cv2.bitwise_or(nose_area, nose_area, mask=nose_mask)
        #cv2.imshow('haha', nose_area_no_nose)
        final_nose = cv2.add(nose_area_no_nose, nose_pig)
        frame[top_left[1]: top_left[1] + nose_height,
        top_left[0]: top_left[0] + nose_width] = final_nose

        # cv2.imshow("final nose", final_nose)
    return frame

# cam = cv2.VideoCapture(0)
# while True:      
#     _, frame = cam.read()
#     frame = attach_nose(frame, nose_image, predictor)
#     cv2.imshow("ok", frame)
#     cv2.waitKey(1)