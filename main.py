import cv2
import dlib
import numpy as np
import requests
import imutils
from ffpyplayer.player import MediaPlayer
from time import *
import keyboard
import threading
from scipy.spatial import distance as dist
from imutils import face_utils
from picamera2.picamera2 import Picamera2


url = "http://192.168.100.6:8080//shot.jpg"
sirenSoundPath = r"mixkit-battleship-alarm-1001.wav"
fullRoadVideo = cv2.VideoCapture(r"pexels-kelly-lacy-5473757.mp4")

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

vehicle_obj_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=False,history=100,varThreshold=40)
detector = dlib.get_frontal_face_detector()
eyeHeight = None
eyeBallThreshold = 35
eyeGazeInTheRange = True
driverOutOfSight = False
blinkCounter = 0
offRoadTime = 0
isOnRoad = True
player = None
readyForBlinkIncrease = True
isTrackerSet = False
leftEyeIsOnRoad = True
rightEyeIsOnRoad = True
sleepTimerHasStarted = False

sleepTimer = 0

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
blobDetector = cv2.SimpleBlobDetector_create(detector_params)

isFaceShowing = False
longEyeShut = False

def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


def gaze_ratio(landmarks, eyePoints, singleFrame):
    eye_region = np.array([(landmarks.part(eyePoints[0]).x, landmarks.part(eyePoints[0]).y),
                           (landmarks.part(eyePoints[1]).x, landmarks.part(eyePoints[1]).y),
                           (landmarks.part(eyePoints[2]).x, landmarks.part(eyePoints[2]).y),
                           (landmarks.part(eyePoints[3]).x, landmarks.part(eyePoints[3]).y),
                           ], np.int32)

    gray = cv2.cvtColor(singleFrame, cv2.COLOR_BGR2GRAY)
    height, width, _ = singleFrame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)


    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_Y = np.min(eye_region[:, 1])
    max_Y = np.max(eye_region[:, 1])

    gray_eye = eye[min_Y:max_Y, min_x:max_x]


    _, thresholdEye = cv2.threshold(gray_eye, 48, 255, cv2.THRESH_BINARY)
    height, width = thresholdEye.shape
    leftsideThreshold = thresholdEye[0:height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(leftsideThreshold)

    rightsideThreshold = thresholdEye[0:height, int(width / 2): width]
    right_side_white = cv2.countNonZero(rightsideThreshold)

    cv2.imshow("eye", thresholdEye)
    midGaze = (left_side_white + right_side_white) / 2

    if 0 <= midGaze <= 25:
        return True
    else:
        return False


def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    return frame


def distractionCountDown():
    global sleepTimer
    global sleepTimerHasStarted
    global player

    while True:

        sleepTimer += 1
        sleep(1)
        if sleepTimer == 3 and (longEyeShut or driverOutOfSight):
            player = MediaPlayer(sirenSoundPath)
        elif sleepTimer >= 3 and (rightEyeIsOnRoad and leftEyeIsOnRoad and not longEyeShut) and not driverOutOfSight:
            if player is not None:
                player.close_player()
                sleepTimerHasStarted = False
                sleepTimer = 0
                break
        elif sleepTimer < 3 and (rightEyeIsOnRoad and leftEyeIsOnRoad and not longEyeShut) and not driverOutOfSight :
            if player is not None:
                player.close_player()
                sleepTimerHasStarted = False
                sleepTimer = 0
                break


def blinkSleep():
    global readyForBlinkIncrease
    global blinkCounter
    sleep(0.2)
    if not eyesClosed :
        blinkCounter += 1

    readyForBlinkIncrease = True


def detect_eyes(img):
    left_eye = None
    right_eye = None
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height

    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]

        else:
            right_eye = img[y:y + h, x:x + w]



    return left_eye, right_eye


def blob_process(img, detector, threshold):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)

    return keypoints


def cut_eyebrows(img):

    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
    return img


def leftAndRightEyeHorLenght(landmarks):
    rightEyeLeftLandmarkX = int((landmarks.part(37).x + landmarks.part(41).x) / 2)
    rightEyeLeftLandmarkY = int((landmarks.part(37).y + landmarks.part(41).y) / 2)
    rightEyeRightLandmarkX = int((landmarks.part(38).x + landmarks.part(40).x) / 2)
    rightEyeRightLandmarkY = int((landmarks.part(38).y + landmarks.part(40).y) / 2)

    leftEyeLeftLandmarkX = int((landmarks.part(43).x + landmarks.part(47).x) / 2)
    leftEyeLeftLandmarkY = int((landmarks.part(43).y + landmarks.part(47).y) / 2)
    leftEyeRightLandmarkX = int((landmarks.part(44).x + landmarks.part(46).x) / 2)
    leftEyeRightLandmarkY = int((landmarks.part(44).y + landmarks.part(46).y) / 2)

    return cv2.norm((rightEyeLeftLandmarkX, rightEyeLeftLandmarkY),
                    (rightEyeRightLandmarkX, rightEyeRightLandmarkY)), cv2.norm(
        (leftEyeLeftLandmarkX, leftEyeLeftLandmarkY),
        (leftEyeRightLandmarkX, leftEyeRightLandmarkY))


def leftAndRightEyeVerLenght(landmarks):
    rightEyeTopLandmarkX = int((landmarks.part(37).x + landmarks.part(38).x) / 2)
    rightEyeTopLandmarkY = int((landmarks.part(37).y + landmarks.part(38).y) / 2)
    rightEyeBottomLandmarkX = int((landmarks.part(41).x + landmarks.part(40).x) / 2)
    rightEyeBottomLandmarkY = int((landmarks.part(41).y + landmarks.part(40).y) / 2)

    leftEyeTopLandmarkX = int((landmarks.part(43).x + landmarks.part(44).x) / 2)
    leftEyeTopLandmarkY = int((landmarks.part(43).y + landmarks.part(44).y) / 2)
    leftEyeBottomLandmarkX = int((landmarks.part(47).x + landmarks.part(46).x) / 2)
    leftEyeBottomLandmarkY = int((landmarks.part(47).y + landmarks.part(46).y) / 2)


    return cv2.norm((rightEyeTopLandmarkX, rightEyeTopLandmarkY),
                    (rightEyeBottomLandmarkX, rightEyeBottomLandmarkY)), cv2.norm(
        (leftEyeTopLandmarkX, leftEyeTopLandmarkY),
        (leftEyeBottomLandmarkX, leftEyeBottomLandmarkY))


def rightEyeHorizontalLineLength(landmarks):
    rightEyeLeftLandmarkX = int((landmarks.part(37).x + landmarks.part(41).x) / 2)
    rightEyeLeftLandmarkY = int((landmarks.part(37).y + landmarks.part(41).y) / 2)
    rightEyeRightLandmarkX = int((landmarks.part(38).x + landmarks.part(40).x) / 2)
    rightEyeRightLandmarkY = int((landmarks.part(38).y + landmarks.part(40).y) / 2)

    return cv2.norm((rightEyeLeftLandmarkX, rightEyeLeftLandmarkY),
                    (rightEyeRightLandmarkX, rightEyeRightLandmarkY))


def showRightEyeRectalngle(singleFrame, landmarks):
    rightEyeTopLandmarkX = landmarks.part(36).x
    rightEyeTopLandmarkY = landmarks.part(37).y
    rightEyeBottomLandmarkX = landmarks.part(39).x
    rightEyeBottomLandmarkY = landmarks.part(40).y

    return singleFrame[rightEyeTopLandmarkY:rightEyeTopLandmarkY + (rightEyeBottomLandmarkY - rightEyeTopLandmarkY) + 2,
           rightEyeTopLandmarkX:rightEyeTopLandmarkX + (rightEyeBottomLandmarkX - rightEyeTopLandmarkX) + 2]


def areEyesShut(shape):

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEyeEar = eye_aspect_ratio(leftEye)
    rightEyeEar = eye_aspect_ratio(rightEye)
    ear = (leftEyeEar + rightEyeEar) / 2.0

    return ear <= 0.27


title_window = 'driver'
alpha_slider_max = 100
import argparse

trackbar_name = 'Alpha x %d' % alpha_slider_max


def on_trackbar(val):
    global eyeBallThreshold
    eyeBallThreshold = val


def initYolo():
    ret, frame = fullRoadVideo.read()

    scale_percent = 15
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    optFrame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    mask = vehicle_obj_detector.apply(optFrame)
    _,mask = cv2.threshold(mask,254,255,cv2.THRESH_BINARY)



    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contourCounter = 0

    for contour in contours :

        area = cv2.contourArea(contour)


        if area >= 2000 :
            contourCounter += 1

            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(optFrame,(x,y),(x+w,y+h),(0,255,0),3)

    cv2.putText(optFrame, f"{contourCounter} Cars in the nearby area", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 23, 32))

    if contourCounter >= 4 :
        cv2.putText(optFrame, f"Heavy Traffic Detected!! ", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255))
    elif 2 <= contourCounter <= 3 :
        cv2.putText(optFrame, f"Possible Traffic Ahead!! ", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255))
    else:
        cv2.putText(optFrame, f"Empty Road ", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0))

    cv2.imshow('cars', optFrame)
cv2.startWindowThread()

picam2 = Picamera2()
picam2.configure(picam2.preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

while True:

    try:
      initYolo()
    except:
       print("yolo video done")


    blinkSleepThread = threading.Thread(target=blinkSleep)
    thread = threading.Thread(target=distractionCountDown)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    elif cv2.waitKey(1) & 0xFF == ord('1'):
        if eyeBallThreshold != 0:
            eyeBallThreshold -= 1

    elif cv2.waitKey(1) & 0xFF == ord('2'):
        if eyeBallThreshold != 255:
            eyeBallThreshold += 1

    singleFrame = picam2.capture_array()

    optFace = cv2.cvtColor(singleFrame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(singleFrame, 1.1, 5)

    gazeIsOnRoad = False
    if(len(faces) == 0):
        cv2.putText(singleFrame, 'Warning!!', (150, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255))
        cv2.putText(singleFrame, 'Drive is out of Sight!! Starting Alarm', (22, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))


        driverOutOfSight = True

    for face in faces:
        driverOutOfSight = False

        topLeftX, topLeftY = face[0], face[1]
        bottomRightX, bottomRightY = face[2], face[3]
        cv2.rectangle(singleFrame, (topLeftX, topLeftY), (topLeftX + bottomRightX, topLeftY + bottomRightY),
                      (0, 222, 122))

        gazeIsOnRoad = True
        landmarks = predictor(optFace,
                              dlib.rectangle(topLeftX, topLeftY, topLeftX + bottomRightX, topLeftY + bottomRightY))

        shape = face_utils.shape_to_np(landmarks)
        eyesClosed = areEyesShut(shape)
        longEyeShut = areEyesShut(shape)
        if eyesClosed and readyForBlinkIncrease:
            readyForBlinkIncrease = False
            blinkSleepThread.start()


        if eyesClosed:
            cv2.putText(singleFrame, 'Eyes Closed', (360, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4,(199, 45, 255) )
        else:
            cv2.putText(singleFrame, 'Eyes Open', (360, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (199, 255, 45))



        leftEyeIsOnRoad = gaze_ratio(landmarks, [37, 38, 40, 41], singleFrame)
        rightEyeIsOnRoad = gaze_ratio(landmarks, [43, 44, 46, 47], singleFrame)

        if leftEyeIsOnRoad and rightEyeIsOnRoad and not longEyeShut:
            cv2.putText(singleFrame, 'On Road', (18, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0))
        else:
            cv2.putText(singleFrame, 'Off Road', (18, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255))

    cv2.putText(singleFrame, f"blink counter: {blinkCounter}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (255, 23, 32))

    if sleepTimer == 3:
        player = MediaPlayer(sirenSoundPath)



    if not leftEyeIsOnRoad or not rightEyeIsOnRoad or longEyeShut or driverOutOfSight:
        if not sleepTimerHasStarted:
            sleepTimerHasStarted = True
            thread.start()
    else:
        sleepTimer = 0

    face_frame = detect_faces(singleFrame, faceCascade)
    if face_frame is not None:
        eyes = detect_eyes(face_frame)

        for eye in eyes:
            if eye is not None:

                eye = cut_eyebrows(eye)

                keypoints = blob_process(eye, blobDetector, eyeBallThreshold)
                pts = cv2.KeyPoint_convert(keypoints)
                if len(pts) != 0:
                    eyeGazeInTheRange = (20 <= pts[0][0] <= 34) and (18 >= pts[0][1] >= 8)
                    # print(f"first {pts[0][0]} second {pts[0][1]}")
                    # cv2.putText(singleFrame, f"second {int(pts[0][1])}", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    #             (0, 0, 223))
                else:
                    eyeGazeInRange = False

                cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("driver", singleFrame)
    if not isTrackerSet:
        isTrackerSet = True
        # cv2.createTrackbar('gela', title_window, 0, 255, on_trackbar)



