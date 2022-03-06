import cv2
import dlib
import numpy as np
import requests
import imutils
from ffpyplayer.player import MediaPlayer
from time import *
import keyboard
import threading

url = "http://192.168.100.6:8080//shot.jpg"
sirenSoundPath = r"C:\Users\User\Videos\mp3\myAudacity\umafade.mp3"

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()
eyeHeight = None
eyeBallThreshold = 48
eyeGazeInTheRange = True
blinkCounter = 0
offRoadTime = 0
isOnRoad = True
player = None
readyForBlinkIncrease = True

sleepTimerHasStarted = False

sleepTimer = 0

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
blobDetector = cv2.SimpleBlobDetector_create(detector_params)

isFaceShowing = False
eyesClosed = False


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
    while True:
        sleepTimer += 1
        sleep(1)


def blinkSleep():
    global readyForBlinkIncrease
    global blinkCounter
    blinkCounter += 1
    sleep(0.4)
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

    cv2.line(singleFrame, (landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(39).x, landmarks.part(39).y),
             (255, 255, 0), 1)
    cv2.line(singleFrame, (landmarks.part(42).x, landmarks.part(42).y), (landmarks.part(45).x, landmarks.part(45).y),
             (255, 255, 0), 1)

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

    cv2.line(singleFrame, (landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(39).x, landmarks.part(39).y),
             (0, 255, 0), 1)
    cv2.line(singleFrame, (landmarks.part(42).x, landmarks.part(42).y), (landmarks.part(45).x, landmarks.part(45).y),
             (0, 255, 0), 1)

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

    # cv2.rectangle(singleFrame, (rightEyeTopLandmarkX, rightEyeTopLandmarkY),
    #               (rightEyeBottomLandmarkX, rightEyeBottomLandmarkY), (0, 222, 122))
    return singleFrame[rightEyeTopLandmarkY:rightEyeTopLandmarkY + (rightEyeBottomLandmarkY - rightEyeTopLandmarkY) + 2,
           rightEyeTopLandmarkX:rightEyeTopLandmarkX + (rightEyeBottomLandmarkX - rightEyeTopLandmarkX) + 2]


def areEyesShut(landmarks):
    RightEyeTopLandmarkX = int((landmarks.part(37).x + landmarks.part(38).x) / 2)
    RightEyeTopLandmarkY = int((landmarks.part(37).y + landmarks.part(38).y) / 2)
    RightEyeBottomLandmarkX = int((landmarks.part(41).x + landmarks.part(40).x) / 2)
    RightEyeBottomLandmarkY = int((landmarks.part(41).y + landmarks.part(40).y) / 2)

    LeftEyeTopLandmarkX = int((landmarks.part(43).x + landmarks.part(44).x) / 2)
    LeftEyeTopLandmarkY = int((landmarks.part(43).y + landmarks.part(44).y) / 2)
    LeftEyeBottomLandmarkX = int((landmarks.part(47).x + landmarks.part(46).x) / 2)
    LeftEyeBottomLandmarkY = int((landmarks.part(47).y + landmarks.part(46).y) / 2)

    cv2.line(singleFrame, (RightEyeTopLandmarkX, RightEyeTopLandmarkY),
             (RightEyeBottomLandmarkX, RightEyeBottomLandmarkY),
             (0, 255, 0), 1)

    cv2.line(singleFrame, (LeftEyeTopLandmarkX, LeftEyeTopLandmarkY),
             (LeftEyeBottomLandmarkX, LeftEyeBottomLandmarkY),
             (0, 255, 0), 1)

    rightEyeHorLenght, leftEyeHorLength = leftAndRightEyeHorLenght(landmarks)
    rightEyeVerLenght, leftEyeVerLength = leftAndRightEyeVerLenght(landmarks)

    eyesHorLengthAvg = (rightEyeHorLenght + leftEyeHorLength) / 2
    eyesVerLengthAvg = (rightEyeVerLenght + leftEyeVerLength) / 2

    return eyesVerLengthAvg < eyesHorLengthAvg - 3


def showEyeLandmarks(landmarks):
    cv2.circle(singleFrame, (landmarks.part(36).x, landmarks.part(36).y), 1, (0, 200, 0))
    cv2.circle(singleFrame, (landmarks.part(37).x, landmarks.part(37).y), 1, (0, 200, 0))
    cv2.circle(singleFrame, (landmarks.part(38).x, landmarks.part(38).y), 1, (0, 200, 0))
    cv2.circle(singleFrame, (landmarks.part(39).x, landmarks.part(39).y), 1, (0, 200, 0))
    cv2.circle(singleFrame, (landmarks.part(40).x, landmarks.part(40).y), 1, (0, 200, 0))
    cv2.circle(singleFrame, (landmarks.part(41).x, landmarks.part(41).y), 1, (0, 200, 0))

    cv2.circle(singleFrame, (landmarks.part(42).x, landmarks.part(42).y), 1, (0, 200, 0))
    cv2.circle(singleFrame, (landmarks.part(43).x, landmarks.part(43).y), 1, (0, 200, 0))
    cv2.circle(singleFrame, (landmarks.part(44).x, landmarks.part(44).y), 1, (0, 200, 0))
    cv2.circle(singleFrame, (landmarks.part(45).x, landmarks.part(45).y), 1, (0, 200, 0))
    cv2.circle(singleFrame, (landmarks.part(46).x, landmarks.part(46).y), 1, (0, 200, 0))
    cv2.circle(singleFrame, (landmarks.part(47).x, landmarks.part(47).y), 1, (0, 200, 0))

    rightEyeLandmarkLine = np.array([[landmarks.part(36).x, landmarks.part(36).y], [landmarks.part(37).x, landmarks.part(37).y],
                                     [landmarks.part(38).x, landmarks.part(38).y], [landmarks.part(39).x, landmarks.part(39).y],
                                     [landmarks.part(40).x, landmarks.part(40).y],[landmarks.part(41).x, landmarks.part(41).y],
                                     [landmarks.part(36).x, landmarks.part(36).y]], np.int32)
    leftEyeLandmarkLine = np.array([[landmarks.part(42).x, landmarks.part(42).y], [landmarks.part(43).x, landmarks.part(43).y],
                                     [landmarks.part(44).x, landmarks.part(44).y], [landmarks.part(45).x, landmarks.part(45).y],
                                     [landmarks.part(46).x, landmarks.part(46).y],[landmarks.part(47).x, landmarks.part(47).y],
                                     [landmarks.part(42).x, landmarks.part(42).y]], np.int32)

    cv2.polylines(singleFrame,[rightEyeLandmarkLine],True,(0,200.200),1)
    cv2.polylines(singleFrame,[leftEyeLandmarkLine],True,(0,200.200),1)


while True:

    # img_resp = requests.get(url)
    # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    # img = cv2.imdecode(img_arr, -1)
    # img = imutils.resize(img, width=1000, height=1800)
    blinkSleepThread = threading.Thread(target=blinkSleep)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    elif cv2.waitKey(1) & 0xFF == ord('1'):
        if eyeBallThreshold != 0:
            eyeBallThreshold -= 1

    elif cv2.waitKey(1) & 0xFF == ord('2'):
        if eyeBallThreshold != 255:
            eyeBallThreshold += 1

    ret, singleFrame = cap.read()

    optFace = cv2.cvtColor(singleFrame, cv2.COLOR_BGR2GRAY)

    faces = detector(singleFrame)

    # cv2.putText(singleFrame, f"threshold {eyeBallThreshold}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (200, 23, 32))
    gazeIsOnRoad = False
    for face in faces:
        topLeftX, topLeftY = face.left(), face.top()
        bottomRightX, bottomRightY = face.right(), face.bottom()
        cv2.rectangle(singleFrame, (topLeftX, topLeftY), (bottomRightX, bottomRightY), (0, 222, 122))

        gazeIsOnRoad = True
        landmarks = predictor(optFace, face)
        showEyeLandmarks(landmarks)
        eyesClosed = areEyesShut(landmarks)

        if eyesClosed and readyForBlinkIncrease:
            readyForBlinkIncrease = False
            blinkSleepThread.start()


        # keypoints = blob_process(rightEye, blobDetector)
        # eye = cv2.drawKeypoints(rightEye, keypoints, rightEye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.putText(singleFrame, f"blink counter: {blinkCounter}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (200, 23, 32))

    isOnRoad = not eyesClosed or (gazeIsOnRoad and eyeGazeInTheRange)
    thread = threading.Thread(target=distractionCountDown)

    # if isOnRoad:
    #     # cv2.putText(singleFrame, "onRoad", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (200, 23, 32))
    #
    #     sleepTimer = 0
    #     if thread.is_alive():
    #         thread.join()
    #     sleepTimerHasStarted = False
    #     if player is not None:
    #         player.close_player()
    #
    #
    # else:
    #     # cv2.putText(singleFrame, "offRoad", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (200, 23, 32))
    #     if not sleepTimerHasStarted:
    #         sleepTimerHasStarted = True
    #         thread.start()

    # if sleepTimer == 3:
    #     print(sleepTimer)
    #     player = MediaPlayer(sirenSoundPath)

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

    # cv2.imshow("Android_cam", img)
