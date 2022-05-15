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


url = "http://192.168.100.6:8080//shot.jpg"
sirenSoundPath = r"C:\Users\User\Downloads\final_627a96974ad7b200913b40c3_598866.mp3"
fullRoadVideo = cv2.VideoCapture(r"C:\Users\User\Downloads\yoloVideo_new.mp4")

cap = cv2.VideoCapture(0)
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

    # cv2.imshow("eye", thresholdEye)
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


def detect_eyes(faceSection):
   leftEye, rightEye = None, None

   grayFrame = cv2.cvtColor(faceSection, cv2.COLOR_BGR2GRAY)
   width, height = np.size(faceSection, 1), np.size(faceSection, 0)
   eyes = eye_cascade.detectMultiScale(grayFrame, 1.3, 5)

   for (x, y, w, h) in eyes:
       if y > height / 2:
           pass
       centerPart = x + w / 2
       if centerPart >= width * 0.5:
           rightEye = faceSection[y:y + h, x:x + w]
       else:
           leftEye = faceSection[y:y + h, x:x + w]

   return leftEye, rightEye


def blob_process(eyePart, detector, threshold):
   grayFrame = cv2.cvtColor(eyePart, cv2.COLOR_BGR2GRAY)
   _, eyePart = cv2.threshold(grayFrame, threshold, 255, cv2.THRESH_BINARY)
   eyePart = cv2.erode(eyePart, None, iterations=2)
   eyePart = cv2.dilate(eyePart, None, iterations=4)
   eyePart = cv2.medianBlur(eyePart, 5)

   return detector.detect(eyePart)


def cut_eyebrows(eyeSection):
   h, w = eyeSection.shape[:2]
   eyebrowHeight = int(h / 4)
   eyeSection = eyeSection[eyebrowHeight:h, 0:w]

   return eyeSection

def leftAndRightEyeHorLenght(landmarks):
    rightEyeLeftLandmarkX = int((landmarks.part(37).x + landmarks.part(41).x) / 2)
    rightEyeLeftLandmarkY = int((landmarks.part(37).y + landmarks.part(41).y) / 2)
    rightEyeRightLandmarkX = int((landmarks.part(38).x + landmarks.part(40).x) / 2)
    rightEyeRightLandmarkY = int((landmarks.part(38).y + landmarks.part(40).y) / 2)

    leftEyeLeftLandmarkX = int((landmarks.part(43).x + landmarks.part(47).x) / 2)
    leftEyeLeftLandmarkY = int((landmarks.part(43).y + landmarks.part(47).y) / 2)
    leftEyeRightLandmarkX = int((landmarks.part(44).x + landmarks.part(46).x) / 2)
    leftEyeRightLandmarkY = int((landmarks.part(44).y + landmarks.part(46).y) / 2)

    # cv2.line(singleFrame, (landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(39).x, landmarks.part(39).y),
    #          (255, 255, 0), 1)
    # cv2.line(singleFrame, (landmarks.part(42).x, landmarks.part(42).y), (landmarks.part(45).x, landmarks.part(45).y),
    #          (255, 255, 0), 1)

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

    # cv2.line(singleFrame, (landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(39).x, landmarks.part(39).y),
    #          (0, 255, 0), 1)
    # cv2.line(singleFrame, (landmarks.part(42).x, landmarks.part(42).y), (landmarks.part(45).x, landmarks.part(45).y),
    #          (0, 255, 0), 1)

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


def areEyesShut(landmarks,shape,):
    RightEyeTopLandmarkX = int((landmarks.part(37).x + landmarks.part(38).x) / 2)
    RightEyeTopLandmarkY = int((landmarks.part(37).y + landmarks.part(38).y) / 2)
    RightEyeBottomLandmarkX = int((landmarks.part(41).x + landmarks.part(40).x) / 2)
    RightEyeBottomLandmarkY = int((landmarks.part(41).y + landmarks.part(40).y) / 2)

    LeftEyeTopLandmarkX = int((landmarks.part(43).x + landmarks.part(44).x) / 2)
    LeftEyeTopLandmarkY = int((landmarks.part(43).y + landmarks.part(44).y) / 2)
    LeftEyeBottomLandmarkX = int((landmarks.part(47).x + landmarks.part(46).x) / 2)
    LeftEyeBottomLandmarkY = int((landmarks.part(47).y + landmarks.part(46).y) / 2)

    # cv2.line(singleFrame, (RightEyeTopLandmarkX, RightEyeTopLandmarkY),
    #          (RightEyeBottomLandmarkX, RightEyeBottomLandmarkY),
    #          (0, 255, 0), 1)
    #
    # cv2.line(singleFrame, (LeftEyeTopLandmarkX, LeftEyeTopLandmarkY),
    #          (LeftEyeBottomLandmarkX, LeftEyeBottomLandmarkY),
    #          (0, 255, 0), 1)

    rightEyeHorLenght, leftEyeHorLength = leftAndRightEyeHorLenght(landmarks)
    rightEyeVerLenght, leftEyeVerLength = leftAndRightEyeVerLenght(landmarks)

    eyesHorLengthAvg = (rightEyeHorLenght + leftEyeHorLength) / 2
    eyesVerLengthAvg = (rightEyeVerLenght + leftEyeVerLength) / 2
    # eyesVerLengthAvg < eyesHorLengthAvg - 3

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEyeEar = eye_aspect_ratio(leftEye)
    rightEyeEar = eye_aspect_ratio(rightEye)
    ear = (leftEyeEar + rightEyeEar) / 2.0

    print(ear)
    return ear <= 0.27


def showEyeLandmarks(landmarks):
    # cv2.circle(singleFrame, (landmarks.part(36).x, landmarks.part(36).y), 1, (0, 200, 0))
    # cv2.circle(singleFrame, (landmarks.part(37).x, landmarks.part(37).y), 1, (0, 200, 0))
    # cv2.circle(singleFrame, (landmarks.part(38).x, landmarks.part(38).y), 1, (0, 200, 0))
    # cv2.circle(singleFrame, (landmarks.part(39).x, landmarks.part(39).y), 1, (0, 200, 0))
    # cv2.circle(singleFrame, (landmarks.part(40).x, landmarks.part(40).y), 1, (0, 200, 0))
    # cv2.circle(singleFrame, (landmarks.part(41).x, landmarks.part(41).y), 1, (0, 200, 0))
    #
    # cv2.circle(singleFrame, (landmarks.part(42).x, landmarks.part(42).y), 1, (0, 200, 0))
    # cv2.circle(singleFrame, (landmarks.part(43).x, landmarks.part(43).y), 1, (0, 200, 0))
    # cv2.circle(singleFrame, (landmarks.part(44).x, landmarks.part(44).y), 1, (0, 200, 0))
    # cv2.circle(singleFrame, (landmarks.part(45).x, landmarks.part(45).y), 1, (0, 200, 0))
    # cv2.circle(singleFrame, (landmarks.part(46).x, landmarks.part(46).y), 1, (0, 200, 0))
    # cv2.circle(singleFrame, (landmarks.part(47).x, landmarks.part(47).y), 1, (0, 200, 0))

    rightEyeLandmarkLine = np.array(
        [[landmarks.part(36).x, landmarks.part(36).y], [landmarks.part(37).x, landmarks.part(37).y],
         [landmarks.part(38).x, landmarks.part(38).y], [landmarks.part(39).x, landmarks.part(39).y],
         [landmarks.part(40).x, landmarks.part(40).y], [landmarks.part(41).x, landmarks.part(41).y],
         [landmarks.part(36).x, landmarks.part(36).y]], np.int32)
    leftEyeLandmarkLine = np.array(
        [[landmarks.part(42).x, landmarks.part(42).y], [landmarks.part(43).x, landmarks.part(43).y],
         [landmarks.part(44).x, landmarks.part(44).y], [landmarks.part(45).x, landmarks.part(45).y],
         [landmarks.part(46).x, landmarks.part(46).y], [landmarks.part(47).x, landmarks.part(47).y],
         [landmarks.part(42).x, landmarks.part(42).y]], np.int32)

    # cv2.polylines(singleFrame,[rightEyeLandmarkLine],True,(0,200.200),1)
    # cv2.polylines(singleFrame,[leftEyeLandmarkLine],True,(0,200.200),1)


title_window = 'driver'
alpha_slider_max = 100
import argparse

trackbar_name = 'Alpha x %d' % alpha_slider_max


def on_trackbar(val):
    global eyeBallThreshold
    eyeBallThreshold = val


def initYolo():

   ret, yoloFrame = fullRoadVideo.read()

   optYoloFrame = yoloFrame[450:900,250:850]

   car_object_cascade = cv2.CascadeClassifier("cars.xml")
   cars = car_object_cascade.detectMultiScale(optYoloFrame,1.1,4)


   contourCounter = 0
   for (x, y, w, h) in cars:
        contourCounter += 1
        cv2.rectangle(optYoloFrame, (x, y), (x + w, y + h), (0, 255, 0), 3)

   cv2.putText(yoloFrame, f"{contourCounter} Cars in the nearby area", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
               (200, 23, 32))

   if contourCounter >= 4:
       cv2.putText(yoloFrame, f"Heavy Traffic Detected!! ", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
   elif 2 <= contourCounter <= 3:
       cv2.putText(yoloFrame, f"Possible Traffic Ahead!! ", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
   else:
       cv2.putText(yoloFrame, f"Empty Road ", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

   cv2.imshow('cars', yoloFrame)

while True:

    try:
      initYolo()
    except Exception as e:
       fullRoadVideo.set(cv2.CAP_PROP_POS_FRAMES,0)




    # img_resp = requests.get(url)
    # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    # img = cv2.imdecode(img_arr, -1)
    # img = imutils.resize(img, width=1000, height=1800)
    blinkSleepThread = threading.Thread(target=blinkSleep)
    thread = threading.Thread(target=distractionCountDown)

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
    if(len(faces) == 0):
        cv2.putText(singleFrame, 'Warning!!', (150, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255))
        cv2.putText(singleFrame, 'Drive is out of Sight!! Starting Alarm', (22, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))


        driverOutOfSight = True

    for face in faces:
        driverOutOfSight = False
        topLeftX, topLeftY = face.left(), face.top()
        bottomRightX, bottomRightY = face.right(), face.bottom()
        cv2.rectangle(singleFrame, (topLeftX, topLeftY), (bottomRightX, bottomRightY), (0, 222, 122))

        gazeIsOnRoad = True
        landmarks = predictor(optFace, face)
        showEyeLandmarks(landmarks)

        shape = face_utils.shape_to_np(landmarks)
        eyesClosed = areEyesShut(landmarks,shape)
        longEyeShut = areEyesShut(landmarks,shape)
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

    # rightEyeGazeRatio = gaze_ratio(landmarks, [43, 44,47, 46],singleFrame)
    # gazeRatio = (leftEyeGazeRatio + rightEyeGazeRatio) / 2
    # cv2.putText(singleFrame, str(leftEyeGazeRatio), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,(200, 23, 32))
    # # if gazeRatio < 1:
    #     cv2.putText(singleFrame, "Right", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
    #                 (200, 23, 32))
    # elif 1 < gazeRatio < 3:
    #     cv2.putText(singleFrame, "Center", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
    #                 (200, 23, 32))
    # else:
    #     cv2.putText(singleFrame, "Left", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
    #                 (200, 23, 32))

    # keypoints = blob_process(rightEye, blobDetector)
    # eye = cv2.drawKeypoints(rightEye, keypoints, rightEye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.putText(singleFrame, f"blink counter: {blinkCounter}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (255, 23, 32))

    # isOnRoad = not eyesClosed or (gazeIsOnRoad and eyeGazeInTheRange)

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

    if sleepTimer == 3:
        print(sleepTimer)
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

    # cv2.imshow("driver", singleFrame)
    if not isTrackerSet:
        isTrackerSet = True
        # cv2.createTrackbar('gela', title_window, 0, 255, on_trackbar)

    # cv2.imshow("Android_cam", img)


