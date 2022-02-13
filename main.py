import cv2
import dlib
import numpy as np
import requests
import imutils

url = "http://192.168.100.6:8080//shot.jpg"

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()

eyeBallThreshold = 48

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
blobDetector = cv2.SimpleBlobDetector_create(detector_params)


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
    print(keypoints)
    return keypoints


def blob(img, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, 48, 255, cv2.THRESH_BINARY)
    keypoints = detector.detect(img)
    return keypoints


def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
    return img


def showLandMark(lendmarkNumb, landmarks):
    rightEyeRightLandX = landmarks.part(lendmarkNumb).x
    rightEyeRightLandY = landmarks.part(lendmarkNumb).y

    # cv2.circle(singleFrame, (rightEyeRightLandX, rightEyeRightLandY), 1, (1, 222, 4), 1)
    return rightEyeRightLandX, rightEyeRightLandY


def showRightEyeHorizontalLine(landmarks):
    rightEyeLeftLandmarkX = int((landmarks.part(37).x + landmarks.part(41).x) / 2)
    rightEyeLeftLandmarkY = int((landmarks.part(37).y + landmarks.part(41).y) / 2)
    rightEyeRightLandmarkX = int((landmarks.part(38).x + landmarks.part(40).x) / 2)
    rightEyeRightLandmarkY = int((landmarks.part(38).y + landmarks.part(40).y) / 2)

    # cv2.line(singleFrame, (rightEyeLeftLandmarkX, rightEyeLeftLandmarkY),
    #          (rightEyeRightLandmarkX, rightEyeRightLandmarkY),
    #          (2, 255, 2), 1)


def showRightEyeRectalngle(singleFrame, landmarks):
    rightEyeTopLandmarkX = landmarks.part(36).x
    rightEyeTopLandmarkY = landmarks.part(37).y
    rightEyeBottomLandmarkX = landmarks.part(39).x
    rightEyeBottomLandmarkY = landmarks.part(40).y

    # cv2.rectangle(singleFrame, (rightEyeTopLandmarkX, rightEyeTopLandmarkY),
    #               (rightEyeBottomLandmarkX, rightEyeBottomLandmarkY), (0, 222, 122))
    return singleFrame[rightEyeTopLandmarkY:rightEyeTopLandmarkY + (rightEyeBottomLandmarkY - rightEyeTopLandmarkY) + 2,
           rightEyeTopLandmarkX:rightEyeTopLandmarkX + (rightEyeBottomLandmarkX - rightEyeTopLandmarkX) + 2]


def showRightEyeVerticalLine(landmarks):
    RightEyeTopLandmarkX = int((landmarks.part(37).x + landmarks.part(38).x) / 2)
    RightEyeTopLandmarkY = int((landmarks.part(37).y + landmarks.part(38).y) / 2)
    RightEyeBottomLandmarkX = int((landmarks.part(41).x + landmarks.part(40).x) / 2)
    RightEyeBottomLandmarkY = int((landmarks.part(41).y + landmarks.part(40).y) / 2)

    # cv2.line(singleFrame, (RightEyeTopLandmarkX, RightEyeTopLandmarkY),
    #          (RightEyeBottomLandmarkX, RightEyeBottomLandmarkY),
    #          (2, 255, 2), 1)

    verticalLineLength = cv2.norm((RightEyeTopLandmarkX, RightEyeTopLandmarkY),
                                  (RightEyeBottomLandmarkX, RightEyeBottomLandmarkY))

    # print(verticalLineLength)
    #
    # if verticalLineLength <= 10:
    #     cv2.putText(singleFrame, "blink", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (200, 23, 32))


while True:

    # img_resp = requests.get(url)
    # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    # img = cv2.imdecode(img_arr, -1)
    # img = imutils.resize(img, width=1000, height=1800)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    elif cv2.waitKey(1) & 0xFF == ord('a'):
        if eyeBallThreshold != 0:
            eyeBallThreshold -= 1

    elif cv2.waitKey(1) & 0xFF == ord('d'):
        if eyeBallThreshold != 255:
            eyeBallThreshold += 1

    ret, singleFrame = cap.read()

    optFace = cv2.cvtColor(singleFrame, cv2.COLOR_BGR2GRAY)

    faces = detector(singleFrame)

    cv2.putText(singleFrame, f"threshold {eyeBallThreshold}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (200, 23, 32))

    for face in faces:
        topLeftX, topLeftY = face.left(), face.top()
        bottomRightX, bottomRightY = face.right(), face.bottom()
        cv2.rectangle(singleFrame, (topLeftX, topLeftY), (bottomRightX, bottomRightY), (0, 222, 122))

        landmarks = predictor(optFace, face)

        # x1, y1 = showLandMark(36, landmarks)
        # x2, y2 = showLandMark(37, landmarks)
        # x3, y3 = showLandMark(38, landmarks)
        # x4, y4 = showLandMark(39, landmarks)
        # x5, y5 = showLandMark(40, landmarks)
        # x6, y6 = showLandMark(41, landmarks)
        #
        #
        #
        # pts = np.array([[x1, y2], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6]])
        # pts = pts.reshape((-1, 1, 2))
        # cv2.polylines(singleFrame, [pts], True, (2, 423, 2), 1)

        showRightEyeVerticalLine(landmarks)
        showRightEyeHorizontalLine(landmarks)


        # keypoints = blob_process(rightEye, blobDetector)
        # eye = cv2.drawKeypoints(rightEye, keypoints, rightEye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    face_frame = detect_faces(singleFrame, faceCascade)
    if face_frame is not None:
        eyes = detect_eyes(face_frame)
        for eye in eyes:
            if eye is not None:
                eye = cut_eyebrows(eye)
                keypoints = blob_process(eye, blobDetector, eyeBallThreshold)
                pts = cv2.KeyPoint_convert(keypoints)
                print(f'keypoint size {pts}')

                cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow("driver", singleFrame)
        # cv2.imshow("Android_cam", img)
