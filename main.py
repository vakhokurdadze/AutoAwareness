
import cv2
import dlib
import numpy as np

cap  = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()


def showLandMark(lendmarkNumb,landmarks) :
    rightEyeRightLandX = landmarks.part(lendmarkNumb).x
    rightEyeRightLandY = landmarks.part(lendmarkNumb).y

    cv2.circle(singleFrame, (rightEyeRightLandX, rightEyeRightLandY), 1, (1, 222, 4), 1)


while True :
    _, singleFrame = cap.read()

    optFace = cv2.cvtColor(singleFrame,cv2.COLOR_BGR2GRAY)

    faces = detector(singleFrame)
    print(faces)
    for face in faces :
        topLeftX, topLeftY = face.left(), face.top()
        bottomRightX, bottomRightY = face.right(), face.bottom()
        cv2.rectangle(singleFrame,(topLeftX,topLeftY),(bottomRightX,bottomRightY),(0,222,122))

        landmarks = predictor(optFace,face)

        for i in range(0,68):
            showLandMark(i, landmarks)


    key = cv2.waitKey(1)
    cv2.imshow("driver",singleFrame)
    if key == 27 :
        break

cap.release()
cv2.destroyAllWindows()
