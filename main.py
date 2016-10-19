import cv2
import numpy as np

# Size constants
kEyePercentTop = 25;
kEyePercentSide = 13;
kEyePercentHeight = 20;
kEyePercentWidth = 35;

# Algorithm Parameters
kFastEyeWidth = 50;
kWeightBlurSize = 5;
kEnableWeight = True;
kWeightDivisor = 1.0;
kGradientThreshold = 50.0;

# Window
MAIN_WINDOW_NAME = "main"
FACE_WINDOW_NAME = "face"
LEFT_EYE_WINDOW_NAME = "left eye"
RIGHT_EYE_WINDOW_NAME = "right eye"
face_cascade_name = "./haarcascade_frontalface_alt.xml"
eye_cascade_name = "./haarcascade_eye_tree_eyeglasses.xml"
eye_cascade = cv2.CascadeClassifier(eye_cascade_name)
face_cascade = cv2.CascadeClassifier(face_cascade_name)

def main():
    # cv2.namedWindow(FACE_WINDOW_NAME, cv2.WINDOW_NORMAL)
    # cv2.moveWindow(FACE_WINDOW_NAME, 10, 100)
    # cv2.namedWindow(LEFT_EYE_WINDOW_NAME, cv2.WINDOW_NORMAL)
    # cv2.moveWindow(LEFT_EYE_WINDOW_NAME, 10, 800)
    # cv2.namedWindow(RIGHT_EYE_WINDOW_NAME, cv2.WINDOW_NORMAL)
    # cv2.moveWindow(RIGHT_EYE_WINDOW_NAME, 10, 600)

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale( gray, 1.1, 2, 0, (150, 150) );
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 1)

        if(len(faces) > 0):
            findEyes(gray, faces[0])

        cv2.imshow("gray", gray)
        cv2.imshow("main", img)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break
        elif k == ord('f'):
            cv2.imwrite("frame.png", frame)
            cv2.flip(frame, frame, 1)
    cv2.destroyAllWindows()

def findEyes(gray, face):
    x,y,w,h = face
    faceROI = gray[y:y+h, x:x+h]
    debugFace = faceROI
    eye_region_width = w * (kEyePercentWidth/100.0);
    eye_region_height = w * (kEyePercentHeight/100.0);
    eye_region_top = h * (kEyePercentTop/100.0);
    leftEyeRegion = (int(w * (kEyePercentSide/100.0)), int(eye_region_top), int(eye_region_width), int(eye_region_height))
    rightEyeRegion = (int(w - eye_region_width - w*(kEyePercentSide/100.0)), int(eye_region_top), int(eye_region_width), int(eye_region_height))
    leftPupil = findEyeCenter(faceROI, leftEyeRegion)
    rightPupil = findEyeCenter(faceROI, rightEyeRegion)

def findEyeCenter(face, eye):
    ex,ey,ew,eh = eye
    eyeROIUnscaled = face[ey:ey+eh, ex: ex+ew]
    # TODO: scale eye region
    cv2.rectangle(face, (ex, ey), (ex+ew, ey+eh), (255,0,0), 1)
    return None


if __name__ == '__main__':
    main()

