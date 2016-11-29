import cv2
import cv2.cv as cv
import numpy as np

# Size constants
EYE_PERCENT_TOP = 32
EYE_PERCENT_SIDE = 18
EYE_PERCENT_HEIGHT = 15
EYE_PERCENT_WIDTH = 25

# EYE_PERCENT_TOP = 25
# EYE_PERCENT_SIDE = 13
# EYE_PERCENT_HEIGHT = 20
# EYE_PERCENT_WIDTH = 35

# Algorithm Parameters
THRESHOLD_SCALE = 3

# Window
MAIN_WINDOW_NAME = "main"
FACE_WINDOW_NAME = "face"
LEFT_EYE_WINDOW_NAME = "left eye"
RIGHT_EYE_WINDOW_NAME = "right eye"
FACE_CASCADE_NAME = "./haarcascade_frontalface_alt.xml"
EYE_CASCADE_NAME = "./haarcascade_eye_tree_eyeglasses.xml"
EYE_CASCADE = cv2.CascadeClassifier(EYE_CASCADE_NAME)
FACE_CASCADE = cv2.CascadeClassifier(FACE_CASCADE_NAME)


def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 2, 0, (150, 150))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

        if len(faces) > 0:
            find_eyes(gray, faces[0])

        cv2.imshow("gray", gray)
        cv2.imshow("main", img)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break
        elif k == ord('f'):
            cv2.imwrite("frame.png", frame)
            cv2.flip(frame, frame, 1)
    cv2.destroyAllWindows()


def find_eyes(gray, face):
    (x, y, w, h) = face
    face_roi = gray[y:y+h, x:x+h]
    eye_region_width = w * (EYE_PERCENT_WIDTH/100.0)
    eye_region_height = w * (EYE_PERCENT_HEIGHT/100.0)
    eye_region_top = h * (EYE_PERCENT_TOP/100.0)
    left_eye_region = (int(w * (EYE_PERCENT_SIDE/100.0)), int(eye_region_top), \
                      int(eye_region_width), int(eye_region_height))
    right_eye_region = (int(w - eye_region_width - w*(EYE_PERCENT_SIDE/100.0)), \
                       int(eye_region_top), int(eye_region_width), int(eye_region_height))
    find_eye_region(face_roi, left_eye_region)
    find_eye_region(face_roi, right_eye_region)

    # Test HoughtCircles with left eye
    # TODO: Thresold eye region to get eye pupil
    # then use HoughCircles to detect open/closed eye
    # Read more here: http://answers.opencv.org/question/32688/finding-the-center-of-eye-pupil/
    eye_left_x, eye_left_y, eye_left_width, eye_left_height = left_eye_region
    left_eye_roi = face_roi[eye_left_y:eye_left_y+eye_left_height, \
                            eye_left_x:eye_left_x+eye_left_width]

    eye_right_x, eye_right_y, eye_right_width, eye_right_height = right_eye_region
    right_eye_roi = face_roi[eye_right_y:eye_right_y+eye_right_height, \
                            eye_right_x:eye_right_x+eye_right_width]

    kernel = np.ones((9, 9), np.uint8)
    (_, thresh_left) = cv2.threshold(left_eye_roi, 50, 255, cv2.THRESH_BINARY_INV)
    thresh_left = cv2.dilate(thresh_left, kernel, iterations=1)
    thresh_left = cv2.erode(thresh_left, kernel, iterations=1)
    thresh_left = cv2.GaussianBlur(thresh_left, (9, 9), 0)
    thresh_left = cv2.resize(thresh_left, (0, 0), fx=THRESHOLD_SCALE, fy=THRESHOLD_SCALE)
    cv2.imshow("thresh_left_eye_roi", thresh_left)

    (_, thresh_right) = cv2.threshold(right_eye_roi, 55, 255, cv2.THRESH_BINARY_INV)
    thresh_right = cv2.dilate(thresh_right, kernel, iterations=1)
    thresh_right = cv2.erode(thresh_right, kernel, iterations=1)
    thresh_right = cv2.GaussianBlur(thresh_right, (9, 9), 0)
    thresh_right = cv2.resize(thresh_right, (0, 0), fx=THRESHOLD_SCALE, fy=THRESHOLD_SCALE)
    cv2.imshow("thresh_right_eye_roi", thresh_right)

    circles_left = cv2.HoughCircles(thresh_left, cv.CV_HOUGH_GRADIENT, 2, 200,
                                    param1=30, param2=10, minRadius=15, maxRadius=25)
    circles_right = cv2.HoughCircles(thresh_right, cv.CV_HOUGH_GRADIENT, 2, 200,
                                     param1=30, param2=10, minRadius=15, maxRadius=25)

    if circles_left is not None:
        circles_left = np.uint16(np.around(circles_left))
        for (x, y, r) in circles_left[0, :]:
            cv2.circle(left_eye_roi, (x/THRESHOLD_SCALE, y/THRESHOLD_SCALE), \
                        r/THRESHOLD_SCALE, (0, 0, 0), 2)
    if circles_right is not None:
        circles_right = np.uint16(np.around(circles_right))
        for (x, y, r) in circles_right[0, :]:
            cv2.circle(right_eye_roi, (x/THRESHOLD_SCALE, y/THRESHOLD_SCALE), \
                        r/THRESHOLD_SCALE, (0, 0, 0), 2)

    cv2.imshow("left_eye_roi", left_eye_roi)
    cv2.imshow("right_eye_roi", right_eye_roi)
    cv2.resizeWindow("left_eye_roi", 300, 300)
    cv2.resizeWindow("right_eye_roi", 300, 300)


def find_eye_region(face, eye):
    eye_x, eye_y, eye_width, eye_height = eye
    cv2.rectangle(face, (eye_x, eye_y), (eye_x + eye_width, eye_y + eye_height), (255, 0, 0), 1)


if __name__ == '__main__':
    main()

