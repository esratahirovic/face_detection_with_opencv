import cv2 as cv

img = cv.resize(cv.imread("kids.jpg"), (0,0), fx=0.3, fy=0.3)
#use cascade classifer for detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")
mount_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_smile.xml")

while True:
    # convert the image into black and white for accuracy
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.5, 5)

    for (x,y, w,h) in faces:
        # draw a rectangle around the detected face
        cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.5, 5)
        for (ex,ey,ew,eh) in eyes:
            eye_center = (x + ex + ew // 2, y + ey + eh // 2)
            radius = int(round((ew + eh) * 0.25))
            # draw circles around the detected eyes
            frame = cv.circle(img, eye_center, radius, (255, 0, 0), 3)

        mounts = mount_cascade.detectMultiScale(roi_gray, 1.5, 9)
        for (mx, my, mw, mh) in mounts:
            # draw a rectangle around the detected mount
            cv.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (50,50,50), 2)


    cv.imshow("faces", img)

    #for ending the loop
    if cv.waitKey(0) == ord("q"):
        break