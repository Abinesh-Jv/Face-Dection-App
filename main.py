import cv2

frontFaceData = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)


while True:
    frame_capture_status,f = cam.read()
    BW_frame = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    face_coords = frontFaceData.detectMultiScale(BW_frame)

    for(x,y,w,h) in face_coords:
        cv2.rectangle(f,(x,y),(x+w,y+h),(0,0,255))
    
    cv2.imshow("Face Deductor",f)
    key=cv2.waitKey(1)

    if key == 27:
        break

print("SUCCESS")
