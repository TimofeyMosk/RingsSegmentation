import cv2
capture = cv2.VideoCapture('video/1e7d9cad-ad5a-4ca5-9d5a-0ca5c4089ddd.avi')
frameNr = 0
while (True):
    success, frame = capture.read()
    n = 1

    if success:
            if frameNr % n == 0:
                cv2.imwrite(f'images/frame_{frameNr // n}.jpg', frame)
    else:
        break
 
    frameNr = frameNr+1
 
capture.release()