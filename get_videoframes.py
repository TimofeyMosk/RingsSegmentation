import cv2
capture = cv2.VideoCapture('video/1e7d9cad-ad5a-4ca5-9d5a-0ca5c4089ddd.avi')
frameNr = 0
while (True):
 
    success, frame = capture.read()
 
    if success:
            cv2.imwrite(f'allFrames/frame_{frameNr}.jpg', frame)
    else:
        break
 
    frameNr = frameNr+1
 
capture.release()