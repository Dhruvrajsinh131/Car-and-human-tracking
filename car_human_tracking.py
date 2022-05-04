import cv2
from cv2 import COLOR_BAYER_BG2GRAY

# video
video = cv2.VideoCapture('xs.mp4')

#pretrained car classifer
car_tracker_file='carhaar.xml'

#create classifier
car_tracker=cv2.CascadeClassifier(car_tracker_file)

#pretrained pededstrian classifer
pedestrian_tracker_file='humanhaar.xml'

#create classifier
pedestrian_tracker=cv2.CascadeClassifier(pedestrian_tracker_file)


#Runs untill cars stops
while True:
    #read current frame
    (read_successful,frame)=video.read()
    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break

 #detecting the cars and pedestians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    print(cars)

## Put rectangle around the cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

# Put rectangle around the pedetrians
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

   
#display the video with cars spotted
    cv2.imshow('My Car Detector',frame)

#wait here and listen for a key press(otherwise it would be closed in blink)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

    
video.release()


print("Cars detected successfully")

# print(cars)