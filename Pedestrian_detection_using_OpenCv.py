#Pedestrians Detection for the video 
#Importing the packages required
import cv2

#Loading the Video into the python
cap = cv2.VideoCapture('D:/Artficial Intelligence/Assignments/7/pedestrians.mp4')

pedestrian_cascade = cv2.CascadeClassifier('D:/Artficial Intelligence/Assignments/7/pedestrian.xml')

while True:
    ret, img = cap.read()
	
    
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pedestrian_cascade = pedestrian_cascade.detectMultiScale(gray,1.3,2)

    for(a,b,c,d) in pedestrian_cascade:
        cv2.rectangle(img,(a,b),(a+c,b+d),(0,255,210),4)
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
