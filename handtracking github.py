#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
#palm detection and identify hand landmarks
import mediapipe as mp
#for the frame rate of the video
import time


# In[ ]:


cap = cv2.VideoCapture(0)
#0 is the camera no. we can use 1 instead of 0
#cap- capture the video 
mpHands = mp.solutions.hands
hands = mpHands.Hands()
#this hands have 4 default parameters
mpDraw = mp.solutions.drawing_utils             #identify the hand mark the points
pTime = 0
cTime = 0          #this is for identify the points on the hand
while True:
    success, img= cap.read()                         #returns the frame
    #flip the screen
    img= cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)   #hands obj only takes rgb images
    results = hands.process(imgRGB)                 #process is a method in hands
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:            #handLms = hand landmark
            for id, lm in enumerate(handLms.landmark):    #returns the finger landmark with their id
                #print(id,lm)
                h, w, c=img.shape
                cx, cy = int(lm.x*w),int(lm.y*h)          #identify position of center of x and y
                print(id,cx,cy)
                if id==8:                                 #8tH landmark
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
            
            mpDraw.draw_landmarks(img,handLms, mpHands.HAND_CONNECTIONS)                  #only for 1 hand  -  put the handlandmarks
    cTime = time.time()                   #this is fixing the fps
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)


# In[ ]:




