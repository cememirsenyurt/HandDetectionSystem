import cv2
import mediapipe as mp
import time

#video capture from camera
cap = cv2.VideoCapture(0)

#hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands()

#to draw line between the points (21) on hand we will need to use
# the function that is provided from mediapipe
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #this print will printout information if there are
    # hand/hands that shown to the camera
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 0:
                    #first landmark
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            #we wanted to draw lines on the image that we display not the RGB one

    cTime = time.time()     #current time
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    #image show
    cv2.imshow("Image", img)
    cv2.waitKey(1)
