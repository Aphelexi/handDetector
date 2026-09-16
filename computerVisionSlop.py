import cv2 as cv
import mediapipe as mp
import math

#create a VideoCapture object to access the webcam, initialize mediapipe hands and drawing utils
cam = cv.VideoCapture(0)
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

#check if a specific finger is pointing by comparing the distance from the fingertip to the wrist with the distance from the knuckle to the wrist
def isPointing(hand_landmarks, whatFinger):
    wrist = hand_landmarks.landmark[0]
    if whatFinger == "index":
        knuckle = hand_landmarks.landmark[6]
        fingertip = hand_landmarks.landmark[8]    
        
    if whatFinger == "middle":
        knuckle = hand_landmarks.landmark[10]
        fingertip = hand_landmarks.landmark[12]
        
    if whatFinger == "ring":
        knuckle = hand_landmarks.landmark[14]
        fingertip = hand_landmarks.landmark[16]
        
    if whatFinger == "pinky":
        knuckle = hand_landmarks.landmark[18]
        fingertip = hand_landmarks.landmark[20] 

    distToWrist = math.hypot(fingertip.x - wrist.x, fingertip.y - wrist.y)
    knuckleToWrist = math.hypot(knuckle.x - wrist.x, knuckle.y - wrist.y)
    return distToWrist > knuckleToWrist
    


with mpHands.Hands(min_tracking_confidence=0.25, min_detection_confidence=0.25) as hands:
    while cam.isOpened():

        #read and verify the camera is working
        ret, frame = cam.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)  #flip the frame horizontally for a mirror effect

        #convert frame to RGB for processing
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        #reset the (x, y) coordinates of the fingertips each frame
        fingertips = []

        #process and draw hand landmarks on the frame
        process_frames = hands.process(rgb_frame)
        if process_frames.multi_hand_landmarks:
            for lm in process_frames.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, lm, mpHands.HAND_CONNECTIONS)

                #check if a finger is pointing & append to fingertips
                if isPointing(lm, "index"):
                    fingertips.append((int(lm.landmark[8].x * frame.shape[1]), int(lm.landmark[8].y * frame.shape[0])))
                elif isPointing(lm, "middle"):
                    fingertips.append((int(lm.landmark[12].x * frame.shape[1]), int(lm.landmark[12].y * frame.shape[0])))
                elif isPointing(lm, "ring"):
                    fingertips.append((int(lm.landmark[16].x * frame.shape[1]), int(lm.landmark[16].y * frame.shape[0])))
                elif isPointing(lm, "pinky"):
                    fingertips.append((int(lm.landmark[20].x * frame.shape[1]), int(lm.landmark[20].y * frame.shape[0])))

        #if there are 2 fingertips pointing, draw a line, calculate distance, and display it on the frame
        if len(fingertips) == 2:
            (x1, y1), (x2, y2) = fingertips
            cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, f"Distance: {int(((x1 - x2)**2 + (y1 - y2)**2)**0.5)} px", ((x1 + x2)//2 - 50,(y1 + y2)//2), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1) 




        #flip & display the resulting frame
        cv.imshow('diddyMustard', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

#release the capture and close windows
cam.release()
cv.destroyAllWindows()