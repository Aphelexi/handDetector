"""Pure gesture-detection logic for handDetector.

No camera, OpenCV, or MediaPipe imports here, so this module can be
imported and unit-tested without any hardware. The functions below are
moved over from handDetector.py unchanged.
"""
import math
from collections import deque

detectionFrames = 25
minDirectionChanges = 3
movementThresh = 2
leftMovementHistory = deque(maxlen=detectionFrames)
rightMovementHistory = deque(maxlen=detectionFrames)


def isWaving(hand_landmarks, label):
    if label == "Left":
        movementHistory = leftMovementHistory
    else:
        movementHistory = rightMovementHistory

    wristY = hand_landmarks.landmark[1].y
    movementHistory.append(wristY)

    if len(movementHistory) < detectionFrames:
        return False

    directions = []
    for i in range(1, len(movementHistory)):
        delta = movementHistory[i] - movementHistory[i - 1]
        if abs(delta) > movementThresh/detectionFrames:
            directions.append(1 if delta > 0 else -1)

    direction_changes = sum(
        1 for i in range(1, len(directions)) if directions[i] != directions[i - 1]
    )
    return direction_changes >= minDirectionChanges


# check if a specific finger is pointing by comparing the distance from the fingertip to the wrist with the distance from the knuckle to the wrist
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


# check if hand is open by verifying that all fingers are pointing
def isOpenHand(hand_landmarks):
    return all(isPointing(hand_landmarks, f) for f in ["index", "middle", "ring", "pinky"])
