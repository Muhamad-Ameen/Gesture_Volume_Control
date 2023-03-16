import cv2  ##Open Camera
import mediapipe as hand_tracking  ##Track The Hand
from math import hypot  ##Calculate The Hypotenuse
from ctypes import cast, POINTER  ## ↓
from comtypes import CLSCTX_ALL  ## connect to sound driver
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  ## ↑
import numpy as np  ##control volume

cam = cv2.VideoCapture(0)
# Hand Tracking
mpHands = hand_tracking.solutions.hands
hands = mpHands.Hands()
mpDraw = hand_tracking.solutions.drawing_utils
# volume
devices = AudioUtilities.GetSpeakers()
# print(devices)
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# print(interface)
volume = cast(interface, POINTER(IAudioEndpointVolume))
print(volume)
volMin, volMax = volume.GetVolumeRange()[:2]

while True:
    done, img = cam.read()
    results = hands.process(img).multi_hand_landmarks
    # print(img)

    lmList = []
    if results:
        for handlandmark in results:
            # print(handlandmark.landmark)
            for id1, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape
                # print(img.shape)
                # print(lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id1, cx, cy])
                # print(id1, cx, cy)
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

            # print(lmList)

            if lmList:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                # print(x1, x2, y1, y2)
                # cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
                # cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

                length = hypot(x2 - x1, y2 - y1)
                vol = np.interp(length, [15, 220], [volMin, volMax])
                # print(vol))
                volume.SetMasterVolumeLevel(vol, None)

    cv2.imshow('Mohammed', img)
    if cv2.waitKey(1) & 0xff == ord('M'):
        break
