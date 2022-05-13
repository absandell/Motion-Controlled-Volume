import math
import cv2
import mediapipe as mp
import numpy as np
import time
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

################################
wCam, hCam = 640, 640
################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

currentTime = 0
previousTime = 0

detector = htm.handDetector()

# Audio Interface
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volumeRange = volume.GetVolumeRange()
minVol = volumeRange[0]
maxVol = volumeRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    if img is not None:
        img = detector.findHands(img)
        landmarkList = detector.findPosition(img, draw=False)
        if len(landmarkList) != 0:
            # Get x/y values of points 4 and 8
            x1, y1 = landmarkList[4][1], landmarkList[4][2]
            x2, y2 = landmarkList[8][1], landmarkList[8][2]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            cv2.circle(img, (x1, y1), 5, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (0, 255, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)

            # Hand Range 15 - 120
            # Volume Range -65 - 0

            vol = np.interp(length, [15, 120], [minVol, maxVol])
            volBar = np.interp(length, [15, 120], [400, 150])
            volPer = np.interp(length, [15, 120], [0, 100])
            volume.SetMasterVolumeLevel(vol, None)
            print(length, vol)

            if length < 15:
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        # FPS Measure
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2, )

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    else:
        break