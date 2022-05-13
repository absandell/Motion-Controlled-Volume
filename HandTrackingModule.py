import cv2
import mediapipe as mp
import time

# 0. WRIST
# 1. THUMB_CMC
# 2. THUMB_MCP
# 3. THUMB_IP
# 4. THUMB_TIP
# 5. INDEX_FINGER_MCP
# 6. INDEX_FINGER_PIP
# 7. INDEX_FINGER_DIP
# 8. INDEX_FINGER_TIP
# 9. MIDDLE_FINGER_MCP
# 10. MIDDLE_FINGER_PIP
# 11. MIDDLE_FINGER_DIP
# 12. MIDDLE_FINGER_TIP
# 13. RING_FINGER_MCP
# 14. RING_FINGER_PIP
# 15. RING_FINGER_DIP
# 16. RING_FINGER_TIP
# 17. PINKY_MCP
# 18. PINKY_PIP
# 19. PINKY_DIP
# 20. PINKY_TIP

class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils # What we use to draw on hands

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Defines RGB version of image to use in processing
        self.results = self.hands.process(imgRGB) # Processes image
        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks: # Extract info from each hand
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS) # Draw on original image (each hand)
        return img

    def findPosition(self, img, handNum=0, draw=True):

        landmarkList = [] #  List of Landmark positions
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for id, landmark in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy, = int(landmark.x * w), int(landmark.y * h)
                landmarkList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 20, (0, 255, 255), cv2.FILLED)  # Draws circle on specific  reference point id
        return landmarkList

def main():
    previousTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()  # Reads video input

        # Detector
        img = detector.findHands(img)
        landmarkList = detector.findPosition(img)
        if len(landmarkList) != 0:
            print(landmarkList[4])

        # FPS Measure
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2, )

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()