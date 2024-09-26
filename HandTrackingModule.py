import cv2
import mediapipe as mp
import time
import math
import numpy as np
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

class HandDetector:
    """
    A class to perform hand detection and tracking using MediaPipe.

    Attributes:
    -----------
    mode : bool
        Whether to treat the input images as a batch or not.
    maxHands : int
        The maximum number of hands to detect.
    detectionCon : float
        Minimum confidence value for hand detection.
    trackCon : float
        Minimum confidence value for hand tracking.

    Methods:
    --------
    findHands(img, draw=True):
        Detects hands in an image and draws landmarks if specified.

    findPosition(img, handNo=0, draw=True):
        Finds the position of hand landmarks in the specified hand and draws bounding box.

    fingersUp():
        Returns a list representing whether each finger is up or down.

    findDistance(p1, p2, img, draw=True, r=10, t=3):
        Finds the distance between two specified landmarks and draws the connection.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        """
        Initializes the HandDetector object with given parameters.
        """
        self.mode = mode
        self.maxHands = maxHands
        # Ensure detectionCon and trackCon are float values
        self.detectionCon = float(detectionCon)
        self.trackCon = float(trackCon)

        # Initialize MediaPipe Hands solution.
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # Landmarks corresponding to finger tips.

    def findHands(self, img, draw=True):
        """
        Detects hands in the given image and optionally draws the hand landmarks.

        :param img: Input image in which hands need to be detected.
        :param draw: Boolean, if True, draws landmarks on the image.
        :return: Image with or without drawn hand landmarks.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # Check if hand landmarks were detected.
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw the hand landmarks and connections on the image.
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        """
        Finds the position of each landmark in the specified hand and draws a bounding box.

        :param img: Input image from which landmark positions need to be extracted.
        :param handNo: Index of the hand to extract landmarks from (default is the first detected hand).
        :param draw: Boolean, if True, draws circles on the detected landmarks and bounding box.
        :return: List of landmark positions as [id, x, y] and the bounding box.
        """
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            # Get the specified hand's landmarks.
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    # Draw circles on the landmark positions.
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Draw a bounding box around the hand.
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        """
        Determines which fingers are up based on landmark positions.

        :return: List of 1s (up) and 0s (down) for each finger.
        """
        fingers = []
        # Check if the thumb is up.
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Check if the other fingers are up.
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=10, t=3):
        """
        Finds the distance between two specified landmarks and draws the connection.

        :param p1: Index of the first landmark.
        :param p2: Index of the second landmark.
        :param img: Image on which to draw.
        :param draw: Boolean, if True, draws the connection.
        :param r: Radius of the circles to draw.
        :param t: Thickness of the line to draw.
        :return: The distance, the image, and the coordinates of the landmarks and midpoint.
        """
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            # Draw the line, circles, and midpoint circle.
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    # Video capture from webcam.
    cap = cv2.VideoCapture(0)  # Change to 0 or 1 based on your camera index.
    detector = HandDetector()
    pTime = 0  # Previous time for FPS calculation.

    while True:
        success, img = cap.read()
        if not success or img is None:
            print("Failed to capture image")
            break

        # Find hands and landmarks.
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if lmList:
            # Print the position of the thumb tip (landmark 4).
            print("Thumb Tip Position:", lmList[4])

        # Calculate FPS and display it.
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Show the image.
        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit.
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()