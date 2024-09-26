import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

# Screen and webcam settings
wCam, hCam = 680, 480  # Width and height of the webcam feed
frameR = 100  # Frame reduction for mouse control area
smoothening = 7  # Smoothening factor for mouse movement

# Initialize previous and current locations for smooth movement
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, wCam)  # Set width of webcam feed
cap.set(4, hCam)  # Set height of webcam feed

# Initialize hand detector
detector = htm.HandDetector(maxHands=1)

# Get screen width and height
wScr, hScr = autopy.screen.size()

# Initialize previous time for FPS calculation
pTime = 0

while True:
    # Capture frame from webcam
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Find hand landmarks in the frame
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Check if landmarks are detected
    if lmList:
        # Get the tip positions of the index and middle fingers
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip

        # Check which fingers are up
        fingers = detector.fingersUp()

        # Draw frame boundary for mouse control area
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # Moving mode: Only index finger is up
        if fingers[1] == 1 and fingers[2] == 0:
            # Convert coordinates to screen size
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # Smoothen the values for smooth mouse movement
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Move mouse pointer
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)

            # Update previous location
            plocX, plocY = clocX, clocY

        # Clicking mode: Both index and middle fingers are up
        if fingers[1] == 1 and fingers[2] == 1:
            # Find the distance between index and middle finger tips
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # If distance is short, perform mouse click
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Show the image with hand tracking and mouse actions
    cv2.imshow("Virtual Mouse", img)

    # Exit the loop if the 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()