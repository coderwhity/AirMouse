import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import mouse

# Initialize MediaPipe Hands
mp_hand = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hand = mp_hand.Hands(max_num_hands=1)
smoothening = 8  # Smoothing factor for cursor movement

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set camera width
cap.set(4, 480)  # Set camera height

# Screen resolution
screen_width, screen_height = pyautogui.size()
frameR = 100  # Region within the camera frame for cursor movement

# Variables for smoothing
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip and process the frame
    image = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pr = hand.process(image_rgb)

    # Draw a boundary box for movement area
    cv2.rectangle(image, (frameR, frameR), (640 - frameR, 480 - frameR), (255, 0, 255), 2)

    if pr.multi_hand_landmarks:
        for multi_lm in pr.multi_hand_landmarks:
            # Extract landmark positions
            lm_list = [(id, lm.x, lm.y) for id, lm in enumerate(multi_lm.landmark)]

            # Index and middle finger positions
            x1, y1 = int(lm_list[8][1] * 640), int(lm_list[8][2] * 480)  # Index finger
            x2, y2 = int(lm_list[12][1] * 640), int(lm_list[12][2] * 480)  # Middle finger

            # Detect which fingers are up
            fingers_up = [lm_list[i][2] < lm_list[i - 2][2] for i in [8, 12]]

            if fingers_up[0] and not fingers_up[1]:

                x3 = np.interp(x1, (frameR, 640 - frameR), (0, screen_width))
                y3 = np.interp(y1, (frameR, 480 - frameR), (0, screen_height))

                curr_x = prev_x + (x3 - prev_x) / smoothening
                curr_y = prev_y + (y3 - prev_y) / smoothening

                mouse.move(curr_x, curr_y)
                cv2.circle(image, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
                prev_x, prev_y = curr_x, curr_y


            if fingers_up[0] and fingers_up[1]:
                length = math.hypot(x2 - x1, y2 - y1)

                if length < 40:
                    cv2.circle(image, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                    mouse.click()

            mp_drawing.draw_landmarks(image, multi_lm, mp_hand.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

    # Display the frame
    cv2.imshow('Virtual Mouse', image)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
