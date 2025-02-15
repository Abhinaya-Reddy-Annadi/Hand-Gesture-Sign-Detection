import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import webbrowser

# Set Streamlit page
st.set_page_config(page_title="Go-With-Gesture", layout="wide")

# UI Header
st.title("✈️ Go-With-Gesture")
st.subheader("Navigate the check-in process with hand gestures!")

# Button to manually open the website
website_url = "https://go-with-gesture.b12sites.com/index"
if st.button("Open Check-in Website"):
    webbrowser.open(website_url)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
FRAME_WINDOW = st.image([])
gesture_text_display = st.empty()  # Placeholder for detected gesture text

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Could not access the camera. Please check permissions.")
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_text = "No Gesture Detected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark positions
            lm_list = []
            for lm in hand_landmarks.landmark:
                lm_list.append(lm)

            if lm_list:
                # Thumb and finger tip indices
                thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip = 4, 8, 12, 16, 20
                thumb_ip, index_dip, middle_dip, ring_dip, pinky_dip = 3, 7, 11, 15, 19

                # Thumb position
                thumb_up = lm_list[thumb_tip].y < lm_list[thumb_ip].y  # Thumb pointing up
                thumb_down = lm_list[thumb_tip].y > lm_list[thumb_ip].y  # Thumb pointing down

                # Finger fold status (True if folded)
                folded_fingers = [
                    lm_list[index_tip].y > lm_list[index_dip].y,
                    lm_list[middle_tip].y > lm_list[middle_dip].y,
                    lm_list[ring_tip].y > lm_list[ring_dip].y,
                    lm_list[pinky_tip].y > lm_list[pinky_dip].y,
                ]

                # Distance between thumb tip and index tip (for OK gesture)
                thumb_x, thumb_y = lm_list[thumb_tip].x, lm_list[thumb_tip].y
                index_x, index_y = lm_list[index_tip].x, lm_list[index_tip].y
                distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

                # 1️⃣ OK Gesture (👌)
                if distance < 0.05 and not folded_fingers[1] and not folded_fingers[2]:
                    gesture_text = "👌 OK Gesture Detected"

                # 2️⃣ LIKE Gesture (👍)
                elif thumb_up and all(folded_fingers):
                    gesture_text = "👍 LIKE Gesture Detected"

                # 3️⃣ DISLIKE Gesture (👎)
                elif thumb_down and all(folded_fingers):
                    gesture_text = "👎 DISLIKE Gesture Detected"

                # 4️⃣ STOP Gesture (✋) - All fingers extended upwards
                elif all(lm_list[i].y < lm_list[i - 1].y for i in [index_tip, middle_tip, ring_tip, pinky_tip]):
                    gesture_text = "✋ STOP Gesture Detected"

                # 5️⃣ PEACE Gesture (✌️) - Index and middle finger up, others folded
                elif not folded_fingers[0] and not folded_fingers[1] and folded_fingers[2] and folded_fingers[3]:
                    gesture_text = "✌️ PEACE Gesture Detected"

                # 6️⃣ CALL ME Gesture (🤙) - Thumb and pinky extended, others folded
                elif not folded_fingers[3] and not thumb_down and all(folded_fingers[:3]):
                    gesture_text = "🤙 CALL ME Gesture Detected"

                # 7️⃣ FORWARD Gesture (👆) - Index up, others folded
                elif not folded_fingers[0] and all(folded_fingers[1:]):
                    gesture_text = "👆 FORWARD Gesture Detected"

                # 8️⃣ LEFT Gesture (👈) - Index extended leftward, others folded
                elif lm_list[index_tip].x < lm_list[index_dip].x and all(folded_fingers[1:]):
                    gesture_text = "👈 LEFT Gesture Detected"

                # 9️⃣ RIGHT Gesture (👉) - Index extended rightward, others folded
                elif lm_list[index_tip].x > lm_list[index_dip].x and all(folded_fingers[1:]):
                    gesture_text = "👉 RIGHT Gesture Detected"

                # 🔟 I LOVE YOU Gesture (🤟) - Thumb, index, and pinky extended
                elif not folded_fingers[0] and folded_fingers[1] and folded_fingers[2] and not folded_fingers[3]:
                    gesture_text = "🤟 I LOVE YOU Gesture Detected"

    # Display Camera Feed
    FRAME_WINDOW.image(frame, channels="BGR")

    # Display Gesture Text
    gesture_text_display.write(f"**Gesture Detected:** {gesture_text}")                             `1`1````1111    

cap.release()
cv2.destroyAllWindows()
