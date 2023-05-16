import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

jump_count = 0
jump_detected = False

foot_position = 0

min_height_difference = 30

cooldown_period = 1.5
last_jump = time.time()


cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame,1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            left_foot_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y
            right_foot_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y
            left_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            left_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
            right_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y
            left_knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
            right_knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y

            if time.time() - last_jump >= cooldown_period:
                # Check if the feet are moving upwards and shoulders, hips, and knees are relatively stable (jumpingggg)
                if left_foot_y < foot_position and right_foot_y < foot_position \
                        and left_shoulder_y > right_shoulder_y and left_hip_y > right_hip_y \
                        and left_knee_y > right_knee_y and not jump_detected:
                    jump_detected = True
                # Not Jumpinggg
                elif left_foot_y > foot_position and right_foot_y > foot_position and jump_detected:
                    # Count a jump if the feet move back to the original position
                    jump_count += 1
                    jump_detected = False
                    last_jump = time.time()

            foot_position = max(left_foot_y, right_foot_y)
        # Display jump count on the frame :))
        cv2.putText(frame, f"Jump Count: {jump_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Display the resulting frame (almost done)
        cv2.imshow('Jump Counter', frame)
        # Break loop on 'q' key press (done)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()