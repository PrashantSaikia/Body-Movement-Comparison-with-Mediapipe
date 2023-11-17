import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Streamlit layout
st.title("Pose Mimicking App")

with st.sidebar:
    st.header("Video Upload")
    benchmark_video_file = st.file_uploader("Upload a benchmark video", type=["mp4", "mov", "avi", "mkv"], key="benchmark")
    uploaded_video = st.file_uploader("Upload your video", type=["mp4", "mov", "avi", "mkv"], key="user_video")

# Initialize Streamlit session state
if 'playing' not in st.session_state:
    st.session_state.playing = False

# Start/Clear button logic
if not st.session_state.playing:
    if st.button('Start'):
        st.session_state.playing = True
else:
    if st.button('Clear'):
        st.session_state.playing = False

# Function to save uploaded file to a temporary file and return the path
def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    return None

# Function to calculate cosine distance
def cosine_distance(landmarks1, landmarks2):
    if landmarks1 and landmarks2:
        points1 = np.array([(lm.x, lm.y, lm.z) for lm in landmarks1.landmark])
        points2 = np.array([(lm.x, lm.y, lm.z) for lm in landmarks2.landmark])
        dot_product = np.dot(points1.flatten(), points2.flatten())
        norm_product = np.linalg.norm(points1.flatten()) * np.linalg.norm(points2.flatten())
        similarity = dot_product / norm_product
        distance = 1 - similarity
        return distance
    else:
        return 1

# Main video processing logic
if st.session_state.playing and benchmark_video_file and uploaded_video:
    # Save uploaded videos to temporary files and read them
    temp_file_path_benchmark = save_uploaded_file(benchmark_video_file)
    temp_file_path_user = save_uploaded_file(uploaded_video)
    cap_benchmark = cv2.VideoCapture(temp_file_path_benchmark)
    cap_user = cv2.VideoCapture(temp_file_path_user)

    # Check if videos are valid
    if not cap_benchmark.isOpened() or not cap_user.isOpened():
        st.error("Failed to open video streams. Please check the video files.")
        st.session_state.playing = False
    else:
        # Layout for videos
        col1, col2, col3 = st.columns([1, 1, 1])

        # Create placeholders for videos and statistics
        benchmark_video_placeholder = col1.empty()
        user_video_placeholder = col2.empty()
        stats_placeholder = col3.empty()

        correct_steps = 0
        total_frames = 0

        # Process and display videos
        while st.session_state.playing:
            ret_benchmark, frame_benchmark = cap_benchmark.read()
            ret_user, frame_user = cap_user.read()

            if not ret_benchmark or not ret_user:
                break

            total_frames += 1

            # Pose detection for benchmark
            image_benchmark = cv2.cvtColor(frame_benchmark, cv2.COLOR_BGR2RGB)

            # Pose detection for user
            image_user = cv2.cvtColor(frame_user, cv2.COLOR_BGR2RGB)
            image_user.flags.writeable = False
            results_user = pose.process(image_user)

            image_user.flags.writeable = True
            image_user = cv2.cvtColor(image_user, cv2.COLOR_RGB2BGR)

            if results_user.pose_landmarks:
                mp_drawing.draw_landmarks(image_user, results_user.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display videos
            benchmark_video_placeholder.image(image_benchmark, channels="RGB", use_column_width=True)
            user_video_placeholder.image(image_user, channels="BGR", use_column_width=True)

            # Calculate error and update statistics
            error = cosine_distance(results_user.pose_landmarks, pose.process(image_benchmark).pose_landmarks) * 100
            correct_step = error < 30
            correct_steps += correct_step

            # Update statistics
            stats = f"""
                Frame Error: {error:.2f}%\n
                Step: {'CORRECT STEP' if correct_step else 'WRONG STEP'}\n
                Cumulative Accuracy: {(correct_steps / total_frames) * 100:.2f}%
            """
            stats_placeholder.markdown(stats)

        cap_benchmark.release()
        cap_user.release()
