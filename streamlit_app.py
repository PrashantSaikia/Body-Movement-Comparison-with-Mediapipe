import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False)

# Streamlit layout
st.title("AI Dance Trainer")

with st.sidebar:
    st.header("Video Upload")
    benchmark_video_file = st.file_uploader("Upload a benchmark video", type=["mp4", "mov", "avi", "mkv"], key="benchmark")
    uploaded_video = st.file_uploader("Upload your video", type=["mp4", "mov", "avi", "mkv"], key="user_video")

if 'playing' not in st.session_state:
    st.session_state.playing = False

if not st.session_state.playing:
    if st.button('Start'):
        st.session_state.playing = True
else:
    if st.button('Clear'):
        st.session_state.playing = False

@st.cache_data
def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    return None

def cosine_distance(landmarks1, landmarks2):
    if landmarks1 and landmarks2:
        points1 = np.array([(lm.x, lm.y, lm.z) for lm in landmarks1.landmark])
        points2 = np.array([(lm.x, lm.y, lm.z) for lm in landmarks2.landmark])
        dot_product = np.dot(points1.flatten(), points2.flatten())
        norm_product = np.linalg.norm(points1.flatten()) * np.linalg.norm(points2.flatten())
        similarity = dot_product / norm_product
        return 1 - similarity
    else:
        return 1

if st.session_state.playing and benchmark_video_file and uploaded_video:
    temp_file_path_benchmark = save_uploaded_file(benchmark_video_file)
    temp_file_path_user = save_uploaded_file(uploaded_video)
    cap_benchmark = cv2.VideoCapture(temp_file_path_benchmark)
    cap_user = cv2.VideoCapture(temp_file_path_user)

    if not cap_benchmark.isOpened() or not cap_user.isOpened():
        st.error("Failed to open video streams. Please check the video files.")
        st.session_state.playing = False
    else:
        col1, col2, col3 = st.columns([1, 1, 1])
        benchmark_video_placeholder = col1.empty()
        user_video_placeholder = col2.empty()
        stats_placeholder = col3.empty()

        correct_steps = 0
        total_frames = 0
        frame_skip_rate = 1  # Process every n'th frame

        while st.session_state.playing:
            for _ in range(frame_skip_rate):
                cap_benchmark.read()
                cap_user.read()

            ret_benchmark, frame_benchmark = cap_benchmark.read()
            ret_user, frame_user = cap_user.read()

            if not ret_benchmark or not ret_user:
                break

            total_frames += 1

            image_benchmark = cv2.cvtColor(frame_benchmark, cv2.COLOR_BGR2RGB)
            image_user = cv2.cvtColor(frame_user, cv2.COLOR_BGR2RGB)

            results_user = pose.process(image_user)
            results_benchmark = pose.process(image_benchmark)

            if results_user.pose_landmarks:
                mp_drawing.draw_landmarks(image_user, results_user.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            benchmark_video_placeholder.image(image_benchmark, channels="RGB", use_column_width=True)
            user_video_placeholder.image(image_user, channels="BGR", use_column_width=True)

            error = cosine_distance(results_user.pose_landmarks, results_benchmark.pose_landmarks) * 100
            correct_step = error < 30
            correct_steps += correct_step

            stats = f"""
                Frame Error: {error:.2f}%\n
                Step: {'CORRECT STEP' if correct_step else 'WRONG STEP'}\n
                Cumulative Accuracy: {(correct_steps / total_frames) * 100:.2f}%
            """
            stats_placeholder.markdown(stats)

        cap_benchmark.release()
        cap_user.release()
