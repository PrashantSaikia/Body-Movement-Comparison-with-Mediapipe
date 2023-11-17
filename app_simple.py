import streamlit as st
import mediapipe as mp
import cv2
import tempfile
import time
import base64

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Function to generate base64 encoding of the video
def get_video_base64(video_file):
    video_encoded = base64.b64encode(video_file.read()).decode()
    return video_encoded

# Initialize Streamlit session state
if 'playing' not in st.session_state:
    st.session_state.playing = False

# Streamlit layout
st.title("Pose Mimicking App")

with st.sidebar:
    st.header("Video Upload")
    benchmark_video_file = st.file_uploader("Upload a benchmark video", type=["mp4", "mov", "avi", "mkv"], key="benchmark")
    uploaded_video = st.file_uploader("Upload your video", type=["mp4", "mov", "avi", "mkv"], key="user_video")

# Start/Clear button logic
if not st.session_state.playing:
    if st.button('Start'):
        st.session_state.playing = True
else:
    if st.button('Clear'):
        st.session_state.playing = False

# When both videos are uploaded and playing is True
if st.session_state.playing and benchmark_video_file and uploaded_video:
    col1, col2 = st.columns(2)

    # Countdown
    countdown_placeholder = st.empty()
    for countdown in range(3, 0, -1):
        countdown_placeholder.header(f"Starting in {countdown}")
        time.sleep(1)
    countdown_placeholder.empty()

    # Display benchmark video
    with col1:
        video_base64 = get_video_base64(benchmark_video_file)
        video_html = f'<video width="100%" height="auto" autoplay loop><source src="data:video/mp4;base64,{video_base64}" type="video/mp4"></video>'
        st.markdown(video_html, unsafe_allow_html=True)

    # Process and display user video
    with col2:
        tfile2 = tempfile.NamedTemporaryFile(delete=False) 
        tfile2.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile2.name)
        
        frameST = st.empty()
        while st.session_state.playing:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Pose detection logic
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                frameST.image(image, channels="BGR", use_column_width=True)

        cap.release()
        
