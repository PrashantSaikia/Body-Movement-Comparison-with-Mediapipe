import cv2, time
import pose_module as pm
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
import streamlit as st

st.title('Dance Comparison')

st.button("Clear", type="primary")

option = st.radio(label='', options=['Example', 'Try Yourself'])

placeholder = st.columns(2, gap="small")

col1 = placeholder[0].empty()
col2 = placeholder[0].empty()

def countdown():
	for sec in range(3,0,-1):
		col1.metric(label='', value=sec)
		time.sleep(1)

def restart():
	st.session_state.page = 0

if st.button("Start"):

	countdown()

	benchmark_video = 'dance_videos/benchmark_dance.mp4'
	
	if option == 'Example':
		user_video = 'dance_videos/right_dance.mp4'
	else:
		user_video = 0

	col1.subheader("Benchmark")
	col2.subheader("User")

	benchmark_cam = cv2.VideoCapture(benchmark_video)
	user_cam = cv2.VideoCapture(user_video) 

	fps_time = 0 #Initializing fps to 0

	detector_1 = pm.poseDetector()
	detector_2 = pm.poseDetector()
	frame_counter = 0
	correct_frames = 0

	frameSkip = 2  # Skip every 2 frames
	frameCount = 0

	while (benchmark_cam.isOpened() or user_cam.isOpened()):
		frameCount += 1
		if frameCount % frameSkip == 0:
			try:
				ret_val, image_1 = benchmark_cam.read()

				#Loop the video if it ended. If the last frame is reached, reset the capture and the frame_counter
				if frame_counter == benchmark_cam.get(cv2.CAP_PROP_FRAME_COUNT):
					frame_counter = 0 #Or whatever as long as it is the same as next line
					correct_frames = 0
					benchmark_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
					benchmark_cam.set(cv2.CAP_PROP_BUFFERSIZE, 2)

				# image_1 = cv2.resize(image_1, (720,640))
				image_1 = detector_1.findPose(image_1)
				lmList_user = detector_1.findPosition(image_1, draw=False)
				# del lmList_user[1:11]

				ret_val_1,image_2 = user_cam.read()
				#Loop the video if it ended. If the last frame is reached, reset the capture and the frame_counter
				if frame_counter == user_cam.get(cv2.CAP_PROP_FRAME_COUNT):
					frame_counter = 0 #Or whatever as long as it is the same as next line
					correct_frames = 0
					user_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
					user_cam.set(cv2.CAP_PROP_BUFFERSIZE, 2)

				# image_2 = cv2.resize(image_2, (720,640))
				image_2 = detector_2.findPose(image_2)
				lmList_benchmark = detector_2.findPosition(image_2, draw=False)
				# del lmList_benchmark[1:11]

				frame_counter += 1


				if ret_val_1 or ret_val:
					error, _ = fastdtw(lmList_user, lmList_benchmark, dist=cosine)

					# Displaying the error percentage
					cv2.putText(image_2, 'Error: {}%'.format(str(round(100*(float(error)),2))), (10, 30),
									cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

					# If the similarity is > 90%, take it as correct step. Otherwise incorrect step.
					if error < 0.3:
						cv2.putText(image_2, "CORRECT STEPS", (40, 600),
									cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
						correct_frames += 1
					else:
						cv2.putText(image_2,  "INCORRECT STEPS", (40, 600),
									cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
					cv2.putText(image_2, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 50),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

					# Display the dynamic accuracy of dance as the percentage of frames that appear as correct
					if frame_counter==0:
						frame_counter = user_cam.get(cv2.CAP_PROP_FRAME_COUNT)
					cv2.putText(image_2, "Dance Steps Accurately Done: {}%".format(str(round(100*correct_frames/frame_counter, 2))), (10, 70), 
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
					
					# Display both the benchmark and the user videos
					col1.image(image_1, channels="BGR", use_column_width=True)
					col2.image(image_2, channels="BGR", use_column_width=True)

					fps_time = time.time()
					if cv2.waitKey(1) & 0xFF == ord('q'):
						break
				else:
					break
			except:
				pass

	benchmark_cam.release()
	user_cam.release()
	cv2.destroyAllWindows()
