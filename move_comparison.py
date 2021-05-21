import cv2, time
import pose_module as pm
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
 
def compare_positions(benchmark_video, user_video):
	benchmark_cam = cv2.VideoCapture(benchmark_video)
	user_cam = cv2.VideoCapture(user_video) 

	fps_time = 0 #Initializing fps to 0

	detector_1 = pm.poseDetector()
	detector_2 = pm.poseDetector()
	frame_counter = 0
	correct_frames = 0

	while (benchmark_cam.isOpened() or user_cam.isOpened()):
		try:
			ret_val, image_1 = user_cam.read()
			#Loop the video if it ended. If the last frame is reached, reset the capture and the frame_counter
			if frame_counter == user_cam.get(cv2.CAP_PROP_FRAME_COUNT):
				frame_counter = 0 #Or whatever as long as it is the same as next line
				correct_frames = 0
				user_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

			winname = "User Video"
			cv2.namedWindow(winname)		   # Create a named window
			cv2.moveWindow(winname, 720,-100)  # Move it to desired location
			image_1 = cv2.resize(image_1, (720,640))
			image_1 = detector_1.findPose(image_1)
			lmList_user = detector_1.findPosition(image_1)
			del lmList_user[1:11]

			ret_val_1,image_2 = benchmark_cam.read()
			#Loop the video if it ended. If the last frame is reached, reset the capture and the frame_counter
			if frame_counter == benchmark_cam.get(cv2.CAP_PROP_FRAME_COUNT):
				frame_counter = 0 #Or whatever as long as it is the same as next line
				correct_frames = 0
				benchmark_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

			image_2 = cv2.resize(image_2, (720,640))
			image_2 = detector_2.findPose(image_2)
			lmList_benchmark = detector_2.findPosition(image_2)
			del lmList_benchmark[1:11]

			frame_counter += 1


			if ret_val_1 or ret_val:
				error, _ = fastdtw(lmList_user, lmList_benchmark, dist=cosine)

				# Displaying the error percentage
				cv2.putText(image_1, 'Error: {}%'.format(str(round(100*(float(error)),2))), (10, 30),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

				# If the similarity is > 90%, take it as correct step. Otherwise incorrect step.
				if error < 0.3:
					cv2.putText(image_1, "CORRECT STEPS", (40, 600),
								cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
					correct_frames += 1
				else:
					cv2.putText(image_1,  "INCORRECT STEPS", (40, 600),
								cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
				cv2.putText(image_1, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 50),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

				# Display the dynamic accuracy of dance as the percentage of frames that appear as correct
				if frame_counter==0:
					frame_counter = user_cam.get(cv2.CAP_PROP_FRAME_COUNT)
				cv2.putText(image_1, "Dance Steps Accurately Done: {}%".format(str(round(100*correct_frames/frame_counter, 2))), (10, 70), 
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
				
				# Display both the benchmark and the user videos
				cv2.imshow('Benchmark Video', image_2)
				cv2.imshow('User Video', image_1)

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
