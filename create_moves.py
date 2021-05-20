import cv2

def create_move(video_name):
    cap = cv2.VideoCapture(0)

    # Get the Default resolutions
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = 24

    # Define the codec and filename.
    out = cv2.VideoWriter('dance_videos/{}.mp4'.format(video_name), cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:

            # write the  frame
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
