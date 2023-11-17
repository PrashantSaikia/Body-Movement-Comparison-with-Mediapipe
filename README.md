# Body Movement Comparison with Mediapipe

This is an AI that gives real-time feedback to the user of how they're performing a body movement (like, workout, dance, etc) against a benchmark video. 

You can give the benchmark video as a pre-saved file, and the user video as either a pre-saved file or with the webcam feed.

Its built using [mediapipe](https://github.com/google/mediapipe) in the backend, so we get a pretty high FPS (around 15 on MacBook Pro 16) when running on CPU only, as opposed to 2-3 FPS when the same application was built with [tf-pose-estimation](https://github.com/ZheC/tf-pose-estimation).

## Usage on stand alone app on local machine

```
from move_comparison import compare_positions

benchmark_video = 'dance_videos/benchmark_dance.mp4'
user_video = 'dance_videos/right_dance.mp4' # replace with 0 for webcam

compare_positions(benchmark_video, user_video)
```

<img width="1439" alt="Screenshot 2021-05-20 at 3 24 54 PM" src="https://user-images.githubusercontent.com/39755678/118936981-a1b65b80-b97f-11eb-8d42-ad11afc0bc2e.png">

You can create your own moves with the `create_moves.py` file. It essentially opens the webcam, lets you do your move, and when you're done, you can press `Q` to save it with the supplied name: `create_move('Move 1')`

## Usage as web app via streamlit

```
streamlit run streamlit_app.py
```

# GCP Deployment in Kubernetes Cluster

In console:

```
1. gcloud services enable containerregistry.googleapis.com
2. git clone https://github.com/PrashantSaikia/Body-Movement-Comparison-with-Mediapipe.git
3. cd Body-Movement-Comparison-with-Mediapipe/
4. docker build -t app .
5. docker tag app gcr.io/dance-comparison/app
6. docker push gcr.io/dance-comparison/app
```

Then create a Kubernetes cluster via the UI, create a deployment with the image pushed, and set the port to 8501.

If you get `Pod errors: Error with exit code 2`, `Pod errors: CrashLoopBackOff` or `Does not have minimum availability`, wait for a couple of minutes, wait for the pods to become available, refresh and check again.
