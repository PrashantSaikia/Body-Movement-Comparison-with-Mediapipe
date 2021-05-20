# Body Movement Comparison with Mediapipe

This is an AI that gives real-time feedback to the user of how they're performing a body movement (like, workout, dance, etc) against a benchmark video. 

You can give the benchmark video as a pre-saved file, and the user video as either a pre-saved file or with the webcam feed.

Its built using [mediapipe](https://github.com/google/mediapipe) in the backend, so we get a pretty high FPS (around 15 on MacBook Pro 16) when running on CPU only, as opposed to 2-3 FPS when the same application was built with [tf-pose-estimation](https://github.com/ZheC/tf-pose-estimation).

## Usage

```
from move_comparison import compare_positions

benchmark_video = 'dance_videos/benchmark_dance.mp4'
user_video = 'dance_videos/right_dance.mp4' # replace with 0 for webcam

compare_positions(benchmark_video, user_video)
```

<img width="1439" alt="Screenshot 2021-05-20 at 3 24 54 PM" src="https://user-images.githubusercontent.com/39755678/118936981-a1b65b80-b97f-11eb-8d42-ad11afc0bc2e.png">

You can create your own moves with the `create_moves.py` file. It essentially opens the webcam, lets you do your move, and when you're done, you can press `Q` to save it with the supplied name: `create_move('Move 1')`
