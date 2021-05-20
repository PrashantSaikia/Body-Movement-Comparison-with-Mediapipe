# Body-Movement-Comparison

This is an AI that gives real-time feedback to the user of how they're performing a body movement (like, workout, dance, etc) against a benchmark video. 

You can give the benchmark video as a pre-saved file, and the user video as either a pre-saved file or with the webcam feed

## Usage

```
from move_comparison import compare_positions

benchmark_video = 'dance_videos/benchmark_dance.mp4'
user_video = 'dance_videos/right_dance.mp4' # replace with 0 for webcam

compare_positions(benchmark_video, user_video)
```
