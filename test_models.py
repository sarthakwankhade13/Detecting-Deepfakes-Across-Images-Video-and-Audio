from video_model import detect_video

video_path = "sample.mp4"

verdict, conf = detect_video(video_path)

print("VIDEO RESULT:", verdict, conf)