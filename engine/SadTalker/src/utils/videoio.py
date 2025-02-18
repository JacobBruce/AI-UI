import shutil
import uuid
import os
import cv2

from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

def load_video_to_cv2(input_path):
    video_stream = cv2.VideoCapture(input_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    full_frames = [] 
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break 
        full_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return full_frames

def save_video_with_watermark(video, audio, save_path, watermark=False):	
	if os.path.isfile(video) and os.path.isfile(audio):
		video_clip = VideoFileClip(video)
		audio_clip = AudioFileClip(audio)
		
		video_clip.audio = CompositeAudioClip([audio_clip])
		video_clip.write_videofile(save_path)
