import pandas as pd
import os
import subprocess
import datetime
import json
import cv2
import time
from moviepy import VideoFileClip

# Load train and test CSV files
train_csv = "/home/project/dataset/VGG_Sound/train.csv"
test_csv = "/home/project/dataset/VGG_Sound/test.csv"

train_df = pd.read_csv(train_csv, header=None, names=["File_Name", "Label"])
test_df = pd.read_csv(test_csv, header=None, names=["File_Name", "Label"])

# Merge train and test datasets
dataset_df = pd.concat([train_df], ignore_index=True)

# Directory to store videos
video_dir = "/home/project/dataset/VGG_Sound_videos/"
os.makedirs(video_dir, exist_ok=True)

def parse_filename(filename):
    parts = filename.split("_")
    youtube_id = parts[0]
    for i in range(1, len(parts)-1):
        youtube_id += "_" + parts[i]
    start_time = parts[-1].split(".")[0]  # Remove .mp4 extension
    return youtube_id, start_time

# Function to download full YouTube video using yt-dlp
def download_video(youtube_id, output_path):
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    
    if os.path.exists(output_path):
        print(f"Video {output_path} already exists, skipping download...")
        return output_path
    
    command = ["yt-dlp", "-f", "mp4", "-o", output_path, url]
    
    try:
        subprocess.run(command, check=True, )
        print(f"Downloaded: {output_path}")
        return output_path
    except subprocess.CalledProcessError:
        print(f"Failed to download {url}")
        return None

# Function to trim video using MoviePy's VideoFileClip (video+audio)
def trim_video_moviepy(input_path, output_path, start_time, duration=10):
    try:
        clip = VideoFileClip(input_path)
        # Create a subclip between start_time and start_time + duration
        end_time = min(clip.duration, float(start_time) + duration)
        subclip = clip.subclipped(float(start_time), end_time)
        subclip.write_videofile(output_path, codec="libx264", logger=None)
        clip.close()
        subclip.close()
        print(f"Trimmed video saved: {output_path}")
        return True
    except Exception as e:
        print(f"Failed to trim video {input_path}: {e}")
        return False

# Load the annotation file
file_path = "/home/project/ImSound/Cycle_Sound/top1_boxes_top10_moments.json"
with open(file_path, "r") as f:
    annotations = json.load(f)[1]  # Extract the dictionary from the list

# Directories
output_dir = "/home/project/dataset/VGG_Sound_extracted/test_data"  # Single directory for frames and audio
os.makedirs(output_dir, exist_ok=True)

# Function to get FPS of a video
def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

# Function to extract frames and audio
counter = 28  # Global counter for naming files sequentially

def extract_data(video_path, frame_numbers, fps):
    global counter
    cap = cv2.VideoCapture(video_path)
    clip = VideoFileClip(video_path)
    
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    for frame_num in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            frame_filename = os.path.join(output_dir, f"{counter:02d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved: {frame_filename}")
        else:
            print(f"Could not extract frame {frame_num} from {video_path}")
        
        if fps:
            start_time = frame_num / fps  # Convert frame number to seconds
            if start_time + 1 > clip.duration:
                start_time = max(0, clip.duration - 1)
            end_time = start_time + 1  # Extract 1 second of audio
            audio_clip = clip.subclipped(start_time, end_time).audio
            audio_path = os.path.join(output_dir, f"{counter:02d}.mp3")
            
            try:
                audio_clip.write_audiofile(audio_path, codec="libmp3lame", logger=None)
                print(f"Extracted audio: {audio_path}")
            except Exception as e:
                if os.path.exists(frame_filename):
                    os.remove(frame_filename)
                    counter -= 1
                    print(f"Deleted frame: {frame_filename} due to audio extraction error.")
                return
        counter += 1  # Increment counter for the next file
    
    cap.release()
    clip.close()

duration = 10
for _, row in dataset_df.iloc[5000:5050].iterrows():
    file_name = row["File_Name"]
    youtube_id, start_time_str = parse_filename(file_name)
    try:
        start_time = float(start_time_str)  # Convert to float
    except ValueError:
        print(f"Invalid start time for file {file_name}")
        continue

    # Convert start time to milliseconds
    start_time_ms = int(start_time * 1000)
    end_time_ms = int((start_time + duration) * 1000)

    # Generate the annotated ID in the required format
    annotated_id = f"{youtube_id}_{start_time_ms}_{end_time_ms}"
    full_video_path = os.path.join(video_dir, f"{youtube_id}.mp4")
    downloaded_path = download_video(youtube_id, full_video_path)
    if downloaded_path:
        trimmed_video_path = full_video_path.replace(".mp4", "_trimmed.mp4")
        trimmed_video = trim_video_moviepy(downloaded_path, trimmed_video_path, start_time, duration)
        if trimmed_video:
            fps = get_fps(trimmed_video_path)
            annotation = annotations.get(annotated_id)  # Safely retrieve the annotation
            if annotation:  # Check if it's found
                extract_data(trimmed_video_path, annotation, fps)
            time.sleep(1)  # Added delay to ensure file handles are released
            os.remove(trimmed_video_path)
            print(f"Deleted trimmed video: {trimmed_video}")
        time.sleep(1)  # Added delay to ensure file handles are released
        os.remove(downloaded_path)
        print(f"Deleted original video: {downloaded_path}")
print("Processing complete!")





