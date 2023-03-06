import os
import re
import subprocess
import numpy as np
import cv2
import math
from ultralytics import YOLO
from IPython.display import display, Image

video_dir = "./test_media/videos/"
clapboard_model = YOLO("./yolo_v8/weights_final/clapboard_best.pt")
clapboard_text_model = YOLO("./yolo_v8/weights_final/clapboard_text_best.pt")

def main():
    # loop through all videos in a certain directory and all subdirectories
    videopaths = scan_dir(video_dir)
    # pass each video to a parser that will get the first 10 seconds (240 frames) of the video (and if tail=True it will get the last 240 frames of the video)
    for videopath in videopaths:
        frames = extract_frames(videopath, 10, 800, 800)
        
        # for each frame of that video, pass that frame to the clapboardrecognition class which will then:
        detected = []
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # check for a clapboard in that image
            clapboard_results = clapboard_model.predict(frame, conf=0.75)
            # if there is a clapboard, it will crop the image to JUST the clapboard

            clapboard_coords = clapboard_results[0]
            if len(clapboard_coords.boxes) > 0:
                cx, cy, cw, ch = clapboard_coords.boxes[0].xywh.cpu().numpy()[0].astype(int)

                ncw = math.ceil(cw/2)
                nch = math.ceil(ch/2)
                frame = frame[max(0, cy-nch):max(0, cy+nch), max(0, cx-ncw):max(0, cx+ncw)]

                # then pass the cropped image to the function to recognize the text on the clapboard
                text_results = clapboard_text_model.predict(frame, conf=0.80)
                # then validate the clapboard text against the ninjav format and against the actual video file name

                # if it cannot find a clapboard, find text, and validate the text:
                #   it will pass the video back to the parser with tail=True
                # if it found all this correctly, it will then move the video to its correct directory
                cv2.imshow("frame", text_results[0].plot())
                cv2.waitKey(41)

# This function scans the directory at the given path for video files
# or subdirectories and prints information about them
def scan_dir(dir_path):
    videos = []
    # Check if the directory exists
    if os.path.exists(dir_path):
        # Loop through each item in the directory
        for file_basename in os.listdir(dir_path):
            # Get the full path of the item
            file_path = os.path.join(dir_path, file_basename)

            # Check if the file name ends in two digits less than 6 (e.g. "video01.mp4")
            regex = re.search("\d{2}$", file_basename)
            # If the item is a subdirectory
            if os.path.isdir(file_path):
                # If the directory matches the criteria
                if regex is not None and int(regex.group()) < 6:
                    # Skip the directory
                    continue
                # Otherwise, print information about the directory
                print("Found Directory:", file_path)
                # Recursively scan the subdirectory
                scan_dir(file_path)
            # If the item is a video file
            elif os.path.splitext(file_basename)[1].lower() in [".mov", ".mp4"] and not file_basename.startswith("."):
                # Print information about the video file
                print("Found Video:", file_path)
                videos.append(file_path)
    else:
        # If the directory does not exist, print information about the current working directory and the absolute path of the given directory
        print(f"The directory '{dir_path}' does not exist.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Absolute path: {os.path.abspath(dir_path)}")
    return videos;

def extract_frames(video_path, sec, width, height, tail=False):
    # Get the duration of the video
    duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    duration = float(subprocess.check_output(duration_cmd))

    # Calculate the start and end times
    if tail:
        start_time = max(duration - sec, 0)
        end_time = duration
    else:
        start_time = 0
        end_time = min(sec, duration)
    
    # Extract frames from the video
    extract_cmd = ['ffmpeg', '-i', video_path, '-ss', str(start_time), '-t', str(end_time - start_time), '-vf', f'scale=w={width}:h={height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2', '-q:v', '2', '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-']
    frames_raw = subprocess.check_output(extract_cmd)
    
    # Convert the raw frames to a numpy array
    frames = np.frombuffer(frames_raw, dtype=np.uint8)
    frames = frames.reshape((-1, height, width, 3))
    
    return frames

main()