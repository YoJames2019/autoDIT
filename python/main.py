import os
import re
import subprocess
import numpy as np
import cv2

video_dir = "./test_media/videos/"

def main():
    # loop through all videos in a certain directory and all subdirectories
    scan_dir(video_dir)
    # pass each video to a parser that will get the first 10 seconds (240 frames) of the video (and if tail=True it will get the last 240 frames of the video)
    # for each frame of that video, pass that frame to the clapboardrecognition class which will then:
    #   check for a clapboard in that image
    #   if there is a clapboard, it will crop the image to JUST the clapboard 
    #   then pass the cropped image to the function to recognize the text on the clapboard
    #   then validate the clapboard text against the ninjav format and against the actual video file name
    #   if it cannot find a clapboard, find text, and validate the text:
    #       it will pass the video back to the parser with tail=True
    #   if it found all this correctly, it will then move the video to its correct directory

# This function scans the directory at the given path for video files
# or subdirectories and prints information about them
def scan_dir(dir_path):
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
    else:
        # If the directory does not exist, print information about the current working directory and the absolute path of the given directory
        print(f"The directory '{dir_path}' does not exist.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Absolute path: {os.path.abspath(dir_path)}")


# This function extracts frames from a video file using the ffmpeg library
def extract_frames(video_path, tail=False):
    # Get the frames per second of the video
    fps = get_video_fps(video_path)
    # Get the total number of frames in the video
    num_frames = get_video_num_frames(video_path)
    if tail:
        # If tail is True, extract the last 10 seconds of the video
        start_frame = max(num_frames - int(fps * 10), 0)
        end_frame = num_frames
    else:
        # Otherwise, extract the first 10 seconds of the video
        start_frame = 0
        end_frame = min(int(fps * 10), num_frames)
    # Define the command to extract the frames using ffmpeg
    cmd = ['ffmpeg', '-i', video_path, '-ss', str(start_frame / fps), '-to', str(end_frame / fps), '-r', str(fps), '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-']
    # Start a subprocess to execute the command
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Capture the stdout and stderr output of the subprocess
    stdout, stderr = process.communicate()
    frames = []
    for i in range(start_frame, end_frame):
        # Calculate the size of each frame in bytes
        frame_size = len(stdout) // num_frames
        # Extract the bytes for the current frame
        frame = stdout[i * frame_size:(i + 1) * frame_size]
        # Convert the bytes to a numpy array
        frame = np.frombuffer(frame, dtype='uint8')
        # Reshape the array to the correct dimensions
        frame = frame.reshape((height, width, 3))
        # Add the frame to the list of frames
        frames.append(frame)
    # Return the array of frames
    return frames

def get_video_fps(video_path):
    # Run ffprobe to get the frames per second of the video
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # Parse the output to get the frames per second
    fps = int(stdout.decode().split('/')[0])
    return fps

def get_video_num_frames(video_path):
    # Run ffprobe to get the total number of frames in the video
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # Parse the output to get the number of frames
    num_frames = int(stdout.decode().strip())
    return num_frames


main()