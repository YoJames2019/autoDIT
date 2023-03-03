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

def extract_frames(video_path, tail=False, height=None, width=None):
    """
    Extracts frames from a video file using ffmpeg.

    Args:
        video_path (str): Path to the video file.
        tail (bool): If True, extract the last 10 seconds of the video.
                     If False, extract the first 10 seconds of the video.
        height (int): Height of the extracted frames in pixels.
                      If None, the original height of the video will be used.
        width (int): Width of the extracted frames in pixels.
                     If None, the original width of the video will be used.

    Returns:
        List of extracted frames as numpy arrays.
    """
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
        if height is not None and width is not None:
            frame = frame.reshape((height, width, 3))
        else:
            # If height and width are not specified, use the original dimensions of the video
            frame = frame.reshape((get_video_height(video_path), get_video_width(video_path), 3))
        # Add the frame to the list of frames
        frames.append(frame)
    # Return the array of frames
    return frames


def get_video_fps(video_path):
    """
    Get the frames per second of a video using ffprobe.

    Args:
        video_path (str): The path to the video file.

    Returns:
        float: The frames per second of the video.
    """
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=avg_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    fps = float(output.decode('utf-8').strip())
    return fps


def get_video_height_width(video_path):
    """
    Get the height and width of a video using ffprobe.

    Args:
        video_path (str): The path to the video file.

    Returns:
        tuple: A tuple containing the height and width of the video.
    """
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=height,width', '-of', 'csv=s=x:p=0', video_path]
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    height, width = map(int, output.decode('utf-8').strip().split('x'))
    return height, width

def get_video_num_frames(video_path):
    # Run ffprobe to get the total number of frames in the video
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # Parse the output to get the number of frames
    num_frames = int(stdout.decode().strip())
    return num_frames


def extract_frames_ffmpeg(video_path, start_time=0, end_time=None, num_frames=None, height=None, width=None):
    """
    Extracts frames from a video file using FFmpeg.

    Args:
        video_path (str): Path to the video file.
        start_time (float): The start time of the frame extraction in seconds (default is 0).
        end_time (float): The end time of the frame extraction in seconds (default is None).
                          If None, extract all frames from the start_time to the end of the video.
        num_frames (int): The total number of frames to extract (default is None).
                          If None, extract all frames from the start_time to the end_time.
        height (int): The height of the extracted frames (default is None).
                      If None, use the original height of the video.
        width (int): The width of the extracted frames (default is None).
                     If None, use the original width of the video.

    Returns:
        List of extracted frames as numpy arrays.
    """
    # Create FFmpeg command to extract frames from the video
    command = ["ffmpeg",
               "-ss", str(start_time),
               "-i", video_path,
               "-an",
               "-loglevel", "error",
               "-f", "image2pipe",
               "-pix_fmt", "rgb24"]

    if end_time is not None:
        # If end_time is specified, add the duration argument to the command
        duration = end_time - start_time
        command += ["-t", str(duration)]

    # Add the output pipe argument to the command
    command += ["pipe:1"]

    # Execute FFmpeg command as a subprocess and capture the output
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)

    # Read the output of the FFmpeg command and extract the frames
    stdout, _ = proc.communicate()

    # Convert the output to a numpy array of RGB pixel values
    frames = np.frombuffer(stdout, dtype=np.uint8)
    if num_frames is not None:
        # If num_frames is specified, extract only the specified number of frames
        frames = frames[:num_frames * height * width * 3]

    # Reshape the numpy array to a list of individual frames
    frames = frames.reshape(-1, height, width, 3)

    return frames.tolist()

main()