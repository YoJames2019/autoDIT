import os
import sys
import subprocess
import numpy as np
import cv2
import math
import fnmatch
from ultralytics import YOLO
import shutil

num_conversions = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'zero': 0
}

# check if the correct arguments were input
if len(sys.argv) < 4:
    print(f"Invalid syntax! Correct syntax is: \n{sys.executable} {os.path.abspath(__file__)} \"replace/with/path/to/folder/with/videos/\" \"replace/with/path/to/folder/with/audio/\" \"replace/with/path/to/output/folder/\"")
    exit()

# define the input video, input audio, and output directory variables from the arguments
input_video_dir = sys.argv[1]
input_audio_dir = sys.argv[2]
output_video_dir = sys.argv[3]

# check if both of those paths exist and if they don't, error
for path in sys.argv[1:4]:
    if not os.path.exists(path):
        print(f"\nThe folder \"{path}\" does not exist!")
        exit()

# Instantiate the clapboard and clapboard text models
clapboard_model = YOLO("./yolo_v8/weights_final/clapboard_best.pt")
clapboard_text_model = YOLO("./yolo_v8/weights_final/clapboard_text_best.pt")

def main():
    # loop through all videos in a certain directory and all subdirectories
    videopaths = find_video_files(input_video_dir)
    # pass each video to a parser that will get the first 10 seconds (240 frames) of the video (and if tail=True it will get the last 240 frames of the video)
    for videopath in videopaths:
        print(f"\n{videopath} - ")
        filename = os.path.basename(videopath)
        proxy = "_proxy" in filename

        frames = extract_frames(videopath, 10, 800, 800)
        
        print(f"{videopath} - Checking for clapboard")
        detected = detect_best_clapboard_info(frames)

        # if it cannot find a clapboard, find text, and validate the text:
        if detected is None:
            print(f"{videopath} - No clapboard detected, checking for tailslate")
            # it will pass the video back to the parser with tail=True
            frames = extract_frames(videopath, 10, 800, 800, True)
            detected = detect_best_clapboard_info(frames)
        
        # if it still couldnt find a clapboard, set all the below variables to none 
        scene, cam, shot, take = detected['results'] if detected else [None, None, None, None]

        print(f"{videopath} - Found Clapboard Data:")
        print("    Scene: " + str(scene))
        print("    Camera: " + str(cam))
        print("    Shot: " + str(shot))
        print("    Take: " + str(take))

        # if it found all this correctly, it will then check
        # if the video file name is the same as predicted:
        if f"s{scene}_s{shot}_t{take}." in filename.lower() or f"s{scene}_s{shot}_t{take}_proxy." in filename.lower():
            print(f"{videopath} - Correct filename")

            # move the video to its correct directory
            copy_file_safe(videopath, os.path.join(output_video_dir, f"Scene {scene}/{'Proxy' if proxy else '4K'}"))
            
            # find the corresponding audio file from the input audio directory
            audio_filepath = search_dir(input_audio_dir, f"{int(scene)}{chr(int(shot) + ord('a') - 1).upper()}_T{int(take)}.wav")
            # rename and move the corresponding audio file to the correct directory
            if audio_filepath is not None:
                copy_file_safe(audio_filepath, os.path.join(output_video_dir, f"Scene {scene}/Audio"), f"S{scene}_S{shot}_T{take}{os.path.splitext(audio_filepath)[1]}")

        else:
            # if not, move the video file to the "Manual Review Required" directory
            print(f"{videopath} - Incorrect filename")
            
            copy_file_safe(videopath, os.path.join(output_video_dir, "Manual Review Required"))



def find_video_files(directory, extensions=['*.mp4', '*.mov']):
    video_files = []
    if os.path.exists(directory):
        for root, dirnames, filenames in os.walk(directory):
            for extension in extensions:
                for filename in fnmatch.filter(filenames, extension):
                    video_files.append(os.path.join(root, filename))

    else:
        print(f"The directory '{directory}' does not exist.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Absolute path: {os.path.abspath(directory)}")
    return video_files

def search_dir(start_dir, target_file):
    for dirpath, dirnames, filenames in os.walk(start_dir):
        for filename in filenames:
            if filename == target_file:
                return os.path.join(dirpath, filename)
    return None

def detect_best_clapboard_info(frames):
    # for each frame of that video, pass that frame to the clapboardrecognition class which will then:
        detected = []
        for i, frame in enumerate(frames):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # check for a clapboard in that image
            clapboard_results = clapboard_model.predict(frame, conf=0.75, verbose=False)
            clapboard_coords = clapboard_results[0]

            # if there is a clapboard, it will crop the image to JUST the clapboard
            if len(clapboard_coords.boxes) > 0:
                cx, cy, cw, ch = clapboard_coords.boxes[0].xywh.cpu().numpy()[0].astype(int)
                ncw = math.ceil(cw/2)
                nch = math.ceil(ch/2)

                frame = frame[max(0, cy-nch):max(0, cy+nch), max(0, cx-ncw):max(0, cx+ncw)]

                # then pass the cropped image to the function to recognize the text on the clapboard
                text_results = clapboard_text_model.predict(frame, conf=0.80, verbose=False)

                # then validate/sort the clapboard text against the ninjav format
                
                if(len(text_results) > 0):
                    result = text_results[0]
                    matches = [list(b) for b in result.boxes.data.cpu().numpy().astype("int")]
                    confs = [c for c in result.boxes.conf.cpu().numpy()]
                    
                    #sort the matches by x distance from each other
                    matches_sorted = sorted(matches, key=lambda x: calculate_distance((x[0] + x[2]/2, x[1] + x[3]/2), (0, 0)))
                    #replace the class ids with the actual class names
                    matches_sorted = [sub[:4] + [num_conversions[clapboard_text_model.names[int(sub[5])]]] for sub in matches_sorted]

                    group = []
                    groups = []
                    i = 0;
                    while len(matches_sorted) > 0 and i < len(matches_sorted):
                        match = matches_sorted[i]

                        if(len(group) > 0 and abs(match[1] - group[-1][1]) > 5):
                            i += 1;
                            continue;
                        
                        group.append(match)
                        matches_sorted.pop(i)

                        if(len(group) == 3):
                            i = 0;
                            groups.append(group)
                            group = []
                            continue;

                    if(len(confs) > 0 and len(groups) == 4 and len(groups[-1]) == 3):
                        detected.append({'frame': frame, 'frame_plotted': result.plot(), 'conf': round(sum(confs)/len(confs), 4), 'boxes': groups, 'results': [str(r[0][4])+str(r[1][4])+str(r[2][4]) for r in groups]})

        if(len(detected) > 0):
            return max(detected, key=lambda x: x['conf'])
        else:
            return None

def extract_frames(videopath, sec, width, height, tail=False):
    # Get the duration of the video
    print(f"{videopath} - Checking video duration")
    duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', videopath]
    duration = float(subprocess.check_output(duration_cmd))

    # Calculate the start and end times
    if tail:
        start_time = max(duration - sec, 0)
        end_time = duration
    else:
        start_time = 0
        end_time = min(sec, duration)
    
    # Extract frames from the video
    print(f"{videopath} - Extracting frames")
    extract_cmd = ['ffmpeg', '-i', videopath, '-ss', str(start_time), '-t', str(end_time - start_time), '-vf', f'scale=w={width}:h={height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2' + (',vflip,hflip' if tail else ''), '-q:v', '2', '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-']
    result = subprocess.run(extract_cmd, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE)
    frames_raw = result.stdout
    
    print(f"{videopath} - Converting buffer to numpy array")
    # Convert the frame buffer to a numpy array
    frames = np.frombuffer(frames_raw, dtype=np.uint8)
    frames = frames.reshape((-1, height, width, 3))
    
    return frames

def copy_file_safe(filepath, dest_dir, filename=None, attempt=1):
    if filename is None:
        filename = os.path.basename(filepath)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    dest_filepath = os.path.join(dest_dir, filename)
    try:
        shutil.copy2(filepath, dest_filepath)
    except Exception as e:
        if attempt <= 5:
            att += 1;
            copy_file_safe(filepath, dest_dir, filename, attempt)
        

        

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

main()