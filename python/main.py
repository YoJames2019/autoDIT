import os
import subprocess
import numpy as np
import cv2
import math
import fnmatch
from ultralytics import YOLO

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

video_dir = "./test_media/videos/"
clapboard_model = YOLO("./yolo_v8/weights_final/clapboard_best.pt")
clapboard_text_model = YOLO("./yolo_v8/weights_final/clapboard_text_best.pt")

def main():
    # loop through all videos in a certain directory and all subdirectories
    videopaths = find_video_files(video_dir)
    # pass each video to a parser that will get the first 10 seconds (240 frames) of the video (and if tail=True it will get the last 240 frames of the video)
    for videopath in videopaths:
        print(videopath + " - ")
        frames = extract_frames(videopath, 10, 800, 800)
        
        print(videopath + " - Checking for clapboard")
        detected = detect_best_clapboard_info(frames)
        # for clapboard in detected:
        #     print(clapboard['confs'])
        #     print(clapboard['boxes'])
        # if it cannot find a clapboard, find text, and validate the text:
        if detected is None:
            print(videopath + " - No clapboard detected, checking for tailslate")
            # it will pass the video back to the parser with tail=True
            frames = extract_frames(videopath, 10, 800, 800, True)
            detected = detect_best_clapboard_info(frames)
        
        scene, cam, shot, take = detected['results']
        # print(detected['conf'])
        print(detected['results'])
        # print(detected['boxes'])

        print(videopath + " - Found Clapboard Data:")
        print("    Scene: " + scene)
        print("    Camera: " + cam)
        print("    Shot: " + shot)
        print("    Take: " + take)

        cv2.imshow("detected", detected['frame_plotted'])
        cv2.waitKey(0)
        # if it found all this correctly, it will then check
        # if the video file name is the same as predicted:
        #   move the video to its correct directory
        #   rename and move the corresponding audio file to the correct directory
        # if not, move the video file to the "Manual Review Required" folder


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

def extract_frames(video_path, sec, width, height, tail=False):
    # Get the duration of the video
    print(video_path + " - Checking video duration")
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
    print(video_path + " - Extracting frames")
    extract_cmd = ['ffmpeg', '-i', video_path, '-ss', str(start_time), '-t', str(end_time - start_time), '-vf', f'scale=w={width}:h={height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2' + (',vflip' if tail else ''), '-q:v', '2', '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-']
    result = subprocess.run(extract_cmd, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE)
    frames_raw = result.stdout
    
    print(video_path + " - Converting buffer to numpy array")
    # Convert the frame buffer to a numpy array
    frames = np.frombuffer(frames_raw, dtype=np.uint8)
    frames = frames.reshape((-1, height, width, 3))
    
    return frames

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

main()