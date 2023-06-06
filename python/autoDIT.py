import subprocess
import re
import os
import math
import fnmatch
import shutil
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from PyQt6.QtCore import pyqtSignal, QThread
class autoDITWorker(QThread):
    progress = pyqtSignal(int)
    image_preview_signal = pyqtSignal(object)
    log_signal = pyqtSignal(str, int)
    progress_bar_signal = pyqtSignal(int)
    progress_bar_range_signal = pyqtSignal(int, int)

    def __init__(self, input_video_dir, input_audio_dir, output_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clapboard_model = YOLO("./weights_final/clapboard_best.pt")
        self.clapboard_text_model = YOLO("./weights_final/clapboard_text_best.pt")
        self.num_conversions = {
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
        # define the input video and input audio from the gui part
        self.video_dir = input_video_dir
        self.audio_dir = input_audio_dir
        self.output_dir = output_dir

        self.log_signal.emit(f"Using GPU: {torch.cuda.is_available()}", 0)


    def run(self):

        self.log_signal.emit(f"Looking for videos in {self.video_dir}", 0)
        # loop through all videos in a certain directory and all subdirectories
        videopaths = self.find_files(self.video_dir, extensions=["*.mp4", "*.mov"])

        self.log_signal.emit(f"Found {len(videopaths)} video files", 0)

        self.log_signal.emit(f"\nLooking for audio files in {self.audio_dir}", 0)
        # find all wav files in the input audio directory
        audio_filepaths = self.find_files(self.audio_dir, extensions=["*.wav"])
        self.log_signal.emit(f"Found {len(audio_filepaths)} audio files", 0)

        self.progress_bar_range_signal.emit(0, len(videopaths)+len(audio_filepaths))
        # pass each video to a parser that will get the first 10 seconds (240 frames) of the video (and if tail=True it will get the last 240 frames of the video)
        for videopath in videopaths:
            self.log_signal.emit(f"\nProcessing {videopath}:", 0)
            filename = os.path.basename(videopath)
            proxy = "_proxy" in filename

            frames = self.extract_frames(videopath, 10, 800, 800)
            
            self.log_signal.emit("Checking for clapboard", 1)
            detected = self.detect_best_clapboard_info(frames)

            # if it cannot find a clapboard, find text, and validate the text:
            if detected is None:
                self.log_signal.emit("No clapboard detected, checking for tailslate", 1)
                # it will pass the video back to the parser with tail=True
                frames = self.extract_frames(videopath, 10, 800, 800, True)
                detected = self.detect_best_clapboard_info(frames)
            
            # if it still couldnt find a clapboard, set all the below variables to none 
            scene, cam, shot, take = detected['clapboard_info'] if detected else [-1, -1, -1, -1]

            self.image_preview_signal.emit(detected['frame'] if detected else frames[-1])

            if take == -1:
                self.log_signal.emit("No clapboard detected", 1)
            else:
                self.log_signal.emit("Found Clapboard Data:", 1)
                self.log_signal.emit(f"Scene: {str(scene)}", 2)
                self.log_signal.emit(f"Camera: {str(cam)}", 2)
                self.log_signal.emit(f"Shot: {str(shot)}", 2)
                self.log_signal.emit(f"Take: {str(take)}", 2)

            # if it found all this correctly, it will then check
            # if the video file name is the same as predicted:
            if detected is not None and (f"{detected['formatted_results']}." in filename.lower() or f"{detected['formatted_results']}_proxy." in filename.lower()):

                self.log_signal.emit("Correct filename", 1)
                # move the video to its correct directory
                self.copy_file_safe(videopath, os.path.join(self.output_dir, f"Scene {scene}/{'Proxy' if proxy else '4K'}"))


            else:
                # if not, move the video file to the "Manual Review Required" directory
                self.log_signal.emit("Incorrect filename", 1)
                
                self.copy_file_safe(videopath, os.path.join(self.output_dir, "Manual Review Required"))
        
            self.progress_bar_signal.emit(1)

        # convert the audio file names (ex. 24B_T1.wav) to the correct format (ex. S024_S002_T001.wav)

        # loop through all the audio files
        for audio_filepath in audio_filepaths:
            
            self.log_signal.emit(f"\nProcessing {audio_filepath}:", 0)
            # get the filename of the audio file
            filename = os.path.basename(audio_filepath)

            # check if the filename is in the correct format
            matched = re.match(r"([\d]+)([A-Z]+)_T([\d]+)", filename)
            
            # if not, move the audio file to the "Manual Review Required" directory and skip to the next audio file
            if not matched or len(matched.groups()) != 3:
                self.log_signal.emit("Incorrect audio format, cannot convert", 1)
                self.copy_file_safe(audio_filepath, os.path.join(self.output_dir, "Manual Review Required"))
                self.progress_bar_signal.emit(1)
                continue
            
            # if it is, rename and move the audio file to the correct directory
            scene, shot, take = matched.groups()

            # convert the scene, shot, and take to the correct format
            cnv_scene = scene.zfill(3)
            cnv_shot = str(ord(shot.lower()) - ord('a') + 1).zfill(3)
            cnv_take = take.zfill(3)

            self.log_signal.emit("Found Scene Data: ", 1)
            self.log_signal.emit(f"Scene: {scene} -> {cnv_scene}", 2)
            self.log_signal.emit(f"Shot: {shot} -> {cnv_shot}", 2)
            self.log_signal.emit(f"Take: {take} -> {cnv_take}", 2)
            new_filename = f"S{cnv_scene}_S{cnv_shot}_T{cnv_take}{os.path.splitext(filename)[1]}"
            
            self.log_signal.emit(f"Correct format, renaming to {new_filename}", 1)
            self.copy_file_safe(audio_filepath, os.path.join(self.output_dir, f"Scene {cnv_scene}/Audio"), new_filename)

            self.progress_bar_signal.emit(1)


    def find_files(self, directory, extensions=['*.mp4', '*.mov']):
        files = []
        if os.path.exists(directory):
            for root, dirnames, filenames in os.walk(directory):
                for extension in extensions:
                    for filename in fnmatch.filter(filenames, extension):
                        files.append(os.path.join(root, filename))

        else:
            self.log_signal.emit(f"The directory '{directory}' does not exist.", 0)
            self.log_signal.emit(f"Current working directory: {os.getcwd()}", 0)
            self.log_signal.emit(f"Absolute path: {os.path.abspath(directory)}", 0)
        return files

    def search_dir(self, start_dir, target_file):
        for dirpath, dirnames, filenames in os.walk(start_dir):
            for filename in filenames:
                if filename == target_file:
                    return os.path.join(dirpath, filename)
        return None

    def extract_frames(self, videopath, sec, width, height, tail=False):
        # Get the duration of the video
        self.log_signal.emit("Checking video duration", 1)
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
        self.log_signal.emit("Extracting frames", 1)
        extract_cmd = ['ffmpeg', '-i', videopath, '-ss', str(start_time), '-t', str(end_time - start_time), '-vf', f'scale=w={width}:h={height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2' + (',vflip,hflip' if tail else ''), '-q:v', '2', '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-']
        result = subprocess.run(extract_cmd, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE)
        frames_raw = result.stdout
        
        self.log_signal.emit("Converting buffer to numpy array", 1)
        # Convert the frame buffer to a numpy array
        frames = np.frombuffer(frames_raw, dtype=np.uint8)
        frames = frames.reshape((-1, height, width, 3))
        
        return frames

    def resize_image(self, img, size=(28,28)):
        # Get the height and width of the image
        h, w = img.shape[:2]

        # Get the number of channels of the image, default to 1 if it's a grayscale image
        c = img.shape[2] if len(img.shape)>2 else 1

        # If the image is already square, just resize it
        if h == w: 
            return cv2.resize(img, size, cv2.INTER_AREA)

        # Find the longer side of the image
        dif = h if h > w else w

        # Decide the interpolation method based on the size of the longer side
        interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

        # Calculate the position to place the original image on the new square image
        x_pos = (dif - w)//2
        y_pos = (dif - h)//2

        # If the image is grayscale
        if len(img.shape) == 2:
            # Create a square image filled with zeros
            mask = np.zeros((dif, dif), dtype=img.dtype)
            # Place the original image in the center of the square image
            mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
        else: # If the image is colored
            # Create a square image filled with zeros with the same number of channels as the original image
            mask = np.zeros((dif, dif, c), dtype=img.dtype)
            # Place the original image in the center of the square image
            mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

        # Resize the square image to the desired size and return it
        return cv2.resize(mask, size, interpolation) # ignore the warning, it works


    def detect_best_clapboard_info(self, frames):
        # for each frame of that video, pass that frame to the clapboardrecognition class which will then:
            raw_results_arr = []
            for i, frame in enumerate(frames):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if i == 0:
                    self.image_preview_signal.emit(frame)

                # check for a clapboard in that image
                clapboard_model_results = self.clapboard_model.predict(frame, conf=0.75, verbose=False)
                clapboard_coords = clapboard_model_results[0]

                # if there is a clapboard
                if len(clapboard_coords.boxes) > 0:

                    # crop the image to JUST the clapboard to eliminate any other text in the image
                    cx, cy, cw, ch = clapboard_coords.boxes[0].xywh.cpu().numpy()[0].astype(int)
                    ncw = math.ceil(cw/2)
                    nch = math.ceil(ch/2)

                    frame = frame[max(0, cy-nch):max(0, cy+nch), max(0, cx-ncw):max(0, cx+ncw)]

                    # resize the image to a fixed size to make it easier to recognize the text
                    frame = self.resize_image(frame, (800, 800))

                    # then pass the cropped image to the function to recognize the text on the clapboard
                    text_results = self.clapboard_text_model.predict(frame, conf=0.80, verbose=False)

                    # then validate/sort the clapboard text against the ninjav format
                    
                    if(len(text_results) > 0):
                        result = text_results[0]
                        matches = [list(b) for b in result.boxes.data.cpu().numpy().astype("int")]
                        confs = [c for c in result.boxes.conf.cpu().numpy()]
                        
                        #sort the matches by x distance from each other
                        matches = sorted(matches, key=lambda x: self.calculate_distance((x[0] + x[2]/2, x[1] + x[3]/2), (0, 0)))

                        #replace the class ids with the actual class names and filter out any detections of text that is not a number
                        matches_filtered = []
                        for sub in matches:
                            class_id = int(sub[5])
                            class_name_text = self.clapboard_text_model.names[class_id] # type: ignore
                            
                            if class_name_text in self.num_conversions:
                                matches_filtered.append(sub[:4] + [self.num_conversions[class_name_text]])

                        group = []
                        groups = []
                        i = 0;
                        while len(matches_filtered) > 0 and i < len(matches_filtered):
                            match = matches_filtered[i]

                            if(len(group) > 0 and abs(match[1] - group[-1][1]) > 5):
                                i += 1;
                                continue;
                            
                            group.append(match)
                            matches_filtered.pop(i)

                            if(len(group) == 3):
                                i = 0;
                                groups.append(group)
                                group = []
                                continue;

                        self.image_preview_signal.emit(result.plot())
                        if(len(confs) > 0 and len(groups) == 4 and len(groups[-1]) == 3):
                            raw_results_arr.append({'frame': frame, 'frame_plotted': result.plot(), 'conf': round(sum(confs)/len(confs), 4), 'boxes': groups, 'clapboard_info': [str(r[0][4])+str(r[1][4])+str(r[2][4]) for r in groups]})

            if(len(raw_results_arr) > 0):
                # TODO: change this so that it counts how many results are the same and then returns the one with the most matches
                clapboard_results_dict = {}
                for res in raw_results_arr:
                    scene, cam, shot, take = res['clapboard_info'] if res else [-1, -1, -1, -1]

                    formatted_info = f"s{scene}_s{shot}_t{take}"


                    if formatted_info not in clapboard_results_dict:
                        clapboard_results_dict[formatted_info] = {'frame': res['frame_plotted'], 'formatted_results': formatted_info, 'clapboard_info': res['clapboard_info'], 'conf': res['conf'], 'count': 1}

                    clapboard_results_dict[formatted_info]['count'] += 1
                
                return max(clapboard_results_dict.values(), key=lambda r: r['count'])
            else:
                return None
            
    def copy_file_safe(self, filepath, dest_dir, filename=None, attempt=1):
        if filename is None:
            filename = os.path.basename(filepath)

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        dest_filepath = os.path.join(dest_dir, filename)
        try:
            shutil.copy2(filepath, dest_filepath)
        except Exception as e:
            if attempt <= 5:
                attempt += 1;
                self.copy_file_safe(filepath, dest_dir, filename, attempt)
        

    def calculate_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)