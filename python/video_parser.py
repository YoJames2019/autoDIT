import cv2, subprocess, numpy as np

class VideoParser:

    def __init__(self, sec=10, height=800, width=800):
        self.INPUT_WIDTH = width
        self.INPUT_HEIGHT = height
        self.SECONDS_TO_SCAN = sec

    def extract_frames(video_path, width, height, tail=False):
        # Get the duration of the video
        duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        duration = float(subprocess.check_output(duration_cmd))

        # Calculate the start and end times
        if tail:
            start_time = max(duration - 10, 0)
            end_time = duration
        else:
            start_time = 0
            end_time = min(10, duration)
        
        # Extract frames from the video
        extract_cmd = ['ffmpeg', '-i', video_path, '-ss', str(start_time), '-t', str(end_time - start_time), '-vf', f'scale=w={width}:h={height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2', '-q:v', '2', '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-']
        frames_raw = subprocess.check_output(extract_cmd)
        
        # Convert the raw frames to a numpy array
        frames = np.frombuffer(frames_raw, dtype=np.uint8)
        frames = frames.reshape((-1, height, width, 3))
        
        return frames

    def resize_frame_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        new_img = cv2.resize(image, dim, interpolation=inter)

        return new_img