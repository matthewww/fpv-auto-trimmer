import os
import time
import cv2
import numpy as np

# Scale down the video by this factor
SCALE_FACTOR = 0.3
# Skip the first seconds of the video
SKIP_SECONDS = 4
# Frames to process per second
FRAMES_PER_SECOND = 30
# Number of iterations for the optical flow calculation
ITERATIONS = 3

def detect_motion(video_path, motion_threshold=8.0):
    start_time = time.time()

    cap = cv2.VideoCapture(video_path)
    fps, total_frames = get_video_metadata(cap)

    print_initial_status(video_path, fps, total_frames)

    if not skip_initial_setup_frames(cap, fps):
        return None, None

    prev_gray = get_initial_frame(cap)
    if prev_gray is None:
        return None, None

    takeoff_frame = process_frames_for_takeoff(
        cap, prev_gray, total_frames,
        start_time,
        motion_threshold)

    print(f"\nAnalysis complete in {time.time() - start_time:.1f} seconds.")
    cap.release()
    return takeoff_frame, None

def get_video_metadata(cap):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, total_frames

def print_initial_status(video_path, fps, total_frames):
    estimated_fps_processing = FRAMES_PER_SECOND
    estimated_total_time = total_frames / estimated_fps_processing
    print(f"Processing {os.path.basename(video_path)}...")
    print(f"Skipping first {SKIP_SECONDS} seconds ({fps * SKIP_SECONDS} frames)...")
    print(f"Estimated processing time: {estimated_total_time:.1f} seconds")

def skip_initial_setup_frames(cap, fps):
    initial_skip_frames = SKIP_SECONDS * fps
    for _ in range(initial_skip_frames):
        ret = cap.read()[0]
        if not ret:
            return False
    return True

def get_initial_frame(cap):
    ret, prev_frame = cap.read()
    if not ret:
        return None
    small_frame = cv2.resize(prev_frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    return cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

def calculate_motion(prev_gray, current_frame):
    h, w = prev_gray.shape[:2]
    small_frame = cv2.resize(current_frame, (w, h))
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        pyr_scale=0.5,
                                        levels=3,
                                        winsize=15,
                                        iterations=ITERATIONS,
                                        poly_n=5,
                                        poly_sigma=1.1,
                                        flags=0)

    h, w = flow.shape[:2]
    center_region = flow[h//4:3*h//4, w//4:3*w//4]
    return np.mean(np.abs(center_region)), gray

def print_progress(frame_num, total_frames, start_time):
    elapsed_time = time.time() - start_time
    progress = (frame_num / total_frames) * 100
    remaining_time = (elapsed_time / frame_num) * (total_frames - frame_num)

    print(f"\rAnalyzing frame {frame_num}/{total_frames} ({progress:.1f}%) - "
          f"Elapsed: {elapsed_time:.1f}s, Remaining: {remaining_time:.1f}s",
          end="", flush=True)

def process_frames_for_takeoff(cap, prev_gray, total_frames, start_time, motion_threshold):
    motion_history = []
    history_size = 5
    takeoff_frame = None

    for frame_num in range(1, total_frames):
        if frame_num % FRAMES_PER_SECOND == 0:
            print_progress(frame_num, total_frames, start_time)

        ret, frame = cap.read()
        if not ret:
            break

        motion_magnitude, gray = calculate_motion(prev_gray, frame)

        motion_history.append(motion_magnitude)
        if len(motion_history) > history_size:
            motion_history.pop(0)

        avg_motion = np.mean(motion_history)

        if takeoff_frame is None and avg_motion > motion_threshold:
            takeoff_frame = max(0, frame_num - history_size)

        prev_gray = gray

    return takeoff_frame

def trim_video(video_path, output_path, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(start_frame, end_frame if end_frame else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

def process_videos(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_extensions = ('.mp4', '.MP4', '.avi', '.mov')
    for video_file in [f for f in os.listdir(folder_path) if f.endswith(video_extensions)]:
        video_path = os.path.join(folder_path, video_file)
        output_path = os.path.join(output_folder, "trimmed_" + video_file)
        takeoff, landing = detect_motion(video_path)

        if takeoff is not None:
            trim_video(video_path, output_path, takeoff, landing if landing else None)
            print(
                f"Processed: {video_file} -> "
                f"Trimmed from frame {takeoff} to {landing if landing else 'end'}."
            )
        else:
            print(f"Takeoff not detected in {video_file}.")

FOLDER_PATH = 'input'
OUTPUT_FOLDER = 'output'
process_videos(FOLDER_PATH, OUTPUT_FOLDER)
