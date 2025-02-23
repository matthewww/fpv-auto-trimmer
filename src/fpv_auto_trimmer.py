import os
import time
import argparse

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
        cap, prev_gray, total_frames, start_time, motion_threshold
    )

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
    gray = convert_frame_to_grayscale(current_frame, prev_gray.shape)
    flow = calculate_optical_flow(prev_gray, gray)
    return calculate_center_motion_magnitude(flow), gray


def convert_frame_to_grayscale(frame, target_shape):
    small_frame = cv2.resize(frame, (target_shape[1], target_shape[0]))
    return cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)


def calculate_optical_flow(prev_gray, gray):
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=ITERATIONS,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )


def calculate_center_motion_magnitude(flow):
    h, w = flow.shape[:2]
    center_region = flow[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    return np.mean(np.abs(center_region))


def print_progress(frame_num, total_frames, start_time):
    elapsed_time = time.time() - start_time
    progress = (frame_num / total_frames) * 100
    remaining_time = (elapsed_time / frame_num) * (total_frames - frame_num)

    print(
        f"\rAnalyzing frame {frame_num}/{total_frames} ({progress:.1f}%) - "
        f"Elapsed: {elapsed_time:.1f}s, Remaining: {remaining_time:.1f}s",
        end="",
        flush=True,
    )


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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(start_frame, end_frame if end_frame else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()


def process_videos(input_path=None, output_folder=None):
    args = parse_command_line_args(input_path, output_folder)
    input_path, output_folder = args.input_path, args.output
    
    files_to_process = get_video_files_to_process(input_path)
    input_folder = os.path.dirname(input_path) if os.path.isfile(input_path) else input_path
    
    output_folder = setup_output_folder(input_folder, output_folder)
    process_video_files(files_to_process, input_folder, output_folder)


def parse_command_line_args(input_path, output_folder):
    if input_path is not None:
        return type('Args', (), {'input_path': input_path, 'output': output_folder})
    
    parser = argparse.ArgumentParser(
        description="Trim FPV drone footage based on motion detection."
    )
    parser.add_argument(
        "input_path", help="Path to folder containing input videos or path to single video file"
    )
    parser.add_argument("--output", "-o", help="Optional custom output folder path")
    return parser.parse_args()


def get_video_files_to_process(input_path):
    if os.path.isfile(input_path):
        return [os.path.basename(input_path)]
    
    video_extensions = (".mp4", ".MP4", ".avi", ".mov", ".MOV")
    return [f for f in os.listdir(input_path) if f.endswith(video_extensions)]


def setup_output_folder(input_folder, output_folder):
    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(input_folder), "output")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    return output_folder


def process_video_files(files_to_process, input_folder, output_folder):
    for video_file in files_to_process:
        video_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, "trimmed_" + video_file)
        process_single_video(video_file, video_path, output_path)


def process_single_video(video_file, video_path, output_path):
    takeoff, landing = detect_motion(video_path)
    
    if takeoff is not None:
        trim_video(video_path, output_path, takeoff, landing if landing else None)
        print_processing_result(video_file, takeoff, landing)
    else:
        print(f"Takeoff not detected in {video_file}.")


def print_processing_result(video_file, takeoff, landing):
    print(
        f"Processed: {video_file} -> "
        f"Trimmed from frame {takeoff} to {landing if landing else 'end'}."
    )


if __name__ == "__main__":
    process_videos()
