import cv2
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm  # For progress bar

# === Configuration Parameters ===
frames_path = 'path to photos.../footage'
output_video = 'path to output folder.../output.mp4'
fps = 30
num_threads = os.cpu_count() or 4

# Ensure output folder exists
os.makedirs(os.path.dirname(output_video), exist_ok=True)

# === Read and sort frame filenames ===
frame_files = sorted([
    f for f in os.listdir(frames_path)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

if not frame_files:
    raise ValueError("❌ No image frames found in the specified directory.")

# === Get size from first frame ===
first_frame = cv2.imread(os.path.join(frames_path, frame_files[0]))
if first_frame is None:
    raise ValueError("❌ First frame could not be read.")
height, width, _ = first_frame.shape

# === Initialize Video Writer ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# === Load and process frames ===
def load_frame(path, filename, target_size):
    full_path = os.path.join(path, filename)
    frame = cv2.imread(full_path)
    if frame is None:
        print(f"⚠️ Skipped unreadable frame: {filename}")
        return None
    if frame.shape[0:2] != (target_size[1], target_size[0]):
        frame = cv2.resize(frame, target_size)
    return frame

# === Start processing ===
print("🚀 Starting video generation...")

start_time = time.time()

load_func = partial(load_frame, frames_path, target_size=(width, height))

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    for frame in tqdm(executor.map(load_func, frame_files), total=len(frame_files), desc="Processing frames", unit="frame"):
        if frame is not None:
            video_writer.write(frame)

video_writer.release()

end_time = time.time()
total_time = end_time - start_time

# === Summary ===
print(f"\n✅ Video successfully saved to: {output_video}")
print(f"🕒 Total time taken: {total_time:.2f} seconds")
print(f"⚙️ Used {num_threads} threads | Total frames: {len(frame_files)} | FPS: {fps}")
