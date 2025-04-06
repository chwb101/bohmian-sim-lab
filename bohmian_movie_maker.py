# === Script 2: make_movie.py ===
import cv2
import os
from glob import glob

image_folder = "exports"
output_file = "bohmian_simulation.mp4"
fps = 20

image_files = sorted(glob(os.path.join(image_folder, "bohm_snapshot_t*.png")))

if not image_files:
    print("No images found.")
    exit()

sample = cv2.imread(image_files[0])
height, width, _ = sample.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

for file in image_files:
    frame = cv2.imread(file)
    video.write(frame)

video.release()
print(f"Movie saved to {output_file}")
# === Script 2 END ===
