import os
from pathlib import Path
import subprocess
import re

def get_first_number(filename: str) -> int:
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        raise ValueError("No number found in the filename.")


def create_ffmpeg_video_from_images(image_folder, output_video="stitched_panorama.mp4", fps=16, cleanup=True):
    image_folder = Path(image_folder)
    image_paths = sorted(
    image_folder.glob("panorama_*.png"),
    key=lambda p: (get_first_number(p.name),len(p.name))
    )  
    if not image_paths:
        print("No panorama images found in folder.")
        return

    list_file = image_folder / "images.txt"

    with open(list_file, "w") as f:
        for path in image_paths:
            f.write(f"file '{path.name}'\n")

    print(f"Created {list_file} with {len(image_paths)} entries.")

    output_path = Path.cwd() / output_video

    command = [
    "ffmpeg",
    "-f", "concat",
    "-safe", "0",
    "-r", str(fps),
    "-i", str(list_file.name),
    "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    str(output_path)
]

    print(f"Running FFmpeg from folder: {image_folder}")
    result = subprocess.run(command, cwd=str(image_folder))

    if cleanup and list_file.exists():
        list_file.unlink()
        print(f"Removed temporary file: {list_file}")

    if result.returncode == 0:
        print(f"Video created successfully: {output_path}")
    else:
        print(f"FFmpeg failed with return code {result.returncode}")

if __name__ == "__main__":
    create_ffmpeg_video_from_images("semantic_results", "stitched_panorama_semantic.mp4", fps=12, cleanup=True)
