import os
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


CAMERA_IMAGE_DATASET_PATH = 'dataset/camera_image' 
CAMERA_SEGMENTATION_DATASET_PATH = 'dataset/semantic/camera_segmentation'
OUTPUT_COLORIZED_IMAGES_PATH = 'colorized_segmentation_overlays'

CAMERA_NAMES_MAP = {
    1: 'FRONT',
    2: 'FRONT_LEFT',
    3: 'FRONT_RIGHT',
    4: 'SIDE_LEFT',
    5: 'SIDE_RIGHT',
}

EXPECTED_CAMERA_IDS = [1, 2, 3, 4, 5]


WAYMO_SEMANTIC_COLORS = {
    0: (0, 0, 0),       # Unlabeled / Background (Black)
    1: (128, 64, 128),  # Road (Purple)
    2: (244, 35, 232),  # Sidewalk (Pink)
    3: (70, 70, 70),    # Building (Dark Gray)
    4: (102, 102, 156), # Wall (Slate Blue)
    5: (190, 153, 153), # Fence (Light Brown)
    6: (153, 153, 153), # Pole (Gray)
    7: (250, 170, 30),  # Traffic Light (Orange)
    8: (220, 220, 0),   # Traffic Sign (Yellow)
    9: (107, 142, 35),  # Vegetation (Olive Green)
    10: (152, 251, 152),# Terrain (Light Green)
    11: (70, 130, 180), # Sky (Light Blue)
    12: (220, 20, 60),  # Person (Red)
    13: (255, 0, 0),    # Rider (Bright Red)
    14: (0, 0, 142),    # Car (Dark Blue)
    15: (0, 0, 70),     # Truck (Very Dark Blue)
    16: (0, 60, 100),   # Bus (Dark Cyan)
    17: (0, 80, 100),   # Train (Medium Cyan)
    18: (0, 0, 230),    # Motorcycle (Blue)
    19: (119, 11, 32),  # Bicycle (Dark Red)
    20: (0, 191, 255),  # Water (Deep Sky Blue) - Example, verify Waymo's actual ID
    21: (255, 165, 0),  # Other Vehicle (Orange) - Example, verify Waymo's actual ID
    22: (160, 32, 240), # Ground (Purple) - Example, verify Waymo's actual ID
    # ... add all 28+ classes from Waymo's definition
}

def colorize_mask(panoptic_mask_array, panoptic_label_divisor, semantic_colors):
    """
    Converts a panoptic mask array into a colorized semantic segmentation mask.

    Args:
        panoptic_mask_array (np.ndarray): The 2D numpy array of panoptic labels.
        panoptic_label_divisor (int): The divisor used to separate semantic from instance IDs.
        semantic_colors (dict): A dictionary mapping semantic IDs to RGB tuples.

    Returns:
        np.ndarray: A 3D (H, W, 3) numpy array representing the colorized mask.
    """
    height, width = panoptic_mask_array.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    semantic_ids = panoptic_mask_array // panoptic_label_divisor

    unique_semantic_ids = np.unique(semantic_ids)

    for sem_id in unique_semantic_ids:
        color = semantic_colors.get(sem_id, (0, 0, 0))
        colored_mask[semantic_ids == sem_id] = color

    return colored_mask

def generate_colorized_waymo_frames(
    camera_image_dir,
    segmentation_dir,
    alpha=0.5
):
    """
    A generator function that yields a dictionary for each unique timestamp
    containing the 5 colorized camera images (original image with segmentation mask overlay).

    Args:
        camera_image_dir (str): Path to the directory containing original camera image parquet files.
        segmentation_dir (str): Path to the directory containing camera segmentation parquet files.
        alpha (float, optional): Transparency level for the mask overlay (0.0 to 1.0).
                                 Defaults to 0.5.

    Yields:
        dict: A dictionary with keys:
            'segment_context_name' (str): The name of the segment.
            'frame_timestamp_micros' (int): The timestamp of the frame.
            'camera_images' (dict): A dictionary where keys are camera names (str)
                                    and values are PIL Image objects of the overlaid image.
    """
    
    print("Loading all camera image data for lookup...")
    camera_images_dfs = []
    camera_image_files = [f for f in os.listdir(camera_image_dir) if f.endswith('.parquet')]
    
    if not camera_image_files:
        print(f"No camera image parquet files found in {camera_image_dir}. Cannot generate frames.")
        return

    for file_name in camera_image_files:
        file_path = os.path.join(camera_image_dir, file_name)
        try:
            df = pq.read_table(file_path).to_pandas()
            camera_images_dfs.append(df)
        except Exception as e:
            print(f"Error reading camera image file {file_path}: {e}")

    all_camera_images_df = pd.concat(camera_images_dfs, ignore_index=True)
    print(f"Loaded {len(all_camera_images_df)} camera image entries.")

    print("Processing segmentation data and yielding colorized frames...")
    segmentation_parquet_files = [f for f in os.listdir(segmentation_dir) if f.endswith('.parquet')]
    
    if not segmentation_parquet_files:
        print(f"No segmentation parquet files found in {segmentation_dir}. Cannot generate frames.")
        return

    for seg_file_name in segmentation_parquet_files:
        seg_file_path = os.path.join(segmentation_dir, seg_file_name)
        print(f"Reading segmentation file: {seg_file_path}")

        try:
            seg_table = pq.read_table(seg_file_path)
            seg_df = seg_table.to_pandas()

            mask_bytes_col = '[CameraSegmentationLabelComponent].panoptic_label'
            divisor_col = '[CameraSegmentationLabelComponent].panoptic_label_divisor'
            camera_name_col = 'key.camera_name'
            timestamp_col = 'key.frame_timestamp_micros'
            segment_context_col = 'key.segment_context_name'
            original_image_bytes_col = '[CameraImageComponent].image' 

            if mask_bytes_col not in seg_df.columns or divisor_col not in seg_df.columns:
                print(f"Required segmentation columns not found in {seg_file_name}. Skipping this file.")
                continue
            
            grouped_frames = seg_df.groupby([segment_context_col, timestamp_col])

            for (segment_context, timestamp), frame_group in grouped_frames:
                current_frame_overlays = {
                    'segment_context_name': segment_context,
                    'frame_timestamp_micros': timestamp,
                    'camera_images': {}
                }
                
                all_cameras_present_and_processed = True 
                
                for camera_id in EXPECTED_CAMERA_IDS:
                    seg_row_for_camera = frame_group[frame_group[camera_name_col] == camera_id]
                    
                    if seg_row_for_camera.empty:
                        all_cameras_present_and_processed = False
                        break

                    matching_image_row = all_camera_images_df[
                        (all_camera_images_df[segment_context_col] == segment_context) &
                        (all_camera_images_df[timestamp_col] == timestamp) &
                        (all_camera_images_df[camera_name_col] == camera_id)
                    ]

                    if matching_image_row.empty:
                        all_cameras_present_and_processed = False
                        break
                    
                    if original_image_bytes_col not in matching_image_row.columns:
                        print(f"'{original_image_bytes_col}' not found in camera image dataframe. Please check column name in camera_image parquet files. Aborting processing of this file.")
                        all_cameras_present_and_processed = False
                        break

                    seg_row = seg_row_for_camera.iloc[0]
                    original_image_bytes = matching_image_row.iloc[0][original_image_bytes_col]
                    panoptic_mask_bytes = seg_row[mask_bytes_col]
                    panoptic_label_divisor = seg_row[divisor_col]

                    try:
                        original_image = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
                        original_image_np = np.array(original_image)

                        panoptic_mask = Image.open(io.BytesIO(panoptic_mask_bytes)).convert("I")
                        panoptic_mask_np = np.array(panoptic_mask)

                        if original_image_np.shape[:2] != panoptic_mask_np.shape[:2]:
                            panoptic_mask_np = np.array(panoptic_mask.resize(original_image.size, Image.NEAREST))

                        colored_semantic_mask = colorize_mask(panoptic_mask_np, panoptic_label_divisor, WAYMO_SEMANTIC_COLORS)

                        original_image_float = original_image_np.astype(np.float32) / 255.0
                        colored_mask_float = colored_semantic_mask.astype(np.float32) / 255.0

                        blended_image_float = (1 - alpha) * original_image_float + alpha * colored_mask_float
                        blended_image_uint8 = (blended_image_float * 255).astype(np.uint8)

                        final_image = Image.fromarray(blended_image_uint8)
                        current_frame_overlays['camera_images'][CAMERA_NAMES_MAP.get(camera_id, f'Camera_{camera_id}')] = final_image

                    except Exception as e:
                        print(f"Error processing overlay for segment {segment_context}, timestamp {timestamp}, camera {CAMERA_NAMES_MAP.get(camera_id, camera_id)}: {e}")
                        all_cameras_present_and_processed = False
                        break

                if all_cameras_present_and_processed and len(current_frame_overlays['camera_images']) == len(EXPECTED_CAMERA_IDS):
                    yield current_frame_overlays
                else:
                    print(f"Skipping frame {segment_context}_{timestamp} due to missing data or processing error for one or more cameras.")

        except Exception as e:
            print(f"Error reading segmentation file {seg_file_path}: {e}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_COLORIZED_IMAGES_PATH, exist_ok=True)

    frame_counter = 0
    for frame_data in generate_colorized_waymo_frames(
        CAMERA_IMAGE_DATASET_PATH,
        CAMERA_SEGMENTATION_DATASET_PATH,
        alpha=0.6
    ):
        print(f"\nProcessing frame: Segment={frame_data['segment_context_name']}, Timestamp={frame_data['frame_timestamp_micros']}")
        
        camera_images = frame_data['camera_images']
        
        for camera_name, img_pil in camera_images.items():
            output_filename = os.path.join(
                OUTPUT_COLORIZED_IMAGES_PATH,
                f"{frame_data['segment_context_name']}_{frame_data['frame_timestamp_micros']}_{camera_name}_overlay.jpg"
            )
            print(camera_name)
            img_pil.save(output_filename, quality=90)
            print(f"Saved {output_filename}")
        
        frame_counter += 1
        if frame_counter >= 2:
            print("Processed 2 full frames. Stopping example usage.")
            break

    if frame_counter == 0:
        print("No complete frames (5 cameras with overlays) were generated.")
    else:
        print(f"\nFinished processing. Total {frame_counter} complete frames (sets of 5 camera images) generated and saved.")