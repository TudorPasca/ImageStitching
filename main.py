import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import io
import warnings

DATASET_BASE_DIR = Path('./dataset')

SEGMENT_IDS = [
    '1005081002024129653_5313_150_5333_150',
    '10017090168044687777_6380_000_6400_000',
    '0023947602400723454_1120_000_1140_000',
    '10061305430875486848_1080_000_1100_000',
    '10072140764565668044_4060_000_4080_000'
]

def load_waymo_data_from_structure(base_dir: Path, segment_id: str):
    if not base_dir.is_dir():
        print(f"Error: Base directory not found: {base_dir.resolve()}")
        return None
        
    if not segment_id:
         print("Error: TARGET_SEGMENT_ID is not set. Please update the variable in the script.")
         return None

    data = {'image': None, 'calibration': None, 'pose': None}

    component_map = {
        'image': 'camera_image',
        'calibration': 'camera_calibration',
        'pose': 'vehicle_pose' 
    }

    for data_key, sub_dir_name in component_map.items():
        component_dir = base_dir / sub_dir_name
        file_path = component_dir / f"{segment_id}.parquet"
        if file_path.is_file():
            try:
                data[data_key] = pd.read_parquet(file_path, engine='pyarrow')
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                data[data_key] = None
        else:
            warnings.warn(f"Warning: File not found: {file_path}", UserWarning)

    if data['image'] is None:
         print(f"Error: Extraction failed for {segment_id}")
         return None
    
    return data

class WaymoFrameParser:
    def __init__(self, waymo_data):
        if not waymo_data or waymo_data.get('image') is None:
            raise ValueError("Invalid Waymo data provided")
        
        self.images_df = waymo_data['image']
        self.images_df = self.images_df.set_index(['key.frame_timestamp_micros', 'key.camera_name'], drop=False)
        self.images_df.sort_index(inplace=True)

        if 'key.frame_timestamp_micros' in self.images_df.index.names:
            self.timestamps = sorted(self.images_df.index.get_level_values('key.frame_timestamp_micros').unique())
        else:
            raise ValueError("Cannot find 'key.frame_timestamp_micros' in image data index")


    def get_timestamps(self):
        """Returns a sorted list of unique frame timestamps."""
        return self.timestamps

    def get_frame_images(self, timestamp_micros):
        """
        Retrieves all camera image data for a specific frame timestamp.

        Args:
            timestamp_micros: The frame timestamp in microseconds.

        Returns:
            A dictionary where keys are camera names and values are pandas Series containing the corresponding row data from the camera_image table.
            Returns an empty dictionary if the timestamp is not found or no images exist for it.
            Returns None in case of critical errors during data retrieval.
        """
        if timestamp_micros not in self.timestamps:
             warnings.warn(f"Timestamp {timestamp_micros} not found.", UserWarning)
             return {}
        
        frame_cameras = {}

        try:
            frame_images_slice = self.images_df.loc[pd.IndexSlice[timestamp_micros,:], :]

            if isinstance(frame_images_slice, pd.DataFrame) and not frame_images_slice.empty:
                 if 'key.camera_name' in frame_images_slice.columns and frame_images_slice.index.name != 'key.camera_name':
                      frame_images_slice.set_index('key.camera_name', inplace=True, drop=False)

            if frame_images_slice.empty:
                 warnings.warn(f"No image data found for timestamp {timestamp_micros} after slicing.", UserWarning)
                 return {}
            else:
                 for camera_name, image_data_series in frame_images_slice.iterrows():
                    frame_cameras[camera_name] = image_data_series
        except Exception as e:
             warnings.warn(f"Error while retrieving camera data for timestamp {timestamp_micros}: {e}", UserWarning)
             return None 
        
        return frame_cameras
    
    def decode_image(self, image_bytes):
        """
        Decodes image bytes into a PIL Image object.
        """
        if not image_bytes or not isinstance(image_bytes, bytes):
             warnings.warn("Invalid image bytes provided for decoding.", UserWarning)
             return None
        try:
            if len(image_bytes) < 100:
                 is_likely_image = False
                 if image_bytes.startswith(b'\xff\xd8'): is_likely_image = True
                 elif image_bytes.startswith(b'\x89PNG\r\n\x1a\n'): is_likely_image = True
                 if not is_likely_image:
                      warnings.warn(f"Image bytes length ({len(image_bytes)}) seems too small and lacks common headers. Skipping decode.", UserWarning)
                      return None
            return Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            warnings.warn(f"Error decoding image: {e}", UserWarning)
            return None


segment_id = '1005081002024129653_5313_150_5333_150'
data = load_waymo_data_from_structure(DATASET_BASE_DIR, segment_id)
frameParser = WaymoFrameParser(data)
timestamps = frameParser.get_timestamps()
print(f"Number of timestamps: {len(timestamps)}")
for timestamp in timestamps[:1]:
    frame_images = frameParser.get_frame_images(timestamp)
    if frame_images:
        for camera_name, image_data in frame_images.items():
            image_bytes = image_data['[CameraImageComponent].image']
            decoded_image = frameParser.decode_image(image_bytes)
            if image_data is not None:
                print(f"Image from {camera_name} at timestamp {timestamp}: {image_data}")
                decoded_image.show()
            else:
                print(f"Failed to decode image from {camera_name} at timestamp {timestamp}")
    else:
        print(f"No images found for timestamp {timestamp}")