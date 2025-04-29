import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import io
import warnings

DATASET_BASE_DIR = Path('./dataset')

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

segment_id = '1005081002024129653_5313_150_5333_150'
data = load_waymo_data_from_structure(DATASET_BASE_DIR, segment_id)
print(data)