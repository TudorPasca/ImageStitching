import time
from typing import List, Optional, Tuple
from matplotlib import pyplot as plt
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

    def get_timestamp_data(self, timestamp_micros):
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
        
    def get_timestamp_images(self, timestamp_micros):
        """
        Retrieves all camera images for a specific frame timestamp.

        Args:
            timestamp_micros: The frame timestamp in microseconds.

        Returns:
            A dictionary where keys are camera names and values are the decoded images (PIL Image objects).
            Returns None if the timestamp is not found or no images exist for it.
        """
        if timestamp_micros not in self.timestamps:
            warnings.warn(f"Timestamp {timestamp_micros} not found.", UserWarning)
            return None
        timestamp_data = self.get_timestamp_data(timestamp_micros)
        if timestamp_data is None or timestamp_data is {}:
            warnings.warn(f"Failed to retrieve data for timestamp {timestamp_micros}.", UserWarning)
            return None
        timestamp_images = {}
        for camera_name, image_data in timestamp_data.items():
            image_bytes = image_data['[CameraImageComponent].image']
            decoded_image = self.decode_image(image_bytes)
            if decoded_image is not None:
                timestamp_images[camera_name] = decoded_image
            else:
                warnings.warn(f"Failed to decode image for camera {camera_name} at timestamp {timestamp_micros}.", UserWarning)
        return timestamp_images

class FASTPCornerDetection:
    def __init__(self, t=20, n=9):
        """
        Initializes the FAST detector parameters.

        Args:
            t (int): Intensity difference threshold.
            n (int): Required number of contiguous pixels in the circle arc (9 or 12).
        """
        self._threshold = t
        self._n = n
        
        self._circle_offsets = np.array([
            [ 0,  3], [ 1,  3], [ 2,  2], [ 3,  1], [ 3,  0], [ 3, -1],
            [ 2, -2], [ 1, -3], [ 0, -3], [-1, -3], [-2, -2], [-3, -1],
            [-3,  0], [-3,  1], [-2,  2], [-1,  3]
        ], dtype=np.int8)
        
        self._n8_offsets = np.array([
            [ 0,  1], [ 1,  1], [ 1,  0], [ 1, -1], [ 0, -1], [-1, -1],
            [-1,  0], [-1,  1]
        ], dtype=np.int8)
        
        self._quick_test_offsets = self._circle_offsets[[0, 4, 8, 12]]
        
        self._quick_test_reject_limit = 3
        self._radius = 3

    def detect_corners(self, image):
        """
        Detects FAST corners in the input image.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            np.ndarray: A boolean NumPy array of the same shape as the image,
                        where True indicates a detected corner.
        """
        if image is None:
            raise ValueError("Input image cannot be None.")
            
        grayscale_image = image.convert("L")
        image_array = np.array(grayscale_image, dtype=np.int16) 
        height, width = image_array.shape
        corners = np.zeros((height, width), dtype=bool)
        corner_score = np.zeros((height, width), dtype=np.int32)
        margin = self._radius

        for i in range(margin, height - margin):
            for j in range(margin, width - margin):
                is_corner_flag, score = self._is_corner(image_array, i, j)
                if is_corner_flag:
                    corners[i, j] = True
                    corner_score[i, j] = score

        final_corners = self._non_maximal_suppression(corners, corner_score)
        return final_corners

    def _is_inside(self, shape, lin, col):
        """Checks if the given (row, col) coordinates are within the image bounds."""
        height, width = shape
        return 0 <= col < width and 0 <= lin < height

    def _non_maximal_suppression(self, corners, corner_score):
        """
        Applies non-maximal suppression to the detected corners.
        Score is computed as the sum of the abs value of the intensity differences.

        Args:
            corners (np.ndarray): Boolean mask of provisionally detected corners.
            corner_score (np.ndarray): Array containing the score for each pixel.

        Returns:
            np.ndarray: Boolean mask of corners after suppression.
        """
        height, width = corners.shape
        suppressed_corners = corners.copy() 
        corner_indices = np.argwhere(corners)
        for i, j in corner_indices:
            for offset in self._n8_offsets:
                ni, nj = i + offset[0], j + offset[1]
                if self._is_inside(corners.shape, ni, nj) and \
                   corners[ni, nj] and \
                   corner_score[ni, nj] > corner_score[i, j]:
                    suppressed_corners[i, j] = False
                    break
        return suppressed_corners

    def _get_corner_score(self, image_array, lin, col, circle_intensities):
        """
        Calculates the corner score (sum of absolute differences).

        Args:
            image_array (np.ndarray): The grayscale image array.
            lin (int): Line index of the center pixel.
            col (int): Column index of the center pixel.
            circle_intensities (np.ndarray): Pre-fetched intensities of the 16 circle pixels.

        Returns:
            int: The calculated corner score.
        """
        center_intensity = image_array[lin, col]
        corner_score = np.sum(np.abs(circle_intensities - center_intensity))
        return corner_score # Return the calculated score

    def _is_corner(self, image_array, lin, col):
        """
        Checks if a pixel is a corner using the FAST algorithm.

        Args:
            image_array (np.ndarray): The grayscale image array.
            lin (int): Row index of the candidate pixel.
            col (int): Column index of the candidate pixel.

        Returns:
            tuple[bool, int]: (True, score) if it's a corner, (False, 0) otherwise.
                                Score is calculated only if it's a corner.
        """
        if self._quick_test_rejects(image_array, lin, col):
            return False, 0

        center_intensity = image_array[lin, col]
        pixel_states = np.zeros(16, dtype=np.int8)
        circle_intensities = np.zeros(16, dtype=np.int16) 
        for i in range(self._circle_offsets.shape[0]):
            lin_offset, col_offset = self._circle_offsets[i]
            pixel_intensity = image_array[lin + lin_offset, col + col_offset]
            circle_intensities[i] = pixel_intensity
            if pixel_intensity > center_intensity + self._threshold:
                pixel_states[i] = 1
            elif pixel_intensity < center_intensity - self._threshold:
                pixel_states[i] = -1

        pixel_states_wrapped = np.concatenate((pixel_states, pixel_states))
        max_contiguous_brighter = 0
        max_contiguous_darker = 0
        current_brighter = 0
        current_darker = 0
        found_arc = False
        for state in pixel_states_wrapped:
            if state == 1:
                current_brighter += 1
                max_contiguous_darker = max(max_contiguous_darker, current_darker)
                current_darker = 0
            elif state == -1:
                current_darker += 1
                max_contiguous_brighter = max(max_contiguous_brighter, current_brighter)
                current_brighter = 0
            else:
                current_brighter = 0
                current_darker = 0
            if current_brighter >= self._n or current_darker >= self._n:
                found_arc = True
                break

        if found_arc:
            score = self._get_corner_score(image_array, lin, col, circle_intensities)
            return True, score
        else:
            return False, 0

    def _quick_test_rejects(self, image_array, lin, col):
        """
        Test to quickly check for non-corners.

        Args:
            image_array (np.ndarray): The grayscale image array.
            lin (int): Row index of the candidate pixel.
            col (int): Column index of the candidate pixel.

        Returns:
            bool: True if the pixel should be REJECTED (quick test failed), 
                  False otherwise (pixel might still be a corner).
        """
        center_intensity = image_array[lin, col]
        similar_count = 0
        for offset in self._quick_test_offsets:
            px_intensity = image_array[lin + offset[0], col + offset[1]]
            if abs(px_intensity - center_intensity) <= self._threshold:
                similar_count += 1
                
        return similar_count >= self._quick_test_reject_limit

class BRIEFDescriptor:
    def __init__(self, descriptor_size=256, patch_size=31, gaussian_sigma=None):
        """
        Initializes the BRIEF descriptor extractor.

        Args:
            descriptor_size (int): The desired length of the descriptor in bits (e.g., 128, 256, 512).
            patch_size (int): The size of the square patch around the keypoint (must be odd).
            gaussian_sigma (float | None): Sigma for Gaussian blur applied to the patch
                                          before sampling. If None, no smoothing is applied.
                                          Requires SciPy installed if not None.
        """
        if patch_size % 2 == 0:
            raise ValueError("patch_size must be odd.")
            
        self.descriptor_size = descriptor_size
        self.patch_size = patch_size
        self.gaussian_sigma = gaussian_sigma
        self._half_patch = patch_size // 2
        
        self._sampling_pairs = self._generate_sampling_pairs()

        if self.gaussian_sigma is not None:
            try:
                from scipy.ndimage import gaussian_filter
                self._gaussian_filter = gaussian_filter
            except ImportWarning:
                warnings.warn("SciPy is not installed. Gaussian smoothing will not be applied.", UserWarning)    
                self._gaussian_filter = None
                self.gaussian_sigma = None
        else:
            self._gaussian_filter = None

    def _generate_sampling_pairs(self):
        """
        Generates the random sampling pairs relative to the patch center.
        Uses a Gaussian distribution, clamped to patch boundaries.
        """
        pairs = np.zeros((self.descriptor_size, 2, 2), dtype=np.int16)
        std_deviation = self.patch_size / 5.0 
        max_offset = self._half_patch
        rng = np.random.default_rng()

        for i in range(self.descriptor_size):
            coords = rng.normal(loc=0.0, scale=std_deviation, size=4)
            coords = np.clip(coords, -max_offset, max_offset)
            pairs[i, 0, 0] = int(coords[0]) 
            pairs[i, 0, 1] = int(coords[1])
            pairs[i, 1, 0] = int(coords[2])
            pairs[i, 1, 1] = int(coords[3])
            #Modify second coordinate if random generation gives an identical pair
            if pairs[i, 0, 0] == pairs[i, 1, 0] and pairs[i, 0, 1] == pairs[i, 1, 1]:
                 if pairs[i, 1, 1] < max_offset:
                     pairs[i, 1, 1] += 1
                 else:
                      pairs[i, 1, 1] -= 1
                      
        return pairs
    
    def compute(self, image_array, keypoints):
        """
        Computes BRIEF descriptors for the given keypoints.

        Args:
            image_array (np.ndarray): The grayscale input image (NumPy array, dtype typically uint8 or int16).
            keypoints (list[tuple[int, int]]): A list of (row, col) coordinates
                                              for which to compute descriptors.

        Returns:
            tuple[list[tuple[int, int]], list[np.ndarray]]:
                - valid_keypoints: A list of (row, col) for which descriptors were successfully computed.
                - descriptors: A list of NumPy arrays (uint8, length=descriptor_size),
                               the computed descriptors corresponding to valid_keypoints.
        """
        if image_array.ndim != 2:
            raise ValueError("Input image must be grayscale (2D array).")
            
        height, width = image_array.shape
        valid_keypoints = []
        descriptors = []
        image_array_int = image_array.astype(np.int16) 

        for lin, col in keypoints:
            if not (self._half_patch <= lin < height - self._half_patch and \
                    self._half_patch <= col < width - self._half_patch):
                continue
            
            lin_start, lin_end = lin - self._half_patch, lin + self._half_patch + 1
            col_start, col_end = col - self._half_patch, col + self._half_patch + 1
            patch = image_array_int[lin_start:lin_end, col_start:col_end]
            if self._gaussian_filter is not None and self.gaussian_sigma is not None:
                 current_patch = self._gaussian_filter(patch, sigma=self.gaussian_sigma).astype(np.int16)
            else:
                 current_patch = patch
            descriptor = np.zeros(self.descriptor_size, dtype=np.uint8)
            center_lin, center_col = self._half_patch, self._half_patch

            for i in range(self.descriptor_size):
                l1, c1 = self._sampling_pairs[i, 0, :]
                l2, c2 = self._sampling_pairs[i, 1, :]
                pl1, pc1 = center_lin + l1, center_col + c1
                pl2, pc2 = center_lin + l2, center_col + c2
                if not (0 <= pl1 < patch.shape[0] and 0 <= pc1 < patch.shape[1] and \
                        0 <= pl2 < patch.shape[0] and 0 <= pc2 < patch.shape[1]):
                    continue
                intensity_p = current_patch[pl1, pc1]
                intensity_q = current_patch[pl2, pc2]
                if intensity_p < intensity_q:
                    descriptor[i] = 1
            
            valid_keypoints.append((lin, col))
            descriptors.append(descriptor)

        return valid_keypoints, descriptors

class FeatureMatcher:
    """
    Matches binary feature descriptors using Hamming distance and Lowe's ratio test.
    """
    def __init__(self, ratio_threshold: float = 0.75):
        """
        Initializes the feature matcher.

        Args:
            ratio_threshold (float): The threshold for Lowe's ratio test. 
                                     A match is kept if distance(best) < ratio * distance(second_best).
        """
        self.ratio_threshold = ratio_threshold

    def match(self, 
              descriptors1: np.ndarray, 
              descriptors2: np.ndarray
             ) -> List[Tuple[int, int, int]]:
        """
        Finds good matches between two sets of descriptors using ratio test.

        Args:
            descriptors1 (np.ndarray): N1 x D array of descriptors (dtype=uint8).
            descriptors2 (np.ndarray): N2 x D array of descriptors (dtype=uint8).

        Returns:
            List[Tuple[int, int, int]]: A list of good matches, where each tuple is
                                        (index_in_descriptors1, index_in_descriptors2, hamming_distance).
        """
        matches: List[Tuple[int, int, int]] = []
        num_descriptors1 = descriptors1.shape[0]
        num_descriptors2 = descriptors2.shape[0]
        if num_descriptors1 == 0 or num_descriptors2 < 2:
            return matches 
        for index1 in range(num_descriptors1):
            d1 = descriptors1[index1]
            best_match, second_match, best_dist, second_dist = self._get_best_2_matches(d1, descriptors2)
            if best_match is not None and second_match is not None:
                if best_dist < self.ratio_threshold * second_dist:
                    matches.append((index1, best_match, best_dist))
        return matches
    
    def _get_best_2_matches(self, 
                           target_descriptor: np.ndarray, 
                           descriptors: np.ndarray
                          ) -> Tuple[Optional[int], Optional[int], int, int]:
        """
        Finds the indices and distances of the best and second-best matches 
        for a target descriptor within a set of descriptors using Hamming distance.

        Args:
            target_descriptor (np.ndarray): The single descriptor (1D array, uint8) to match.
            descriptors (np.ndarray): The set of descriptors (2D array, uint8) to search within.

        Returns:
            Tuple[Optional[int], Optional[int], int, int]: 
                (best_match_index, second_match_index, best_distance, second_distance).
                Indices can be None if fewer than two descriptors were provided.
                Distances are initialized to infinity.
        """
        best_match_index: Optional[int] = None
        second_match_index: Optional[int] = None
        best_distance: int = np.iinfo(np.int32).max 
        second_best_distance: int = np.iinfo(np.int32).max
        for i, descriptor in enumerate(descriptors):
            distance = self.hamming_distance(target_descriptor, descriptor)
            if distance < best_distance:
                second_best_distance = best_distance
                second_match_index = best_match_index
                best_distance = distance
                best_match_index = i
            elif distance < second_best_distance:
                second_best_distance = distance
                second_match_index = i
        return best_match_index, second_match_index, best_distance, second_best_distance
    
    def hamming_distance(self, 
                         descriptor1: np.ndarray, 
                         descriptor2: np.ndarray
                        ) -> int:
        """
        Computes the Hamming distance (number of differing bits) between two 
        binary descriptors represented as NumPy arrays of uint8.

        Args:
            descriptor1 (np.ndarray): First descriptor (1D array, uint8).
            descriptor2 (np.ndarray): Second descriptor (1D array, uint8).

        Returns:
            int: The Hamming distance.
        """
        if descriptor1.shape != descriptor2.shape:
            raise ValueError("Descriptors must have the same shape.")
        xor_result = np.bitwise_xor(descriptor1, descriptor2)
        distance = np.sum(np.unpackbits(xor_result))
        return int(distance)

# segment_id = SEGMENT_IDS[0]
# data = load_waymo_data_from_structure(DATASET_BASE_DIR, segment_id)
# frameParser = WaymoFrameParser(data)
# timestamps = frameParser.get_timestamps()
# print(f"Number of timestamps: {len(timestamps)}")
# for timestamp in timestamps[:1]:
#     frame_images = frameParser.get_timestamp_images(timestamp)
#     if frame_images:
#         for camera_name, image in frame_images.items():
#             print(f"Camera: {camera_name}, Image size: {image.size}")
#     else:
#         print(f"No images found for timestamp {timestamp}")