import time
from typing import Dict, List, Optional, Tuple
import cv2
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
        found = False
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
                found = True
                break

        if found:
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

        descriptors_np = np.array(descriptors)
        return valid_keypoints, descriptors_np

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
    
class RANSACHomography:
    """
    Estimates the Homography matrix between two sets of corresponding points
    using the RANSAC algorithm with the Direct Linear Transform (DLT).
    """
    def __init__(self, num_iterations=1000, inlier_threshold=3.0):
        """
        Initializes the RANSAC Homography estimator.

        Args:
            num_iterations (int): Number of RANSAC iterations to perform.
            inlier_threshold (float): Maximum reprojection error (in pixels)
                                        for a point pair to be considered an inlier.
        """
        self.num_iterations = num_iterations
        self.inlier_threshold = inlier_threshold
        self.min_samples = 4
        
    def _compute_homography_dlt(self, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        """
        Computes the Homography matrix using the Direct Linear Transform (DLT).

        Args:
            pts1 (np.ndarray): Source points (shape: Nx2).
            pts2 (np.ndarray): Destination points (shape: Nx2).

        Returns:
            np.ndarray: The computed Homography matrix (3x3).
        """
        A = []
        for i in range(pts1.shape[0]):
            x1, y1 = pts1[i]
            x2, y2 = pts2[i]
            A.append([-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2])
            A.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])
        A = np.array(A)
        try:
            _, _, Vt = np.linalg.svd(A)
            H = Vt[-1].reshape(3, 3)
            return H
        except np.linalg.LinAlgError:
            raise ValueError("SVD failed to compute Homography matrix.")
        
    def get_homography(self, 
                    keypoints1: List[Tuple[int, int]], 
                    keypoints2: List[Tuple[int, int]], 
                    matches: List[Tuple[int, int, int]]
                    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Finds the best Homography matrix using RANSAC.

        Args:
            keypoints1 (list): List of (row, col) tuples for image 1.
            keypoints2 (list): List of (row, col) tuples for image 2.
            matches (list): List of good matches [(idx1, idx2, distance), ...].

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                - best_H: The estimated 3x3 Homography matrix (or None if failed).
                - best_inlier_mask: A boolean mask indicating which matches were inliers (or None).
        """

        if len(matches) < self.min_samples:
            return None, None
        
        pts1 = np.float32([ [keypoints1[m[0]][1], keypoints1[m[0]][0]] for m in matches ])
        pts2 = np.float32([ [keypoints2[m[1]][1], keypoints2[m[1]][0]] for m in matches ])
        num_matches = len(matches)
        best_H: Optional[np.ndarray] = None
        max_inliers: int = -1
        best_inlier_mask: Optional[np.ndarray] = None
        rng = np.random.default_rng()

        for k in range(self.num_iterations):
            if num_matches >= self.min_samples:
                 sample_indices = rng.choice(num_matches, self.min_samples, replace=False)
            else:
                 continue
            sample_pts1 = pts1[sample_indices]
            sample_pts2 = pts2[sample_indices]
            H_candidate = self._compute_homography_dlt(sample_pts1, sample_pts2)
            if H_candidate is None:
                continue
            try:
                pts1_h = np.hstack((pts1, np.ones((num_matches, 1))))
                pts1_transformed_h = (H_candidate @ pts1_h.T).T
                
                z_coords = pts1_transformed_h[:, 2]

                valid_z = np.abs(z_coords) > 1e-8
                
                pts1_transformed = np.zeros_like(pts1)
                
                pts1_transformed[valid_z] = pts1_transformed_h[valid_z, :2] / z_coords[valid_z, np.newaxis]
                
                diff = pts1_transformed - pts2
                squared_distances = np.sum(diff**2, axis=1)

                inlier_mask = (squared_distances < self.inlier_threshold**2) & valid_z
                num_inliers = np.sum(inlier_mask)

                if num_inliers > max_inliers:
                    max_inliers = num_inliers
                    best_H = H_candidate
                    best_inlier_mask = inlier_mask
            except Exception:
                 continue
             
        return best_H, best_inlier_mask
    
class ImageTransformer:
    """ Transforms an image using a the Homography matrix. """
    
    def transform(self, 
                         image_src: np.ndarray, 
                         H: np.ndarray, 
                         output_shape: Tuple[int, int], 
                         interpolation_flag: int = cv2.INTER_LINEAR, 
                         border_mode: int = cv2.BORDER_CONSTANT, 
                         border_value: int = 0):
        """
        Applies transform using cv2.warpPerspective.

        Args:
            image_src (np.ndarray): The source image (H x W or H x W x C, uint8).
                                    Must be a NumPy array.
            H (np.ndarray): The 3x3 Homography matrix mapping points 
                            FROM source TO output canvas coordinates.
            output_shape (Tuple[int, int]): Desired (height, width) of the 
                                            warped output image.
            interpolation_flag (int): OpenCV interpolation flag, e.g., 
                                      cv2.INTER_LINEAR (default), cv2.INTER_NEAREST, 
                                      cv2.INTER_CUBIC.
            border_mode (int): OpenCV border mode, e.g., cv2.BORDER_CONSTANT (default).
            border_value (int): Value used for border pixels if 
                                                    border_mode is cv2.BORDER_CONSTANT. 
                                                    Use int for grayscale.

        Returns:
            Optional[np.ndarray]: The warped image (NumPy array), or None if OpenCV is unavailable.
        """
    
        if image_src is None or H is None:
            print("Error: Source image or Homography matrix is None.")
            return None
            
        output_width = output_shape[1]
        output_height = output_shape[0]
        dsize = (output_width, output_height)
        try:
            warped_image = cv2.warpPerspective(
                src=image_src,
                M=H,
                dsize=dsize,
                flags=interpolation_flag,
                borderMode=border_mode,
                borderValue=border_value
            )
            return warped_image
        except Exception as e:
            print(f"Error during cv2.warpPerspective: {e}")
            return np.full(output_shape, border_value, dtype=image_src.dtype)
        
    def calculate_composite_homographies(
        self,
        pairwise_results: Dict[Tuple[int, int], Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
        all_cam_nums: List[int],
        ref_cam_num: int
    ) -> Optional[Dict[int, np.ndarray]]:
        """
        Calculates homographies mapping each camera view TO the reference camera view.

        Args:
            pairwise_results (Dict): Dictionary from RANSAC:
                Keys=(target_cam, source_cam), Values=(H_matrix | None, mask | None).
            all_cam_nums (List): List of all camera numbers (e.g., [1, 2, 3, 4, 5]).
            ref_cam_num (int): The number of the reference camera.

        Returns:
            Optional[Dict[int, np.ndarray]]: A dictionary mapping each camera number
                to its 3x3 composite homography matrix (H_ref <- i).
                Returns None if essential homographies are missing.
        """
        composite_H = {}
        identity_matrix = np.eye(3, dtype=np.float32)

        all_available_H = {}
        for (target, source), (H_ts, _) in pairwise_results.items():
            if H_ts is not None:
                all_available_H[(target, source)] = H_ts
                try:
                    H_st = np.linalg.inv(H_ts)
                    all_available_H[(source, target)] = H_st
                except np.linalg.LinAlgError:
                    warnings.warn(f"Could not invert H mapping {source} to {target}.")
                    
        paths_to_ref = {
            1: [], 
            2: [1], 
            3: [1],
            4: [2, 1], 
            5: [3, 1]
        }
        
        if ref_cam_num not in all_cam_nums: 
            return None
        
        print(f"\nCalculating homographies relative to camera {ref_cam_num}...")
        for cam_num in all_cam_nums:
            if cam_num == ref_cam_num:
                composite_H[cam_num] = identity_matrix
                continue
            path = paths_to_ref.get(cam_num)
            if path is None:
                print(f"  Warning: No path defined for {cam_num} -> {ref_cam_num}. Skipping.")
                continue
            H_comp = identity_matrix
            current_cam = cam_num
            possible = True
            for step_target_cam in path:
                needed_pair = (step_target_cam, current_cam) 
                
                if needed_pair in all_available_H:
                    H_step = all_available_H[needed_pair]
                    H_comp = H_step @ H_comp 
                    current_cam = step_target_cam
                else:
                    print(f"  Warning: Missing required pair H for {current_cam} -> {step_target_cam}. "
                        f"Cannot compute H for {cam_num}.")
                    possible = False
                    break 
            if possible:
                composite_H[cam_num] = H_comp
        
        if len(composite_H) != len(all_cam_nums):
            print("Warning: Could not compute composite homographies for all cameras.")
        return composite_H
    
    def calculate_canvas_geometry(
        self,
        composite_H: Dict[int, np.ndarray],
        image_shapes: Dict[int, Tuple[int, int]],
        camera_names: List[int]
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Calculates the required canvas size and offset to fit all transformed images.

        Args:
            composite_H (Dict): Maps camera number to the 3x3 homography matrix
                                (H_ref <- i) transforming points from that camera
                                TO the reference camera's coordinate system.
                                Should contain np.eye(3) for the reference camera.
            image_shapes (Dict): Maps camera number to its original (height, width).
            camera_names (List): List of all camera numbers to consider.

        Returns:
            Optional[Tuple[Tuple[int, int], Tuple[int, int]]]: 
                Returns ((canvas_height, canvas_width), (offset_x, offset_y))
                or None if calculation fails (e.g., no valid transformed corners).
                offset_x, offset_y define the top-left corner's position relative
                to the reference frame's origin (0,0).
        """
        all_transformed_corners = []

        for cam_num in camera_names:
            if cam_num not in composite_H or composite_H[cam_num] is None:
                warnings.warn(f"Skipping camera {cam_num} in canvas calculation: Missing composite homography.")
                continue
            if cam_num not in image_shapes:
                warnings.warn(f"Skipping camera {cam_num} in canvas calculation: Missing original image shape.")
                continue
                
            H_ref_i = composite_H[cam_num]
            h, w = image_shapes[cam_num]
            original_corners = np.array([
                [0,   0],
                [w-1, 0],
                [w-1, h-1],
                [0,   h-1] 
            ], dtype=np.float32)
            
            corners_orig_h = np.hstack((original_corners, np.ones((4, 1))))
            transformed_corners_h = (H_ref_i @ corners_orig_h.T).T
            scale = transformed_corners_h[:, 2]
            valid_scale_mask = np.abs(scale) > 1e-8
            
            transformed_corners = np.zeros((4, 2), dtype=np.float32)
            transformed_corners[valid_scale_mask] = (
                transformed_corners_h[valid_scale_mask, :2] / scale[valid_scale_mask, np.newaxis]
            )
            
            all_transformed_corners.append(transformed_corners[valid_scale_mask])
        if not all_transformed_corners:
            print("Error: Could not determine shape. No valid transformed corners found.")
            return None

        all_corners_np = np.vstack(all_transformed_corners)

        min_coords = np.min(all_corners_np, axis=0)
        max_coords = np.max(all_corners_np, axis=0)
        
        min_x = np.floor(min_coords[0]).astype(int)
        max_x = np.ceil(max_coords[0]).astype(int)
        min_y = np.floor(min_coords[1]).astype(int)
        max_y = np.ceil(max_coords[1]).astype(int)

        canvas_width = max_x - min_x
        canvas_height = max_y - min_y

        offset_x = -min_x
        offset_y = -min_y

        print(f"  Calculated Canvas Size (HxW): {canvas_height} x {canvas_width}")
        print(f"  Calculated Offset (x, y): ({offset_x}, {offset_y})")
        
        canvas_shape = (canvas_height, canvas_width)
        canvas_offset = (offset_x, offset_y)

        return canvas_shape, canvas_offset
    
    def adjust_homographies_for_canvas(
        self,
        composite_H: Dict[int, np.ndarray],
        canvas_offset: Tuple[int, int]
    ) -> Dict[int, np.ndarray]:
        """
        Adjusts composite homographies to map directly to the final canvas coordinates.

        Args:
            composite_H (Dict): Maps camera number -> H matrix mapping TO REFERENCE frame.
            canvas_offset (Tuple[int, int]): (offset_x, offset_y) translation required
                                            to align reference origin with canvas origin.

        Returns:
            Dict[int, np.ndarray]: Maps camera number -> H matrix mapping TO FINAL CANVAS.
        """
        offset_x, offset_y = canvas_offset
        T_offset = np.array([
            [1, 0, offset_x],
            [0, 1, offset_y],
            [0, 0, 1]
        ], dtype=np.float32)
        final_homographies = {}
        for cam_num, H_ref_i in composite_H.items():
            if H_ref_i is not None:
                H_canvas_i = T_offset @ H_ref_i
                final_homographies[cam_num] = H_canvas_i
            else:
                final_homographies[cam_num] = None
                print(f"Skipping adjustment for Camera {cam_num} (missing composite H).")
        return final_homographies
    
    def warp_images_and_create_masks(
        self,
        original_images: Dict[int, np.ndarray],
        final_homographies: Dict[int, Optional[np.ndarray]],
        canvas_shape: Tuple[int, int],
        camera_names: List[int],
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, Tuple[int, int]]]:
        """
        Warps all original images onto the final canvas size using adjusted homographies
        and creates corresponding binary masks.

        Args:
            original_images (Dict): Maps camera number to original image (NumPy BGR array).
            final_homographies (Dict): Maps camera number to the 3x3 homography matrix
                                    (H_canvas <- i) mapping to the final canvas.
            canvas_shape (Tuple[int, int]): Final canvas (height, width).
            camera_names (List): List of camera numbers to process.
        Returns:
            Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, Tuple[int, int]]]:
                - warped_images: Dict mapping cam_num to warped image (on canvas).
                - masks: Dict mapping cam_num to binary mask (on canvas).
                - corners: Dict mapping cam_num to top-left (x, y) corner on canvas.
        """
        
        warped_images: Dict[int, np.ndarray] = {}
        masks: Dict[int, np.ndarray] = {}
        corners: Dict[int, Tuple[int, int]] = {}

        canvas_height, canvas_width = canvas_shape
        dsize_cv = (canvas_width, canvas_height)
        
        for cam_num in camera_names:
            if cam_num not in original_images:
                warnings.warn(f"Original image for camera {cam_num} not found. Skipping warp.")
                continue
            if cam_num not in final_homographies or final_homographies[cam_num] is None:
                warnings.warn(f"Final homography for camera {cam_num} not found. Skipping warp.")
                continue

            pil_image = original_images[cam_num]
            img_src_numpy = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            H_canvas_i = final_homographies[cam_num]
            src_h, src_w = img_src_numpy.shape[:2]

            print(f"  Warping Camera {cam_num}...")
            warped_img = self.transform(
                image_src=img_src_numpy,
                H=H_canvas_i,
                output_shape=canvas_shape,
                interpolation_flag=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                border_value=(0, 0, 0)
            )

            if warped_img is None:
                warnings.warn(f"Warping failed for camera {cam_num}.")
                continue

            mask_src = np.ones((src_h, src_w), dtype=np.uint8) * 255
            mask = self.transform(
                image_src=mask_src,
                H=H_canvas_i,
                output_shape=canvas_shape,
                interpolation_flag=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                border_value=0
            )

            if mask is None:
                warnings.warn(f"Mask warping failed for camera {cam_num}.")
                continue

            orig_corner = np.array([[0, 0, 1]], dtype=np.float32).T
            transformed_corner_h = H_canvas_i @ orig_corner
            
            tl_x, tl_y = 0, 0
            if abs(transformed_corner_h[2, 0]) > 1e-8:
                tl_x = int(round(transformed_corner_h[0, 0] / transformed_corner_h[2, 0]))
                tl_y = int(round(transformed_corner_h[1, 0] / transformed_corner_h[2, 0]))
                
            warped_images[cam_num] = warped_img
            masks[cam_num] = mask
            corners[cam_num] = (tl_x, tl_y)
        return warped_images, masks, corners
    
    def blend_images_opencv(
        self,
        warped_images: Dict[int, Optional[np.ndarray]],
        masks: Dict[int, Optional[np.ndarray]],
        corners: Dict[int, Optional[Tuple[int, int]]],
        canvas_shape: Tuple[int, int],
        camera_order: List[int],
    ) -> Optional[np.ndarray]:
        """Blends images in a final panormic photo using OpenCV's MultiBandBlender."""

        blender = cv2.detail.MultiBandBlender()

        canvas_h, canvas_w = canvas_shape
        canvas_rect = (0, 0, canvas_w, canvas_h)
        blender.prepare(canvas_rect)

        for cam_num in camera_order:
            img = warped_images.get(cam_num)
            mask = masks.get(cam_num)
            corner = corners.get(cam_num)
            if img is not None and mask is not None and corner is not None:
                blender.feed(img.astype(np.int16), mask, corner) 
            else:
                warnings.warn(f"Skipping feed for Cam {cam_num} - missing data.")
        result_pano, result_mask = blender.blend(None, None)
        panorama_final = np.clip(result_pano, 0, 255).astype(np.uint8)
        return panorama_final
    
FAST_THRESHOLD = 30
FAST_N = 12
BRIEF_DESC_SIZE = 256
BRIEF_PATCH_SIZE = 31
MATCHER_RATIO_THRESH = 0.75
RANSAC_ITERATIONS = 1000
RANSAC_RESIDUAL_THRESH = 3.0

CAMERA_NAMES = [1, 2, 3, 4, 5]

ADJACENT_PAIRS = [
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5)
]
    
segment_id = SEGMENT_IDS[-1]
data = load_waymo_data_from_structure(DATASET_BASE_DIR, segment_id)
frameParser = WaymoFrameParser(data)
timestamps = frameParser.get_timestamps()

target_timestamp = timestamps[0]
frame_images = frameParser.get_timestamp_images(target_timestamp)

fast_detector = FASTPCornerDetection(t=FAST_THRESHOLD, n=FAST_N)
brief_descriptor = BRIEFDescriptor(descriptor_size=BRIEF_DESC_SIZE, patch_size=BRIEF_PATCH_SIZE)
feature_matcher = FeatureMatcher(ratio_threshold=MATCHER_RATIO_THRESH)
ransac_homography = RANSACHomography(num_iterations=RANSAC_ITERATIONS, inlier_threshold=RANSAC_RESIDUAL_THRESH)
image_transformer = ImageTransformer()

all_valid_keypoints = {}
all_descriptors = {}
image_shapes = {}
camera_to_image = {}

print("COMPUTING DESCRIPTORS")

for camera in CAMERA_NAMES:
    if camera in frame_images:
        image = frame_images[camera]
        scale_percent = 50 # example: 50%
        new_width = int(image.width * scale_percent / 100)
        new_height = int(image.height * scale_percent / 100)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        image_shapes[camera] = (image.height, image.width)
        camera_to_image[camera] = image
        print(f"Camera {camera}: {image.size}")
        grayscale_image = image.convert("L")
        image_array = np.array(grayscale_image, dtype=np.int16) 
        keypoints = fast_detector.detect_corners(image)
        keypoint_coords = list(zip(*np.where(keypoints))) 
        valid_keypoints, descriptors = brief_descriptor.compute(image_array, keypoint_coords)
        if descriptors is not None and len(valid_keypoints) > 0:
            all_valid_keypoints[camera] = valid_keypoints
            all_descriptors[camera] = descriptors
    else:
        print(f"Camera {camera}: No image data available.")
        
print("COMPUTING HOMOGRAPHIES")
        
adjacent_homographies = {}
for target_camera, source_camera in ADJACENT_PAIRS:
    if target_camera not in all_descriptors or source_camera not in all_descriptors:
        print(f"Skipping pair ({target_camera}, {source_camera}) due to missing descriptors.")
        adjacent_homographies[(target_camera, source_camera)] = None
        continue
    matches = feature_matcher.match(all_descriptors[source_camera], all_descriptors[target_camera])
    print(f"Found {len(matches)} matches for {target_camera}, {source_camera}.")
    if len(matches) < ransac_homography.min_samples:
        print(f"Skipping pair: Not enough matches ({len(matches)}) found for homography (need >= {ransac_homography.min_samples}).")
        adjacent_homographies[(target_camera, source_camera)] = None
        continue
    H_target_from_source, inlier_mask = ransac_homography.get_homography(
        all_valid_keypoints[source_camera],
        all_valid_keypoints[target_camera],
        matches
    )
    adjacent_homographies[(target_camera, source_camera)] = (H_target_from_source, inlier_mask) 
    if H_target_from_source is None:
        print(f"Failed to compute homography for pair ({target_camera}, {source_camera}).")
        continue
    
composite_homographies = image_transformer.calculate_composite_homographies(adjacent_homographies, CAMERA_NAMES, ref_cam_num=1)
if composite_homographies is None:
    print("Failed to compute composite homographies.")
    
canvas_result = image_transformer.calculate_canvas_geometry(composite_homographies, image_shapes, CAMERA_NAMES)
if not canvas_result: 
    print("Error: Failed to calculate canvas geometry. Stopping.")
    exit(1)
canvas_shape, canvas_offset = canvas_result

final_homographies = image_transformer.adjust_homographies_for_canvas(composite_homographies, canvas_offset)

warped_images, masks, corners = image_transformer.warp_images_and_create_masks(
        camera_to_image,
        final_homographies,
        canvas_shape,
        CAMERA_NAMES,
)

camera_order = sorted(CAMERA_NAMES)
final_panorama = image_transformer.blend_images_opencv(
        warped_images,
        masks,
        corners,
        canvas_shape,
        camera_order
)

# --- Display Final Result ---
if final_panorama is not None:
    print("\n--- Final Panorama Generated ---")
    output_filename = f"final_panorama_{segment_id}_{target_timestamp}_multiband.png"
    try:
        cv2.imwrite(output_filename, final_panorama)
        print(f"Final panorama saved to: {output_filename}")

        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(final_panorama, cv2.COLOR_BGR2RGB))
        plt.title(f"Final Stitched Panorama (Cam Ref 1, multiband blend)")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error saving or displaying final panorama: {e}")
else:
    print("\n--- Final Panorama Generation Failed ---")


