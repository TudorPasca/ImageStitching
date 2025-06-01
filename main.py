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
import os
from datetime import datetime

DATASET_BASE_DIR = Path('./dataset/camera_image')

SEGMENT_IDS = [
    '1005081002024129653_5313_150_5333_150',
    '10017090168044687777_6380_000_6400_000',
    '10023947602400723454_1120_000_1140_000',
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

    data = {'image': None}

    file_path = base_dir / f"{segment_id}.parquet"
    if file_path.is_file():
        try:
            df = pd.read_parquet(file_path, engine='pyarrow')
            data['image'] = df[df['component_type'] == 'camera_image'] if 'component_type' in df.columns else df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    else:
        warnings.warn(f"Warning: File not found: {file_path}", UserWarning)
        return None

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
        ], dtype=np.int16)
        
        self._n8_offsets = np.array([
            [ 0,  1], [ 1,  1], [ 1,  0], [ 1, -1], [ 0, -1], [-1, -1],
            [-1,  0], [-1,  1]
        ], dtype=np.int16)
        
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
    def __init__(self, descriptor_size=256, patch_size=31, gaussian_sigma=None, seed=42):
        """
        Initializes the BRIEF descriptor extractor.

        Args:
            descriptor_size (int): The desired length of the descriptor in bits (e.g., 128, 256, 512).
            patch_size (int): The size of the square patch around the keypoint (must be odd).
            gaussian_sigma (float | None): Sigma for Gaussian blur applied to the patch
                                          before sampling. If None, no smoothing is applied.
                                          Requires SciPy installed if not None.
            seed (int): Random seed for generating sampling pairs.
        """
        if patch_size % 2 == 0:
            raise ValueError("patch_size must be odd.")
            
        self.descriptor_size = descriptor_size
        self.patch_size = patch_size
        self.gaussian_sigma = gaussian_sigma
        self._half_patch = patch_size // 2
        self._seed = seed
        
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
        rng = np.random.default_rng(seed=self._seed)

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
    def __init__(self, num_iterations=1000, inlier_threshold=3.0, seed=42):
        """
        Initializes the RANSAC Homography estimator.

        Args:
            num_iterations (int): Number of RANSAC iterations to perform.
            inlier_threshold (float): Maximum reprojection error (in pixels)
                                        for a point pair to be considered an inlier.
            seed (int): Random seed for RANSAC sampling.
        """
        self.num_iterations = num_iterations
        self.inlier_threshold = inlier_threshold
        self.min_samples = 4
        self._seed = seed
        self._rng = np.random.default_rng(seed=seed)
        
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

        for k in range(self.num_iterations):
            if num_matches >= self.min_samples:
                 sample_indices = self._rng.choice(num_matches, self.min_samples, replace=False)
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
             
        # if best_H is not None and np.sum(best_inlier_mask) > self.min_samples:
        #     all_inlier_pts1 = pts1[best_inlier_mask]
        #     all_inlier_pts2 = pts2[best_inlier_mask]
        #     try:
        #         # Re-estimate H using all inliers from the best model
        #         refined_H = self._compute_homography_dlt(all_inlier_pts1, all_inlier_pts2)
        #         if refined_H is not None:
        #             # You might want to ensure the refined_H doesn't drastically increase error,
        #             # but usually, it's better or very similar for the inlier set.
        #             best_H = refined_H
        #     except ValueError: # SVD can fail if inliers are degenerate
        #         pass # Keep the original best_H from the minimal sample
        #     except Exception as e:
        #         print(f"Warning: Error during homography refinement: {e}") # Keep original best_H
        #         pass

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
        Cameras are arranged in order: 4, 2, 1, 3, 5 from left to right, all at same height.
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
                    
        expected_positions = {
            1: (0, 0),
            2: (-1, 0),
            3: (1, 0),
            4: (-2, 0),
            5: (2, 0)
        }
        
        if ref_cam_num not in all_cam_nums: 
            return None
        
        print(f"\nCalculating homographies relative to camera {ref_cam_num}...")
        composite_H[ref_cam_num] = identity_matrix

        for cam_num in all_cam_nums:
            if cam_num == ref_cam_num:
                continue

            if (ref_cam_num, cam_num) in all_available_H:
                H = all_available_H[(ref_cam_num, cam_num)]
                if self._validate_homography(H, expected_positions[cam_num]):
                    composite_H[cam_num] = H
                    print(f"  Using direct homography for camera {cam_num}")
                    continue

            if cam_num in [2, 3]:
                if (ref_cam_num, cam_num) in all_available_H:
                    H = all_available_H[(ref_cam_num, cam_num)]
                    if self._validate_homography(H, expected_positions[cam_num]):
                        composite_H[cam_num] = H
                        print(f"  Using validated direct homography for camera {cam_num}")
                        continue

            if cam_num == 4 and 2 in composite_H:
                if (2, 4) in all_available_H:
                    H_2_4 = all_available_H[(2, 4)]
                    H = composite_H[2] @ H_2_4
                    if self._validate_homography(H, expected_positions[cam_num]):
                        composite_H[cam_num] = H
                        print(f"  Using path through camera 2 for camera {cam_num}")
                        continue

            if cam_num == 5 and 3 in composite_H:
                if (3, 5) in all_available_H:
                    H_3_5 = all_available_H[(3, 5)]
                    H = composite_H[3] @ H_3_5
                    if self._validate_homography(H, expected_positions[cam_num]):
                        composite_H[cam_num] = H
                        print(f"  Using path through camera 3 for camera {cam_num}")
                        continue

            # # For camera 5, try a more direct approach if the indirect path failed
            # if cam_num == 5 and (ref_cam_num, 5) in all_available_H:
            #     H = all_available_H[(ref_cam_num, 5)]
            #     if np.sign(H[0, 2]) > 0:  # Check if it moves right
            #         composite_H[cam_num] = H
            #         print(f"  Using direct homography for camera {cam_num} (relaxed validation)")
            #         continue

            print(f"  Warning: Could not find valid homography for camera {cam_num}")

        if len(composite_H) != len(all_cam_nums):
            print("Warning: Could not compute composite homographies for all cameras.")
        return composite_H

    def _validate_homography(self, H: np.ndarray, expected_position: Tuple[int, int]) -> bool:
        """
        Validates if a homography matrix produces a reasonable transformation.
        Now focuses on horizontal alignment since all cameras are at the same height.
        """
        try:
            test_points = np.array([
                [0, 0],
                [100, 0],
                [0, 100],
                [100, 100]
            ], dtype=np.float32)
            
            test_points_h = np.hstack((test_points, np.ones((4, 1))))
            transformed_points_h = (H @ test_points_h.T).T
            transformed_points = transformed_points_h[:, :2] / transformed_points_h[:, 2:]
        
            translation = np.mean(transformed_points - test_points, axis=0)
            
            expected_x, _ = expected_position
            if expected_x != 0:
                if expected_x < 0:
                    if translation[0] > 0:
                        return False
                else:
                    if np.sign(translation[0]) != np.sign(expected_x):
                        return False
            
            # Much more lenient vertical translation check for cameras 4 and 5
            max_vertical = 1200 if expected_x in [-2, 2] else 800  # Increased for outer cameras
            if abs(translation[1]) > max_vertical:
                return False
            
            # More lenient threshold for outer cameras
            max_translation = 6000 if abs(expected_x) == 2 else 4000  # Increased for outer cameras
            if np.any(np.abs(translation) > max_translation):
                return False
                
            return True
            
        except Exception:
            return False
    
    def calculate_canvas_geometry(
        self,
        composite_H: Dict[int, np.ndarray],
        image_shapes: Dict[int, Tuple[int, int]],
        camera_names: List[int]
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Calculates the required canvas size and offset to fit all transformed images.
        Now uses a more compact approach to minimize black space.
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
        
        # Calculate dimensions with padding
        padding = 200  # Add padding to prevent edge clipping
        width = max_coords[0] - min_coords[0] + 2 * padding
        height = max_coords[1] - min_coords[1] + 2 * padding

        # Ensure minimum size
        min_size = 2000
        if width < min_size:
            width = min_size
        if height < min_size:
            height = min_size

        # Scale down if the canvas is too large
        max_dimension = 6000  # Increased from 4000
        if width > max_dimension or height > max_dimension:
            scale_factor = max_dimension / max(width, height)
            width = int(width * scale_factor)
            height = int(height * scale_factor)

        # Calculate offset to center the content
        # First, find the center of the reference camera (camera 1)
        ref_cam_corners = None
        for cam_num in camera_names:
            if cam_num == 1 and cam_num in composite_H and composite_H[cam_num] is not None:
                h, w = image_shapes[cam_num]
                ref_corners = np.array([
                    [0,   0],
                    [w-1, 0],
                    [w-1, h-1],
                    [0,   h-1] 
                ], dtype=np.float32)
                ref_corners_h = np.hstack((ref_corners, np.ones((4, 1))))
                ref_transformed_h = (composite_H[cam_num] @ ref_corners_h.T).T
                ref_scale = ref_transformed_h[:, 2]
                valid_scale_mask = np.abs(ref_scale) > 1e-8
                ref_transformed = np.zeros((4, 2), dtype=np.float32)
                ref_transformed[valid_scale_mask] = (
                    ref_transformed_h[valid_scale_mask, :2] / ref_scale[valid_scale_mask, np.newaxis]
                )
                ref_cam_corners = ref_transformed
                break

        if ref_cam_corners is not None:
            # Use the reference camera's center as the anchor point
            ref_center = np.mean(ref_cam_corners, axis=0)
            offset_x = int(width/2 - ref_center[0])
            offset_y = int(height/2 - ref_center[1])
        else:
            # Fallback to using the center of all corners
            center_x = (min_coords[0] + max_coords[0]) / 2
            center_y = (min_coords[1] + max_coords[1]) / 2
            offset_x = int(width/2 - center_x)
            offset_y = int(height/2 - center_y)

        canvas_height_int = int(np.ceil(height))
        canvas_width_int = int(np.ceil(width))

        print(f"  Calculated Canvas Size (HxW): {canvas_height_int} x {canvas_width_int}")
        print(f"  Calculated Offset (x, y): ({offset_x}, {offset_y})")
        
        canvas_shape = (canvas_height_int, canvas_width_int)
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
        """
        warped_images: Dict[int, np.ndarray] = {}
        masks: Dict[int, np.ndarray] = {}
        corners: Dict[int, Tuple[int, int]] = {}

        canvas_height, canvas_width = canvas_shape
        print(f"\nWarping images to canvas size: {canvas_width}x{canvas_height}")
        
        for cam_num in camera_names:
            if cam_num not in original_images:
                print(f"  Skipping camera {cam_num}: No original image")
                continue
            if cam_num not in final_homographies or final_homographies[cam_num] is None:
                print(f"  Skipping camera {cam_num}: No homography matrix")
                continue

            try:
                pil_image = original_images[cam_num]
                img_src_numpy = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                H_canvas_i = final_homographies[cam_num]
                src_h, src_w = img_src_numpy.shape[:2]

                print(f"  Processing camera {cam_num}")
                print(f"    Source image size: {src_w}x{src_h}")
                
                # Warp the image
                warped_img = cv2.warpPerspective(
                    img_src_numpy,
                    H_canvas_i,
                    (canvas_width, canvas_height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0)
                )

                if warped_img is None:
                    print(f"    Warping failed for camera {cam_num}")
                    continue

                # Create and warp the mask
                mask_src = np.ones((src_h, src_w), dtype=np.uint8) * 255
                mask = cv2.warpPerspective(
                    mask_src,
                    H_canvas_i,
                    (canvas_width, canvas_height),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

                if mask is None:
                    print(f"    Mask warping failed for camera {cam_num}")
                    continue

                # Calculate corner position
                orig_corner = np.array([[0, 0, 1]], dtype=np.float32).T
                transformed_corner_h = H_canvas_i @ orig_corner
                
                tl_x, tl_y = 0, 0
                if abs(transformed_corner_h[2, 0]) > 1e-8:
                    tl_x = int(round(transformed_corner_h[0, 0] / transformed_corner_h[2, 0]))
                    tl_y = int(round(transformed_corner_h[1, 0] / transformed_corner_h[2, 0]))
                
                print(f"    Warped image size: {warped_img.shape}")
                print(f"    Corner position: ({tl_x}, {tl_y})")
                
                warped_images[cam_num] = warped_img
                masks[cam_num] = mask
                corners[cam_num] = (tl_x, tl_y)
                
            except Exception as e:
                print(f"    Error processing camera {cam_num}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
                
        return warped_images, masks, corners
    
    def blend_images_opencv(
        self,
        warped_images: Dict[int, Optional[np.ndarray]],
        masks: Dict[int, Optional[np.ndarray]],
        corners: Dict[int, Optional[Tuple[int, int]]],
        canvas_shape: Tuple[int, int],
        camera_order: List[int],
    ) -> Optional[np.ndarray]:
        """Blends images using alpha blending with masks and removes black borders."""
        try:
            canvas_h, canvas_w = canvas_shape
            # Initialize the final panorama with zeros
            final_panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
            # Initialize the weight mask
            weight_mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)
            
            print("\nBlending images...")
            for cam_num in camera_order:
                img = warped_images.get(cam_num)
                mask = masks.get(cam_num)
                corner = corners.get(cam_num)
                
                if img is None or mask is None or corner is None:
                    print(f"  Skipping camera {cam_num}: Missing data")
                    continue
                
                # print(f"  Processing camera {cam_num}")
                # print(f"    Image shape: {img.shape}")
                # print(f"    Mask shape: {mask.shape}")
                # print(f"    Corner position: {corner}")
                
                # Convert mask to float32 and normalize
                mask_float = mask.astype(np.float32) / 255.0
                
                # Convert image to float32
                img_float = img.astype(np.float32)
                
                # Add the weighted image to the panorama
                for c in range(3):  # For each color channel
                    final_panorama[:, :, c] += img_float[:, :, c] * mask_float
                
                # Update the weight mask
                weight_mask += mask_float
            
            # Avoid division by zero
            weight_mask = np.maximum(weight_mask, 1e-6)
            
            # Normalize the panorama
            for c in range(3):
                final_panorama[:, :, c] /= weight_mask
            
            # Convert back to uint8
            panorama_final = np.clip(final_panorama, 0, 255).astype(np.uint8)
            
            # Remove black borders
            gray = cv2.cvtColor(panorama_final, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add some padding
                padding = 50
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(panorama_final.shape[1] - x, w + 2*padding)
                h = min(panorama_final.shape[0] - y, h + 2*padding)
                
                # Crop the image
                panorama_final = panorama_final[y:y+h, x:x+w]
            
            # print("Blending completed successfully")
            return panorama_final
            
        except Exception as e:
            print(f"Error during blending: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
FAST_THRESHOLD = 25
FAST_N = 12
BRIEF_DESC_SIZE = 256
BRIEF_PATCH_SIZE = 31
MATCHER_RATIO_THRESH = 0.8
RANSAC_ITERATIONS = 3000
RANSAC_RESIDUAL_THRESH = 2.5
GLOBAL_SEED = 45

CAMERA_NAMES = [1, 2, 3, 4, 5]

ADJACENT_PAIRS = [
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5)
]

def liniar_interpolate(image: np.ndarray) -> np.ndarray:
    for x in range(image.shape[1]):
        start = 0
        fin = 0
        colored = []
        for y in range(image.shape[0]):
            r , g, b = image[y, x]
            if r != 0 or g != 0 or b != 0:
                if start == 0:
                    start = y
                colored.append((r,g,b))
            else:
                if start != 0:
                    fin = y
                    break
        # print(start, fin, int(start +  (fin-start)))
        for y in range(image.shape[0]):
            position = int(y * (fin-start)/(image.shape[0]))
            r , g, b = colored[position]
            image[y,x] = (r,g,b)
    return image

def stitch_segment(segment_id: str, frame_parser: WaymoFrameParser, timestamp: int, output_dir: Path, frame_index: int, composite_homographies_cache = None, images_semantic = None) -> bool:
    """
    Process a single segment and save the stitched panorama.
    
    Args:
        segment_id: The ID of the segment to process
        frame_parser: The WaymoFrameParser instance
        timestamp: The timestamp to process
        output_dir: Directory to save the results
        frame_index: The index of the frame being processed (1-based)
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # print(f"\nProcessing segment: {segment_id}, frame {frame_index}")
        if images_semantic is None:
            frame_images = frame_parser.get_timestamp_images(timestamp)
        else:
            frame_images = images_semantic
        if not frame_images:
            print(f"No images found for segment {segment_id} at timestamp {timestamp}")
            return False
        # IMAGE_SAVED_DIR = ".\\input_images"
        # Initialize feature detection and matching components
        fast_detector = FASTPCornerDetection(t=FAST_THRESHOLD, n=FAST_N)
        brief_descriptor = BRIEFDescriptor(descriptor_size=BRIEF_DESC_SIZE, patch_size=BRIEF_PATCH_SIZE, seed=GLOBAL_SEED)
        feature_matcher = FeatureMatcher(ratio_threshold=MATCHER_RATIO_THRESH)
        ransac_homography = RANSACHomography(num_iterations=RANSAC_ITERATIONS, inlier_threshold=RANSAC_RESIDUAL_THRESH, seed=GLOBAL_SEED)
        image_transformer = ImageTransformer()

        all_valid_keypoints = {}
        all_descriptors = {}
        image_shapes = {}
        camera_to_image = {}

        # print("COMPUTING DESCRIPTORS")
        for camera in CAMERA_NAMES:
            if camera in frame_images:
                image = frame_images[camera]
                scale_percent = 50
                new_width = int(image.width * scale_percent / 100)
                new_height = int(image.height * scale_percent / 100)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                image_shapes[camera] = (image.height, image.width)
                camera_to_image[camera] = image
                # print(f"Camera {camera}: {image.size}")
                if composite_homographies_cache is None:
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

        
        if composite_homographies_cache is None:
            # print("COMPUTING HOMOGRAPHIES")
            adjacent_homographies = {}
            for target_camera, source_camera in ADJACENT_PAIRS:
                if target_camera not in all_descriptors or source_camera not in all_descriptors:
                    print(f"Skipping pair ({target_camera}, {source_camera}) due to missing descriptors.")
                    adjacent_homographies[(target_camera, source_camera)] = None
                    continue
                matches = feature_matcher.match(all_descriptors[source_camera], all_descriptors[target_camera])
                # print(f"Found {len(matches)} matches for {target_camera}, {source_camera}.")
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
                return False
            else:
                composite_homographies_cache = composite_homographies
        else:
            composite_homographies = composite_homographies_cache
        canvas_result = image_transformer.calculate_canvas_geometry(composite_homographies, image_shapes, CAMERA_NAMES)
        if not canvas_result: 
            print("Error: Failed to calculate canvas geometry.")
            return False
        canvas_shape, canvas_offset = canvas_result

        final_homographies = image_transformer.adjust_homographies_for_canvas(composite_homographies, canvas_offset)

        warped_images, masks, corners = image_transformer.warp_images_and_create_masks(
            camera_to_image,
            final_homographies,
            canvas_shape,
            CAMERA_NAMES,
        )

        camera_order = [4, 2, 1, 3, 5]
        final_panorama = image_transformer.blend_images_opencv(
            warped_images,
            masks,
            corners,
            canvas_shape,
            camera_order
        )

        if final_panorama is not None:
            # print(type(final_panorama))       # Should print: <class 'numpy.ndarray'>
            # print(final_panorama.shape)       # (Height, Width) for grayscale or (Height, Width, 3) for RGB
            # print(final_panorama.dtype)       # e.g., uint8, float32
            # final_panorama = liniar_interpolate(final_panorama)
            output_filename = output_dir / f"panorama_{segment_id}_frame{frame_index}.png"
            cv2.imwrite(str(output_filename), final_panorama)
            # print(f"Panorama saved to: {output_filename}")
            return True, composite_homographies_cache
        else:
            print("Failed to generate final panorama.")
            return False

    except Exception as e:
        print(f"Error processing segment {segment_id}, frame {frame_index}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main(limit: int):
    composite_homographies_cache = None
    i_limit = 0
    for TARGET_SEGMENT_INDEX in SEGMENT_IDS[::-1]:
        i_limit+=1
        if i_limit > limit:
            break
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results") 
        results_dir.mkdir(parents=True, exist_ok=True)
        # TARGET_SEGMENT_INDEX = SEGMENT_IDS[-1]
        NUM_FRAMES = float('inf') 
        
        if TARGET_SEGMENT_INDEX not in SEGMENT_IDS:
            print(f"Error: Segment {TARGET_SEGMENT_INDEX} not found in available segments")
            return
            
        print(f"\nProcessing segment: {TARGET_SEGMENT_INDEX}")
        
        data = load_waymo_data_from_structure(DATASET_BASE_DIR, TARGET_SEGMENT_INDEX)
        if data is None:
            print(f"Failed to load data for segment {TARGET_SEGMENT_INDEX}")
            return
            
        frame_parser = WaymoFrameParser(data)
        timestamps = frame_parser.get_timestamps()
        if not timestamps:
            print(f"No timestamps found for segment {TARGET_SEGMENT_INDEX}")
            return
        
        num_frames_to_process = min(NUM_FRAMES, len(timestamps))
        print(f"Processing {num_frames_to_process} frames from segment {TARGET_SEGMENT_INDEX}")
        for i, timestamp in enumerate(timestamps[:num_frames_to_process], 1):
            if i_limit > limit:
                break
            i_limit+=1
            print(f"\nProcessing frame {i} of {num_frames_to_process}")
            success, composite_homographies_cache = stitch_segment(TARGET_SEGMENT_INDEX, frame_parser, timestamp, results_dir, i, composite_homographies_cache)
            if success:
                print(f"Successfully processed frame {i}")
            else:
                print(f"Failed to process frame {i}")
    return composite_homographies_cache


from semantic import generate_colorized_waymo_frames


def main_semantic(composite_homographies_cache):
    SEMANTIC_PANORAMA_DIR = "semantic_results"
    CAMERA_IMAGE_DATASET_PATH = 'dataset/camera_image' 
    # Path to the directory containing camera segmentation parquet files
    CAMERA_SEGMENTATION_DATASET_PATH = 'dataset/semantic/camera_segmentation'
    # Iterate through the generator
    frame_counter = 0
    for frame_data in generate_colorized_waymo_frames(
        CAMERA_IMAGE_DATASET_PATH,
        CAMERA_SEGMENTATION_DATASET_PATH,
        alpha=0.6 # Adjust transparency as needed
    ):
        print(f"\nProcessing frame: Segment={frame_data['segment_context_name']}, Timestamp={frame_data['frame_timestamp_micros']}")
        
        # Access the 5 images for this timestamp
        camera_images = frame_data['camera_images']
        
        # Example: Save each of the 5 images to disk
        images_semantic = {}
        for camera_name, img_pil in camera_images.items():
            # output_filename = os.path.join(
            #     OUTPUT_COLORIZED_IMAGES_PATH,
            #     f"{frame_data['segment_context_name']}_{frame_data['frame_timestamp_micros']}_{camera_name}_overlay.jpg"
            # )
            if camera_name == "SIDE_LEFT":
                cn = 4
            elif camera_name == "FRONT_LEFT":
                cn = 2
            elif camera_name == "FRONT":
                cn = 1
            elif camera_name == "FRONT_RIGHT":
                cn = 3
            else:
                cn = 5
            images_semantic[cn] = img_pil
        success, composite_homographies_cache = stitch_segment("0", None, 1, Path(SEMANTIC_PANORAMA_DIR), frame_counter, composite_homographies_cache, images_semantic)
        frame_counter += 1
        # Optional: Stop after processing a few frames for demonstration
        # if frame_counter >= 2: # Uncomment and adjust to process only a few frames
        #     print("Processed 2 full frames. Stopping example usage.")
        #     break

    if frame_counter == 0:
        print("No complete frames (5 cameras with overlays) were generated.")
    else:
        print(f"\nFinished processing. Total {frame_counter} complete frames (sets of 5 camera images) generated and saved.")

if __name__ == "__main__":
    composite_homographies_cache = main(1)
    main_semantic(composite_homographies_cache)

