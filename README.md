# ImageStitching
A personal image-stitching algorithm implementation for the Image Processing laboratory. Uses the Waymo Public Dataset.


## Instructions
---

To obtain the panoramic images you should have a folder named `dataset/camera_image` where you should load the `.parquet` files. Also very important for the semantic panoramas you should have a folder `dataset/semantic/camera_segmentation` and also the result forlder `semantic_results`.
To create both the semantic and normal panoramas you should give the `flaot('inf')` argument of the `main()` function inside the main.py. If youn want just the semantic give the argument `1`. To run create a video from the panorama images you should change the target directories inside the `video.py` file then run the command `python video.py`.