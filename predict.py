# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import fnmatch
import mimetypes
import os
import re
import tarfile
import tempfile
from typing import List, Tuple, Optional
from zipfile import ZipFile

import cv2
import mediapipe as mp
import numpy as np
from cog import BasePredictor, Input, Path
from PIL import Image, ImageFilter


def face_mask_google_mediapipe(
    images: List[Image.Image], blur_amount: float = 0.0, bias: float = 50.0
) -> List[Image.Image]:
    """
    Returns a list of images with masks on the face parts.
    """
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.1
    )
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.1
    )

    masks = []
    for image in images:
        image_np = np.array(image)

        # Perform face detection
        results_detection = face_detection.process(image_np)
        ih, iw, _ = image_np.shape
        if results_detection.detections:
            this_im_masks = []
            for detection in results_detection.detections:
                bboxC = detection.location_data.relative_bounding_box

                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )

                # make sure bbox is within image
                bbox = (
                    max(0, bbox[0]),
                    max(0, bbox[1]),
                    min(iw - bbox[0], bbox[2]),
                    min(ih - bbox[1], bbox[3]),
                )

                print(bbox)

                # Extract face landmarks
                face_landmarks = face_mesh.process(
                    image_np[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
                ).multi_face_landmarks

                # https://github.com/google/mediapipe/issues/1615
                # This was def helpful
                indexes = [
                    10,
                    338,
                    297,
                    332,
                    284,
                    251,
                    389,
                    356,
                    454,
                    323,
                    361,
                    288,
                    397,
                    365,
                    379,
                    378,
                    400,
                    377,
                    152,
                    148,
                    176,
                    149,
                    150,
                    136,
                    172,
                    58,
                    132,
                    93,
                    234,
                    127,
                    162,
                    21,
                    54,
                    103,
                    67,
                    109,
                ]

                if face_landmarks:
                    mask = Image.new("L", (iw, ih), 0)
                    mask_np = np.array(mask)

                    for face_landmark in face_landmarks:
                        face_landmark = [face_landmark.landmark[idx] for idx in indexes]
                        landmark_points = [
                            (int(l.x * bbox[2]) + bbox[0], int(l.y * bbox[3]) + bbox[1])
                            for l in face_landmark
                        ]
                        mask_np = cv2.fillPoly(
                            mask_np, [np.array(landmark_points)], 255
                        )

                    mask = Image.fromarray(mask_np)

                    # Apply blur to the mask
                    if blur_amount > 0:
                        mask = mask.filter(ImageFilter.GaussianBlur(blur_amount))

                    # Apply bias to the mask
                    if bias > 0:
                        mask = np.array(mask)
                        mask = mask + bias * np.ones(mask.shape, dtype=mask.dtype)
                        mask = np.clip(mask, 0, 255)
                        mask = Image.fromarray(mask)

                    # Convert mask to 'L' mode (grayscale) before saving
                    mask = mask.convert("L")

                    this_im_masks.append(mask)
                else:
                    # If face landmarks are not available, add a black mask of the same size as the image
                    this_im_masks.append(Image.new("L", (iw, ih), 255))
            masks.append(this_im_masks)

        else:
            print("No face detected, adding full mask")
            # If no face is detected, add a white mask of the same size as the image
            masks.append([Image.new("L", (iw, ih), 255)])

    return masks


def _crop_to_square(
    image: Image.Image, com: List[Tuple[int, int]], resize_to: Optional[int] = None
):
    cx, cy = com
    width, height = image.size
    if width > height:
        left_possible = max(cx - height / 2, 0)
        left = min(left_possible, width - height)
        right = left + height
        top = 0
        bottom = height
    else:
        left = 0
        right = width
        top_possible = max(cy - width / 2, 0)
        top = min(top_possible, height - width)
        bottom = top + width

    image = image.crop((left, top, right, bottom))

    if resize_to:
        image = image.resize((resize_to, resize_to), Image.Resampling.LANCZOS)

    return image


# def _center_of_mass(mask: Image.Image):
#     """
#     Returns the center of mass of the mask
#     """
#     x, y = np.meshgrid(np.arange(mask.size[0]), np.arange(mask.size[1]))
#     mask_np = np.array(mask) + 0.01
#     x_ = x * mask_np
#     y_ = y * mask_np

#     x = np.sum(x_) / np.sum(mask_np)
#     y = np.sum(y_) / np.sum(mask_np)

#     return x, y


def _find_files(pattern, dir="."):
    """Return list of files matching pattern in a given directory, in absolute format.
    Unlike glob, this is case-insensitive.
    """

    rule = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
    return [os.path.join(dir, f) for f in os.listdir(dir) if rule.match(f)]


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        images: Path = Input(
            description="Input image as png or jpeg, or zip/tar of input images"
        ),
        blur_amount: float = Input(description="Blur to apply to mask", default=0.0),
        bias: float = Input(description="Bias to apply to mask (lightens background)", ge=0.0, le=255.0, default=0.0),
        output_transparent_image: bool = Input(description="if true, outputs face image with transparent background", default=False),
        # output_cropped_image: bool = Input(description="if true, outputs image cropped to face", default=False)
    ) -> List[Path]:
        """Run a single prediction on the model"""

        output_cropped_image = False

        tmp_in_dir = tempfile.mkdtemp()

        tmp_out_dir = tempfile.mkdtemp()

        mt = mimetypes.guess_type(str(images))
        if mt and mt[0] and mt[0].startswith("image/"):
            images = [Image.open(str(images)).convert("RGB")]
        else:
            if mt and mt[0] and mt[0].startswith("application/zip"):
                with ZipFile(str(images), "r") as zip_ref:
                    for zip_info in zip_ref.infolist():
                        if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                            "__MACOSX"
                        ):
                            continue
                        mt = mimetypes.guess_type(zip_info.filename)
                        if mt and mt[0] and mt[0].startswith("image/"):
                            zip_info.filename = os.path.basename(zip_info.filename)
                            zip_ref.extract(zip_info, tmp_in_dir)
            elif mt and mt[0] and mt[0].startswith("application/x-tar"):
                with tarfile.open(str(images), "r") as tar_ref:
                    for tar_info in tar_ref:
                        if tar_info.name[-1] == "/" or tar_info.name.startswith(
                            "__MACOSX"
                        ):
                            continue

                        mt = mimetypes.guess_type(tar_info.name)
                        if mt and mt[0] and mt[0].startswith("image/"):
                            tar_info.name = os.path.basename(tar_info.name)
                            tar_ref.extract(tar_info, tmp_in_dir)
            else:
                assert (
                    False
                ), "`images` must be a png or jpeg image, or zip or tar of images"

            files = tmp_in_dir
            # load images
            if isinstance(files, str):
                # check if it is a directory
                if os.path.isdir(files):
                    # get all the .png .jpg in the directory
                    files = (
                        _find_files("*.png", files)
                        + _find_files("*.jpg", files)
                        + _find_files("*.jpeg", files)
                    )

                if len(files) == 0:
                    raise Exception(
                        f"No files found in {files}. Either {files} is not a directory or it does not contain any .png or .jpg/jpeg files."
                    )
                files = sorted(files)
                print("Image files: ", files)
            images = [Image.open(file).convert("RGB") for file in files]

        seg_masks = face_mask_google_mediapipe(
            images=images, blur_amount=blur_amount, bias=bias
        )

        # if output_transparent_image or output_cropped_image:
        #     coms = [(image.size[0] / 2, image.size[1] / 2) for image in images]
        #     # based on the center of mass, crop the image to a square
        #     images = [
        #         _crop_to_square(image, com, resize_to=None)
        #         for image, com in zip(images, coms)
        #     ]
        #     seg_masks = [
        #         _crop_to_square(mask, com, resize_to=None)
        #         for mask, com in zip(seg_masks, coms)
        #     ]
        
        if output_transparent_image:
            # TODO: Not a great way of dealing with multiple faces in one image
            tmp_images = []
            for i, image in enumerate(images):
                for j, seg_mask in enumerate(seg_masks[i]):
                    arr = np.array(image.convert('RGBA'))
                    arr[:,:,3] = np.array(seg_mask.convert('L'))
                    tmp_images.append(Image.fromarray(arr))
            images = tmp_images

        if output_cropped_image or output_transparent_image:
            im_paths = []
            for idx, im in enumerate(images):
                im_file = f"{idx}.image.png"

                im_file = os.path.join(tmp_out_dir, im_file)
                im_paths.append(Path(im_file))
                im.save(im_file)
            return im_paths

        else:
            mask_paths = []
            for i, mask_list in enumerate(seg_masks):
                for j, mask in enumerate(mask_list):
                    mask_file = f"{i}_{j}.mask.png"

                    mask_path = os.path.join(tmp_out_dir, mask_file)
                    mask_paths.append(Path(mask_path))
                    mask.save(mask_path)

            return mask_paths
