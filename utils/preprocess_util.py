#!/usr/bin/env python3

from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T


from utils import feature_util
from utils import logging
from utils.misc import array_to_tensor


logger: logging.Logger = logging.get_logger()


class ImageCropper(torch.nn.Module):
    """
    Crops the image and the mask around the amodal mask, applys padding and rescaling.

    crop_size: the size of the cropped image, it should be divisible by 14 due to DINO features.
    """

    def __init__(self, crop_size) -> None:
        super().__init__()

        self.crop_size = crop_size

        self.to_tensor = T.ToTensor()

        # Normalize with the ImageNet mean/std.
        self.normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

    def forward_v0(
        self,
        image: np.ndarray,
        modal_mask: np.ndarray,
        image_size: Tuple[int, int],
        grid_cell_size: float,
        black_background: bool = False,
    ):

        # Get the bounding box around the modal mask.
        x1, x2, y1, y2 = extract_box_from_mask(modal_mask)

        # Generate the real 2D points representing the sampled features.
        image_grid_x, image_grid_y = feature_util.generate_grid_points_matrices(
            grid_size=image_size,
            cell_size=grid_cell_size,
        )

        # Converts a HWC numpy array to a CHW tensor.
        image_tensor_chw = self.to_tensor(image)
        mask_tensor = array_to_tensor(modal_mask)

        # Crop and pad the object:
        if black_background:
            image_grid_x = image_grid_x[y1:y2, x1:x2]
            image_grid_y = image_grid_y[y1:y2, x1:x2]

            # # Crop the image and the mask within the bounding box.
            image_cropped = image_tensor_chw[:, y1:y2, x1:x2]
            mask_cropped = mask_tensor[y1:y2, x1:x2]

            # Apply the mask on the image, this will black-out the background.
            image_cropped = image_cropped * mask_cropped

            # # Add padding to crops to make them square.
            image_tensor_chw = make_square_padding(image_cropped)
            mask_tensor = make_square_padding(mask_cropped)
            image_grid_x = make_square_padding(image_grid_x)
            image_grid_y = make_square_padding(image_grid_y)

        else:
            (
                image_tensor_chw,
                mask_tensor,
                image_grid_x,
                image_grid_y,
            ), _ = crop_and_make_square_padding(
                [image_tensor_chw, mask_tensor, image_grid_x, image_grid_y],
                (x1, x2, y1, y2),
            )
            print(
                image_tensor_chw.shape,
                mask_tensor.shape,
                image_grid_x.shape,
                image_grid_y.shape,
            )

        # Resize the shortest side of the cropped box to the crop_size (for both
        # the image and the bounding box).
        image_out = T.functional.resize(
            image_tensor_chw,
            int(self.crop_size),
            T.InterpolationMode.BICUBIC,
        )
        mask_out = T.functional.resize(
            mask_tensor.unsqueeze(0),
            int(self.crop_size),
            T.InterpolationMode.NEAREST,
        )[0]

        image_grid_x = T.functional.resize(
            image_grid_x.unsqueeze(0),
            int(self.crop_size),
            T.InterpolationMode.NEAREST,
        )[0]

        image_grid_y = T.functional.resize(
            image_grid_y.unsqueeze(0),
            int(self.crop_size),
            T.InterpolationMode.NEAREST,
        )[0]

        # Normalize the image values.
        image_out = self.normalize(image_out)

        # # Get the resize factor so that it can be used to fix query points.
        # resize_factor = self.crop_size / mask_cropped.shape[0]

        # # Bounding box.
        # bbox = (x1, x2, y1, y2)

        # Stack the query points.
        image_cropped_grid_points = torch.vstack(
            (image_grid_x.flatten(), image_grid_y.flatten())
        ).T

        # Return the resized image and mask crops.
        return image_out, mask_out, image_cropped_grid_points

    def forward(
        self,
        image: np.ndarray,
        modal_mask: np.ndarray,
        proj_sampling_points: torch.Tensor,
        image_size: Tuple[int, int],
        black_background: bool = False,
    ):

        # Get the bounding box around the modal mask.
        x1, x2, y1, y2 = extract_box_from_mask(modal_mask)

        # Converts a HWC numpy array to a CHW tensor.
        image_tensor_chw = self.to_tensor(image)
        mask_tensor = array_to_tensor(modal_mask)

        # Crop and pad the object:
        if black_background:

            # # Crop the image and the mask within the bounding box.
            image_cropped = image_tensor_chw[:, y1:y2, x1:x2]
            mask_cropped = mask_tensor[y1:y2, x1:x2]

            # Apply the mask on the image, this will black-out the background.
            image_cropped = image_cropped * mask_cropped

            # # Add padding to crops to make them square.
            image_tensor_chw, padding = make_square_padding(image_cropped)
            mask_tensor, _ = make_square_padding(mask_cropped)

            # Update the shift locations according to the padding.
            (pad_left, pad_top, pad_right, pad_bottom) = padding
            x1 += pad_left
            y1 += pad_top
        else:
            (
                image_tensor_chw,
                mask_tensor,
            ), (x1, y1) = crop_and_make_square_padding(
                [image_tensor_chw, mask_tensor],
                (x1, x2, y1, y2),
            )

        logger.info(
            f"Cropped image and mask  sizes {image_tensor_chw.shape} , {mask_tensor.shape}"
        )

        # Resize the shortest side of the cropped box to the crop_size (for both
        # the image and the bounding box).
        image_out = T.functional.resize(
            image_tensor_chw,
            int(self.crop_size),
            T.InterpolationMode.BICUBIC,
        )
        mask_out = T.functional.resize(
            mask_tensor.unsqueeze(0),
            int(self.crop_size),
            T.InterpolationMode.NEAREST,
        )[0]

        # Normalize the image values.
        image_out = self.normalize(image_out)

        # Get the resize factor so that it can be used to fix query points.
        resize_factor = mask_out.shape[0] / mask_tensor.shape[0]

        # Shift the sampling points and rescale with the resize factor of images.
        proj_sampling_points[:, 0] -= x1
        proj_sampling_points[:, 1] -= y1
        proj_sampling_points *= resize_factor

        # Return the resized image and mask crops.
        return image_out, mask_out, proj_sampling_points


def extract_box_from_mask(binary_mask: np.ndarray):
    # Extracts the bounding box around a binary mask.

    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return x1, x2, y1, y2


def crop_and_make_square_padding(image_list, bbox):
    """
    Pad an image tensor to make it square.

    Parameters:
    - image_list (List[torch.Tensor]): a 2D or 3D tensor.
    - bbox (Tuple[int, int, int, int]): bounding box coordinates.

    Returns:
    - torch.Tensor: square padded tensor.
    """

    # Determine croppind boundaries and padding.
    x1, x2, y1, y2 = bbox
    h = y2 - y1 + 1
    w = x2 - x1 + 1

    image = image_list[0]
    if len(image.shape) == 2:
        H, W = image.shape
    elif len(image.shape) == 3:
        _, H, W = image.shape

    if h > w:
        pad_left = pad_right = (h - w) // 2
        if (h - w) % 2 == 1:
            pad_right += 1
        cy1 = y1
        cy2 = y2
        cx1 = max(0, x1 - pad_left)
        cx2 = min(W, x2 + pad_right)
        w = cx2 - cx1 + 1
        pad_left = pad_right = (h - w) // 2
        pad_top = pad_bottom = 0
        if (h - w) % 2 == 1:
            pad_right += 1
    else:
        pad_top = pad_bottom = (w - h) // 2
        if (w - h) % 2 == 1:
            pad_bottom += 1
        cx1 = x1
        cx2 = x2
        cy1 = max(0, y1 - pad_top)
        cy2 = min(H, y2 + pad_bottom)
        h = cy2 - cy1 + 1
        pad_top = pad_bottom = (w - h) // 2
        pad_left = pad_right = 0
        if (w - h) % 2 == 1:
            pad_bottom += 1

    padded_image_list = []
    # Apply the crop and padding to all images given in the list.
    for image_to_pad in image_list:

        # Crop the image within the bounding box.
        if len(image_to_pad.shape) == 2:
            image_to_pad = image_to_pad[cy1:cy2, cx1:cx2]
        elif len(image_to_pad.shape) == 3:
            image_to_pad = image_to_pad[:, cy1:cy2, cx1:cx2]

        # Add padding so that image is centered.
        padding = (pad_left, pad_top, pad_right, pad_bottom)

        # Apply padding
        padded_image = T.functional.pad(image_to_pad, padding, 0, "constant")

        # Update the list with the padded image.
        padded_image_list.append(padded_image)

    return padded_image_list, (x1 + pad_left, y1 + pad_top)


def make_square_padding(image):
    """
    Pad an image tensor to make it square.

    Parameters:
    - image (torch.Tensor): a 2D or 3D tensor.

    Returns:
    - torch.Tensor: square padded tensor.
    """
    # Check the number of dimensions to determine if it's grayscale or RGB
    if len(image.shape) == 2:
        h, w = image.shape
    elif len(image.shape) == 3:
        _, h, w = image.shape
    else:
        raise ValueError("Input tensor must be 2D (HxW) or 3D (CxHxW)")

    # Determine padding
    if h > w:
        pad_left = pad_right = (h - w) // 2
        pad_top = pad_bottom = 0
        if (h - w) % 2 == 1:
            pad_right += 1
    else:
        pad_top = pad_bottom = (w - h) // 2
        pad_left = pad_right = 0
        if (w - h) % 2 == 1:
            pad_bottom += 1

    # Add padding so that image is centered.
    padding = (pad_left, pad_top, pad_right, pad_bottom)

    # Add padding so that image is located on upper left corner
    # padding = (0, 0, pad_left + pad_right, pad_top + pad_bottom)

    # Apply padding
    padded_image = T.functional.pad(image, padding, 0, "constant")

    return padded_image, padding
