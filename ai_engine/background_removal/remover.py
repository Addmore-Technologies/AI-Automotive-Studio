"""
Background Removal and Transparent Cutout Extraction.
Extracts high-resolution foreground RGBA images from vehicle masks.
Applies soft alpha feathering and edge fringe de-spill.
"""

import os
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from ai_engine.common.types import ColorOrder


class BackgroundRemover:
    """
    Extracts isolated vehicle cutouts with alpha channel transparency.
    Performs boundary feathering and color fringe de-spill.
    """

    def __init__(self, feather_radius: int = 2, despill_edges: bool = True):
        """
        Initialize the BackgroundRemover.

        Args:
            feather_radius: Radius for edge anti-aliasing (in pixels).
            despill_edges: If True, attenuates background color bleeding on vehicle boundaries.
        """
        self.feather_radius = feather_radius
        self.despill_edges = despill_edges

    def extract_foreground(
        self,
        image_or_path: Union[str, np.ndarray],
        mask: np.ndarray,
        color_order: ColorOrder = ColorOrder.BGR,
    ) -> np.ndarray:
        """
        Produce an RGBA foreground image with transparent background.

        Args:
            image_or_path: Image path or numpy array (shape HxWx3).
            mask: Binary mask (uint8 0 or 255, shape HxW).
            color_order: BGR (OpenCV) or RGB.

        Returns:
            RGBA numpy array (uint8, shape HxWx4, values 0-255).
            Channel order is [R, G, B, A] where A is alpha (0=transparent, 255=opaque).
        """
        if isinstance(image_or_path, str):
            if not os.path.exists(image_or_path):
                raise FileNotFoundError(f"Image not found at {image_or_path}")
            bgr_img = cv2.imread(image_or_path)
            if bgr_img is None:
                raise ValueError(f"Failed to load image from {image_or_path}")
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        else:
            if color_order == ColorOrder.BGR:
                rgb_img = cv2.cvtColor(image_or_path, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = image_or_path.copy()

        h, w = rgb_img.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 1. Generate soft alpha matte with edge anti-aliasing
        alpha = self._create_soft_alpha(mask)

        # 2. Edge fringe de-spill to clean up background color bleeding
        cleaned_rgb = rgb_img.copy()
        if self.despill_edges and np.any(mask > 0):
            cleaned_rgb = self._despill_edges(cleaned_rgb, mask)

        # 3. Create RGBA image
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = cleaned_rgb
        rgba[:, :, 3] = alpha

        return rgba

    def _create_soft_alpha(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Produce a smooth alpha channel from binary mask using subtle distance transform
        or Gaussian blur on the boundary for anti-aliasing.
        """
        if self.feather_radius <= 0 or np.count_nonzero(binary_mask) == 0:
            return binary_mask.astype(np.uint8)

        # Calculate mask boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary_mask, kernel, iterations=self.feather_radius)
        eroded = cv2.erode(binary_mask, kernel, iterations=self.feather_radius)
        boundary_zone = cv2.subtract(dilated, eroded)

        # Blur the whole mask gently
        ksize = self.feather_radius * 2 + 1
        blurred = cv2.GaussianBlur(binary_mask.astype(np.float32), (ksize, ksize), 0)

        # Alpha is 255 in solid core, 0 in exterior, and smooth inside boundary zone
        alpha = np.where(boundary_zone > 0, blurred, binary_mask.astype(np.float32))
        return np.clip(alpha, 0, 255).astype(np.uint8)

    def _despill_edges(self, rgb_img: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
        """
        Perform inward color bleeding along the border so edge pixels blend naturally
        without halos from the old background.
        """
        # Outer boundary where original background color might spill
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        inner_core = cv2.erode(binary_mask, kernel, iterations=2)
        fringe = cv2.subtract(binary_mask, inner_core)

        if np.count_nonzero(fringe) == 0 or np.count_nonzero(inner_core) == 0:
            return rgb_img

        # Inpaint fringe using inner vehicle colors
        inpaint_mask = fringe.copy()
        despilled_bgr = cv2.inpaint(
            cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR),
            inpaint_mask,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA,
        )
        despilled_rgb = cv2.cvtColor(despilled_bgr, cv2.COLOR_BGR2RGB)

        # Blend: preserve original inside core, use despilled only at edge fringe
        fringe_float = (fringe > 0).astype(np.float32)[:, :, None]
        out_rgb = rgb_img * (1.0 - fringe_float * 0.75) + despilled_rgb * (fringe_float * 0.75)
        return np.clip(out_rgb, 0, 255).astype(np.uint8)

    def save_cutout(self, rgba: np.ndarray, output_path: str) -> str:
        """
        Save the RGBA image to disk as a PNG preserving full alpha transparency.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img = Image.fromarray(rgba, mode="RGBA")
        img.save(output_path, format="PNG")
        return output_path
