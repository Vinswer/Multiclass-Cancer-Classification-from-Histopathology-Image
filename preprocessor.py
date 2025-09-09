import numpy as np
import os
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk
from scipy.ndimage.morphology import binary_fill_holes

class MedicalImagePreprocessor:
    """
    Advanced medical image preprocessing including H&E stain normalization
    and color standardization for histopathological images.
    """

    def __init__(self, output_dir="preprocessed_outputs", show_plots=True):
        # Reference stain vectors for H&E normalization (pre-computed)
        self.stain_ref = np.array([
            [0.54598845, 0.322116],
            [0.72385198, 0.76419107],
            [0.42182333, 0.55879629]
        ])
        self.max_sat_ref = np.array([[0.82791151], [0.61137274]])

        self.output_dir = output_dir
        self.show_plots = show_plots
        os.makedirs(self.output_dir, exist_ok=True)

    def _savefig(self, filename, dpi=200, bbox_inches="tight"):
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
        # if self.show_plots:
        #     plt.show()
        # else:
        #     plt.close()
        print(f"[Saved] {save_path}")
        return save_path

    def optical_density_conversion(self, image):
        """
        Convert RGB image to optical density values.
        Essential for H&E stain analysis in histopathology.
        """
        image = image.astype(np.float64)
        od = -np.log10((image + 1e-6) / 255.0)
        return od

    def he_stain_separation(self, image, beta=0.15, alpha=1, light_intensity=255):
        """
        H&E stain normalization using Macenko et al. method (2009).

        Reference: "A method for normalizing histology slides for quantitative analysis"
        ISBI 2009. Critical for standardizing images across different scanners.
        """
        try:
            # Setup
            x = np.asarray(image, dtype=np.float64)
            original_shape = x.shape
            x = x.reshape(-1, 3)

            # Convert to optical density
            od = -np.log10((x + 1e-8) / light_intensity)

            # Remove transparent pixels
            od_thresh = od[np.all(od >= beta, axis=1)]

            if len(od_thresh) < 2:
                return image  # Return original if insufficient tissue

            # SVD for stain vector estimation
            try:
                _, _, V = np.linalg.svd(od_thresh, full_matrices=False)
                top_eigvecs = V[:2, :].T * -1
            except np.linalg.LinAlgError:
                return image

            # Project onto stain space and find extremes
            proj = od_thresh @ top_eigvecs
            angles = np.arctan2(proj[:, 1], proj[:, 0])

            min_angle = np.percentile(angles, alpha)
            max_angle = np.percentile(angles, 100 - alpha)

            # Convert to stain vectors
            extreme_angles = np.array([
                [np.cos(min_angle), np.cos(max_angle)],
                [np.sin(min_angle), np.sin(max_angle)]
            ])
            stains = top_eigvecs @ extreme_angles

            # Ensure hematoxylin is first (darker stain)
            if stains[0, 0] < stains[0, 1]:
                stains = stains[:, [1, 0]]

            # Calculate stain saturations
            sats, _, _, _ = np.linalg.lstsq(stains, od.T, rcond=None)
            max_sat = np.percentile(sats, 99, axis=1, keepdims=True)

            # Avoid division by zero
            max_sat[max_sat == 0] = 1
            sats = sats / max_sat * self.max_sat_ref

            # Reconstruct normalized image
            od_norm = self.stain_ref @ sats
            x_norm = 10 ** (-od_norm) * light_intensity
            x_norm = np.clip(x_norm, 0, 255).astype(np.uint8)
            x_norm = x_norm.T.reshape(original_shape)

            return x_norm

        except Exception as e:
            print(f"Stain normalization failed: {e}")
            return image

    def color_normalization_pipeline(self, image):
        """Complete color normalization pipeline for histopathology."""
        normalized = self.he_stain_separation(image)
        return normalized

    def tissue_detection(self, image, tissue_threshold=0.8):
        """
        Detect tissue regions using morphological operations.
        Filters out background and low-quality regions.
        """
        # Convert to grayscale and invert
        gray = rgb2gray(image)
        gray = 1 - gray  # Tissue becomes bright

        # Edge detection and morphological cleanup
        try:
            edges = canny(gray, sigma=1.0)
            cleaned = binary_closing(edges, disk(10))
            cleaned = binary_dilation(cleaned, disk(10))
            cleaned = binary_fill_holes(cleaned)

            tissue_percentage = cleaned.mean()
            return tissue_percentage >= tissue_threshold, tissue_percentage
        except:
            return True, 1.0  # Default to accepting image if processing fails

    def preprocess_image(self, image, apply_stain_norm=True, check_tissue=True):
        """
        Complete preprocessing pipeline for a single image.
        """
        if check_tissue:
            has_tissue, tissue_pct = self.tissue_detection(image)
            if not has_tissue:
                return None, tissue_pct

        if apply_stain_norm:
            processed_image = self.color_normalization_pipeline(image)
        else:
            processed_image = image

        return processed_image, 1.0