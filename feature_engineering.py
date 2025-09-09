import cv2
from preprocessor import MedicalImagePreprocessor
import numpy as np

class MedicalFeatureExtractor:
    """
    Medical-specific feature engineering for histopathological images.
    """

    def __init__(self):
        self.preprocessor = MedicalImagePreprocessor()

    def extract_color_features(self, image):
        """Extract color-based features relevant for H&E staining."""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        features = {}

        # RGB statistics
        for i, channel in enumerate(['R', 'G', 'B']):
            features[f'{channel}_mean'] = np.mean(image[:, :, i])
            features[f'{channel}_std'] = np.std(image[:, :, i])

        # HSV statistics
        for i, channel in enumerate(['H', 'S', 'V']):
            features[f'{channel}_mean'] = np.mean(hsv[:, :, i])
            features[f'{channel}_std'] = np.std(hsv[:, :, i])

        # Color ratios (important for H&E)
        features['RG_ratio'] = features['R_mean'] / (features['G_mean'] + 1e-8)
        features['BR_ratio'] = features['B_mean'] / (features['R_mean'] + 1e-8)

        return features

    def extract_texture_features(self, image):
        """Extract texture features using statistical methods."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Gradient-based features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        features = {
            'gradient_mean': np.mean(gradient_magnitude),
            'gradient_std': np.std(gradient_magnitude),
            'edge_density': np.sum(gradient_magnitude > np.mean(gradient_magnitude)) / gradient_magnitude.size
        }

        return features

    def create_feature_vector(self, image):
        """Create comprehensive feature vector for traditional ML approaches."""
        color_features = self.extract_color_features(image)
        texture_features = self.extract_texture_features(image)

        # Combine all features
        feature_vector = {**color_features, **texture_features}
        return feature_vector