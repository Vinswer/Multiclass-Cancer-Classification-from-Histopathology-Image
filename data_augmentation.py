from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from preprocessor import MedicalImagePreprocessor
from PIL import Image
import numpy as np

class MedicalImageDataset(Dataset):
    """
    Custom dataset class for histopathological images with medical preprocessing.
    """

    def __init__(self, image_paths, labels, transform=None,
                 apply_medical_preprocessing=True, preprocessor=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.apply_medical_preprocessing = apply_medical_preprocessing
        self.preprocessor = preprocessor or MedicalImagePreprocessor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image (simulate for demo)
        image = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
        label = self.labels[idx]

        # Apply medical preprocessing
        if self.apply_medical_preprocessing:
            processed_image, _ = self.preprocessor.preprocess_image(image)
            if processed_image is not None:
                image = processed_image

        # Convert to PIL for transforms
        image = Image.fromarray(image.astype(np.uint8))

        if self.transform:
            image = self.transform(image)

        return image, label


def create_medical_transforms():
    """
    Create medical-specific data augmentation transforms.
    Designed for histopathological images considering rotation invariance
    and color variations in H&E staining.
    """

    # Training transforms with medical-appropriate augmentations
    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),  # Histology slides can be rotated
        transforms.ColorJitter(
            brightness=0.2,  # Compensate for staining variations
            contrast=0.2,
            saturation=0.1,
            hue=0.05
        ),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet statistics
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform
