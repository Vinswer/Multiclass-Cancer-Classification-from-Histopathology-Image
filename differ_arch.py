import torch
import torch.nn as nn
import torchvision.models as models

class SpatialAttentionModule(nn.Module):
    """
    Spatial attention mechanism for focusing on diagnostically relevant regions.
    Critical for medical images where specific tissue patterns matter.
    """

    def __init__(self, in_channels):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_weights = self.sigmoid(self.conv(x))
        return x * attention_weights


class ChannelAttentionModule(nn.Module):
    """Channel attention for feature recalibration."""

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class MedicalResNet50(nn.Module):
    """
    Enhanced ResNet50 with medical-specific modifications:
    - Spatial and channel attention mechanisms
    - Medical-grade classification head with dropout
    - Optimized for histopathological image analysis
    """

    def __init__(self, num_classes=2, pretrained=True):
        super(MedicalResNet50, self).__init__()

        # Load pretrained ResNet50 backbone
        self.backbone = models.resnet50(pretrained=pretrained)

        # Remove original classification layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Add attention mechanisms
        self.channel_attention = ChannelAttentionModule(2048)
        self.spatial_attention = SpatialAttentionModule(2048)

        # Medical classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)

        # Apply attention mechanisms
        features = self.channel_attention(features)
        features = self.spatial_attention(features)

        # Global pooling and classification
        pooled = self.global_pool(features)
        output = self.classifier(pooled)

        return output, features  # Return features for Grad-CAM


class MedicalEfficientNet(nn.Module):
    """
    EfficientNet-based model optimized for medical imaging.
    Efficient architecture suitable for clinical deployment.
    """

    def __init__(self, num_classes=2):
        super(MedicalEfficientNet, self).__init__()

        # Load EfficientNet-B0 backbone
        self.backbone = models.efficientnet_b0(pretrained=True)

        # Extract features before classifier
        self.features = self.backbone.features
        self.avgpool = self.backbone.avgpool

        # Medical classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        pooled = self.avgpool(features)
        pooled = torch.flatten(pooled, 1)
        output = self.classifier(pooled)

        return output, features


class EnsembleModel(nn.Module):
    """
    Ensemble of ResNet50 and EfficientNet for robust medical predictions.
    """

    def __init__(self, num_classes=2):
        super(EnsembleModel, self).__init__()
        self.resnet = MedicalResNet50(num_classes)
        self.efficientnet = MedicalEfficientNet(num_classes)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        resnet_out, resnet_features = self.resnet(x)
        efficientnet_out, efficientnet_features = self.efficientnet(x)

        # Combine predictions
        combined = torch.cat([resnet_out, efficientnet_out], dim=1)
        ensemble_out = self.fusion(combined)

        return ensemble_out, resnet_features  # Use ResNet features for Grad-CAM

