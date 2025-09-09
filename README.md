# Multiclass-Cancer-Classification-from-Histopathology-Image
This project used the PatchCamelyon dataset to develop a deep learning-based automated histopathology cancer detection system, designed to assist pathologists in cancer diagnosis. It implemented a complete pipeline from medical image preprocessing to deep learning classification and interpretable AI, and output clinical-grade performance metrics.

## Key Features
- **Medical Image Preprocessing**
  - H&E Stain Normalization: Stain normalization based on the classic method of Macenko et al. (2009)
  - Addressing image variability between scanners and ensuring staining consistency
  - Quality control: Automatic filtering of low-quality and background areas

- **Deep learning model development**
  - Enhanced ResNet50: Integrating spatial and channel attention mechanisms
  - Medical-specific EfficientNet: A lightweight architecture optimized for histopathology images
  - Transfer learning: Adapting to the medical domain based on ImageNet pre-trained models

- **Clinical Clinic-Level Evaluation Metrics**
  - Sensitivity: The ability to detect malignant cases
  - Specificity: The ability to correctly identify benign cases
  - AUC-ROC: Overall discriminative performance assessment
  - Precision and Negative Predictive Value (NPV): Clinical decision support metrics
  - Statistical significance testing: Ensures the clinical reliability of the results
    
- **Model Interpretability**
  - Grad-CAM Visualization: Generates heatmaps to highlight the discriminative regions of the model's attention
  - Attention Distribution Analysis: Quantifies the model's focus on different tissue regions

## Dataset Structure
```bash
Histopathology-Cancer-Classification/
├── main.py # Main program entry
├── preprocessor.py # Medical image preprocessing class
├── differ_arch.py # Deep learning model architecture
├── data_augmentation.py # Data loading and augmentation
├── evaluation_metric.py # Medical evaluation metric calculation
├── grad.py # Grad-CAM visualization implementation
├── requirements.txt # Dependency list
├── results/ # Training results and model storage
    ├── best_model.pth # Best performing model
    ├── training_history.png # Training curve graph
    └──  gradcam_analysis.png # Grad-CAM analysis results
└── data/ # Data Directory
    ├── train.zip/ # Training Data
    ├── test.zip/ # Test Data
    └── sample_submission.csv # Image name and label

```

## Grad-CAM Interpretability Analysis
The model's attention mechanism demonstrates good medical logic:
- **Central Region Attention:** 69.8%, aligns with diagnostic criteria for the central 32x32 region of the PatchCamelyon dataset.
- **Peripheral Region Attention:** 30%, shows attention focused on the central region (red/yellow areas).
The heatmap demonstrates that the model has learned to focus on the correct spatial region.
Attention decreases towards the periphery, which is appropriate for this task.
- **Prediction Confidence:** 92% (high confidence in the malignant prediction, consistent with the true label)
<img width="836" height="284" alt="image" src="https://github.com/user-attachments/assets/1b057caa-b664-4735-9283-e0e937b70529" />




