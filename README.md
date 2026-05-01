#  Arabic License Plate Recognition for Egyptian Vehicles

An AI-powered system for automatic recognition of Egyptian Arabic license plates, combining classical image processing techniques with a custom-trained Convolutional Neural Network (CNN).

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-97.8%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

##  Highlights

-  **97.8% Test Accuracy** on Egyptian license plates
-  **Lightweight Model** with only 2.4 million parameters (26× smaller than Mask R-CNN)
-  **Fast Inference** - less than 100ms per plate
-  **Runs on Consumer Hardware** - tested on NVIDIA RTX 4050 Laptop GPU
-  **Modular Pipeline** - 5-stage architecture for easy debugging and improvement
-  **Compact** - only 10 MB model size

##Demo
Input:  Egyptian License Plate Image
Output: Recognized Plate Text
Example: "4 2 9 8 ج ن"  (Numbers and Arabic letters)
## System Architecture

The system uses a 5-stage pipeline:
Input Image → Augmentation → Preprocessing → Segmentation → CNN → Output Text
### Stages:

1. **Data Augmentation** - Generates 24,816 training images from 2,069 originals
2. **Image Preprocessing** - CLAHE + Adaptive Thresholding for clean binary output
3. **Character Segmentation** - Contour detection with intelligent filtering
4. **CNN Classification** - Custom 2.4M parameter network for character recognition
5. **Result Assembly** - Combines predictions into final plate text

## Performance Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | **97.8%** |
| Segmentation Success Rate | 94.8% |
| Model Parameters | ~2.4 million |
| Model Size | 10 MB |
| Training Time | 10 minutes |
| Inference Time | <100ms per plate |
| GPU Memory Usage | <1 GB |

## Comparison with Existing Work

| Approach | Accuracy | Parameters | Hardware |
|----------|----------|------------|----------|
| Hefnawy et al. (2024) - Mask R-CNN + LSTM | 99% | ~63M | High-end GPU |
| Sayedelahl (2024) - Two-stage DL | 99.3% | Heavy | High-end GPU |
| Abdellatif et al. (2023) - IoT-based | 93% | Image processing only | Raspberry Pi |
| **Our Work (2026)** | **97.8%** | **2.4M** | **Consumer GPU** |

## Tech Stack

### Languages & Frameworks
- **Python 3.11**
- **PyTorch** - Deep learning framework
- **OpenCV (cv2)** - Image processing
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization

### AI/ML Techniques
- Convolutional Neural Networks (CNN)
- Data Augmentation
- Adaptive Thresholding
- Contour Detection

### Image Processing
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Bilateral Filtering
- Adaptive Gaussian Thresholding
- Morphological Operations

##  Installation

### Prerequisites

- Python 3.11
- NVIDIA GPU with CUDA support (optional, but recommended)
- Windows/Linux/macOS

### Setup

```bash
# Clone the repository
git clone https://github.com/refa3ey/Egyptian-License-plate.git
cd Egyptian-License-plate

# Install dependencies
pip install -r requirements.txt
```

##  Usage

### Quick Test - Predict from Sample Image

```bash
python predict.py samples/plate_001.jpg
```

### Train Your Own Model

```bash
# Run augmentation first
python augmentation.py

# Run preprocessing
python preprocessing.py

# Run segmentation
python segmentation.py

# Label characters manually
python labeling.py

# Train the CNN
python train.py
```

### Check System Health

```bash
python diagnostic.py
```

## 📊 Dataset

- **2,069** original Egyptian vehicle license plate images
- **24,816** augmented images through 5 augmentation techniques
- **1,903** manually labeled characters
- **26** classes (16 Arabic letters + 10 numerical digits)

### Augmentation Techniques

| Technique | Parameters |
|-----------|-----------|
| Rotation | ±5°, ±10° |
| Brightness | 0.6, 0.8, 1.2, 1.4 |
| Gaussian Noise | Random variance |
| Gaussian Blur | Various kernels |
| Combined Dark + Noise | Simulates night conditions |

The `samples/` folder contains representative plates for testing.

## 🧠 CNN Architecture
NPUT (64×64 grayscale)
↓
[Conv Block 1] - 32 filters
Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPool → Dropout(0.25)
↓
[Conv Block 2] - 64 filters
Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPool → Dropout(0.25)
↓
[Conv Block 3] - 128 filters
Conv2D → BatchNorm → ReLU → MaxPool → Dropout(0.25)
↓
[Classification Head]
Flatten → FC(8192→256) → ReLU → Dropout(0.5) → FC(256→26)
↓
OUTPUT (26 classes - probabilities)
## ⚙️ Training Configuration

```python
Framework: PyTorch
Input Size: 64×64 grayscale
Batch Size: 64
Optimizer: Adam (lr=0.0005)
Loss Function: Cross-Entropy
Scheduler: StepLR (gamma=0.5, step=10)
Epochs: 20
Train/Test Split: 80/20
```

## Sample Results

| Plate Image | Predicted Output | Confidence | Result |
|-------------|------------------|------------|--------|
| Plate 0001 | 7 1 2 6 ص ط ن | 91.7% | ✅ Correct |
| Plate 0044 | 6 6 6 6 م م | 95.9% | ✅ Correct |
| Plate 0144 | 4 4 4 4 م م م | 99.8% | ✅ Correct |

## Common Confusion Patterns

We identified these confusion patterns through analysis:
- **ن vs ق** - Similar curved shapes, dots can be lost during preprocessing
- **Plate dividers** - Sometimes misclassified as أ
- **Merged characters** - Adjacent characters can connect, causing segmentation errors

## Future Improvements

- [ ] Add YOLO-based plate detection from full vehicle images
- [ ] Expand dataset with more diverse plates
- [ ] Implement attention mechanisms for similar character disambiguation
- [ ] Try transfer learning with ResNet or EfficientNet
- [ ] Support for plates from other Arab countries (Saudi, UAE, Kuwait)
- [ ] Real-world deployment testing
- [ ] Edge device optimization (NVIDIA Jetson, Raspberry Pi)
- [ ] Mobile app deployment

## Research Paper

This project is based on our research paper:

> **"Arabic License Plate Recognition Using Image Processing and Custom CNN: A Lightweight Approach for Egyptian Vehicles"**  
> Authors: Bilal Mahmoud Alrifaee, Moatasem El Dieb, Youssef Khaled, Youssef Mohamed Hassan

### Key Contributions:

1. ✅ Complete preprocessing pipeline using CLAHE + Adaptive Gaussian Thresholding
2. ✅ Contour-based character segmentation with intelligent filtering rules
3. ✅ Lightweight CNN classifier (97.8% accuracy with 2.4M parameters)
4. ✅ End-to-end pipeline runnable on consumer hardware

## Limitations

We acknowledge the following limitations:
- Class imbalance (some characters have fewer training samples)
- No plate detection module (assumes pre-cropped input)
- Visual similarity confusion (e.g., ن vs ق)
- Dataset size (2,069 plates is relatively small)

## Authors

- **Bilal Mahmoud Alrifaee** - Lead Developer & Researcher
- **Moatasem El Dieb** - Researcher
- **Youssef Khaled** - Researcher
- **Youssef Mohamed Hassan** - Researcher

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Our project supervisor for guidance
- The OpenCV and PyTorch communities
- Authors of the research papers we referenced

## 🌟 Star this Repository

If you find this project useful, please consider giving it a star! ⭐

---

**Built with ❤️ for Egyptian smart transportation systems**
