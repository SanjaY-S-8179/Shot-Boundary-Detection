# Enhanced Shot Boundary Detection using Deep Learning

This project implements an advanced Shot Boundary Detection system using a hybrid deep learning architecture based on **ResNet18 + BiLSTM**. The system is capable of detecting **abrupt** and **gradual** transitions in videos with high accuracy.

## ✨ Features
- Hybrid model: **ResNet18** for spatial feature extraction + **BiLSTM** for temporal sequence modeling.
- Gradual transition detection using enhanced training with additional gradual samples.
- Optical flow techniques for handling complex transitions (planned extension).
- Temporal smoothing for false positive reduction (planned extension).
- High accuracy, precision, recall, and F1-score.

## 📂 Dataset

We used the [**ClipShots Dataset**](https://github.com/Tangshitao/ClipShots) — a large-scale dataset specifically designed for video shot boundary detection containing:
- Abrupt and gradual transitions
- A wide variety of real-world scenarios

**Local folder structure** used for extracted frames:
```
/extracted_frames
    /train
        /transition
        /non_transition
    /only_gradual
        /transition
        /non_transition
    /test
        /transition
        /non_transition
```

## 🛠️ Technology Stack
- **Python 3.8+**
- **PyTorch**
- **TorchVision**
- **OpenCV**
- **NumPy**
- **scikit-learn**
- **Matplotlib**
- **TQDM** (for progress visualization)

## 🏗️ Model Architecture

| Component | Details |
|-----------|---------|
| Feature Extractor | Pretrained **ResNet18** (without final FC layer) |
| Sequence Model | **BiLSTM** (hidden size = 256, bidirectional) |
| Final Classifier | Fully Connected layer (output: 2 classes) |

### Custom `ShotBoundaryDataset`
- Handles loading images from `transition` and `non_transition` folders.
- Applies data augmentations like random flips, rotations, and color jittering.

### Data Augmentation Techniques
- Resize images to (224×224)
- Random horizontal flip
- Random rotation (±30°)
- Random brightness and contrast changes
- Normalization (mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/shot-boundary-detection.git
cd shot-boundary-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset
- Download and extract the [ClipShots dataset](https://github.com/Tangshitao/ClipShots).
- Organize frames as per the folder structure shown above.

### 4. Train the Model
```bash
python main.py
```
- Training is done on the **combined train + only_gradual datasets**.
- Best model checkpoint (`best_shot_boundary_model.pth`) will be automatically saved.

### 5. Evaluate the Model
After training, the script automatically evaluates the model on the **test dataset** and prints:
- Accuracy
- Precision
- Recall
- F1 Score

## 📈 Results

| Metric | Score (Example) |
|:------|:------|
| Accuracy | 90% |
| Precision | 92% |
| Recall | 95% |
| F1 Score | 91% |

*Note: Actual results may slightly vary depending on the train/test split and hyperparameter tuning.*

## 📜 File Structure

```bash
.
├── main.py                  # Main training and evaluation script
├── dataset_loader.py         # Custom dataset class and dataloaders
├── model.py                  # ResNet + BiLSTM hybrid model
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
├── best_shot_boundary_model.pth  # Saved best model (after training)
└── /extracted_frames         # Extracted frame datasets
```

## ⚙️ Future Work
- Integrate **optical flow-based gradual transition detection**.
- Introduce **Attention Mechanisms** for better key-frame selection.
- Apply **Temporal Smoothing** techniques to further reduce false positives.
- Deployment as a video-processing API or web application.

## 👌 Acknowledgements
- [ClipShots Dataset](https://github.com/Tangshitao/ClipShots) for providing annotated video transitions.
- [PyTorch](https://pytorch.org/) and [TorchVision](https://pytorch.org/vision/stable/index.html) for deep learning frameworks.

## 📬 Contact
For any queries or contributions, feel free to connect!

# 🚀 Thank you for visiting!

