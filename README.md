# EchoCNN: CNN-Based Audio Classification Framework

## 📌 Overview
EchoCNN is a deep learning-based **audio classification framework** designed to classify real-time audio samples with high accuracy. Built using **Convolutional Neural Networks (CNNs)**, this model was trained on **3,000 labeled audio samples from IIT Jammu**, achieving an impressive **94% accuracy**.

## 🎯 Features
- **High Accuracy**: Achieves **94% prediction accuracy** on real-world audio datasets.
- **Data Augmentation**: Increases dataset size by **50%**, enhancing model generalization and robustness.
- **Optimized Training**: Cross-validation and hyperparameter tuning reduce training time by **20%** while improving accuracy by **5%**.
- **Scalability**: Can be adapted for different audio classification tasks, including speech recognition, wildlife monitoring, and environmental noise detection.

## 🔧 Technologies Used
- **Python** (NumPy, Pandas, Matplotlib)
- **TensorFlow/Keras** (CNN-based architecture)
- **Librosa** (Audio processing & feature extraction)
- **Scikit-learn** (Data preprocessing & evaluation metrics)

## 📂 Dataset & Preprocessing
- **Dataset**: 3,000 labeled audio samples collected from IIT Jammu.
- **Preprocessing Steps**:
  - Converted raw audio signals into **Mel spectrograms**.
  - Applied **noise reduction and normalization**.
  - Implemented **data augmentation** (time-stretching, pitch shifting, and noise addition).

## 🏗️ Model Architecture
- **Input Layer**: Mel spectrograms as feature inputs.
- **CNN Layers**: Extracts meaningful features from spectrograms.
- **Dense Layers**: Fully connected layers for classification.
- **Softmax Output**: Predicts the audio class probabilities.

## 📊 Performance & Evaluation
| Metric  | Value  |
|---------|--------|
| Accuracy | 94% |
| Training Time Reduction | 20% |
| Accuracy Improvement | 5% |

## 🚀 How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/EchoCNN.git
   cd EchoCNN
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python train.py
   ```
4. Test the model:
   ```bash
   python test.py
   ```

## 📌 Applications
- **Speech & Sound Recognition**
- **Wildlife & Birdsong Classification**
- **Environmental Noise Monitoring**
- **Industrial Sound Fault Detection**

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request.

## 📜 License
This project is licensed under the **MIT License**.

---

🚀 **EchoCNN** – Transforming Audio into Intelligent Insights! 🎙️
