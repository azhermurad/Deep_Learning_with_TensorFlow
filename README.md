# 🔬 Learn TensorFlow for Deep Learning

This repository contains all my work from the Learn TensorFlow for Deep Learning. Practical, project-based approach to Lean deep learning using TensorFlow and covers everything from fundamentals to deployment.

---

## 📚 Course Curriculum Overview

| Module | Description |
|--------|-------------|
| `00` | TensorFlow Fundamentals (Tensors, Variables, Devices) |
| `01` | Deep Learning Workflow (Build, Compile, Fit, Evaluate) |
| `02` | Neural Network Classification |
| `03` | Computer Vision with CNNs |
| `04` | Transfer Learning with TensorFlow Hub |
| `05` | Natural Language Processing (NLP) |
| `06` | Time Series Forecasting |
| `07` | TensorFlow Serving and Deployment |

---

## 🧠 Model Summaries and Architectures

### 1. 🔢 TensorFlow Fundamentals
- Basics: `tf.constant`, `tf.Variable`, broadcasting, device placement
- ✅ Built basic tensor operations and tracked gradients manually

### 2. 🔁 Deep Learning Workflow
- Used `tf.keras.Sequential`, `compile`, `fit`, `evaluate`, `predict`
- ✅ Developed full ML pipeline for classification

### 3. 🧠 Neural Network Classification
- Dataset: FashionMNIST
- Model:
  ```python
  model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  ```
- Optimizer: Adam | Loss: CategoricalCrossentropy

### 4. 🖼️ Computer Vision
- Dataset: Food101, CIFAR-10
- Used: `Conv2D`, `MaxPooling2D`, `BatchNormalization`, `Dropout`
- ✅ Created custom CNNs and improved generalization with data augmentation

### 5. 📦 Transfer Learning
- Pretrained Models: `EfficientNet`, `MobileNetV2`, etc.
- Source: `tensorflow_hub`
- Strategy:
  - Freeze base layers, train classifier head
  - Fine-tune top layers later
- ✅ Achieved high performance with fewer training steps

### 6. 📝 Natural Language Processing
- Text Preprocessing: `TextVectorization`
- Models: RNN, GRU, LSTM, 1D Conv
- ✅ Built spam detection and sentiment analysis pipelines

### 7. 📈 Time Series Forecasting
- Dataset: Daily temperature series
- Windowing strategies and model inputs
- Used: Dense, Conv1D, LSTM layers for sequence prediction
- ✅ Built and evaluated models using MAE and visual inspection

### 8. 🌍 Model Deployment
- Tools: `TensorFlow Serving`, `SavedModel`, `TFLite`, `TFJS`
- ✅ Converted models to `SavedModel`, deployed with Docker or used in browsers

---

## 🧪 Tools & Libraries Used

- `TensorFlow`, `TensorFlow Hub`, `Keras`
- `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`
- `Scikit-learn`, `Gradio`, `TensorBoard`

---

## 🏆 Projects

| Project | Description | Tech |
|--------|-------------|------|
| `FoodVision` | Image classifier for 101 food classes | CNN / Transfer Learning |
| `Sentiment Classifier` | Text classification using LSTM & Conv1D | NLP |
| `Time Series Predictor` | Temperature forecasting using LSTM | Time Series |
| `TensorFlow Deployment` | Converted and served models using Docker/Fast API | TF Serving 

---

## 📂 Repository Structure

```bash
.
├── 00_tensorflow_fundamentals/
├── 01_tensorflow_workflow/
├── 02_neural_network_classification/
├── 03_cnn_computer_vision/
├── 04_transfer_learning/
├── 05_nlp/
├── 06_time_series/
├── 07_model_serving/
├── extras/
└── README.md
```

