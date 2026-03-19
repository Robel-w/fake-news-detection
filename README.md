# Fake News Detection Using Neural Network

Assignment project: Building a feedforward neural network to classify news headlines as Real (0) or Fake (1).

## Dataset
- File: `fake_news_dataset.csv`
- ~4000 samples, balanced classes
- Using **title** column only (article text is placeholder → no information)

## Technologies
- Python
- Pandas, Scikit-learn (TF-IDF)
- TensorFlow / Keras (Neural Network)
- Google Colab

## Model Architecture
- Input (TF-IDF features) → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
- Loss: Binary Crossentropy
- Optimizer: Adam (lr=0.001)
- Epochs: 20

## Results
- Test Accuracy: ~50% (expected due to synthetic dataset with generic titles)
- Limitation: Titles are "Breaking News 1/2/..." → no real patterns to learn

## How to Run
1. Open in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/fake-news-detection/blob/main/notebooks/Fake_News_Detection_Neural_Network.ipynb)
2. Run all cells

## Answers to Assignment Questions
- **Architecture**: Feedforward NN with 2 hidden layers (128 + 64 neurons)
- **Epochs**: 20
- **Activations**: ReLU (hidden), Sigmoid (output)
- **Accuracy**: ~50% on test set (random baseline due to dataset nature)

