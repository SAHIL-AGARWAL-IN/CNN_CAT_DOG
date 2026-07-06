# 🐱 Cat vs Dog Image Classifier 🐶

A deep learning project implementing a custom Convolutional Neural Network (CNN) in **PyTorch** to classify images of cats and dogs. The project also includes an interactive web interface built using **Gradio** for real-time model predictions.

## 🚀 Live Demo & Deployment
You can try the model live on Hugging Face Spaces:
🔗 **[Live Demo: Cat vs Dog Classifier](https://huggingface.co/spaces/SAHIL-AGARWAL/CNN_CAT_DOG)**

### How to Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/SAHIL-AGARWAL-IN/CNN_CAT_DOG.git
   cd CNN_CAT_DOG
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Gradio web application:
   ```bash
   python app.py
   ```
4. Open the local address (typically `http://127.0.0.1:7860`) in your web browser to test predictions.

---

## 🧠 Model Architecture
The network is a custom CNN with the following layout:
* **4 Convolutional Blocks**:
  * `Conv2d` -> `ReLU` -> `MaxPool2d` (downsamples resolution from 128x128 to 8x8)
  * Increasing channel size: `3 -> 8 -> 16 -> 32 -> 64`
* **Fully Connected Layers**:
  * Dense mapping: `4096 (8*8*64) -> 256 -> 64 -> 2`
  * Outputs raw logits for 2 classes (Cat/Dog).

---

## 📈 Training Details
* **Dataset**: Kaggle Cats vs Dogs Dataset (~25,000 images).
* **Preprocessing**: Images are resized to `128x128` pixels and normalized.
* **Epochs**: Trained for 10 epochs using the Adam optimizer and Cross Entropy Loss.
* **Model Checkpoint**: The best validation model weights are saved in `best_model.pt`.

---

## 📁 Repository Structure
* `app.py`: Standalone Python web application script for Gradio interface.
* `best_model.pt`: Saved model weights from the best training epoch.
* `Untitled.ipynb`: Jupyter notebook containing data loading, training loop, validation code, and evaluation metrics (accuracy, precision, recall, confusion matrix).
* `requirements.txt`: List of dependencies.
* `.gitignore`: Excludes the raw dataset and cache files.
