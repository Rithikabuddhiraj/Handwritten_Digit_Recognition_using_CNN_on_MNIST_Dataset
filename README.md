**Handwritten\_Digit\_Recognition\_using\_CNN\_on\_MNIST\_Dataset**


## 🧠 Handwritten Digit Recognition using CNN on MNIST Dataset

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to accurately classify handwritten digits from the MNIST dataset. The model is trained to recognize digits (0–9) based on grayscale images and achieves high accuracy on unseen data. This application can be extended to real-world use cases like postal automation, form digitization, and check processing.


## 🚀 Key Highlights

* ✅ Built a robust CNN architecture tailored for digit recognition tasks.
* ✅ Leveraged TensorFlow and Keras for model design, training, and evaluation.
* ✅ Achieved over **98% validation accuracy**, demonstrating strong generalization on test data.
* ✅ Applied data normalization and one-hot encoding for optimal preprocessing.
* ✅ Evaluated model performance using classification report and confusion matrix.

---

## 📂 Dataset Used

* **MNIST Handwritten Digit Dataset**

  * 60,000 training samples
  * 10,000 test samples
  * Each image is 28x28 pixels, grayscale

---

## 🛠️ Model Architecture

* Input Layer (28x28x1)
* Conv2D → ReLU → MaxPooling
* Conv2D → ReLU → MaxPooling
* Flatten → Dense → Dropout → Output Softmax Layer

---

## 📈 Results

* **Training Accuracy**: \~99%
* **Validation Accuracy**: \~98%
* **Confusion Matrix**: Highlights minimal misclassification
* **Classification Report**: High precision, recall, and F1-score across all classes

---

## 💻 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/Rithikabuddhiraj/Handwritten_Digit_Recognition_using_CNN_on_MNIST_Dataset.git
   cd Handwritten_Digit_Recognition_using_CNN_on_MNIST_Dataset
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook in Jupyter or Google Colab to train and evaluate the model.

Dependencies

* Python 3.8+
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-learn
