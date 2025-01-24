# Eyes_Detection_Tensorflow

This project demonstrates a machine learning model for detecting whether a person's eyes are open or closed. The model is trained using TensorFlow and MobileNet with transfer learning to classify images of eyes. This can be extended for applications like drowsiness detection systems.

## Table of Contents
- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [How to Run](#how-to-run)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

---

## Introduction

The project uses transfer learning with the MobileNet model to classify eye images into two categories:
- **Closed Eyes**
- **Open Eyes**
---

## Technologies Used

- **Python** 
- **TensorFlow** (Keras API)
- **OpenCV** 
- **NumPy**
- **Matplotlib**
- **Pickle**

---

## Dataset

- The dataset consists of two classes: `Closed_Eyes` and `Open_Eyes`.
- Images are preprocessed and resized to `224x224` to match the input requirements of the MobileNet model.
- The dataset is organized as follows:
<pre>
Test_Dataset/
├── Closed_Eyes/
├── Open_Eyes/
</pre>


---

## Workflow

### 1. Preprocessing the Dataset
- **Convert Grayscale to RGB:** Images are converted from grayscale to RGB to match the input format required by the MobileNet model.
- **Resize Images:** Each image is resized to `224x224`.
- **Label Encoding:** Labels are assigned as:
- `0`: Closed Eyes
- `1`: Open Eyes
- Images and labels are stored in `training_Data`.

### 2. Data Shuffling and Splitting
- The data is shuffled using Python's `random` module to ensure randomness.
- Features (`X`) and labels (`y`) are separated and stored as NumPy arrays for efficient processing.

### 3. Saving Data with Pickle
- Preprocessed data (`X` and `y`) is serialized and saved to disk using Pickle for faster reloading.

### 4. Model Development
- **Transfer Learning:** 
- A pre-trained MobileNet model is loaded. 
- All layers are frozen to retain previously learned features.
- **Custom Layers:** Fully connected layers are added for binary classification (Open vs. Closed Eyes).
- **Activation Function:** The final layer uses a `sigmoid` activation function.

### 5. Training 
- The model is trained on the preprocessed dataset with appropriate loss and optimization functions.

### 6. Prediction
- OpenCV detects the face and eyes, extracts the ROI (Region of Interest), and predicts the eye status.

---

## How to Run

### Prerequisites
1. Install the required Python libraries:
   
### Steps
1. Clone the repository:
   `https://github.com/Shreyansh301/Eyes_Detection_Tensorflow.git`
3. Preprocess the dataset:
- Run the data preprocessing script to create `X` and `y` Pickle files.
3. Train the model:
- Run the training script to train and save the model (`my_model.h5`).
4. Run the detection script:

## Results

- The model achieves high accuracy on the validation dataset.

## Acknowledgments

- **TensorFlow/Keras**
- **OpenCV**
