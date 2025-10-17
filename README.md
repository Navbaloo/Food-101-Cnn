# Food-101 Image Classification using TensorFlow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview 📝

This project demonstrates how to build and train an image classification model using TensorFlow and Keras to classify images from the popular Food-101 dataset. It utilizes transfer learning with a pre-trained EfficientNetB0 model for feature extraction. The notebook covers data loading, preprocessing, model building, training with mixed precision, and evaluation.

## Key Features ✨

* **Dataset:** Uses the Food-101 dataset loaded via TensorFlow Datasets (TFDS).
* **Preprocessing:** Implements image resizing and data type conversion using `tf.data`.
* **Model:** Employs a feature extraction approach using a pre-trained `EfficientNetB0` base model (frozen).
* **Training:** Leverages `tf.keras` functional API, mixed precision (`mixed_float16`) for potentially faster training on compatible GPUs, and efficient `tf.data` input pipelines.
* **Evaluation:** Measures model performance using accuracy on a validation/test split.

## Technologies Used 💻

* TensorFlow 2.x
* TensorFlow Datasets (TFDS)
* Keras (within TensorFlow)
* Matplotlib (for visualization)
* Python 3

## Dataset 🍔🍕🍎

The project uses the **Food-101** dataset, which consists of 101 food categories, with 1000 images per category (750 for training, 250 for testing). The notebook loads this dataset directly using TFDS.

* **Classes:** 101
* **Training Images:** 75,750
* **Testing/Validation Images:** 25,250

## Workflow ⚙️

1.  **Import Libraries:** Necessary libraries like TensorFlow, TFDS, and Matplotlib are imported.
2.  **Load Data:** The Food-101 dataset is loaded using `tfds.load`, splitting it into training and validation sets. Dataset information (`ds_info`) is also retrieved.
3.  **Explore Data:** Basic exploration is done to understand image shapes, data types, class names, and visualize a sample image.
4.  **Preprocessing:**
    * A `preprocess_img` function is defined to resize images to (224, 224) and cast the data type to `tf.float32`. *Note: Pixel normalization (dividing by 255) is commented out in the provided notebook; EfficientNet models often handle scaling implicitly or require input in the [0, 255] range.*
    * This function is mapped to the datasets using `tf.data.Dataset.map`.
5.  **Create Data Pipelines:** Training and test datasets are shuffled (training only), batched, and prefetched for efficient loading using `tf.data`.
6.  **Enable Mixed Precision:** Global policy is set to `mixed_float16` to leverage potential performance gains on GPUs.
7.  **Build Model (Feature Extraction):**
    * `EfficientNetB0` (pre-trained on ImageNet, without the top classification layer) is used as the base model.
    * The base model's layers are frozen (`trainable = False`).
    * A `GlobalAveragePooling2D` layer is added.
    * A final `Dense` output layer with 101 units (one for each class) and a `softmax` activation (set to `float32` for stability with mixed precision) is added.
8.  **Compile Model:** The model is compiled using the Adam optimizer, `sparse_categorical_crossentropy` loss, and the accuracy metric.
9.  **Train Model:** The model is trained for 3 epochs using `model.fit()`, evaluating on a subset of the validation data during training.
10. **Evaluate Model:** The final performance is evaluated on the full test dataset using `model.evaluate()`.

## Model Architecture 🏗️
The model uses the Keras Functional API and employs a feature extraction approach with a pre-trained base model.
Input Layer: Defines the input shape for the images ((224, 224, 3)).
inputs = layers.Input(shape = input_shape, name = 'input_layer')
Base Model (Feature Extractor): Uses EfficientNetB0 pre-trained on ImageNet, with its top classification layer removed (include_top=False).
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
The layers of this base model are frozen (base_model.trainable = False), meaning their weights won't be updated during the initial training phase.
x = base_model(inputs, training=False)
Pooling Layer: A GlobalAveragePooling2D layer is applied to reduce the spatial dimensions of the feature maps coming from the base model.
x = layers.GlobalAveragePooling2D()(x)

Output Layer:
A Dense (fully connected) layer with units equal to the number of food classes (101) acts as the classifier head.
x = layers.Dense(len(class_names))(x)
A softmax activation function is applied to the output of the dense layer to get probability scores for each class. The data type is explicitly set to float32 for stability when using mixed precision training.
output = layers.Activation('softmax', dtype=tf.float32, name='softmax_float32')(x)
Model Definition: The complete model is defined by specifying the input and output layers.
model = tf.keras.Model(inputs, output)

## Results 📊

After training for 3 epochs using feature extraction:

* **Validation Accuracy (during training):** ~72.38%
* **Test Accuracy (final evaluation):** ~72.74%

*(Note: These results are based on only 3 epochs of training with a frozen base model. Further improvements can be achieved through fine-tuning and longer training.)*

## How to Run ▶️

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow tensorflow-datasets matplotlib
    ```
3.  **Run the Jupyter Notebook:**
    Open and run the `Food_101.ipynb` notebook in an environment like Jupyter Lab, Jupyter Notebook, or Google Colab.
    * *Recommendation:* Use a GPU-enabled environment for faster training, especially with mixed precision enabled. TFDS will download the Food-101 dataset on the first run (this may take some time and requires significant disk space).
