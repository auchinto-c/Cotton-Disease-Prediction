# Cotton-Disease-Prediction

## Introduction

This project aims to classify images of cotton plant and cotton plant leaves into one of 4 categories:
1. Diseased Cotton Plant
2. Diseased Cotton Leaf
3. Fresh Cotton Plant
4. Fresh Cotton Leaf

This is essentially a **Multi-category classification** problem. We are dealing with this objective using a pre-designed CNN architecture, ResNet50, made out of 50 layers. This architecture is available as part of Keras applications.

<br><hr>

## Credits

|              | Details |
|--------------|--------------------------------------------------------------------------------------------------------------------|
| **Tutorial** | [Krish Naik - Cotton Disease Prediction](https://www.youtube.com/watch?v=-vDwY1kOfNw)                              |
| **Code**     | [Krish Naik - Cotton-Disease-Prediction](https://github.com/krishnaik06/Cotton-Disease-Prediction-Deep-Learning)   |
| **Dataset**  | [Drive - Collected by Akash Zade](https://drive.google.com/drive/folders/1vdr9CC9ChYVW2iXp6PlfyMOGD-4Um1ue)        |
| **Dataset Owner**  | [LinkedIn - Akash Zade](https://www.linkedin.com/in/ai-engineer-az/)                                         |

<hr>

## Brief

|                   | **Details**                  |
|-------------------|------------------------------|
| **Training Set**  | 1951 images across 4 classes |
| **Test Set**      | 18 images across 4 classes   |
|                   |                              |
| **Version**       | **v1**                       |
| **Model**         | ResNet50                     |
| **Training Time** | 1052s                        |
| **Epochs**        | 20                           |
|                   |                              |

<hr>

## Procedure

1. **Model Initialisation** - Initialise ResNet50 architecture, which is a custom CNN with 50 layers
   - Image size in the dataset is (224, 224), and given that the images have 3 channels, RGB, we will consider the input layer as (224, 224, 3).
   - Use `imagenet` weights to use its pretrained weights
   - We are not going to include the top layer from the ResNet, instead we are providing our own input layer.
   - We are also removing the last layer, since we will be giving our own output layer of 4 nodes, representing each of the categories.
   - We are choosing activation function as `softmax` since we have multiple categories. Softmax gives the probability of each of the category, unlike `sigmoid` which can be used for binary classification.
   - Choose the cost, optimization method and metric.
     - Cost: Categorical Cross Entropy
     - Optimizer: Adam
     - Metrics: Accuracy
2. **Data Augmentation**
      - Initialise Image Data Generators for Training and Test data
      - Create the train and test set from these image data generators.
3. **Fit the model**
   - Epochs: 20
   - Steps per epoch: length of training set
   - Validation steps: length of test set
4. **Visualizations**
   - Comparison of Train Loss and Validation Loss
   - Comparison of Train Accuracy and Validation Accuracy
5. **Predictions**
   - Get the probabilities of each of the classes for each test image
   - Capture the class with the maximum probability for a given test image
6. **Save the model**

<br><hr>

## Local Setup

1. Clone the repository to your local system.
2. Run the `cotton-price-prediction.ipynb` file on Jupyter notebook or Google Colab, to get the script of the model development and training.

### System Specifications used for Training
- This model has been implemented and trained in a Google Colab notebook, enabled with GPU runtime. The general specifications for Google Colab are as follows:
  - GPU : Nvidia K80/ T4
  - GPU Memory : 12GB / 16GB
  - No. of CPU Cores : 2
  - Available RAM : 12GB

<br><hr>

## Library and Tools

### Language
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

### Data Analysis
![numpy](https://img.shields.io/badge/numpy-%23548ecc.svg?style=for-the-badge&logo=numpy&logoColor=white)

### Model Development
![tensorflow-keras](https://img.shields.io/badge/tensorflow-keras-%23cc8854.svg?style=for-the-badge&logo=tensorflow&logoColor=white)

### Visualizations
![matplotlib](https://img.shields.io/badge/matplotlib-%230e4e5e.svg?style=for-the-badge&logoColor=white)

### Development Tools
![google-colab](https://img.shields.io/badge/google-colab-%23e38330.svg?style=for-the-badge&logo=google-colab&logoColor=white)

<br><hr>

## Results

### Training Loss v/s Validation Loss
| **Model** | **Training Loss** | **Validation Loss** |
|-----------|-------------------|---------------------|
| ResNet50  | 0.7112            | 0.6557              |

We observe a steady fall in the Training loss through the beginning epochs, followed by gradual decline in the loss through the rest of the epochs. While Validation loss remains on the bottom end of the plot through the epochs. 

![img](https://github.com/auchinto-c/Cotton-Disease-Prediction/blob/main/Visualizations/loss_and_val_loss.png)

### Training Accuracy v/s Validation Accuracy
| **Model** | **Training Accuracy** | **Validation Accuracy** |
|-----------|-----------------------|-------------------------|
| ResNet50  | 0.7458                | 0.7222                  |

We observe a steep rise in the Training accuracy through the beginning epochs, followed by steady rise through the rest of the epochs. While Validation accuracy remains on the upper portion of the plot through the epochs.

![img](https://github.com/auchinto-c/Cotton-Disease-Prediction/blob/main/Visualizations/accuracy_and_val_accuracy.png)

<br><hr>

## Future Scope

With ResNet50, which has around 25.6M parameters, we are still at around `74.5%` Training accuracy with `20 epochs over 1951 training images`.
1. We can explore the effects on accuracy if we increase the number of epochs over the given data set.
2. We can explore the effects on accuracy and training time, if we use an architecture which has high number of parameters as that of ResNet50. For instance, `ResNet152V2 with 60.4M parameters`.
3. We can explore the effects on accuracy and training time, if we use an architecture with almost similar number of parameters as ResNet50. For instance, `InceptionV3 with 23.9M parameters`.
4. We can explore the effects on accuracy and training time, if we use an architecture with lesser number of parameters as that of ResNet50. For instance, `MobileNetV2 with 3.5M parameters`.
5. The results of these investigations can be compared to provide insights on the trade-off between accuracy and training time.
6. We can also explore solving this problem statement using a generic CNN, and observe the results in terms of accuracies and training time.

<hr>
