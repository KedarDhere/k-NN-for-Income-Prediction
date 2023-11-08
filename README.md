# k-NN for Income Prediction

This repository contains the implementation of the k-Nearest Neighbors (k-NN) algorithm for the purpose of income prediction. The project employs the k-NN algorithm to predict whether individuals earn more or less than $50,000 per year. We have approached this problem using both a custom implementation of the k-NN algorithm and the k-NN implementation from Scikit-learn.

## Data Preprocessing

The preprocessing of the data is carried out using Pandas and Scikit-learn's ColumnTransformer. The data is prepared with the following steps:

- **Pandas**: For initial data handling and preprocessing.
- **OneHotEncoder**: Both Pandas get_dummies and Scikit-learn's OneHotEncoder are used for converting categorical data into a format that can be provided to the ML algorithms.
- **Scaling**: Numerical data is scaled to standardize the range of independent variables.
- **ColumnTransformer**: This scikit-learn tool combines both passthrough for numerical data and One Hot Encoding for categorical data.

## Feature Extraction

The feature extraction phase is critical as it directly impacts the performance of the k-NN algorithm. We use the aforementioned preprocessing tools to select and transform features that are most relevant to the income prediction task.

## Distance Metrics Learning

In the domain of k-NN, the choice of distance metric can significantly affect the performance. We have explored different distance metrics, focusing on the following:

- **Manhattan Distance**: Also known as L1 norm, this distance measures the absolute differences between coordinates.
- **Euclidean Distance**: Also known as L2 norm, this is the 'ordinary' straight-line distance between two points in Euclidean space.

Understanding and implementing these distance metrics is crucial as they play a key role in the behavior of the k-NN algorithm.

## k-NN Implementation

We have implemented the k-NN algorithm in two ways:

1. **Scikit-learn Implementation**: Utilizing the robust and efficient tools provided by Scikit-learn.
2. **Custom Implementation**: A custom Python class named `knnClassifier` within the `knn_income_prediction.ipynb` notebook, which encapsulates the logic of the k-NN algorithm built from scratch.

Both implementations have been trained on a subset of 5000 data points and tested on a separate set of 1000 data points. The development error for the custom implementation with `k=41` and using the Manhattan distance metric is 14.1%.

## Results

The prediction results are saved in the `income.test.predicted` file. This file contains the income predictions made by the k-NN models.

## Repository Structure

- `knn_income_prediction.ipynb`: Jupyter notebook containing the custom k-NN implementation.
- `income.test.predicted`: Output file containing the income predictions.
- `data`: This directory contains the dataset used for training and testing the models.
- `preprocessing.py`: The Python script with data preprocessing functions.
- `README.md`: The README file providing the project overview and setup instructions.

## Setup Instructions

To run this project, you will need a Python environment with the necessary libraries installed. You can install the dependencies using the following command:

```sh
pip install -r requirements.txt
```
