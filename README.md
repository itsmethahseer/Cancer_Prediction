# Lung Cancer Prediction Web Application

This is a web application built with FastApi that allows users to predict the likelihood of lung cancer based on various input features. The application utilizes a machine learning model trained on a dataset of survey responses related to lung cancer.
![Cancer Predictor App](/Cancer Prdictor.png)

## Installation

1. Clone the repository:


2. Install the required Python packages:


3. Place the `cancer_model.pkl` file containing the trained machine learning model in the root directory of the project.

## Usage

1. Run the fastapi development server:


2. Access the web application in your web browser at `http://localhost:8000/`.

3. Fill in the input form with the required information (e.g., gender, age, smoking habits, etc.).

4. Click the "Predict" button to see the prediction result, which indicates the likelihood of lung cancer.

## Project Structure

- `Cancer_Prediction`: The main FastApi project directory.

  - `main.py`: Defines the fastapi main functions
  - `templates`: Directory containing the HTML templates for rendering the web pages.
- `cancer_model.pkl`: The trained machine learning model used for prediction.

## Data Preprocessing

The original dataset used for training the model was preprocessed to convert categorical variables into numerical representations using Label Encoding. Additionally, the imbalanced class distribution was addressed using the ADASYN (Adaptive Synthetic Sampling) technique to generate synthetic samples of the minority class.

## Model Training

The machine learning model used for lung cancer prediction is a logistic regression classifier. It was trained on the preprocessed dataset, and its performance was validated using various evaluation metrics.

## Acknowledgments

The Lung Cancer Prediction web application was developed as part of a data science project. The dataset used for training the model was obtained from an open-source data repository.

