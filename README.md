# House Price Prediction using Machine Learning

## Project Overview
This project predicts house prices based on key features such as the average number of rooms, percentage of lower status population, and pupil-teacher ratio. It is a simple regression problem using the Boston Housing Dataset and a Linear Regression model.

## Dataset
- **Source:** [Boston Housing Dataset (CSV)](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv)  
- **Features Used:**
  - `RM` – Average number of rooms per dwelling
  - `LSTAT` – Percentage of lower status population
  - `PTRATIO` – Pupil-teacher ratio by town
- **Target Variable:** `MEDV` – Median value of homes in $1000s

## Tech Stack
- **Python** – Programming language
- **Pandas** – Data loading and preprocessing
- **Matplotlib** – Visualization of predictions
- **Scikit-learn** – Machine learning model and evaluation

## Steps Performed
1. Loaded and explored the dataset using Pandas.
2. Selected key features for prediction.
3. Split the dataset into training and testing sets (80% train, 20% test).
4. Trained a Linear Regression model using Scikit-learn.
5. Predicted house prices on the test set.
6. Evaluated the model using Mean Squared Error (MSE) and R² Score.
7. Visualized **Actual vs Predicted Prices** using Matplotlib.

## Model Performance
- **Mean Squared Error (MSE):** ~24.3  
- **R² Score:** ~0.71  
> The model performs reasonably well for a beginner project and captures the relationship between features and house prices.
