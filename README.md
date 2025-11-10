# Ensemble Learning for Bike Share Demand Forecasting

*(Based on DA5401 - Assignment 8 by Basavaraj A Naduvinamani, Roll NO: DA25C005)*

## Objective

This project explores the application of ensemble learning techniques (Bagging, Boosting, and Stacking) to solve a complex, non-linear, time-series regression problem. The goal is to accurately predict hourly bike rental demand while understanding how different ensemble approaches manage bias and variance.

## Problem Statement

As a data scientist for a city's bike-sharing program, the task is to accurately forecast the total count of rented bikes (`cnt`) per hour. This is crucial for managing inventory and logistics. The prediction is challenging because demand is highly dependent on non-linear and variable factors like weather, time of day, and seasonal variations.

## Dataset

This model uses the **Bike Sharing Demand Dataset**, which contains over 17,000 hourly observations of bike rentals along with associated temporal and weather features.

## Methodology

The project follows a structured approach from data preprocessing to comparative model evaluation:

1.  **Data Preprocessing:**
    * **Feature Engineering:** Dropped leaky features (`casual`, `registered`) and irrelevant identifiers (`instant`, `dteday`).
    * **Encoding:** Performed One-Hot Encoding on categorical features such as `season`, `weathersit`, `mnth`, `hr`, and `weekday` to make them suitable for regression models.

2.  **Train/Test Split:**
    * The data was split into training (80%) and test (20%) sets using a **time-based split**. This ensures that the model is trained on past data and validated on future data, preventing data leakage and respecting the time-series nature of the problem.

3.  **Model Implementation:**
    * **Baseline:** A **Linear Regression** model was trained as a baseline, achieving an RMSE of **100.446**.
    * **Bagging Regressor:** A `BaggingRegressor` was implemented using Decision Tree base estimators to reduce variance.
    * **Gradient Boosting Regressor:** A `GradientBoostingRegressor` was trained to reduce bias by sequentially building models that correct the errors of the previous ones.
    * **Stacking Regressor:** A final `StackingRegressor` was built to combine the strengths of diverse models. It used **KNeighbors Regressor**, **Bagging Regressor**, and **Gradient Boosting Regressor** as base learners, and a **Ridge Regressor** as the final meta-learner.

## Final Results

All models were evaluated on the held-out test set using Root Mean Squared Error (RMSE). The Stacking Regressor achieved the lowest error, demonstrating the most effective approach.

| Model | RMSE |
| :--- | :--- |
| **Stacking Regressor** | **47.880** |
| Gradient Boosting Regressor | 48.410 |
| Baseline (Linear Regression) | 100.446 |
| Bagging Regressor | 112.390 |

## Conclusion

The **Stacking Regressor** provided the best performance, achieving a final **RMSE of 47.88**.

This success is attributed to its ability to integrate multiple perspectives on the data. While Boosting (Gradient Boosting) significantly reduced bias compared to the baseline, the Stacking model was able to harmonize the predictions from diverse base learners (capturing locality with KNN, stabilizing variance with Bagging, and correcting bias with Boosting) into a single, more accurate, and generalizable forecasting model.
