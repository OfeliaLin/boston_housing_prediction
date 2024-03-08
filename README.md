# Boston Housing Price Prediction with XGBoost

This repository contains code for predicting Boston housing prices using the XGBoost algorithm. The project utilizes Python libraries such as Pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, and SHAP (SHapley Additive exPlanations).

## Introduction

The Boston housing dataset is a well-known dataset in the field of machine learning and statistics. It contains various features such as crime rate, nitric oxides concentration, number of rooms, etc., to predict the median value of owner-occupied homes.

## Dataset

The dataset consists of two files: `train.csv` and `test.csv`. The training data contains both features and labels, while the test data only contains features. 

## Usage

1. **Data Preprocessing**: The `data_preprocess()` function reads the data and preprocesses it according to the specified mode (Train/Test). Features are selected, and labels are separated for the training mode. 

2. **One-Hot Encoding**: Categorical features like 'chas' and 'rad' are one-hot encoded using the `one_hot_encoding()` function.

3. **Binning**: The `binning()` function is used to bin numerical features like 'zn' into discrete intervals.

4. **Normalization**: Features are normalized using MinMaxScaler from scikit-learn to bring them within a similar scale.

5. **Model Training**: XGBoost regressor is trained using GridSearchCV to find the best hyperparameters. Cross-validation is utilized for model evaluation.

6. **Model Evaluation**: The best model is evaluated on a validation set using Root Mean Square Error (RMSE) and Accuracy (R-squared score).

7. **Feature Importance**: SHAP library is employed to visualize feature importance using SHAP values.

## Results

The best model achieved an RMSE of `RMSE_value` and an accuracy of `accuracy_value`.

## Conclusion

This project demonstrates the use of XGBoost for predicting Boston housing prices. Further improvements and experimentation can be done by tuning hyperparameters, trying different algorithms, or exploring feature engineering techniques.

## Requirements

- Python 3.x
- Pandas
- NumPy
- scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- SHAP

## Author

Ofelia Lin

## License

This project is licensed under the [MIT License](LICENSE).
