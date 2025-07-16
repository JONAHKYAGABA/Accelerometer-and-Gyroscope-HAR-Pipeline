# Accelerometer & Gyroscope Activity Recognition

This project implements a robust pipeline for binary human activity recognition using inertial sensor data. It processes accelerometer and gyroscope signals to classify short activity windows as either **"Home Activities" (class 0)** or **"Walking Away" (class 1)** using a pre-trained XGBoost classifier.

## 🚀 Features

- Consistent preprocessing across datasets (filtering, windowing, resampling)
- Feature engineering from raw accelerometer and gyroscope signals
- Support for multiple benchmark datasets:
  - WEDA-FALL (10Hz)
  - IPIN2017
  - UCI HAR
  - GeoTecINIT
- Binary classification using a pre-trained XGBoost model
- Evaluation via accuracy, F1 score, confusion matrix, and visual analytics
- Outputs prediction results to CSV for further analysis

## 📁 Datasets

| Dataset     | Label Used          | Class Assignment   |
|-------------|---------------------|--------------------|
| WEDA-FALL   | All activities      | Home Activities (0)|
| IPIN2017    | All walking trials  | Walking Away (1)   |
| UCI HAR     | All activities      | Home Activities (0)|
| GeoTecINIT  | All activities      | Home Activities (0)|

## 🛠️ Setup

```bash
pip install pandas numpy scipy scikit-learn xgboost joblib matplotlib seaborn rarfile
