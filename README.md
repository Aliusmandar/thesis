# Wind Turbine RUL Prediction with Machine Learning

![GitHub Banner](https://user-images.githubusercontent.com/3997099/121780442-799a0e80-cb3b-11eb-9880-a2f0412613b9.png) This project focuses on the predictive maintenance of wind turbines by estimating the Remaining Useful Life (RUL) of their components. The primary goal is to build and evaluate various time-series forecasting models to predict the degradation of a wind turbine's health over time, which is crucial for scheduling maintenance and preventing unexpected failures.

This repository contains the code and analysis for a thesis on predicting the RUL of a wind turbine using vibration data. The project explores feature engineering, the creation of a Health Indicator (HI), and the implementation and comparison of three different forecasting models: **Long Short-Term Memory (LSTM)**, **Autoregressive Integrated Moving Average (ARIMA)**, and **Prophet**.

---

## 📋 Table of Contents
* [Dataset](#-dataset)
* [Methodology](#-methodology)
* [File Descriptions](#-file-descriptions)
* [Results](#-results)
* [How to Run](#-how-to-run)
* [Dependencies](#-dependencies)
* [Acknowledgments](#-acknowledgments)
* [License](#-license)

---

## 💿 Dataset

The project uses the **"Wind Turbine High-Speed Bearing Prognosis"** dataset provided by MathWorks. This dataset contains vibration data collected from a wind turbine, which is used for developing prognostic models. The data is available in `.mat` files.

You can find and download the data from the official repository:
[mathworks/WindTurbineHighSpeedBearingPrognosis-Data](https://github.com/mathworks/WindTurbineHighSpeedBearingPrognosis-Data)

---

## 🛠️ Methodology

### Feature Engineering and Health Indicator (HI)
To effectively monitor the health of the wind turbine, a comprehensive set of features is extracted from the raw vibration signals. These features capture both the time-domain and frequency-domain characteristics of the data.

* **Time-Domain Features**: Mean, Standard Deviation, Skewness, Kurtosis, Peak-to-Peak, RMS, Crest Factor, Shape Factor, Impulse Factor, Margin Factor, and Energy.
* **Frequency-Domain Features**: Spectral Kurtosis (SK) features, including SKMean, SKStd, SKSkewness, and SKKurtosis, are extracted using Short-Time Fourier Transform (STFT).

These features are then used to create a unified **Health Indicator (HI)** using Principal Component Analysis (PCA). The first principal component is selected as the HI, as it captures the most significant variance in the data and represents the overall degradation trend of the turbine.

### Models
Three different time-series forecasting models are implemented and benchmarked to predict the RUL:

1.  **LSTM (Long Short-Term Memory)**: A type of recurrent neural network (RNN) that is well-suited for learning from sequential data.
2.  **ARIMA (Autoregressive Integrated Moving Average)**: A classical statistical model for time-series forecasting.
3.  **Prophet**: A forecasting tool developed by Facebook that is designed to handle time-series data with seasonal patterns.

---

## 📁 File Descriptions

* `P1_Data_Analysis_and_Model_Performance.ipynb`: Covers the initial data analysis, feature extraction, and the creation of the Health Indicator (HI).
* `P2_prediction_intervals_lstm_RUL_Pred.ipynb`: Details the implementation of the LSTM model for RUL prediction and the generation of prediction intervals.
* `P4_benchmark_lstm_arima_prophet.ipynb`: Provides a comparative analysis of the LSTM, ARIMA, and Prophet models.

---

## 📈 Results

The performance of the three models is evaluated using **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, and **R-squared ($R^2$) score**. The results indicate that the Prophet model generally outperforms the LSTM and ARIMA models in predicting the RUL for this dataset, providing more accurate forecasts and tighter prediction intervals.

The notebooks contain detailed visualizations of the model predictions, including the fitted curves on the training data and the forecasted values for the test data, along with 95% prediction intervals.

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Aliusmandar/thesis.git](https://github.com/Aliusmandar/thesis.git)
    cd thesis
    ```
2.  **Download the data:**
    Download the data from [this link](https://github.com/mathworks/WindTurbineHighSpeedBearingPrognosis-Data) and place the `wind_turbine_data` folder in the root of the project directory.

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to create a `requirements.txt` file based on the dependencies listed below.)*

4.  **Run the Jupyter Notebooks:**
    To reproduce the results, run the notebooks in the following order:
    1.  `P1_Data_Analysis_and_Model_Performance.ipynb`
    2.  `P2_prediction_intervals_lstm_RUL_Pred.ipynb`
    3.  `P4_benchmark_lstm_arima_prophet.ipynb`

---

## 📦 Dependencies

The following Python libraries are required to run the notebooks:

* `scipy`
* `numpy`
* `matplotlib`
* `pandas`
* `seaborn`
* `scikit-learn`
* `torch`
* `statsmodels`
* `prophet`
  
## 🙏 Acknowledgments

This work is the result of my thesis project, "Predictive Maintenance for Wind Turbines: RUL Prediction using Time-Series Models," submitted in partial fulfillment of the requirements for the **Master's degree in Risk Analysis, with a specialization in Engineering Risk Analysis and Management**, at the **University of Stavanger, Norway**.

I would like to express my sincere gratitude to my supervisor, and colleagues, for their invaluable guidance, support, and encouragement throughout this research. Their expertise was instrumental in shaping this project.

My special thanks go to **MathWorks** for making the "Wind Turbine High-Speed Bearing Prognosis" dataset publicly available. This resource was fundamental to the analysis and development of the predictive models presented in this work.


You can install them via pip:
```bash
pip install scipy numpy matplotlib pandas seaborn scikit-learn torch statsmodels prophet
