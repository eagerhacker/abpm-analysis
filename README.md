# 24h Blood Pressure Monitoring & AI Early Warning

An interactive web app to analyze 24-hour ambulatory blood pressure (ABPM) data, detect physiological anomalies, and forecast systolic blood pressure (`Sys`) 20 minutes ahead.

## Key Features
* **Time-Series Processing:** Aligns midnight transitions, resamples to 20-min intervals, and extracts hemodynamic features (`MAP`, `PP`, lag features, and cyclical `Sin/Cos` time).
* **Anomaly Detection:** Uses **Isolation Forest** to flag abnormal blood pressure and heart rate spikes.
* **Short-Term Forecasting:** Predicts the next 20-minute `Sys` reading using **Random Forest Regressor** and **Linear Regression**(chronological 80/20 split).
