import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Holter BP AI Analytics", layout="wide")
st.title("ABPM Analysis & Prediction")
st.markdown("Process real-world medical time-series data project by applying Machine Learning.")

@st.cache_data
def load_and_process_data(file):
    df = pd.read_csv(file)
    
    times = pd.to_datetime(df['Time'], format='%H:%M')
    day_offset = (times.diff() < pd.Timedelta(0)).cumsum()
    df['Timestamp'] = pd.to_datetime('2026-07-12') + pd.to_timedelta(day_offset, unit='D') + (times - pd.to_datetime('1900-01-01'))
    
    df = df.set_index('Timestamp').sort_index()
    #df.index = df.index.strftime('%H:%M')
    
    cols = ['Sys', 'Dia', 'HR', 'MAP']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df[cols].resample('20min').mean().interpolate(method='linear')
    
    df['PP'] = df['Sys'] - df['Dia'] # Pulse Pressure
    df['Hour'] = df.index.hour
    df['Period'] = df['Hour'].apply(lambda h: 'Night' if (h >= 22 or h < 6) else 'Day')
    
    # Cyclical Encoding
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)
    
    # Lag features
    df['Sys_lag1'] = df['Sys'].shift(1)
    df['Sys_lag2'] = df['Sys'].shift(2)
    df['HR_lag1'] = df['HR'].shift(1)
    df['MAP_lag1'] = df['MAP'].shift(1)

    with st.expander("Input data"):
        st.dataframe(df)    
    return df

DEFAULT_FILE = "holter-bp.csv" 
df = load_and_process_data(DEFAULT_FILE)

#1 Cardiovascular metrics
st.header("1. Cardiovascular Metrics Overview")
day_df = df[df['Period'] == 'Day']
night_df = df[df['Period'] == 'Night']

sys_mean = df['Sys'].mean()
dia_mean = df['Dia'].mean()
dipping = ((day_df['Sys'].mean() - night_df['Sys'].mean()) / day_df['Sys'].mean()) * 100 if len(night_df) > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Systolic", f"{sys_mean:.1f} mmHg", delta=f"{sys_mean - 130:.1f} versus 130mmHg threshold", delta_color="inverse")
c2.metric("Diastolic", f"{dia_mean:.1f} mmHg", delta=f"{dia_mean - 80:.1f} versus 80mmHg threshold", delta_color="inverse")
c3.metric("Heart Rate", f"{df['HR'].mean():.1f} bpm")
c4.metric("Noctural Dipping", f"{dipping:.1f}%", delta="Standard: 10% - 20%", delta_color="normal" if 10 <= dipping <= 20 else "off")

#2 Anomaly Detection
st.header("2. Anomaly Detection")
iso = IsolationForest(contamination=0.05, random_state=42)
df['Anomaly'] = iso.fit_predict(df[['Sys', 'Dia', 'HR', 'PP']])
anomalies = df[df['Anomaly'] == -1]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.index, df['Sys'], label='Systolic', color='#1f77b4', linewidth=2)
ax.plot(df.index, df['Dia'], label='Diastolic', color='#2ca02c', linewidth=1.5)
ax.scatter(anomalies.index, anomalies['Sys'], color='red', s=80, zorder=5, label='Anomaly Point')
ax.axhline(135, color='orange', linestyle='--', alpha=0.7, label='Sys Threshold(135)')
ax.set_ylabel("mmHg")
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
st.pyplot(fig)

with st.expander("Anomaly Detail"):
    st.dataframe(anomalies[['Sys', 'Dia', 'HR', 'MAP', 'PP', 'Period']])

#3 BP Forecasting 
st.header("3. Blood Pressure Forecasting in the next 20 minutes using RandomForestRegressor")
ml_df = df.dropna().copy() 

features = ['Sys_lag1', 'Sys_lag2', 'HR_lag1', 'MAP_lag1', 'Hour_sin', 'Hour_cos']

split_idx = int(len(ml_df) * 0.8)
X_train, X_test = ml_df[features].iloc[:split_idx], ml_df[features].iloc[split_idx:]
y_train, y_test = ml_df['Sys'].iloc[:split_idx], ml_df['Sys'].iloc[split_idx:]

model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

col_left, col_right = st.columns([1, 2])
with col_left:
    st.info(f"**Model Accuracy (Test Set):**\n\nMean Absolute Error (**MAE**): **{mae:.2f} mmHg**")

with col_right:    
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.plot(y_test.index, y_test.values, label='Actual', marker='o', color='black')
    ax2.plot(y_test.index, preds, label='Predicted', marker='x', linestyle='--', color='red')
    ax2.set_title("Validate Prediction by the last 20% of dataset")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    st.pyplot(fig2)

st.divider()

# --- Forecasting with linear model ----
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)
split_idx = int(len(ml_df) * 0.8)
X_train, X_test = ml_df[features].iloc[:split_idx], ml_df[features].iloc[split_idx:]
y_train, y_test = ml_df['Sys'].iloc[:split_idx], ml_df['Sys'].iloc[split_idx:]

st.header("Blood Pressure Forecasting in the next 20 minutes using Linear Regression")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predict = lr_model.predict(X_test)
mae2 = mean_absolute_error(y_test, lr_predict)

c_left, c_right = st.columns([1, 2])
with c_left:
    st.info(f"**Model Accuracy (Test Set):**\n\nMean Absolute Error (**MAE**): **{mae2:.2f} mmHg**")    

with c_right:    
    fig3, ax3 = plt.subplots(figsize=(8, 3.5))
    ax3.plot(y_test.index, y_test.values, label='Actual', marker='o', color='black')
    ax3.plot(y_test.index, lr_predict, label='Predicted', marker='x', linestyle='--', color='red')
    ax3.set_title("Validate Prediction by the last 20% of dataset")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    st.pyplot(fig3)


