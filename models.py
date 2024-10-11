import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import pandas as pd
import yfinance as yf
import numpy as np
import io
import base64
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def stock_prediction_and_analysis(ticker):
    filtered_df = pd.read_csv(r'C:\Users\sathw\OneDrive\Desktop\M\Sentiment_Analysis\data\headlines_sentiment.csv')
    filtered_df['date'] = pd.to_datetime(filtered_df['date'])
    
    stock_data = yf.download(ticker, start="2017-01-01", end="2020-06-05")
    stock_data.reset_index(inplace=True)
    stock_data.rename(columns={'Date': 'date'}, inplace=True)

    final_df = stock_data.merge(filtered_df[['date', 'sentiment']], on='date', how='left')
    
    final_df['Lag_1'] = final_df['Close'].shift(1)
    final_df.dropna(inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_df[['Close', 'Lag_1', 'Open', 'High', 'Low', 'Volume', 'sentiment']])
    
    X = []
    y = []
    look_back = 60
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i, 0])  

    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, batch_size=32, epochs=30, verbose=0)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], 6)))))[:, 0]

    y_test_scaled = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 6)))))[:, 0]

    small_input = X_test[-1].reshape(1, 60, 7)[0]
    prediction = []

    for _ in range(7):  
        small_input = small_input.reshape(1, 60, 7)
 
        next_day_pred = model.predict(small_input)

        next_day_pred_inverse = scaler.inverse_transform(np.hstack((next_day_pred.reshape(-1, 1), np.zeros((next_day_pred.shape[0], 6)))))[:, 0]
        prediction.append(next_day_pred_inverse[0])

        new_row = np.hstack((next_day_pred_inverse, small_input[0][-1, 1:]))  
        small_input = np.vstack((small_input[0][1:], new_row)) 

    decision = []
    for i in range(1, len(prediction)):
        percent_change = (prediction[i] - prediction[i-1]) / prediction[i-1] * 100
        if percent_change > 2:
            decision.append('buy')
        elif percent_change < -2:
            decision.append('sell')
        else:
            decision.append('hold')

    buy_percentage = decision.count('buy') / len(decision) * 100
    sell_percentage = decision.count('sell') / len(decision) * 100
    hold_percentage = decision.count('hold') / len(decision) * 100

    plt.figure(figsize=(10, 5))
    plt.plot(y_test_scaled, label='Actual Stock Prices')
    plt.plot(predictions, label='Predicted Stock Prices')
    plt.legend()
    plt.title(f'Comparing Stock Price Prediction with Actual for {ticker}')
    plt.xlabel('Test Data Points')
    plt.ylabel('Stock Price (Close)')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()  

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(y_test_scaled), len(y_test_scaled) + 7), prediction, marker='o', color='orange', label='Predicted Prices for Next 7 Days')
    plt.title(f'Predicted Stock Prices for Next 7 Days for {ticker}')
    plt.xlabel('test data points')
    plt.ylabel('Stock Price (Close)')
    plt.legend()

    img_next_7_days = io.BytesIO()
    plt.savefig(img_next_7_days, format='png')
    img_next_7_days.seek(0)
    next_7_days_graph_url = base64.b64encode(img_next_7_days.getvalue()).decode()
    plt.close() 

    return graph_url, predictions[-1], next_7_days_graph_url, buy_percentage, sell_percentage, hold_percentage, decision # Return both graphs and the final prediction