#****** IMPORT PACKAGES ********
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
from datetime import datetime
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#****** STREAMLIT APP ********
st.set_page_config(page_title="Stock Prediction App", page_icon="📈", layout="wide")

# Add a sidebar with navigation options
st.sidebar.title("Stock Prediction Navigation")
sidebar_option = st.sidebar.radio("Choose an option", ["Stock Info", "Predict", "Charts & Stats"])

st.title("Stock Prediction App")
st.markdown("""
    This app predicts stock prices using different machine learning models such as ARIMA, Linear Regression, and LSTM. 
    Enter the stock symbol to explore historical data, charts, and forecasts.
""")

#****** FUNCTIONS TO FETCH DATA *********
def get_historical(quote):
    end = datetime.now()
    start = datetime(end.year - 2, end.month, end.day)
    data = yf.download(quote, start=start, end=end)
    df = pd.DataFrame(data=data)
    return df

#******* ARIMA SECTION *******

def ARIMA_ALGO(df):
    df = df.set_index('Date')
    df['Price'] = df['Close']
    df = df.fillna(df.bfill())
    quantity = df['Price'].values
    size = int(len(quantity) * 0.80)
    train, test = quantity[0:size], quantity[size:len(quantity)]
    predictions = []

    history = [x for x in train]
    for t in range(len(test)):
        model = ARIMA(history, order=(6, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        history.append(test[t])

    # Forecast next 7 days
    model = ARIMA(history, order=(6, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=7)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(test,color ='blue', label='Actual Price')
    plt.plot(predictions, color = 'red', label='Predicted Price')
    plt.legend()
    st.pyplot(plt)

    # Calculate error
    error_arima = math.sqrt(mean_squared_error(test, predictions))
    return forecast, error_arima

#****** LINEAR REGRESSION SECTION *******

def LIN_REG_ALGO(df):
    forecast_out = 7
    df['Close after n days'] = df['Close'].shift(-forecast_out)
    df = df.dropna()

    y = np.array(df['Close after n days'])
    X = np.array(df['Close']).reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train = X[:int(0.8 * len(X))]
    X_test = X[int(0.8 * len(X)):]
    y_train = y[:int(0.8 * len(y))]
    y_test = y[int(0.8 * len(y)):]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Forecast next 7 days
    forecast = model.predict(X[-7:])

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, color = 'blue', label='Actual Price')
    plt.plot(y_pred, color = 'red', label='Predicted Price')
    plt.legend()
    st.pyplot(plt)

    error_lr = math.sqrt(mean_squared_error(y_test, y_pred))
    return forecast, error_lr

#****** LSTM SECTION ********

def LSTM_ALGO(ticker):
    # Fetch the stock data from Yahoo Finance
    start_date =  pd.to_datetime('2010-01-01')
    end_date =  pd.to_datetime(datetime.now())
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Preprocess the data
    close_prices = data['Close'].values
    close_prices = close_prices.reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices_scaled = scaler.fit_transform(close_prices)
    
    # Prepare the data into sequences
    def create_dataset(data, time_step):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60  # Time step for LSTM (can adjust based on your preference)
    X, y = create_dataset(close_prices_scaled, time_step)

    # Reshape X to be 3D for LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train-Test Split (80-20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.3))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Make predictions on the test data
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Reverse normalization

    # Reverse scaling of actual stock prices for comparison
    y_test = y_test.reshape(-1, 1)
    y_test = scaler.inverse_transform(y_test)

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, color='blue', label='Actual Stock Price')
    plt.plot(predictions, color='red', label='Predicted Stock Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot(plt)

    # Forecast the next 7 days
    last_data = close_prices_scaled[-time_step:]  # Last 60 days (time_step) data
    forecast = []

    for i in range(7):  # Forecast the next 7 days
        last_data = last_data.reshape((1, time_step, 1))
        pred_price = model.predict(last_data)
        forecast.append(pred_price[0, 0])
        last_data = np.append(last_data[:, 1:, :], pred_price)  # Append the predicted price for the next day

    # Inverse transform the forecasted prices
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    # Generate the dates for the forecasted values
    forecast_dates = pd.date_range(start=data.index[-1], periods=8, freq='D')[1:]

    # Create a DataFrame for the forecasted values
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Predicted Price': forecast.flatten()
    })

    # Display forecast table
    st.write("7-Day Stock Price Forecast (LSTM):")
    st.table(forecast_df)
    error_lstm = math.sqrt(mean_squared_error(y_test, predictions))
    st.write(f"*Root Mean Square Error (RMSE):* {error_lstm:.2f}")
    return forecast, error_lstm

#****** Helper function to display forecast results *****
def display_forecast_results(forecast, model_name, error):
    st.write(f"### {model_name} Model 7-Day Forecast")
    forecast_df = pd.DataFrame({'Day': [f'Day {i+1}' for i in range(7)], 'Predicted Price': forecast})
    st.table(forecast_df)
    st.write(f"*Root Mean Square Error (RMSE):* {error:.2f}")

#****** APP FUNCTIONALITY ******

# Sidebar Inputs
quote = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
st.sidebar.markdown("### Select Models to Predict")
predict_arima = st.sidebar.checkbox("ARIMA")
predict_lr = st.sidebar.checkbox("Linear Regression")
predict_lstm = st.sidebar.checkbox("LSTM")

if sidebar_option == "Stock Info":
    # Show stock information and statistics
    if st.button("Show Stock Info and Stats"):
        try:
            df = get_historical(quote)
            
            # Display basic stock information and statistics
            st.subheader(f"Stock Information for {quote}")
            st.write(df.describe())
            
            # Plot Closing Price Over Time
            st.line_chart(df['Close'])
            
            # Moving Average (7-day and 30-day)
            df['7-day MA'] = df['Close'].rolling(window=7).mean()
            df['30-day MA'] = df['Close'].rolling(window=30).mean()

            plt.figure(figsize=(10, 6))
            plt.plot(df['Close'], label='Closing Price', color='blue')
            plt.plot(df['7-day MA'], label='7-Day Moving Average', color='orange')
            plt.plot(df['30-day MA'], label='30-Day Moving Average', color='green')
            plt.title(f'{quote} - Closing Price and Moving Averages')
            plt.legend()
            st.pyplot(plt)
            
        except Exception as e:
            st.error(f"Error fetching data for symbol {quote}: {e}")

elif sidebar_option == "Predict":
    if st.button("Run Predictions"):
        try:
            df = get_historical(quote)
            df.reset_index(inplace=True)

            if predict_arima:
                st.write("### ARIMA Model Prediction")
                arima_forecast, error_arima = ARIMA_ALGO(df)
                display_forecast_results(arima_forecast, "ARIMA", error_arima)

            if predict_lr:
                st.write("### Linear Regression Prediction")
                lr_forecast, error_lr = LIN_REG_ALGO(df)
                display_forecast_results(lr_forecast, "Linear Regression", error_lr)

            if predict_lstm:
                st.write("### LSTM Model Prediction")
                lstm_forecast, error_lstm = LSTM_ALGO(quote)
                # display_forecast_results(lstm_forecast, "LSTM", error_lstm)

        except Exception as e:
            st.error(f"Error fetching data for symbol {quote}: {e}")

elif sidebar_option == "Charts & Stats":
    # Display additional charts and statistics
    try:
        df = get_historical(quote)

        # **Additional Functionality**

        # 1. Pie chart showing the percentage of days with positive and negative returns
        df['Daily Return'] = df['Close'].pct_change()
        positive_days = (df['Daily Return'] > 0).sum()
        negative_days = (df['Daily Return'] < 0).sum()

        pie_data = [positive_days, negative_days]
        pie_labels = ['Positive Returns', 'Negative Returns']
        
        plt.figure(figsize=(7, 7))
        plt.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
        plt.title(f'{quote} - Positive vs Negative Days')
        st.pyplot(plt)

        # 2. Histogram of Closing Prices
        plt.figure(figsize=(10, 6))
        plt.hist(df['Close'], bins=50, color='blue', alpha=0.7)
        plt.title(f'{quote} - Distribution of Closing Prices')
        plt.xlabel('Closing Price')
        plt.ylabel('Frequency')
        st.pyplot(plt)

        # 3. Volatility (Standard Deviation) Plot
        df['Volatility'] = df['Close'].rolling(window=30).std()
        plt.figure(figsize=(10, 6))
        plt.plot(df['Volatility'], label='30-Day Volatility', color='purple')
        plt.title(f'{quote} - Stock Volatility')
        plt.legend()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error fetching data for symbol {quote}: {e}")
