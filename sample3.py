# stock_prediction_app.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# --- PAGE NAVIGATION FUNCTIONS ---
def show_signup():
    st.title("Sign Up")
    with st.form(key="signup_form"):
        username = st.text_input("Create Username")
        password = st.text_input("Create Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        signup_button = st.form_submit_button("Sign Up")
        if signup_button:
            if password == confirm_password:
                st.success("Signup successful! Please login.")
                st.session_state['users'][username] = password
            else:
                st.error("Passwords do not match!")

def show_login():
    st.title("Login")
    with st.form(key="login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")
        if login_button:
            if username in st.session_state['users'] and st.session_state['users'][username] == password:
                st.session_state['logged_in'] = True
                st.session_state['current_user'] = username
                st.success(f"Welcome back, {username}!")
            else:
                st.error("Invalid username or password!")

def show_home():
    st.title("Home Page")
    st.write("Welcome to the Stock Prediction App! You can predict stock prices, view charts, and generate buy/sell signals.")
    
    # Display stock image
    st.image("C:\\ML project cat-2\\stock image.jpg", 
             caption="Stock Market", use_column_width=True)

    # Display introduction content
    st.subheader("Introduction")
    st.write("""
        This application allows you to analyze stock data using various deep learning models such as LSTM. 
        You can visualize stock trends with candlestick charts and generate buy/sell signals based on moving averages.
    """)
    st.write("Use the navigation menu to start predicting stock prices or view additional features!")

def show_introduction():
    st.title("Introduction")
    st.write("""
        The Stock Prediction App is designed to help you predict stock prices using advanced machine learning techniques. 
        You can choose the LSTM model for stock price forecasting.
        
        Additionally, this app provides visualizations in the form of candlestick charts and offers buy/sell signals based on 
        moving average strategies to aid your trading decisions. 
    """)

def show_feedback():
    st.title("Feedback Page")
    with st.form(key="feedback_form"):
        feedback = st.text_area("Your Feedback", placeholder="Write your feedback here...")
        feedback_button = st.form_submit_button("Submit Feedback")
        if feedback_button:
            st.success("Thank you for your feedback!")

# Preprocess the stock data
def preprocess_data(stock_data, seq_length=60):
    stock_data = stock_data[['Open', 'High', 'Low', 'Close']].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data_scaled = scaler.fit_transform(stock_data)
    
    def create_sequences(data, seq_length):
        x, y = [], []
        for i in range(len(data) - seq_length):
            x.append(data[i:i + seq_length])
            y.append(data[i + seq_length, 3])  # Predict the 'Close' price
        return np.array(x), np.array(y)
    
    x_data, y_data = create_sequences(stock_data_scaled, seq_length)
    return x_data, y_data, scaler

# Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train model
def train_model(model, x_train, y_train, epochs=10, batch_size=32):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

# Make predictions and inverse the scaling
def predict_and_inverse_transform(model, x_test, scaler):
    predictions = model.predict(x_test)
    predictions_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], 3)), predictions), axis=1))[:, 3]
    return predictions_rescaled

# Generate Buy/Sell signals using Moving Averages
def buy_sell_signals(data):
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    buy_signal = []
    sell_signal = []
    for i in range(len(data)):
        if data['SMA50'][i] > data['SMA200'][i]:  # Buy signal
            buy_signal.append(data['Close'][i])
            sell_signal.append(np.nan)
        elif data['SMA50'][i] < data['SMA200'][i]:  # Sell signal
            buy_signal.append(np.nan)
            sell_signal.append(data['Close'][i])
        else:
            buy_signal.append(np.nan)
            sell_signal.append(np.nan)
    return buy_signal, sell_signal

# Perform EDA
def perform_eda(stock_data):
    st.subheader("Exploratory Data Analysis (EDA)")
    
    st.write("## Summary Statistics")
    st.write(stock_data.describe())

    st.write("## Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(stock_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)

    st.write("## Distribution of Stock Prices")
    plt.figure(figsize=(10, 6))
    for col in ['Open', 'High', 'Low', 'Close']:
        sns.histplot(stock_data[col], kde=True, label=col)
    plt.legend()
    st.pyplot(plt)

# --- MAIN APP LOGIC ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'users' not in st.session_state:
    st.session_state['users'] = {}

if not st.session_state['logged_in']:
    choice = st.sidebar.selectbox("Navigation", ["Home", "Login", "Sign Up"])
    if choice == "Home":
        show_home()
    elif choice == "Login":
        show_login()
    elif choice == "Sign Up":
        show_signup()
else:
    # Horizontal navigation in the Home page
    pages = ["Home", "Introduction", "Predict Stock Prices", "Exploratory Data Analysis (EDA)", "Feedback", "Logout"]
    choice = st.radio("Navigation", pages, horizontal=True)  # Horizontal radio button for navigation

    if choice == "Home":
        show_home()
    elif choice == "Introduction":
        show_introduction()
    elif choice == "Predict Stock Prices":
        st.title("Stock Price Prediction App with Candlestick Analysis")

        # User input for stock ticker
        ticker = st.text_input("Enter stock ticker:", value="AAPL")  # Take input from the user

        # Time period selection
        time_period = st.selectbox("Choose Time Period", ["1d", "1h", "5d", "1mo", "6mo", "1y"])

        model_choice = "LSTM"  # Only LSTM is available now

        # Fetch stock data using yfinance based on selected time period
        if st.button("Predict"):
            if time_period == "1d":
                stock_data = yf.download(ticker, period="1d", interval="1m")
            elif time_period == "1h":
                stock_data = yf.download(ticker, period="1d", interval="1h")
            elif time_period == "5d":
                stock_data = yf.download(ticker, period="5d", interval="30m")
            elif time_period == "1mo":
                stock_data = yf.download(ticker, period="1mo", interval="1d")
            elif time_period == "6mo":
                stock_data = yf.download(ticker, period="6mo", interval="1d")
            elif time_period == "1y":
                stock_data = yf.download(ticker, period="1y", interval="1d")

            # Plot candlestick chart using plotly
            fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                        open=stock_data['Open'],
                        high=stock_data['High'],
                        low=stock_data['Low'],
                        close=stock_data['Close'])])

            fig.update_layout(title=ticker + ' Candlestick Chart',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            xaxis_rangeslider_visible=False)
            st.plotly_chart(fig)

            st.subheader("Generating Buy/Sell Signals Based on Moving Averages")
            stock_data['Buy'], stock_data['Sell'] = buy_sell_signals(stock_data)

            # Plot the signals
            plt.figure(figsize=(10, 6))
            plt.plot(stock_data['Close'], label='Close Price', alpha=0.5)
            plt.plot(stock_data['Buy'], marker='^', color='g', label='Buy Signal', alpha=1)
            plt.plot(stock_data['Sell'], marker='v', color='r', label='Sell Signal', alpha=1)
            plt.title(f'{ticker} Buy/Sell Signals')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.legend()
            st.pyplot(plt)

            # Predicting stock prices using LSTM
            if model_choice == "LSTM":
                seq_length = 60  # Sequence length
                x_data, y_data, scaler = preprocess_data(stock_data, seq_length)

                # Split data into training and test sets
                train_size = int(len(x_data) * 0.8)
                x_train, y_train = x_data[:train_size], y_data[:train_size]
                x_test, y_test = x_data[train_size:], y_data[train_size:]

                model = build_lstm_model((x_train.shape[1], x_train.shape[2]))
                model = train_model(model, x_train, y_train, epochs=10, batch_size=32)

                # Predict and plot
                predictions_rescaled = predict_and_inverse_transform(model, x_test, scaler)

                st.subheader(f'{ticker} Stock Price Prediction with LSTM')
                plt.figure(figsize=(10, 6))
                plt.plot(stock_data.index[train_size + seq_length:], stock_data['Close'][train_size + seq_length:], color='blue', label='Actual Stock Price')
                plt.plot(stock_data.index[train_size + seq_length:], predictions_rescaled, color='orange', label='Predicted Stock Price')
                plt.title(f'{ticker} Stock Price Prediction')
                plt.xlabel('Date')
                plt.ylabel('Stock Price')
                plt.legend()
                st.pyplot(plt)

    elif choice == "Exploratory Data Analysis (EDA)":
        # User input for stock ticker
        ticker = st.text_input("Enter stock ticker:", value="AAPL")

        if st.button("Perform EDA"):
            stock_data = yf.download(ticker, period="1y", interval="1d")
            perform_eda(stock_data)

    elif choice == "Feedback":
        show_feedback()

    elif choice == "Logout":
        st.session_state['logged_in'] = False
        st.success("You have been logged out.")
