import streamlit as st
import datetime as date
import yfinance as yf
import prophet
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as grph


start= '2014-01-01'
end = date.date.today().strftime("%Y-%m-%d") 

st.title("Stockyy: The Price Predictor")
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'NVDA')
selected_stock = st.selectbox('Select dataset for prediction', stocks)
years = st.slider('Years of prediction:', 1, 10)
period = years * 365

def load_data(ticker):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)


st.subheader('Raw data, Starting and Current')
st.write(data.head(),data.tail())

def raw_data():
    chart= grph.Figure()
    chart.add_trace(grph.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    chart.add_trace(grph.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    chart.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(chart)

raw_data()
    
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {years} years')
chart2 = plot_plotly(m, forecast)
st.plotly_chart(chart2)

st.write("Forecast components")
chart3 = m.plot_components(forecast)
st.write(chart3)
