import yfinance as yf
from datetime import datetime

ticker_symbol = "^VIX"
start_date = "2013-10-01"
end_date = datetime.today().strftime('%Y-%m-%d')
data = yf.download(ticker_symbol, start=start_date, end=end_date)
close_prices = data[['Close']]
close_prices.to_csv("TotalEnergies_Close_Prices.csv")

