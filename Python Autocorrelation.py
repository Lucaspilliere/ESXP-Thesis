import yfinance as yf
from hurst import compute_Hc
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox

# Yahoo finance ticker list
ticker = ["TTE.PA", "GOLD", "AAPL", "BABA", "WMT", "MCHI", "MELI", "TSM", "LCAL.L", "EEM"]

for i in ticker:
    #Compute returns
    data = yf.download(i, start="2010-01-01", end="2024-09-01")
    close_prices = data['Adj Close']
    returns = close_prices.pct_change().dropna()
    
    #Autocorrelation
    sm.graphics.tsa.plot_acf(returns, lags=30)
    plt.title('Autocorrelation ' + i)
    plt.show()
    
    # Ljung-Box test
    ljung_box_results = acorr_ljungbox(returns, lags=[1, 2, 10, 30], return_df=True)
    print(ljung_box_results)
    
    # Hurst test
    H, c, data_hurst = compute_Hc(returns, kind='change')
    print(f"Exponent Hurst {i}: {H}")


    
    
    
    
