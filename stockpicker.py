import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Fudamental Analysis





#is meathod of analying stocks by examining the underlying financial
# and economic facotrs thagt influence their value.

# Earning per Share (EPS)
# This is a companys net income divided by its outstanding shares.
# it gives an idea of how much profit each share of
# stock is generatin

#Price to Earning Share (EPS)
#This is a companys current stock price divided by its EPS
#It gives an idea of how much investors are willing to pay
#for each dollar if earnings

#Price to book ratio (P/B Ratio)
#This is the Companys current stock price divided by its
#book value for its share.
#The P/B ratio gives an idea of how much investors
#are paying for each dollar of tangible assets

#Return on Equity (ROE)
#this is a companys net income divided by its shareholders equity
#it gives an idea of how effciently a companys is using
#ts equity to genarate profits

#load the companys financial data
#Takes CSV Files!
#df = pd.read_cvv("Companys_Financials.csv")

#calculate the EPS
#df['EPS'] = df["Net Income"] / df['EPS']

#Calculate the P/E
#df["P/E Ratio"] = df["Stock Price"] / (df['Total Assets']) - df['Total Liabilities']) / df['Shares Outstanding']

#Calculate the ROE
#df['ROE'] = df['Net Income'] / df["ShareHolder Equity"]

#print the results
#print(df[["EPS", "P/E Ratio", "ROE"]])

#Technical Analysis


#Moving Averages
#This is a meathod of smoothing out fluctuations in stock prices
#by calculating the average price over a specified peroids of time
#this can help idenify trends and support/resistance levels

#Relative Strength Index(RSI)
#This is a momentum indicator that measure the strength of a stocks price
#action by comparing thr magnitude of its recent gaines to recent loses
#RSI vaules ranges from 0 to 100, with values about 70 indicating an overbought conditions
#and values below below 30 indicatinf an oversold condition

#Bollinger Bands
#This is a volatillty indicator that uses moving averages to define
#upper and lower bounds around a stocks price
#Bollinger Bands can help identify periods of high and low vilatilly

# Load the stock's price data
#df = pd.read_csv('stock_prices.csv')

# Calculate the 50-day moving average
#df['MA50'] = df['Close'].rolling(window=50).mean()

# Calculate the RSI
#delta = df['Close'].diff()
#gain = delta.where(delta > 0, 0)
#loss = -delta.where(delta < 0, 0)
#avg_gain = gain.rolling(window=14).mean()
#avg_loss = loss.rolling(window=14).mean()
#rs = avg_gain / avg_loss
#df['RSI'] = 100 - (100 / (1 + rs))

# Calculate the upper and lower Bollinger Bands
#df['MA20'] = df['Close'].rolling(window=20).mean()
#df['UpperBand'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
#df['LowerBand'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()

# Plot the results
#fig, ax = plt.subplots()
#ax.plot(df['Date'], df['Close'], label='Close')
#ax.plot(df['Date'], df['MA50'], label='MA50')
#ax.legend()
#plt.show()

#fig, ax = plt.subplots()
#ax.plot(df['Date'], df['RSI'], label='RSI')
#ax.axhline(y=70, color='red', linestyle='--')
#ax.axhline(y=30, color='green', linestyle='--')
#ax.legend()
#plt.show()

#fig, ax = plt.subplots()
#ax.plot(df['Date'], df['Close'], label='Close')
#ax.plot(df['Date'], df['MA20'], label='MA20')
#ax.plot(df['Date'], df['UpperBand'], label='UpperBand')
#ax.plot(df['Date'], df['LowerBand'], label='LowerBand')
#ax.fill_between(df['Date'], df['UpperBand'], df['LowerBand'], alpha=0.1)
#ax.legend()
#plt.show()

Quantitative Analysis
is a method of analyzing stocks by using mathematical and statistical
models to identify patterns and trends in the market.
This approach is based on the assumption that market behavior
can be predicted using quantitative models that are based on
historical data, fundamental data, or other data sources.


# Load the stock's price data
df = pd.read_csv('stock_prices.csv')

# Calculate the daily returns
df['Returns'] = np.log(df['Close']) - np.log(df['Close'].shift(1))

# Calculate the moving average
df['MA50'] = df['Close'].rolling(window=50).mean()

# Calculate the standard deviation
df['STD50'] = df['Close'].rolling(window=50).std()

# Calculate the Bollinger Bands
df['UpperBand'] = df['MA50'] + 2 * df['STD50']
df['LowerBand'] = df['MA50'] - 2 * df['STD50']

# Calculate the linear regression
X = np.arange(len(df)).reshape(-1, 1)
y = df['Close'].values.reshape(-1, 1)
reg = LinearRegression().fit(X, y)
df['Trendline'] = reg.predict(X)

# Plot the results
fig, ax = plt.subplots()
ax.plot(df['Date'], df['Close'], label='Close')
ax.plot(df['Date'], df['MA50'], label='MA50')
ax.plot(df['Date'], df['UpperBand'], label='UpperBand')
ax.plot(df['Date'], df['LowerBand'], label='LowerBand')
ax.fill_between(df['Date'], df['UpperBand'], df['LowerBand'], alpha=0.1)
ax.plot(df['Date'], df['Trendline'], label='Trendline')
ax.legend()
plt.show()










