import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Download S&P 500 data
sp500 = yf.Ticker("^GSPC")
sp500_data = sp500.history(period="max")

# Plotting the closing prices
plt.figure(figsize=(10, 6))
plt.plot(sp500_data['Close'], label='Close Price')
plt.title('S&P 500 Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()  # Display the plot

# Preprocessing data for the model
sp500_data.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
sp500_data["Tomorrow"] = sp500_data["Close"].shift(-1)
sp500_data["Target"] = (sp500_data["Tomorrow"] > sp500_data["Close"]).astype(int)
sp500_data = sp500_data.loc["1990-01-01":].copy()

# Training the model
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
train = sp500_data.iloc[:-100]
test = sp500_data.iloc[-100:]
predictors = ["Close", "Volume", "Open", "High", "Low"]

model.fit(train[predictors], train["Target"])

# Making predictions
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

# Combining predictions with actual data and plotting
combined = pd.concat([test["Target"], preds], axis=1)
plt.figure(figsize=(10, 6))
combined.plot()
plt.title('Actual vs Predicted')
plt.show()  # Display the plot
