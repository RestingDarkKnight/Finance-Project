#!/usr/bin/env python
# coding: utf-8

# In[15]:



get_ipython().system('pip install pulp')
get_ipython().system('pip install pandas-datareader')


# In[21]:


import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import yfinance as yf


# In[ ]:


# Minimum Variance Portfolio and Equally Weighted Portfolio.


# In[22]:


df = data.DataReader([
                        'ADANITRANS.NS',
                        'AXISBANK.NS',
                        'CIPLA.NS',
                        'BPCL.NS',
                        'BOSCHLTD.NS',
                        'EICHERMOT.NS',
                        'HDFCBANK.NS',
                        'HEROMOTOCO.NS',
                        'IOC.NS',
                        'ITC.NS',
                        'M&M.NS',
                        'MARUTI.NS',
                        'MRF.NS',
                        'ONGC.NS',
                        'PIDILITIND.NS',
                        'SUNPHARMA.NS',
                        'SBIN.NS',
                        'TATAELXSI.NS',
                        'TATAMOTORS.NS',
                        'TITAN.NS'
],
'yahoo', start='2019/09/11', end='2021/09/11')

df.head()


# In[23]:



# In this data closing price is important  to us. hence, we tabulate the closing price.
df = df['Adj Close']
df.head()


# In[ ]:


# Now,we find the covariance and correlation matrix.


# In[25]:


# Covariance Matrix:
cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
cov_matrix


# In[27]:


# correlation matrix:
corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()
corr_matrix


# In[ ]:


# # Q 1(b): Equally weighted portfolio (weights and portfolio variance)


# In[28]:


# To calculate the portfolio variance we need to give certain weights to the stocks. 
# lets, calculte the portfolio variance for equally weighted portfolio.
# Weights for the equally weighted portfolio will be 0.05(1/20) for each stock.



w = {
    'AXISBANK.NS': 0.05,
    'ADANITRANS.NS': 0.05,
    'BOSCHLTD.NS': 0.05,
    'BPCL.NS': 0.05,
    'CIPLA.NS': 0.05,
    'EICHERMOT.NS': 0.05,
    'HDFCBANK.NS': 0.05,
    'HEROMOTOCO.NS': 0.05,
    'ITC.NS': 0.05,
    'IOC.NS': 0.05,
    'ONGC.NS': 0.05,
    'MRF.NS': 0.05,
    'PIDILITIND.NS': 0.05,
    'SUNPHARMA.NS': 0.05,
    'M&M.NS': 0.05,
    'MARUTI.NS': 0.05,
    'SBIN.NS': 0.05,
    'TATAELXSI.NS': 0.05,
    'TATAMOTORS.NS': 0.05,
    'TITAN.NS': 0.05
}

portfolio_var = cov_matrix.mul(w, axis=0).mul(w, axis=1).sum().sum()

print("Portfolio Variance for equally weighted portfolio is: ",portfolio_var)


# In[ ]:


# # Q1(a): Minimum variance portfolio (weights and portfolio variance)
# # Q2: Efficient Frontier plot and Sharpe Ratio  portfolio.


# In[29]:


# Now, for Minimum variance portfolio, we cannot randomly put values of weights. 
# Hence, to calculate it we need to find the maximum portfolio expected return.
# Portfolio Expected Return: It is the sum of all individual expected returns further multiplied by the weight of assets give us expected return for the portfolio.

# Yearly expected returns for an individual stock is given as;
yer = df.resample('Y').last().pct_change().mean()
print(yer)


# In[30]:


# Plotting the efficient frontier:

# For this we need to first calculate the VOLATILITY which is the annual standard deviation.

# To calculate the annual standard deviation we need to multiply the standard deviation from square root of variance,
# by a factor of 252(Number trading days in an indian trading calander), as it gives the daily standard deviation.

# Annual Standard Deviation:

asd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(252))
asd


# In[32]:



stocks = pd.concat([yer, asd], axis=1) # Creating a table for visualising returns and volatility of assets
stocks.columns = ['Returns', 'Volatility']
stocks


# In[ ]:


# Now, for plotting the efficient frontier plot , we need run a loop. In each cycle of the loop it will consider different weights for assets and will calculate the return and volatility of that particular portfolio combination


# In[ ]:


# Now, we run this loop 20000 times.

# For assigning random weights everytime, np.random.random(), function can be used.


# In[33]:


# Define an empty array for portfolio returns

portfolio_return = []

# Define an empty array for portfolio volatility

portfolio_volatility = []

# Define an empty array for asset weights

portfolio_weights = []

total_stocks = len(df.columns)
total_portfolios = 20000

for portfolio in range(total_portfolios):
    weights = np.random.random(total_stocks)
    weights = weights/np.sum(weights)
    portfolio_weights.append(weights)
    
    # Returns are the product of individual expected returns of asset and its weights.
    
    returns = np.dot(weights, yer)
    portfolio_return.append(returns)
    
    # Portfolio Variance
    
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
    
     # Daily standard deviation
        
    sd = np.sqrt(var)
    
    # Annual standard deviation = volatility
    
    annual_stddevi = sd*np.sqrt(252) 
    portfolio_volatility.append(annual_stddevi)
    

data = {'Returns':portfolio_return, 'Volatility':portfolio_volatility}

for counter, symbol in enumerate(df.columns.tolist()):
    #print(counter, symbol)
    data[symbol+' weight'] = [w[counter] for w in portfolio_weights]
portfolios  = pd.DataFrame(data)
portfolios.head()


# In[34]:


# Plotting efficient frontier
portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=False, figsize=[5,5])


# In[35]:


# To find the minimum variance portfolio,  we find the Minimum Volatility portfolio, which are same. 
# For achieveing this we use the idxmin() function, which gives us the minimum value in specified column.
min_var_portfolio = portfolios.iloc[portfolios['Volatility'].idxmin()]

print(min_var_portfolio )
                              


# In[36]:


# Minimum variance portfolio marked in the efficient frontier.
plt.subplots(figsize=[5,5])
plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
plt.scatter(min_var_portfolio[1], min_var_portfolio[0], color='r', marker='*', s=500)


# In[37]:


# Portfolio Variance of minimum variance portfolio.
   
print(cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum())


# In[ ]:


# Sharpe Ratio: The ratio is the average return earned in excess of the risk-free rate per unit of volatility or total risk. 
# Sharpe Ratio Portfolio: The portfolio which has the highest value of sharpe ratio.


# In[38]:


# Finding the Sharpe Ratio Portfolio(Tangency Portfolio).
# We define the risk-free rate to be 0% or 0.00.
rf = 0.00
sharpe_ratio_portfolio = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]

print(sharpe_ratio_portfolio)


# In[39]:



# Sharpe Ratio Portfolio(Tangency portfolio) marked in the efficient frontier.

plt.subplots(figsize=(5, 5))
plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
plt.scatter(min_var_portfolio[1], min_var_portfolio[0], color='r', marker='*', s=500)
plt.scatter(sharpe_ratio_portfolio[1], sharpe_ratio_portfolio[0], color='g', marker='*', s=500)


# In[40]:


# Minimum Var portfolio ka portfolio variance:
print(cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum())


# In[ ]:




