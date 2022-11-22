# %% [markdown]
# # Portfolio Optimisation using DeepLearning.
# 
# Authors.
# - Sri Balaaji Natarajan Kalaivendan.
# - Prashanth Jayaraman.
# 

# %% [markdown]
# # Imports

# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pandas_datareader import data as pdr
from cryptocmd import CmcScraper
import holidays
from random import randint
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
%config InlineBackend.figure_format = 'retina'
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tqdm
from __future__ import print_function
import tflearn


# %% [markdown]
# # Fetching the data

# %%
crypto = ['BTC', 'ETH', 'LTC', 'BCH', 'XMR', 'XRP']

def dataFetch(ticker):
    scraper = CmcScraper(ticker)
    df = scraper.get_dataframe()
    df['tickerCrypto'] = ticker
    return df

dataCrypto = []
for ticker in crypto:
    df = dataFetch(ticker)
    dataCrypto.append(df)

dataCrypto = pd.concat(dataCrypto)
dataCrypto.to_csv('dataCrypto.csv')

# %%
dataCryptoPath = 'dataCrypto.csv'
dataCrypto = pd.read_csv(dataCryptoPath)
dataCrypto  = dataCrypto.sort_values(['tickerCrypto', 'Date'])
print('crypto currenicies considered : ' + str(np.unique(dataCrypto['tickerCrypto'])))

minTickerPoints = 100000000
for ticker in np.unique(dataCrypto['tickerCrypto']):
    if len(dataCrypto['Date'].loc[dataCrypto['tickerCrypto']==ticker]) < minTickerPoints:
        minTickerPoints = len(dataCrypto['Date'].loc[dataCrypto['tickerCrypto']==ticker])
        smallest_date = dataCrypto['Date'].loc[dataCrypto['tickerCrypto']==ticker].min()
    
dataCrypto_filtered = dataCrypto.loc[dataCrypto['Date'] >= smallest_date]
dataCrypto_filtered  = dataCrypto_filtered.sort_values(['tickerCrypto', 'Date'])

pivot_table = pd.pivot_table(dataCrypto_filtered, values='Close', index=['Date'],columns=['tickerCrypto'])
returnsCrypto = pivot_table.pct_change().dropna()
crypto_returns_mean = returnsCrypto.mean()
cryptoCovarianceMatrix = returnsCrypto.cov()


def get_input_data(data_table, ticker_column_name):
    openV = list()
    closeV = list()
    highV = list()
    lowV = list()

    for ticker in np.unique(data_table[ticker_column_name]):
        open_value_list = data_table['Open'].loc[data_table[ticker_column_name]==ticker]
        openV.append(open_value_list[1:].reset_index(drop = True) / open_value_list[:-1].reset_index(drop = True))
        closeV.append(data_table['Close'].loc[data_table[ticker_column_name]==ticker][:-1] / open_value_list[:-1])
        highV.append(data_table['High'].loc[data_table[ticker_column_name]==ticker][:-1] / open_value_list[:-1])
        lowV.append(data_table['Low'].loc[data_table[ticker_column_name]==ticker][:-1] / open_value_list[:-1])
    
    return np.array([closeV,
                     highV,
                     lowV,
                     openV])
    
crypto_input_array = get_input_data(dataCrypto_filtered, 'tickerCrypto')
np.save('dataCrypto_input.npy', crypto_input_array)
print('shape of crypto data input : ' + str(crypto_input_array.shape))

# %%
dataCrypto_filtered.to_csv('dataCrypto_Filtered.csv')

# %% [markdown]
# #Modern Portfolio theory
# 
# 

# %%
def ReadData(TypeOfData):
    CryptoCont = pd.read_csv('dataCrypto_Filtered.csv')
    CryptoPath = 'dataCrypto.csv'

    if TypeOfData == "Crypto":
        Data = pd.read_csv(CryptoPath)
        Ticker = "tickerCrypto"
    elif TypeOfData == "CryptoCont":
        Data = CryptoCont
        Ticker = "tickerCrypto"
    else:
        print("Error")

    return Data, Ticker

plt.rcParams.update({'font.sans-serif':'Helvetica'})
Timeperiod = 365
TypeOfData = "CryptoCont"
Data, Ticker = ReadData(TypeOfData)
Data = Data[['Date', Ticker, 'Close']]
Data['Date'] = pd.to_datetime(Data['Date'])
Data['Date'] = Data['Date'].dt.date

if TypeOfData == "CryptoCont":
    Data = Data.groupby(['Date', Ticker])['Close'].agg('mean').reset_index()


Table = Data.pivot(index='Date',columns=Ticker,values='Close') 
ColNames = Table.columns.values
Returns = Table.pct_change()
plt.figure(figsize=(16, 8))
for Col in ColNames:
    plt.plot(Returns.index, Returns[Col],label=Col)
plt.ylabel('Returns')
plt.legend(loc='upper left')
plt.title("Daily returns for cryptocurrency")

def EfficientFrontier(AvgReturns, Covariance, CountPortfolio, RFF):
    
    WeightHolder = []
    Res = np.zeros((3,CountPortfolio))
    for j in range(CountPortfolio):   
        Wei = np.random.random(6)
        Wei /= np.sum(Wei)
        WeightHolder.append(Wei)  
        Por_SD = np.sqrt(np.dot(Wei.T, np.dot(Covariance, Wei))) * np.sqrt(Timeperiod)
        Por_Return = np.sum(AvgReturns*Wei ) *Timeperiod
        Res[0,j] = Por_SD
        Res[1,j] = Por_Return
        Res[2,j] = (Por_Return - RFF) / Por_SD

    
    VolatilityIndexMin = np.argmin(Res[0])
    MinVolSD, MinVolReturn = Res[0,VolatilityIndexMin], Res[1,VolatilityIndexMin]
    MinVolAllocation = pd.DataFrame(WeightHolder[VolatilityIndexMin],index=Table.columns,columns=['Alloc'])
    MinVolAllocation.Alloc = [round(i*100,2)for i in MinVolAllocation.Alloc]
    MinVolAllocation = MinVolAllocation.T
    SharpeIndexMax = np.argmax(Res[2])
    SDMaxSharpe, RetMaxSharpe = Res[0,SharpeIndexMax], Res[1,SharpeIndexMax]
    SharpeAllocationMax = pd.DataFrame(WeightHolder[SharpeIndexMax],index=Table.columns,columns=['Alloc'])
    SharpeAllocationMax.Alloc = [round(i*100,2)for i in SharpeAllocationMax.Alloc]
    SharpeAllocationMax = SharpeAllocationMax.T
    print(" - " * 45)
    print("Allocation for Portfolio with Minimum volatility\n")
    print("Portfolio Return Annual:", round(MinVolReturn,4))
    print("Portfolio Volatility Annual", round(MinVolSD,4))
    print("\n")
    print(MinVolAllocation)
    print(" -" * 45)
    print("Allocation for Portfolio with Maximum sharpe ratio\n")
    print("Portfolio Return Annual:", round(RetMaxSharpe,4))
    print("Portfolio Volatility Annual", round(SDMaxSharpe,4))
    print("\n")
    print(SharpeAllocationMax)
    plt.figure(figsize=(14, 7))
    plt.scatter(Res[0,:],Res[1,:],c=Res[2,:],cmap='plasma', marker='o', s=6, alpha=0.5)
    plt.colorbar()
    plt.scatter(SDMaxSharpe,RetMaxSharpe, marker='.',color='g',s=300, label='Portfolio with Maximum Sharpe ratio')
    plt.scatter(MinVolSD,MinVolReturn, marker='.',color='r',s=300, label='Portfolio with Minimum Volatility')
    plt.title('Modern Portfolio Theory using Efficient Frontier Theory')
    plt.xlabel('Annual Volatility')
    plt.ylabel('Annual Returns')
    plt.legend(labelspacing=0.6)

CountPortfolio = 30000
RFF = 0.0178
AvgReturns = Returns.mean()
Covariance = Returns.cov()
EfficientFrontier(AvgReturns, Covariance, CountPortfolio, RFF)

# %% [markdown]
# # EDA

# %%
cryptoData = pd.read_csv("dataCrypto.csv")
print("Earliest Date for Crypto currency", cryptoData['Date'].min())
print("Latest Date for Crypto currency", cryptoData['Date'].max())
print("Number of Holdings in Crypto data:", len(set(cryptoData['tickerCrypto'])))
pd.pivot_table(cryptoData, values ='Close', index =['Date'], columns =['tickerCrypto']).isnull().values.any()
crypto_null = pd.pivot_table(cryptoData, values ='Close', index =['Date'], columns =['tickerCrypto'])
crypto_null[crypto_null.isnull().any(axis=1)].sample(5)

# %%
us_holidays = holidays.US()
cryptoData['Holiday'] = cryptoData.Date.apply(lambda x: x in us_holidays)
cryptoData['Date'] = pd.to_datetime(cryptoData['Date'])
cryptoData['Day'] = cryptoData['Date'].dt.day_name()
cryptoAssets = cryptoData.loc[cryptoData['Holiday'] == False]
cryptoData = cryptoAssets.loc[~cryptoAssets.Day.isin(['Saturday', 'Sunday'])]

# %%
dataCrypto = []
for ticker in list(set(cryptoData['tickerCrypto'])):
    temp = cryptoData.loc[cryptoData['tickerCrypto'] == ticker]
    temp["crypto returns"] = temp['Close'].pct_change()
    dataCrypto.append(temp)
    
cryptoData = pd.concat(dataCrypto)

# %%
for ticker in list(set(cryptoData['tickerCrypto'])):
    temp = cryptoData.loc[cryptoData['tickerCrypto'] == ticker]
    print(ticker+' has data for '+str(len(set(temp.index)))+' days')

# %%
cryptoData['Date'] = pd.to_datetime(cryptoData['Date'])
filter_mask = cryptoData['Date'] > pd.Timestamp(2016, 1, 1)
filteredData = cryptoData[filter_mask]
cryptoDataReshaped = filteredData.pivot(index='Date', columns='tickerCrypto', values='crypto returns')
cryptoDataReshaped = cryptoDataReshaped.reset_index()
cryptoDataReshaped.tail()

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['BTC'],
                name="Bitcoin",
                line_color='red',
                opacity=0.5))

fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['ETH'],
                name="Ethereum",
                line_color='blue',
                opacity=0.5))

fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['LTC'],
                name="Litecoin",
                line_color='green',
                opacity=0.5))

fig.update_layout(xaxis_range=['2022-01-01','2022-07-31'],
                 title_text = 'Returns for Crypto Currencies [2022]')
fig.update_yaxes(title_text = 'Returns')
fig.update_xaxes(title_text = 'Time period')
fig.show()

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['BTC'],
                name="Bitcoin",
                line_color='red',
                opacity=0.5))

fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['ETH'],
                name="Ethereum",
                line_color='blue',
                opacity=0.5))

fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['LTC'],
                name="Litecoin",
                line_color='green',
                opacity=0.5))


fig.update_layout(xaxis_range=['2021-01-01','2021-12-31'], title_text = 'Returns for Crypto Currencies [2021]')
fig.update_yaxes(title_text = 'Returns')
fig.update_xaxes(title_text = 'Time period')
fig.show()

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['BTC'],
                name="Bitcoin",
                line_color='red',
                opacity=0.5))

fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['ETH'],
                name="Ethereum",
                line_color='blue',
                opacity=0.5))

fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['LTC'],
                name="Litecoin",
                line_color='green',
                opacity=0.5))

fig.update_layout(xaxis_range=['2020-01-01','2020-12-31'], title_text = 'Returns for Crypto Currencies [2020]')
fig.update_yaxes(title_text = 'Returns')
fig.update_xaxes(title_text = 'Time period')
fig.show()

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['BTC'],
                name="Bitcoin",
                line_color='red',
                opacity=0.5))

fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['ETH'],
                name="Ethereum",
                line_color='blue',
                opacity=0.5))

fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['LTC'],
                name="Litecoin",
                line_color='green',
                opacity=0.5))


fig.update_layout(xaxis_range=['2019-01-01','2019-12-31'], title_text = 'Returns for Crypto Currencies [2019]')
fig.update_yaxes(title_text = 'Returns')
fig.update_xaxes(title_text = 'Time period')
fig.show()

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['BTC'],
                name="Bitcoin",
                line_color='red',
                opacity=0.5))

fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['ETH'],
                name="Ethereum",
                line_color='blue',
                opacity=0.5))

fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['LTC'],
                name="Litecoin",
                line_color='green',
                opacity=0.5))

fig.update_layout(xaxis_range=['2018-01-01','2018-12-31'], title_text = 'Returns for Crypto Currencies [2018]')
fig.update_yaxes(title_text = 'Returns')
fig.update_xaxes(title_text = 'Time period')
fig.show()

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['BTC'],
                name="Bitcoin",
                line_color='red',
                opacity=0.5))

fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['ETH'],
                name="Ethereum",
                line_color='blue',
                opacity=0.5))

fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['LTC'],
                name="Litecoin",
                line_color='green',
                opacity=0.5))


fig.update_layout(xaxis_range=['2017-01-01','2017-12-31'], title_text = 'Returns for Crypto Currencies [2017]')
fig.update_yaxes(title_text = 'Returns')
fig.update_xaxes(title_text = 'Time period')
fig.show()

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['BTC'],
                name="Bitcoin",
                line_color='red',
                opacity=0.5))

fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['ETH'],
                name="Ethereum",
                line_color='blue',
                opacity=0.5))

fig.add_trace(go.Scatter(
                x=cryptoDataReshaped.Date,
                y=cryptoDataReshaped['LTC'],
                name="Litecoin",
                line_color='green',
                opacity=0.5))

fig.update_layout(xaxis_range=['2016-01-01','2016-12-31'], title_text = 'Returns for Crypto Currencies [2016]')
fig.update_yaxes(title_text = 'Returns')
fig.update_xaxes(title_text = 'Time period')
fig.show()

# %%
corr_crypto = pd.pivot_table(cryptoData, values ='Close', index =['Date'], columns =['tickerCrypto'])
correlation_crypto = corr_crypto.corr()
plt.figure(figsize=(10, 5))
sns_plot = sns.heatmap(correlation_crypto, annot=True, cmap="RdYlGn_r")
plt.savefig('images/Correlation_crypto.png')

# %% [markdown]
# # Reinforcement Learning Environment 

# %%
class RLEnv():
    def __init__(self, Path, PortfolioValue, TransCost, ReturnRate, WindowSize, TrainTestSplit):
        self.Dataset = np.load(Path)
        self.NumCrypto = self.Dataset.shape[1]
        self.NumValues = self.Dataset.shape[0]
        self.PortfolioValue = PortfolioValue
        self.TransCost = TransCost
        self.ReturnRate = ReturnRate
        self.WindowSize = WindowSize
        self.Done = False
        self.state = None
        self.TimeLength = None
        self.Terminate = False
        self.TerminateRows = int((self.Dataset.shape[2] - self.WindowSize) * TrainTestSplit)
        
    def UpdatedOpenValues(self, T):
        return np.array([1+self.ReturnRate]+self.Dataset[-1,:,T].tolist())
    
    def InputTensor(self, Tensor, T):
        return Tensor[: , : , T - self.WindowSize:T]
    
    def ResetEnvironment(self, InitWeight, InitPortfolio, T):
        self.state= (self.InputTensor(self.Dataset, self.WindowSize) , InitWeight , InitPortfolio)
        self.TimeLength = self.WindowSize + T
        self.Done = False
        return self.state, self.Done
    
    def Step(self, Action):
        Dataset = self.InputTensor(self.Dataset, int(self.TimeLength))
        weightVectorOld = self.state[1]
        PortfolioValue_old = self.state[2]
        NewOpenValues = self.UpdatedOpenValues(int(self.TimeLength))
        WeightAllocation = Action
        PortfolioAllocation = PortfolioValue_old
        TransactionCost = PortfolioAllocation * self.TransCost * np.linalg.norm((WeightAllocation-weightVectorOld),ord = 1)
        ValueAfterTransaction = (PortfolioAllocation * WeightAllocation) - np.array([TransactionCost]+ [0] * self.NumCrypto)
        NewValueofCrypto = ValueAfterTransaction * NewOpenValues
        NewPortfolioValue = np.sum(NewValueofCrypto)
        NewWeightVector = NewValueofCrypto / NewPortfolioValue
        RewardValue = (NewPortfolioValue - PortfolioValue_old) / (PortfolioValue_old)
        self.TimeLength = self.TimeLength + 1
        self.state = (self.InputTensor(self.Dataset, int(self.TimeLength)), NewWeightVector, NewPortfolioValue) 
        if self.TimeLength >= self.TerminateRows:
            self.Done = True    
        return self.state, RewardValue, self.Done


# %% [markdown]
# # Portfolio Vector Memory

# %%
class PVM(object):
    def __init__(self, tickerNum, beta_pvm, trainingSteps, trainingBatchSize, wt_vector_init):
        self.pvm = np.transpose(np.array([wt_vector_init] * int(trainingSteps)))
        self.beta_pvm = beta_pvm
        self.trainingSteps = trainingSteps
        self.trainingBatchSize = trainingBatchSize

    def getWeights_t(self, t):
        return self.pvm[:, t]

    def updateWeight_vector_t(self, t, weight):
        self.pvm[:, int(t)] = weight
            
    def test(self):
        return self.pvm

# %% [markdown]
# # HyperParameters for LSTM and CNN

# %%
data_path = 'dataCrypto_input.npy'
ticker_list = ['BCH','BTC','ETH','LTC','XMR','XRP']
data = np.load(data_path)
ohlcFeaturesNum = data.shape[0]
tickerNum = data.shape[1]
tradingDaysCaptured = data.shape[2]
print('number of ohlc features : ' + str(ohlcFeaturesNum))
print('number of crypto currencies considered : ' + str(tickerNum))
print('number of trading days captured : ' + str(tradingDaysCaptured))
numFiltersLayer_1 = 2
num_filters_layer_2 = 20
kernel_size = (1, 3)
train_data_ratio = 0.8
trainingSteps = 0.8 * tradingDaysCaptured
validationSteps = 0.1 * tradingDaysCaptured
testSteps = 0.1 * tradingDaysCaptured
trainingBatchSize = 50
beta_pvm = 5e-5  
numTradingPeriods = 20
weight_vector_init = np.array(np.array([1] + [0] * tickerNum))
PortfolioValueInit = 10000
weightVector = np.array(np.array([1] + [0] * tickerNum))
PortfolioValueInit_test = 10000
num_episodes = 1
numBatches = 1
equiweight_vector = np.array(np.array([1/(tickerNum + 1)] * (tickerNum + 1)))
epsilon = 0.8
adjusted_rewards_alpha = 0.1
l2_reg_coef = 1e-8
adam_opt_alpha = 9e-2
optimizer = tf.train.AdamOptimizer(adam_opt_alpha)
tradingCost = 1/100000
interestRate = 0.02/250
cashBias_init = 0.7

# %% [markdown]
# # Plots for the model

# %%
def plot_cpv(final_PortfolioValues_opt, final_PortfolioValues_eq, initial_PortfolioValue, plot_name):
	plt.plot(final_PortfolioValues_opt, label = 'optimizing agent', color = 'darkgreen')
	plt.plot(final_PortfolioValues_eq, label = 'equiweight agent', color = 'mediumpurple')
	plt.plot([initial_PortfolioValue] * len(final_PortfolioValues_opt), label = 'no investment')
	plt.legend()
	plt.title('cumulative portfolio values over test steps')
	plt.xlabel('test steps')
	plt.ylabel('cumulative portfolio value')
	plt.savefig('images/test_result_plots/' + plot_name)

def plotWeights_assigned(wt_vector, num_assets, ticker_list, plot_name):
	plt.bar(np.arange(num_assets + 1), wt_vector, color='slateblue')
	plt.title('Final Portfolio weights (test set)', fontsize=13)
	plt.rcParams["figure.figsize"] = (10,6)
	plt.xlabel("Tickers ['BCH', 'BTC', 'ETH', 'LTC', 'XMR', 'XRP']")
	plt.savefig('images/test_result_plots/' + plot_name)

# %% [markdown]
# # CNN

# %%
class PolicyCNN(object):
    def __init__(self, ohlc_feature_num, tickerNum, numTradingPeriods, sess, optimizer, tradingCost, cashBias_init, interestRate, 
        equiweight_vector, adjusted_rewards_alpha, kernel_size, numFilterLayer_1, numFilterLayer_2):
        self.ohlc_feature_num = ohlc_feature_num
        self.tickerNum = tickerNum
        self. numTradingPeriods =  numTradingPeriods
        self.tradingCost = tradingCost
        self.cashBias_init = cashBias_init
        self.interestRate = interestRate
        self.equiweight_vector = equiweight_vector
        self.adjusted_rewards_alpha = adjusted_rewards_alpha 
        self.kernel_size = kernel_size
        self.numFilterLayer_1 = numFilterLayer_1
        self.numFilterLayer_2 = numFilterLayer_2
        self.optimizer = optimizer
        self.sess = sess

        self.X_t = tf.placeholder(tf.float32, [None, self.ohlc_feature_num, self.tickerNum, self.numTradingPeriods])
        self.weightsPrevious_t = tf.placeholder(tf.float32, [None, self.tickerNum + 1])
        self.PortfolioPrevious_t = tf.placeholder(tf.float32, [None, 1])
        self.dailyReturns_t = tf.placeholder(tf.float32, [None, self.tickerNum]) 
        cashBias = tf.get_variable('cashBias', shape=[1, 1, 1, 1], initializer = tf.constant_initializer(self.cashBias_init))
        shape_X_t = tf.shape(self.X_t)[0]
        self.cashBias = tf.tile(cashBias, tf.stack([shape_X_t, 1, 1, 1]))
        def convolution_layers(X_t, numFilterLayer_1, kernel_size, numFilterLayer_2, numTradingPeriods):
            with tf.variable_scope("Convolution1"):
                convolution1 = tf.layers.conv2d(
                inputs = tf.transpose(X_t, perm=[0, 3, 2, 1]),
                activation = tf.nn.tanh,
                filters = numFilterLayer_1,
                strides = (1, 1),
                kernel_size = kernel_size,
                padding = 'same')


            with tf.variable_scope("Convolution2"):
                convolution2 = tf.layers.conv2d(
                inputs = convolution1,
                activation = tf.nn.tanh,
                filters = numFilterLayer_2,
                strides = (numTradingPeriods, 1),
                kernel_size = (1, numTradingPeriods),
                padding = 'same')

            with tf.variable_scope("Convolution3"):
                self.convolution3 = tf.layers.conv2d(
                    inputs = convolution2,
                    activation = tf.nn.relu,
                    filters = 1,
                    strides = (numFilterLayer_2 + 1, 1),
                    kernel_size = (1, 1),
                    padding = 'same')

            return self.convolution3


        def policy_output(convolution, cashBias):
            with tf.variable_scope("Policy-Output"):
                tensor_squeeze = tf.squeeze(tf.concat([cashBias, convolution], axis=2), [1, 3])
                self.action = tf.nn.softmax(tensor_squeeze)
            return self.action


        def reward(shape_X_t, actionChosen, interestRate, weightsPrevious_t, PortfolioPrevious_t, dailyReturns_t, tradingCost):
            with tf.variable_scope("Reward"):
                cash_return = tf.tile(tf.constant(1 + interestRate, shape=[1, 1]), tf.stack([shape_X_t, 1]))
                y_t = tf.concat([cash_return, dailyReturns_t], axis=1)
                portfolioVector_t = actionChosen * PortfolioPrevious_t
                portfolioVector_previous = weightsPrevious_t * PortfolioPrevious_t

                total_tradingCost = tradingCost * tf.norm(portfolioVector_t - portfolioVector_previous, ord=1, axis=1) * tf.constant(1.0, shape=[1])
                total_tradingCost = tf.expand_dims(total_tradingCost, 1)

                zero_vector = tf.tile(tf.constant(np.array([0.0] * tickerNum).reshape(1, tickerNum), shape=[1, tickerNum], dtype=tf.float32), tf.stack([shape_X_t, 1]))
                cost_vector = tf.concat([total_tradingCost, zero_vector], axis=1)

                portfolioVector_second_t = portfolioVector_t - cost_vector
                final_portfolioVector_t = tf.multiply(portfolioVector_second_t, y_t)
                PortfolioValue = tf.norm(final_portfolioVector_t, ord=1)
                self.instantaneous_reward = (PortfolioValue - PortfolioPrevious_t) / PortfolioPrevious_t
                
            
            with tf.variable_scope("Reward-Equiweighted"):
                cash_return = tf.tile(tf.constant(1 + interestRate, shape=[1, 1]), tf.stack([shape_X_t, 1]))
                y_t = tf.concat([cash_return, dailyReturns_t], axis=1)
                portfolioVector_eq = self.equiweight_vector * PortfolioPrevious_t
                PortfolioValue_eq = tf.norm(tf.multiply(portfolioVector_eq, y_t), ord=1)
                self.instantaneous_reward_eq = (PortfolioValue_eq - PortfolioPrevious_t) / PortfolioPrevious_t

            with tf.variable_scope("Reward-adjusted"):
                self.adjusted_reward = self.instantaneous_reward - self.instantaneous_reward_eq - self.adjusted_rewards_alpha * tf.reduce_max(actionChosen)
                
            return self.adjusted_reward


        self.convolution = convolution_layers(self.X_t, self.numFilterLayer_1, self.kernel_size, self.numFilterLayer_2, self.numTradingPeriods) 
        self.actionChosen = policy_output(self.convolution, self.cashBias)
        self.adjusted_reward = reward(shape_X_t, self.actionChosen, self.interestRate, self.weightsPrevious_t, self.PortfolioPrevious_t, self.dailyReturns_t, self.tradingCost)
        self.train_op = optimizer.minimize(-self.adjusted_reward)

    def compute_weights(self, X_t_, weightsPrevious_t_):
        return self.sess.run(tf.squeeze(self.actionChosen), feed_dict={self.X_t: X_t_, self.weightsPrevious_t: weightsPrevious_t_})

    def train_cnn(self, X_t_, weightsPrevious_t_, PortfolioPrevious_t_, dailyReturns_t_):
        self.sess.run(self.train_op, feed_dict={self.X_t: X_t_,
                                                self.weightsPrevious_t: weightsPrevious_t_,
                                                self.PortfolioPrevious_t: PortfolioPrevious_t_,
                                                self.dailyReturns_t: dailyReturns_t_})

# %%
def getRandAction(num_tickers):
    vector_rand = np.random.rand(num_tickers + 1)
    return vector_rand / np.sum(vector_rand)

def maxDrawDown(weights, time_period=numTradingPeriods, portfolioDataFrame=pivot_table):
    weights = weights.reshape(len(weights[0],))
    weights = weights[1:]
    simulated_portfolio = weights[0]*portfolioDataFrame.iloc[:,0]
    for i in range(1, len(portfolioDataFrame.columns)):
        simulated_portfolio += weights[i]*portfolioDataFrame.iloc[:,i]
    maxDrawDown_value = float('-inf')
    for i in range(int(len(simulated_portfolio)/time_period)-1):
        if min(simulated_portfolio[i*time_period:(i+1)*time_period]) > 0:
            biggestVariation = max(simulated_portfolio[i*time_period:(i+1)*time_period])/min(simulated_portfolio[i*time_period:(i+1)*time_period])
        else:
            biggestVariation = 0
        if(biggestVariation > maxDrawDown_value):
            maxDrawDown_value = biggestVariation
    return maxDrawDown_value

def RoMad(weights, mdd, returns_mean=crypto_returns_mean,time_period=tradingDaysCaptured):
    weights = weights.reshape(len(weights[0],))
    weights = weights[1:]
    portfolioReturn = np.sum(returns_mean.values * weights * tradingDaysCaptured)
    romad = portfolioReturn/mdd
    return romad

def sharpeCrypto(w ,returnsCov=cryptoCovarianceMatrix, returns_mean=crypto_returns_mean):
    w = w.reshape(len(w[0],))
    w = w[1:]
    portfolioReturn = np.sum(returns_mean.values * w * tradingDaysCaptured)
    portfolio_volatility = np.sqrt(np.dot(w.T, np.dot(returnsCov.values, w))* tradingDaysCaptured)
    sharpe_ratio = portfolioReturn/portfolio_volatility
    return sharpe_ratio

def main(assets = True):
    
    env_pf_optimizer = RLEnv(Path = data_path, PortfolioValue = PortfolioValueInit, TransCost = tradingCost, 
        ReturnRate = interestRate, WindowSize = numTradingPeriods, TrainTestSplit = train_data_ratio)

    envPortfolioEquiweight = RLEnv(Path = data_path, PortfolioValue = PortfolioValueInit, TransCost = tradingCost, 
        ReturnRate = interestRate, WindowSize = numTradingPeriods, TrainTestSplit = train_data_ratio)
    
    tf.reset_default_graph()
    sess = tf.Session()
    PortfolioOptimizingAgent = PolicyCNN(ohlcFeaturesNum, tickerNum, numTradingPeriods, sess, optimizer, tradingCost, cashBias_init, interestRate, 
            equiweight_vector, adjusted_rewards_alpha, kernel_size, numFiltersLayer_1, num_filters_layer_2)
    sess.run(tf.global_variables_initializer())
    trainPortfolioValues = []
    trainPortfolioValues_equiweight = []
    for ep_num in tqdm.tnrange(num_episodes, desc = 'Episodes'):
        pvm = PVM(tickerNum, beta_pvm, trainingSteps, trainingBatchSize, weight_vector_init)
        for batch_num in tqdm.tnrange(numBatches, desc = 'Batches'):
            list_X_t, listWeight_previous, listPortfolioValue_previous, listDailyReturns_t, listPortfolioValue_previous_equiweight, listSharpeTraining = [], [], [], [], [], []
            training_batch_t_selection = False
            while training_batch_t_selection == False:
                training_batch_t = trainingSteps - trainingBatchSize + 1 - np.random.geometric(p = beta_pvm)
                if training_batch_t >= 0: 
                    training_batch_t_selection = True

            state, done = env_pf_optimizer.ResetEnvironment(pvm.getWeights_t(int(training_batch_t)), PortfolioValueInit , training_batch_t  )
            stateEquiweight, done_equiweight = envPortfolioEquiweight.ResetEnvironment(equiweight_vector, PortfolioValueInit, training_batch_t)
            for training_batch_num in tqdm.tnrange(trainingBatchSize, desc = 'Training Batches'): 
                X_t = state[0].reshape([-1] + list(state[0].shape))
                WeigtPrevious = state[1].reshape([-1] + list(state[1].shape))
                PortfolioValue_previous = state[2]
                action = PortfolioOptimizingAgent.compute_weights(X_t, WeigtPrevious) if (np.random.rand() < epsilon) else getRandAction(tickerNum)
                state, reward, done = env_pf_optimizer.Step(action)
                stateEquiweight, rewardEquiweight, done_equiweight = envPortfolioEquiweight.Step(equiweight_vector)                
                X_next = state[0]
                Weight_t = state[1]
                PortfolioValue_t = state[2]  
                PortfolioValue_t_equiweight = stateEquiweight[2]
                dailyReturns_t = X_next[-1, :, -1]
                pvm.updateWeight_vector_t(training_batch_t + training_batch_num, Weight_t)
                list_X_t.append(X_t.reshape(state[0].shape))
                listWeight_previous.append(WeigtPrevious.reshape(state[1].shape))
                listPortfolioValue_previous.append([PortfolioValue_previous])
                listDailyReturns_t.append(dailyReturns_t)
                listPortfolioValue_previous_equiweight.append(PortfolioValue_t_equiweight)
                sharpe_ratio = sharpeCrypto(w=WeigtPrevious)
                mdd = maxDrawDown(weights=WeigtPrevious)
                romad = RoMad(weights=WeigtPrevious, mdd=mdd).astype(int)
                listSharpeTraining.append(sharpe_ratio)
                print('------------------ training -----------------------')
                print('current portfolio value : ' + str(PortfolioValue_previous))
                print('weights assigned : ' + str(WeigtPrevious))
                print('sharpe_ratio:', sharpe_ratio)
                print('RoMad:', romad)
                print('equiweight portfolio value : ' + str(PortfolioValue_t_equiweight))
            trainPortfolioValues.append(PortfolioValue_t)
            trainPortfolioValues_equiweight.append(PortfolioValue_t_equiweight)
            PortfolioOptimizingAgent.train_cnn(np.array(list_X_t), 
                np.array(listWeight_previous),
                np.array(listPortfolioValue_previous), 
                np.array(listDailyReturns_t))

    print('------ train final value -------')
    print(trainPortfolioValues)
    print(trainPortfolioValues_equiweight)
    state, done = env_pf_optimizer.ResetEnvironment(weightVector, PortfolioValueInit_test, training_batch_t)
    stateEquiweight, done_equiweight = envPortfolioEquiweight.ResetEnvironment(equiweight_vector, PortfolioValueInit_test, training_batch_t)
    testPortfolioValues = [PortfolioValueInit_test]
    testPortfolioValues_equiweight = [PortfolioValueInit_test]
    weightVectors = [weightVector]
    test_sharpe_ratio = []
    test_mdd = []
    startStepNum = int(trainingSteps  + validationSteps - int(numTradingPeriods/2))
    end_step_num = int(trainingSteps  + validationSteps + testSteps - numTradingPeriods)

    for test_step_num in range(startStepNum, end_step_num):
        X_t = state[0].reshape([-1] + list(state[0].shape))
        wt_previous = state[1].reshape([-1] + list(state[1].shape))
        PortfolioValue_previous = state[2]
        action = PortfolioOptimizingAgent.compute_weights(X_t, wt_previous)
        state, reward, done = env_pf_optimizer.Step(action)
        stateEquiweight, rewardEquiweight, done_equiweight = envPortfolioEquiweight.Step(equiweight_vector)
        X_next_t, wt_t, PortfolioValue_t = state[0], state[1], state[2]
        PortfolioValue_t_equiweight = stateEquiweight[2]
        dailyReturns_t = X_next_t[-1, :, -1]
        sharpe_ratio = sharpeCrypto(w=wt_previous)
        mdd = maxDrawDown(weights=WeigtPrevious)
        romad = RoMad(weights=WeigtPrevious, mdd=mdd)
        print('------------------ testing -----------------------')
        print('sharpe_ratio:', sharpe_ratio)
        print('RoMad:', romad)
        testPortfolioValues.append(PortfolioValue_t)
        testPortfolioValues_equiweight.append(PortfolioValue_t_equiweight)
        weightVectors.append(wt_t)
        test_sharpe_ratio.append(sharpe_ratio)
        test_mdd.append(romad)

    print('------ test final value -------')
    print('Mean sharpe ratio : ' + str(np.nanmean(test_sharpe_ratio)))
    print('Mean Max Drawdown:', str(np.mean(test_mdd)))
    plotWeights_assigned(weightVectors[-1], tickerNum, ticker_list, 'cnnWeight_crypto.png')
    plot_cpv(testPortfolioValues, testPortfolioValues_equiweight, PortfolioValueInit_test, 'cnn_cpv_crypto.png')

main()

# %% [markdown]
# # LSTM

# %%
class PolicyLSTM(object):
    def __init__(self, ohlc_feature_num, tickerNum, numTradingPeriods, sess, optimizer, tradingCost, cashBias_init, interestRate, equiweight_vector, adjusted_rewards_alpha, numFilterLayer):
        self.ohlc_feature_num = ohlc_feature_num
        self.tickerNum = tickerNum
        self.numTradingPeriods =  numTradingPeriods
        self.tradingCost = tradingCost
        self.cashBias_init = cashBias_init
        self.interestRate = interestRate
        self.equiweight_vector = equiweight_vector
        self.adjusted_rewards_alpha = adjusted_rewards_alpha 
        self.optimizer = optimizer
        self.sess = sess
        self.numFilterLayer = numFilterLayer
        layers=2
        lstm_neurons = 20
        self.X_t = tf.placeholder(tf.float32, [None, self.ohlc_feature_num, self.tickerNum, self.numTradingPeriods])
        self.weightsPrevious_t = tf.placeholder(tf.float32, [None, self.tickerNum + 1])
        self.PortfolioPrevious_t = tf.placeholder(tf.float32, [None, 1])
        self.dailyReturns_t = tf.placeholder(tf.float32, [None, self.tickerNum]) 
        cashBias = tf.get_variable('cashBias', shape=[1, 1, 1, 1], initializer = tf.constant_initializer(self.cashBias_init))
        shape_X_t = tf.shape(self.X_t)[0]
        self.cashBias = tf.tile(cashBias, tf.stack([shape_X_t, 1, 1, 1]))
        def lstm(X_t):
            network = tf.transpose(X_t, [0, 1, 3, 2])
            network = network / network[:, :, -1, 0, None, None]
            for layer_number in range(layers):
                resultlist = []
                reuse = False
                for i in range(self.tickerNum):
                    if i > 0:
                        reuse = True
                    result = tflearn.layers.lstm(X_t[:,:,:, i],
                                                    lstm_neurons,
                                                    dropout=0.3,
                                                    scope="lstm"+str(layer_number),
                                                    reuse=reuse)
                    resultlist.append(result)
                network = tf.stack(resultlist)
                network = tf.transpose(network, [1, 0, 2])
                network = tf.reshape(network, [-1, 1, self.tickerNum, lstm_neurons])
            return network

        def policy_output(network, cashBias):
            with tf.variable_scope("Convolution_Layer"):
                self.conv = tf.layers.conv2d(
                    inputs = network,
                    activation = tf.nn.relu,
                    filters = 1,
                    strides = (numFilterLayer + 1, 1),
                    kernel_size = (1, 1),
                    padding = 'same')
            with tf.variable_scope("Policy-Output"):
                tensor_squeeze = tf.squeeze(tf.concat([cashBias, self.conv], axis=2), [1,3])
                self.action = tf.nn.softmax(tensor_squeeze)
            return self.action


        def reward(shape_X_t, actionChosen, interestRate, weightsPrevious_t, PortfolioPrevious_t, dailyReturns_t, tradingCost):
            
            with tf.variable_scope("Reward"):
                cash_return = tf.tile(tf.constant(1 + interestRate, shape=[1, 1]), tf.stack([shape_X_t, 1]))
                y_t = tf.concat([cash_return, dailyReturns_t], axis=1)
                portfolioVector_t = actionChosen * PortfolioPrevious_t
                portfolioVector_previous = weightsPrevious_t * PortfolioPrevious_t

                total_tradingCost = tradingCost * tf.norm(portfolioVector_t - portfolioVector_previous, ord=1, axis=1) * tf.constant(1.0, shape=[1])
                total_tradingCost = tf.expand_dims(total_tradingCost, 1)

                zero_vector = tf.tile(tf.constant(np.array([0.0] * tickerNum).reshape(1, tickerNum), shape=[1, tickerNum], dtype=tf.float32), tf.stack([shape_X_t, 1]))
                cost_vector = tf.concat([total_tradingCost, zero_vector], axis=1)

                portfolioVector_second_t = portfolioVector_t - cost_vector
                final_portfolioVector_t = tf.multiply(portfolioVector_second_t, y_t)
                PortfolioValue = tf.norm(final_portfolioVector_t, ord=1)
                self.instantaneous_reward = (PortfolioValue - PortfolioPrevious_t) / PortfolioPrevious_t
                
            with tf.variable_scope("Reward-Equiweighted"):
                cash_return = tf.tile(tf.constant(1 + interestRate, shape=[1, 1]), tf.stack([shape_X_t, 1]))
                y_t = tf.concat([cash_return, dailyReturns_t], axis=1)  
                portfolioVector_eq = self.equiweight_vector * PortfolioPrevious_t
                PortfolioValue_eq = tf.norm(tf.multiply(portfolioVector_eq, y_t), ord=1)
                self.instantaneous_reward_eq = (PortfolioValue_eq - PortfolioPrevious_t) / PortfolioPrevious_t

            with tf.variable_scope("Reward-adjusted"):
                self.adjusted_reward = self.instantaneous_reward - self.instantaneous_reward_eq - self.adjusted_rewards_alpha * tf.reduce_max(actionChosen)
            return self.adjusted_reward


        self.lstm_layer = lstm(self.X_t)
        self.actionChosen = policy_output(self.lstm_layer, self.cashBias)
        self.adjusted_reward = reward(shape_X_t, self.actionChosen, self.interestRate, self.weightsPrevious_t, self.PortfolioPrevious_t, self.dailyReturns_t, self.tradingCost)
        self.train_op = optimizer.minimize(-self.adjusted_reward)

    def compute_weights(self, X_t_, weightsPrevious_t_):
        return self.sess.run(tf.squeeze(self.actionChosen), feed_dict={self.X_t: X_t_, self.weightsPrevious_t: weightsPrevious_t_})

    def train_lstm(self, X_t_, weightsPrevious_t_, PortfolioPrevious_t_, dailyReturns_t_):
        self.sess.run(self.train_op, feed_dict={self.X_t: X_t_,
                                                self.weightsPrevious_t: weightsPrevious_t_,
                                                self.PortfolioPrevious_t: PortfolioPrevious_t_,
                                                self.dailyReturns_t: dailyReturns_t_})

# %%
numFilterLayer= 20

def maxDrawDown(weights, time_period=numTradingPeriods, portfolioDataFrame=pivot_table):
    weights = weights.reshape(len(weights[0],))
    weights = weights[1:]
    simulated_portfolio = weights[0]*portfolioDataFrame.iloc[:,0]
    for i in range(1, len(portfolioDataFrame.columns)):
        simulated_portfolio += weights[i]*portfolioDataFrame.iloc[:,i]
    maxDrawDown_value = float('-inf')
    for i in range(int(len(simulated_portfolio)/time_period)-1):
        if min(simulated_portfolio[i*time_period:(i+1)*time_period]) > 0:
            biggestVariation = max(simulated_portfolio[i*time_period:(i+1)*time_period])/min(simulated_portfolio[i*time_period:(i+1)*time_period])
        else:
            biggestVariation = 0
        if(biggestVariation > maxDrawDown_value):
            maxDrawDown_value = biggestVariation
    return maxDrawDown_value

def getRandAction(num_tickers):
    vector_rand = np.random.rand(num_tickers + 1)
    return vector_rand / np.sum(vector_rand)


def sharpeCrypto(w ,returnsCov=cryptoCovarianceMatrix, returns_mean=crypto_returns_mean):
    w = w.reshape(len(w[0],))
    w = w[1:]
    portfolioReturn = np.sum(returns_mean.values * w * tradingDaysCaptured)
    portfolio_volatility = np.sqrt(np.dot(w.T, np.dot(returnsCov.values, w))* tradingDaysCaptured)
    sharpe_ratio = portfolioReturn/portfolio_volatility
    return sharpe_ratio

def RoMad(weights, mdd, returns_mean=crypto_returns_mean,time_period=tradingDaysCaptured):
    weights = weights.reshape(len(weights[0],))
    weights = weights[1:]
    portfolioReturn = np.sum(returns_mean.values * weights * time_period)
    if mdd>0:
        romad = portfolioReturn/mdd
    else:
        romad=0
    return romad


def main(assets = True):
    env_pf_optimizer = RLEnv(Path = data_path, PortfolioValue = PortfolioValueInit, TransCost = tradingCost, 
        ReturnRate = interestRate, WindowSize = numTradingPeriods, TrainTestSplit = train_data_ratio)
    envPortfolioEquiweight = RLEnv(Path = data_path, PortfolioValue = PortfolioValueInit, TransCost = tradingCost, 
        ReturnRate = interestRate, WindowSize = numTradingPeriods, TrainTestSplit = train_data_ratio)
    tf.reset_default_graph()
    sess = tf.Session()
    PortfolioOptimizingAgent = PolicyLSTM(ohlcFeaturesNum, tickerNum, numTradingPeriods, sess, optimizer, tradingCost, cashBias_init, interestRate, equiweight_vector, adjusted_rewards_alpha, numFilterLayer)
    sess.run(tf.global_variables_initializer())
    trainPortfolioValues = []
    trainPortfolioValues_equiweight = []
    for ep_num in tqdm.tnrange(num_episodes, desc = 'Episodes'):
        pvm = PVM(tickerNum, beta_pvm, trainingSteps, trainingBatchSize, weight_vector_init)
        for batch_num in tqdm.tnrange(numBatches, desc = 'Batches'):
            list_X_t, listWeight_previous, listPortfolioValue_previous, listDailyReturns_t, listPortfolioValue_previous_equiweight, listSharpeTraining = [], [], [], [], [], []
            training_batch_t_selection = False
            while training_batch_t_selection == False:
                training_batch_t = trainingSteps - trainingBatchSize + 1 - np.random.geometric(p = beta_pvm)
                if training_batch_t >= 0: 
                    training_batch_t_selection = True
            state, done = env_pf_optimizer.ResetEnvironment(pvm.getWeights_t(int(training_batch_t)), PortfolioValueInit , training_batch_t  )
            stateEquiweight, done_equiweight = envPortfolioEquiweight.ResetEnvironment(equiweight_vector, PortfolioValueInit, training_batch_t)
            for training_batch_num in tqdm.tnrange(trainingBatchSize, desc = 'Training Batches'): 
                X_t = state[0].reshape([-1] + list(state[0].shape))
                WeigtPrevious = state[1].reshape([-1] + list(state[1].shape))
                PortfolioValue_previous = state[2]
                action = PortfolioOptimizingAgent.compute_weights(X_t, WeigtPrevious) if (np.random.rand() < epsilon) else getRandAction(tickerNum)
                state, reward, done = env_pf_optimizer.Step(action)
                stateEquiweight, rewardEquiweight, done_equiweight = envPortfolioEquiweight.Step(equiweight_vector)                
                X_next = state[0]
                Weight_t = state[1]
                PortfolioValue_t = state[2]  
                PortfolioValue_t_equiweight = stateEquiweight[2]
                dailyReturns_t = X_next[-1, :, -1]
                pvm.updateWeight_vector_t(training_batch_t + training_batch_num, Weight_t)
                list_X_t.append(X_t.reshape(state[0].shape))
                listWeight_previous.append(WeigtPrevious.reshape(state[1].shape))
                listPortfolioValue_previous.append([PortfolioValue_previous])
                listDailyReturns_t.append(dailyReturns_t)
                listPortfolioValue_previous_equiweight.append(PortfolioValue_t_equiweight)
                sharpe_ratio = sharpeCrypto(w=WeigtPrevious)
                mdd = maxDrawDown(weights=WeigtPrevious)
                romad = RoMad(weights=WeigtPrevious, mdd=mdd)
                listSharpeTraining.append(sharpe_ratio)
                print('------------------ training -----------------------')
                print('current portfolio value : ' + str(PortfolioValue_previous))
                print('weights assigned : ' + str(WeigtPrevious))
                print('sharpe_ratio:', sharpe_ratio)
                print('RoMad', romad)
                print('equiweight portfolio value : ' + str(PortfolioValue_t_equiweight))            
            trainPortfolioValues.append(PortfolioValue_t)
            trainPortfolioValues_equiweight.append(PortfolioValue_t_equiweight)
            PortfolioOptimizingAgent.train_lstm(np.array(list_X_t), 
                np.array(listWeight_previous),
                np.array(listPortfolioValue_previous), 
                np.array(listDailyReturns_t))

    print('------ train final value -------')
    print(trainPortfolioValues)
    print(trainPortfolioValues_equiweight)
    state, done = env_pf_optimizer.ResetEnvironment(weightVector, PortfolioValueInit_test, training_batch_t)
    stateEquiweight, done_equiweight = envPortfolioEquiweight.ResetEnvironment(equiweight_vector, PortfolioValueInit_test, training_batch_t)
    testPortfolioValues = [PortfolioValueInit_test]
    testPortfolioValues_equiweight = [PortfolioValueInit_test]
    weightVectors = [weightVector]
    test_sharpe_ratio = []
    test_mdd = []
    startStepNum = int(trainingSteps  + validationSteps - int(numTradingPeriods/2))
    end_step_num = int(trainingSteps  + validationSteps + testSteps - numTradingPeriods)

    for test_step_num in range(startStepNum, end_step_num):
        X_t = state[0].reshape([-1] + list(state[0].shape))
        wt_previous = state[1].reshape([-1] + list(state[1].shape))
        PortfolioValue_previous = state[2]
        action = PortfolioOptimizingAgent.compute_weights(X_t, wt_previous)
        state, reward, done = env_pf_optimizer.Step(action)
        stateEquiweight, rewardEquiweight, done_equiweight = envPortfolioEquiweight.Step(equiweight_vector)
        X_next_t, wt_t, PortfolioValue_t = state[0], state[1], state[2]
        PortfolioValue_t_equiweight = stateEquiweight[2]
        dailyReturns_t = X_next_t[-1, :, -1]
        sharpe_ratio = sharpeCrypto(w=wt_previous)
        mdd = maxDrawDown(weights=WeigtPrevious)
        romad = RoMad(weights=WeigtPrevious, mdd=mdd)
        print('------------------ testing -----------------------')
        print('sharpe_ratio:', sharpe_ratio)
        print('RoMad', romad)

        testPortfolioValues.append(PortfolioValue_t)
        testPortfolioValues_equiweight.append(PortfolioValue_t_equiweight)
        weightVectors.append(wt_t)
        test_sharpe_ratio.append(sharpe_ratio)
        test_mdd.append(romad)

    print('------ test final value -------')
    print('Mean sharpe ratio : ' + str(np.nanmean(test_sharpe_ratio)))
    print('Mean Max Drawdown:', str(np.mean(test_mdd)))
    
    plotWeights_assigned(weightVectors[-1], tickerNum, ticker_list, 'lstmWeight_crypto.png')
    plot_cpv(testPortfolioValues, testPortfolioValues_equiweight, PortfolioValueInit_test, 'lstm_cpv_crypto.png')

main()


