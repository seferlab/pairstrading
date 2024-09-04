from datetime import date
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def get_Return(price):
    '''
    Calculate the return of the price series
    Notice: Fill the null value with 0
    
    Parameters
    ----------
    price: a list/series of price

    Returns
    -------
    return: a list/series of return

    '''
    ret = (price - price.shift(1)) / price.shift(1)
    ret = ret.drop(ret.index[0])
    # fill the nan values with 0
    ret = ret.fillna(value=0)
    return ret
def find_Factor(ret, delay, fac_num):
    '''
    Parameters:
    ----------------
    ret: a list/series of return
    delay: the delay of the factor
    fac_num: the number of PCA factors

    Return:
    ----------------
    factor_ret: a list/series of factor
    weight: the weight of each factor

    '''
    # standardize the return
    mean = ret.mean(axis=0)
    std = ret.std(axis=0)
    std_ret = (ret - mean) / std
    ##
    ##    # PCA process
    pca = PCA(n_components=fac_num)
    pca.fit(std_ret)
    # pca.fit(std_ret[0:delay])
    factor_ret = pd.DataFrame(pca.components_)
    factor_ret.columns = std.index
    factor_ret = factor_ret / std
    weight = pd.DataFrame(np.dot(ret, factor_ret.transpose()), index=ret.index)
    return factor_ret, weight
def find_Parameter(X):
    '''
    '''

    rid_train_params = []

    for i in X.columns:
        model = AutoReg(endog=X[i], lags=1)
        res = model.fit()
        rid_train_params.append([res.params[0], res.params[1], np.var(res.resid)])
        
    return rid_train_params

delay = 252
window = 60
factor = 20
K = 8.4
#sbo = 1.45
#sso = 1.45
#sbc = 0.45
#ssc = 0.75
sbo=1.25
sso=1.25
sbc=0.50
ssc=0.75
r = 0.02
tran_cost = 0.000
leverage = 1.0
start_val = 100
bo = 1
so = -1
stocks_number= 740 #445

df = pd.read_csv('./Data/sp500_2012_2022_1.csv', index_col=['symbol'])
timeline = pd.read_csv('./Data/timeline_modified.csv')


# return_data is return_data
# return_data contains daily precentage return of all stocks
data = df.T
return_data = data.apply(lambda x: get_Return(x)).T

# get s_score for five stocks
s_score_pd = pd.DataFrame(index=return_data.index)
# factor_rets=pd.DataFrame(index=return_data_five.index)
weight_pd = pd.DataFrame(index=return_data.index)  # save the lambda i


for i in range(0, len(return_data.columns) - delay):    

    return_matrix = return_data.iloc[:, i:i + delay]
    time = return_data.columns[i + delay - 1]
    # get pca matrix
    factor_ret, weight = find_Factor(return_matrix, delay, factor)
    pca_return = pd.DataFrame(np.dot(weight, factor_ret), columns=return_matrix.columns, index=return_matrix.index)
    residual = return_matrix - pca_return
    # factor_rets[time]=factor_ret.iloc[:,0]

    weight.columns = [time + ' ' + str(x) for x in range(1, factor+1)]

    weight_pd = weight_pd.join(weight)

    residual = residual.iloc[:, -window:]

    X = pd.DataFrame(columns=range(1, window + 1))

    for j in range(1, window + 1):
        if (j == 1):
            X[j] = residual.iloc[:, j - 1]
        else:
            X[j] = X[j - 1] + residual.iloc[:, j - 1]


    X_T = X.drop(window, axis=1).T 
    X_T.reset_index(drop=True, inplace=True)
    rid_train_params = find_Parameter(X_T)


    # calculate the k, m, sigma
    k_list = []
    m_list = []
    sigma_list = []
    for k in rid_train_params:
        if k[1] < 0:
            k_list.append(None)
            m_list.append(None)
            sigma_list.append(None)
        else:
            k_list.append(-np.log(k[1]) * 252)
            m_list.append(k[0] / (1 - k[1]))
            sigma_list.append(np.sqrt(k[2] / (1 - k[1] ** (2))))
    # get the s-score s=x -m / sigma
    s_score_list = []
    count = 0

    for item in X.index:
        if (sigma_list[count] == 0 or k_list[count] is not None and k_list[count] < K or k_list[count] is None):
            s_score_list.append(0)
            count += 1
            
        else:
            s_score = (X.loc[item][window] - m_list[count]) / sigma_list[count]
            if s_score > 5 or s_score < -5:
                print(
                    "The abnormal one s_score is%f, item is%s, X is %.3f, m is %.3f, sigma is %.3f, a is %.3f, b is %.3f" % (
                        s_score, item, X.loc[item][window], m_list[count], sigma_list[count],
                        rid_train_params[count][0],
                        rid_train_params[count][1]))
            s_score_list.append(s_score)
            count += 1
    print("got the %s time s-score" % (time))
    s_score_pd[time] = s_score_list
    
    

s_score_pd.to_pickle('./Data/nb_s_score.pkl')
weight_pd.to_pickle('./Data/nb_weights.pkl')
# read s_score and weight
s_score_pd = pd.read_pickle('./Data/nb_s_score.pkl')
weight_pd = pd.read_pickle('./Data/nb_weights.pkl')


def call_dollar_amounts(position_stock, stocks_number, i, weight_pd, cash, tran_cost):
    buying_dollars = 0
    selling_dollars = 0
    columns = ['s' + str(x) for x in range(1, stocks_number+1)]
    stock_situation = pd.DataFrame(0, index=position_stock.index, columns=columns)
    pca_list = ['pca' + str(x) for x in range(1, factor+1)]
    # enumerate the psition_stock
    for k in position_stock.index:
        # if we have trading signal, then we calculating the buying and selling
        if (position_stock.loc[k]['s_score'] != 0 and position_stock.loc[k]['tag'] == 1):
            # stock_situation[k]+=position_stock.loc[k]['s_score']#first add leverage then calculate the pca part
            for j in position_stock.columns:
                if j in pca_list:
                    weight = weight_pd.loc[:, i + ' ' + j[3:]] * position_stock.loc[k][j] # 3: is the pca number
                    # weight = weight_pd.loc[:, i + ' ' + j[3]] * position_stock.loc[k][j]
                    stock_situation.loc[k, :] += weight.values
             
    total_dollars = stock_situation.abs().sum().sum()
    # add trasaction cost
    cash = cash - cash * tran_cost
    for index, row in stock_situation.iterrows():
        row['s1':'s'+str(stocks_number)] = row['s1':'s'+str(stocks_number)] / total_dollars * cash
    for k in position_stock.index:
        if (position_stock.loc[k]['tag'] == 1):
            position_stock.loc[k, 's1':'s'+str(stocks_number)] = stock_situation.loc[k, :]
    cash = 0

    return cash, position_stock


# back_testing
# pnl = cash +holding
# we need update holding at the begining and then use the s-score, if no cash, no trading
pnl = pd.Series(0,index=s_score_pd.loc[:,'2016-09-20':].columns)
pnl_initial = 10000.0
holding_pd = pd.Series(0,index=s_score_pd.loc[:,'2016-09-20':].columns)
cash = pnl_initial  # cash = pnl_initial
pnl.iloc[0] = pnl_initial
cash_pd = pd.Series(0,index=s_score_pd.loc[:,'2016-09-20':].columns)
columns = ['tag', 's_score'] + ['pca' + str(x) for x in range(1, factor+1)] + ['s' + str(x) for x in range(1, stocks_number+1)]
# if tag =1 means we buy to open this portfolio, if tag >1 means we cannot trade it because we didn't meet close signal yet.
position_stock = pd.DataFrame(0, index=s_score_pd.index,
                              columns=columns)
# position_stock_before = pd.Series(0, index=s_score_pd.index)
pnl_before = 0
time_count = 0  # tag the first, updating the pnl from the second day.
count = 0
buy_count=0
sell_count=0


for i in s_score_pd.loc[:,'2016-09-20':].columns:
    trading_count=0
    s_list = ['s' + str(x) for x in range(1, stocks_number+1)]
    if (count > 0):
        # updating the holding accorind to the stock return
        for name, values in position_stock.iteritems():
            # check if name in the s1, s2, s3, s4 .... s20 if name is s1, update [0][i]
            if (name in s_list):
                # position_stock.loc[:, name] += values * return_data.iloc[int(name[1]) - 1][i]
                position_stock.loc[:, name] += values * return_data.iloc[int(name[1:]) - 1][i]
    tag = 0  # if we have bo or so tag++ then we will calculate the call_dollar_amounts
    for j in s_score_pd.index:
        pca_list = ['pca' + str(x) for x in range(1, factor+1)]

        # buy to open or sell to open
        if (position_stock.loc[j]['s_score'] == 0):
            # buy to open :buying  1 dollar stock selling R ETF
            if (s_score_pd[i][j] < -sbo and cash > 0):
                trading_count+=1
                tag += 1
                position_stock.loc[j, 'tag'] = 1
                position_stock.loc[j, 's_score'] = bo
                buy_count+=1
                # pca1= time +' 1'
                for pca in pca_list:

                    position_stock.loc[j, pca] = float(leverage * weight_pd.loc[j][i + ' ' + pca[3:]])

            # sell to open: selling 1 dollar stock buying R ETF
            elif (s_score_pd[i][j] > sso and cash > 0):
                trading_count+=1
                tag += 1
                buy_count+=1
                position_stock.loc[j, 'tag'] = 1
                position_stock.loc[j, 's_score'] = so
                for pca in pca_list:

                    position_stock.loc[j, pca] = float(-leverage * weight_pd.loc[j][i + ' ' + pca[3:]])
 
            # close long position: selling stock buying ETF
        elif (position_stock.loc[j]['s_score'] > 0 and s_score_pd[i][j] > -ssc):
            trading_count+=1
            cash += abs(position_stock.loc[j]['s1':'s'+str(stocks_number)]).sum()
            # add transaction cost
            cash = cash - tran_cost * cash
            sell_count+=1
            position_stock.loc[j, :] = 0
            # close short position: buying stock selling ETF
        elif (position_stock.loc[j]['s_score'] < 0 and s_score_pd[i][j] < sbc):
            trading_count+=1
            cash += abs(position_stock.loc[j]['s1':'s'+str(stocks_number)]).sum()
            cash = cash - tran_cost * cash
            position_stock.loc[j, :] = 0
            sell_count+=1
        else:
            position_stock.loc[j, 'tag'] = 2

    # if tag>0 means we have open position this iteration.
    print("we have %s times trading at %s time" % (trading_count, i))
    print("we have %.4f$ cash at %s time" % (cash, i))

    if (tag > 0):
        cash, position_stock = call_dollar_amounts(position_stock, stocks_number, i, weight_pd, cash, tran_cost)
    holding = position_stock.loc[:, 's1':'s'+str(stocks_number)].abs().sum().sum()
    holding_pd[i] = holding
    cash_pd[i] = cash
    print("we have %.4f$ holding at %s time" % (holding, i))
    pnl_before = pnl.loc[i]

    pnl[i] = holding + cash
    print("we have %.4f$ total amount at %s" % (pnl[i], i))
    count += 1


cash_pd.to_csv("./Data/nbrv_cash.csv")
holding_pd.to_csv("./Data/nbrv_holding.csv")
pnl.to_csv("./Data/nbrv_pnl.csv")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
pnl.plot(ax=ax, style='k-', label='pnl') 
ax.legend(loc='best')
plt.savefig('./Data/nbrv_pnl_amount.jpg')
plt.show()

# calculate the sharpe ratio
pnl = pd.read_csv("./Data/nbrv_pnl.csv",names=['time','pnl'])
pnl=pnl.drop([0])
pnl['time']=pd.to_datetime(pnl['time'])
last_list=pnl.groupby([pd.Grouper(key = 'time', freq = 'Y')])['time'].last()
first_list=pnl.groupby([pd.Grouper(key = 'time', freq = 'Y')])['time'].first()

annual_return = []

for first,last in zip(first_list,last_list):
    last_day=float(pnl.loc[(pnl['time']==last,['pnl'])].values)

    first_day=float(pnl.loc[(pnl['time']==first,['pnl'])].values)

    annual_return.append(((last_day-first_day)/first_day)*100)
    
print(annual_return)

pnl=pnl.set_index('time')
pnl['return'] = get_Return(pnl)
pnl['return'] = pnl['return'].fillna(value=0)
sharpe_ratio = []
for first,last in zip(first_list,last_list):
    days=np.shape(pnl[first:last])[0]
    average_return = pnl[first:last]['return'].mean() * days
    std_return = pnl[first:last]['return'].std() * ((days) ** 0.5)
    sharpe_ratio.append(average_return / std_return)
    
print(sharpe_ratio)
print("sell count:%f"%(sell_count))
print("buy count:%f"%(buy_count))
