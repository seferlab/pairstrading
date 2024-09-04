import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import os

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Global settings
delay = 252
window = 60
factor = 15
r = 0.02
tran_cost = 0.0002
leverage = 1.0
start_val = 100
bo = 1
so = -1
ol=0.5
co=0.8
cs=0.2
ol_label=1
os_label=-1
cs_label=-2
co_label=2
training_number=3

# Functions
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

def find_Factor(ret, fac_num):
    '''
    Parameters:
    ----------------
    ret: a list/series of return
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
    factor_ret = pd.DataFrame(pca.components_)
    factor_ret.columns = std.index
    factor_ret = factor_ret / std
    weight = pd.DataFrame(np.dot(ret, factor_ret.transpose()), index=ret.index)
    return factor_ret, weight

def call_dollar_amounts(position_stock, stocks_number, i, weight_pd, cash, tran_cost):
    '''
    Parameters:
    ----------------
    position_stock: the position table
    stocks_number: the number of stocks to be selected
    i: the index of the position table
    weight_pd: the weight of each factor
    cash: the initial cash
    tran_cost: the transaction cost

    Return:
    ----------------
    cash: the cash after the transaction
    position_stock: the position table after the transaction
    '''
    buying_dollars = 0
    selling_dollars = 0
    columns = ['s' + str(x) for x in range(1, stocks_number+1)]
    stock_situation = pd.DataFrame(0, index=position_stock.index, columns=columns)
    pca_list = ['pca' + str(x) for x in range(1, factor+1)]
    # enumerate the psition_stock
    for k in position_stock.index:
        # if we have trading signal, then we calculating the buying and selling
        if (position_stock.loc[k]['up_down'] != 0 and position_stock.loc[k]['tag'] == 1):
            # stock_situation[k]+=position_stock.loc[k]['s_score']#first add leverage then calculate the pca part
            for j in position_stock.columns:
                if j in pca_list:
                    weight = weight_pd.loc[:, i + ' ' + j[3]] * position_stock.loc[k][j]
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

def trainAdaBoost(X,Y,previous_model):
    '''
    X: training data
    Y: training label
    previous_model: the previous model
    '''
    # if there is no previous model, then we train a new model
    if previous_model == None:
        base = DecisionTreeClassifier(max_depth=1) #5
        model=AdaBoostClassifier(learning_rate=0.3,n_estimators=50,base_estimator=base)
        model.fit(X,Y.ravel())
    # if there is a previous model, then we update the model
    else:
        model=previous_model
        model.fit(X,Y.ravel())
    return model

def select_stocks(return_data, timeline, date_start, date_end, period):
    '''
    Parameters:
    ----------------
    return_data: a dataframe of return
    timeline: a list of date, the timeline of the stocks existence on S&P 500
    date_start: the start date of the predict period
    date_end: the end date of the predict period, default to be a 60 days interval
    period: length of historical data
        - if 'hist' : all historical data
        - if n: the number of years before the date_start

    Return:
    ----------------
    selected_stocks: the past years data of stocks that are in the S&P 500 during the predict period

    '''
    time_list = pd.to_datetime(return_data.columns)

    if period == 'hist':
        # start from the 3rd year and get all historical data
        # select_time = time_list[time_list <= pd.Timestamp(date_start)]
        select_time = time_list[time_list <= pd.Timestamp(date_end)]
    else:
        # start from the 3rd year and get 3-year historical data each time
        # select_time = time_list[(time_list <= pd.Timestamp(date_start)) & (time_list >= pd.Timestamp(date_start) - pd.DateOffset(years=period))]
        select_time = time_list[(time_list <= pd.Timestamp(date_end)) & (time_list >= pd.Timestamp(date_end) - pd.DateOffset(years=period))]

        # ensure that each time extract 4 x 252 days of data
        if len(select_time) != period*252:
            shift_days = period*252 - len(select_time)
            select_time = time_list[(time_list <= pd.Timestamp(date_end)) & 
                                    (time_list >= pd.Timestamp(date_end) - pd.DateOffset(years=period) - pd.DateOffset(days=shift_days))]

    select_time = select_time.strftime("%Y-%m-%d %H:%M:%S")

    window_time = time_list[(time_list <= pd.Timestamp(date_end)) & (time_list >= pd.Timestamp(date_start))]
    window_time = window_time.strftime("%Y-%m-%d %H:%M:%S")

    stocks_ticker = timeline[(timeline['Added'] <= date_start) & (timeline['Removed'] >= date_end)]['Ticker'].tolist()

    # remove stocks without values in this window
    period_stock_data = return_data.loc[stocks_ticker, window_time]
    period_stock_data = period_stock_data.loc[~(period_stock_data==0).all(axis=1)]
    stocks_ticker = period_stock_data.index.tolist()

    stocks_data = return_data.loc[stocks_ticker, select_time]
    
    return stocks_data

def pca_residual(return_data, delay, factor):
    '''
    Parameters:
    ----------------
        return_data: the return data of this period
        delay: the delay
        factor: the number of PCA factors
    
    Return:
    ----------------
        residual: the residual data of `len(return_data.columns) - delay`
        weight: the weight of each factor
        
    '''

    ###get X_pd
    weight_pd = pd.DataFrame(index=return_data.index)  # save the lambda i
    X_pd = pd.DataFrame(columns=return_data.index)

    for i in range(0, len(return_data.columns) - delay + 1): # no first 252 days, thus need the length of data at least 252
        return_matrix = return_data.iloc[:, i:i + delay]
        time = return_data.columns[i + delay - 1]
        # get pca matrix
        factor_ret, weight = find_Factor(return_matrix, factor)
        # factor_ret is the dimension-reduced matrix
        pca_return = pd.DataFrame(np.dot(weight, factor_ret), columns=return_matrix.columns, index=return_matrix.index)
        residual = return_matrix - pca_return

        # this should be adjustable with the factor number
        # version of adjustable
        weight.columns = [time + ' ' + str(x) for x in range(1, factor+1)]

        weight_pd = weight_pd.join(weight)
        
        residual = residual.iloc[:, -window:]

        X = pd.DataFrame(columns=range(1, window + 1))

        # get the cumulative residual data
        for j in range(1, window + 1):
            if (j == 1):
                X[j] = residual.iloc[:, j - 1] # residual.iloc[:,0] 
            else:
                X[j] = X[j - 1] + residual.iloc[:, j - 1] # X[1] + residual.iloc[:,1] 
                
        X_T = X.T
        #    X_pd = X_pd.append(X_T, ignore_index=True)
        X_pd = pd.concat((X_pd, X_T), axis=0)
    
        # print("add %s X " % (time))
    
    X_pd.reset_index(drop=True, inplace=True)

    return X_pd, weight_pd
    
def get_train_test_data(X_pd, return_data, training_number, delay, window):
    '''
    Parameters:
    ------------
        X_pd: the residual data output from `pca_residual()`
        training_number: the number of training years
        delay: trading period - default 252 days
        window: default 60 days

    Returns:
    ------------
        X_training: the training data
        Y_training: the training label
        X_testing: the testing data
        Y_testing: the testing label

    '''
    

    # X_training = pd.DataFrame(columns=range(0, window))
    X_training = pd.Series(np.zeros(window))
    Y_training = []
    # X_testing = pd.DataFrame(columns=range(0, window))
    X_testing = pd.Series(np.zeros(window))
    Y_testing = {}
    # num = training_number * delay * window # each day of a year generates a window 252 x 60
    num = (training_number * delay - window) * window # training days except last window days

    for tag in X_pd.columns:
        #     #i=0 ....len(X_pd.index)-window
        for i in range(0, num, window):
            X_example = X_pd.loc[i:i + window - 1][tag].values
            #    X_training = X_training.append(pd.Series(X_example), ignore_index=True) 
            X_training = pd.concat((X_training,pd.Series(X_example)), axis=1, ignore_index=True)
            Y_training.append(X_pd.loc[i + window][tag]) # the value of the next day 61th day
            
        # X_training 

        #   # get the testing data
        # r = -int(len(return_data.columns) - (num / window + delay))
        r = -60

        for i in range(num, len(X_pd.index) - window, window):
            X_example = X_pd.loc[i:i + window - 1][tag].values
            # get time index
            time = return_data.columns[r]
            index_name = time + " " + tag
            r += 1
            # X_testing = X_testing.append(pd.Series(X_example, name=index_name))
            X_testing = pd.concat((X_testing, pd.Series(X_example)), axis=1, ignore_index=True)
            Y_testing[index_name] = X_pd.loc[i + window][tag]
        print('done %s stocks' % (tag))

    X_training = X_training.T.iloc[1:,:].reset_index(drop=True)
    X_testing = X_testing.T.iloc[1:,:].reset_index(drop=True)
    
    Y_training = pd.DataFrame(Y_training)
    Y_testing = pd.DataFrame.from_dict(Y_testing, orient='index', columns=['values'])
    
    X_testing.index = Y_testing.index

    return X_training, Y_training, X_testing, Y_testing

def back_testing_tables(return_data, pnl_initial, factor, stocks_number):
    '''
    Parameters:
    ------------
        return_data: the return data of all stocks
        pnl_initial: the initial amount of money
        num(default): -60, the last 60 days of data
        factor: the number of PCA factors
        stocks_number: the number of stocks to be selected
     
    Returns:
    ------------
        pnl: the profit and loss
        holding_pd: the holding table
        cash: the initial cash
        cash_pd: the cash table
        up_down_stock: the up and down table
        holding_stock: the holding stock table
        position_stock: the position table
        ensemble_stock: the ensemble table
        
    '''
    num = -60

    pnl = pd.Series(index=return_data.columns[num:], dtype='float64')
    pnl.iloc[0] = pnl_initial   

    holding_pd = pd.Series(index=return_data.columns[num:], dtype='float64')
    # cash = pnl_initial
    cash = pnl_initial  
    cash_pd = pd.Series(index=return_data.columns[num:], dtype='float64')
    
    columns = ['s' + str(x) for x in range(1, stocks_number+1)]
    # if tag =1 means we hold, if up_down =1 means it will go up, we will buy it.
    # if tag =0 means we don't hold, if up_down =0 means it will go down, we will sell it. # from copilot
    # 1 = holding | 0 = not holding
    up_down_stock = pd.DataFrame(0, columns=return_data.index, index=return_data.columns[num:])
    holding_stock = pd.Series(0, index=columns)

    ensemble_stock = pd.DataFrame(0, columns=return_data.index, index=return_data.columns[num:])

    position_columns = ['tag', 'up_down'] + ['pca' + str(x) for x in range(1, factor+1)] + ['s' + str(x) for x in range(1, stocks_number+1)]
    # if tag =1 means we buy to open this portfolio, if tag >1 means we cannot trade it because we didn't meet close signal yet.
    position_stock = pd.DataFrame(0, index=return_data.index, columns=position_columns)
    

    return  pnl, holding_pd, cash, cash_pd, up_down_stock, holding_stock, position_stock, ensemble_stock

def training_predict(X_training, Y_training, X_testing, Y_testing, saved_model, pnl, up_down_stock, ensemble_stock, stocks_number, ol, co, cs, ol_label, cs_label, os_label, co_label):
    '''
    
    Parameters
    ----------
    X_training : 
    Y_training :
    X_testing :
    Y_testing :
    saved_model :
    stocks_number :
    ol : open long
    co : close long
    cs : close short
    ol_label : +1
    cs_label : +2
    os_label : 0
    co_label : -1

    Returns
    -------
    up_down_stock : the predict result
    saved_model : the trained model
    ensemble_stock : the ensemble result
    
    '''
    time_list = pnl.index
    X_test = pd.DataFrame()
    Y_test = pd.DataFrame()

    # update the data
    for j in time_list:
        X_test = pd.concat((X_test, X_testing[X_testing.index.str.startswith(j)]), axis=0)
        Y_test = pd.concat((Y_test, Y_testing[Y_testing.index.str.startswith(j)]), axis=0)

    Y_tag = np.array(Y_training.values)
    X_train = np.array(X_training)
    Y_train = np.array(Y_tag)
    X_test_1 = np.array(X_test)
    Y_test_tag=np.array(Y_test)

    saved_model=trainAdaBoost(X_train,Y_train, saved_model)
    print("this is %s - %s time adboost"%(pnl.index[0], pnl.index[-1]),flush=True)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(saved_model, X_test_1, Y_test_tag.ravel(), scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    print("test acc:{0:.3f}% loss:{1:.3f}".format(np.mean(n_scores)*100,np.std(n_scores)), flush=True)

    # adaboost based on the decision tree does not output the probability, so we need to use the predict_proba
    predict_Y = saved_model.predict_proba(X_test_1)

    ensemble_Y = predict_Y[:,1] # save for ensemble
    ensemble_stock.loc[time_list[0]:time_list[-1], :] = ensemble_Y.reshape(len(time_list), stocks_number)

    predict_Y = predict_Y[:,1]

    # if predict>1 means they will up, else go down
        #ol=0.5 co=0.8 cs=0.2
    #1. "open long" if number is in (0.5,0.8) +1
    #2. "close long" if the number is in (0.8, ) -1
    #3. "open short" if the number is in (0.2,0.5) 0
    #4. "close short" if the number is in (,0.2) +2

    for x in range(0, len(predict_Y)):
        if predict_Y[x] >= ol and predict_Y[x]<co:
            predict_Y[x] = ol_label
        elif predict_Y[x]>=co:
            predict_Y[x] = cs_label
        elif predict_Y[x] < ol and predict_Y[x]>cs:
            predict_Y[x] = os_label
        else:
            predict_Y[x] = co_label
            
    predict_Y = predict_Y.reshape(len(time_list), stocks_number)
    up_down_stock.loc[time_list[0]:time_list[-1], :] = predict_Y

    return  up_down_stock, saved_model, ensemble_stock

def results(return_data, pnl, stocks_number, up_down_stock, factor, position_stock, tran_cost, cash, cash_pd, holding_pd, weight_pd, leverage, ol_label, cs_label, os_label, co_label):
    '''
    Parameters
    ----------

    
    Returns
    -------
    pnl : 
    position_stock :
    cash_pd : 
    holding_pd :
    
    '''

    pnl_before = 0
    # tag the first, updating the pnl from the second day.
    count = 0

    for i in pnl.index:
        s_list = ['s' + str(x) for x in range(1, stocks_number+1)]
        if (count > 0):
            # updating the holding accorind to the stock return
            for name, values in position_stock.iteritems():
                # check if name in the s1, s2, s3, s4 .... s20 if name is s1, update [0][i]
                if (name in s_list):
                    position_stock.loc[:, name] += values * return_data.iloc[int(name[1]) - 1][i]
                        

        tag = 0  # if we have bo or so tag++ then we will calculate the call_dollar_amounts

        # buy to open or sell to open

        for j in up_down_stock.columns:  # s1 s2 s3
            # holding =0 and up_down=1 and cash >0
            pca_list = ['pca' + str(x) for x in range(1, factor+1)]
            if(position_stock.loc[j]['up_down']==0):
                #open long
                if (up_down_stock.loc[i][j] == ol_label and cash > 0):
                    tag += 1
                    position_stock.loc[j,'tag'] = 1
                    position_stock.loc[j,'up_down'] = ol_label
                    for pca in pca_list:
                        position_stock.loc[j, pca] = float(leverage * weight_pd.loc[j][i + ' ' + pca[3]])
    
            # if flag =-1 open_short
                elif (up_down_stock.loc[i][j] == os_label and cash > 0):
                    tag += 1
                    position_stock.loc[j,'tag'] = 1
                    position_stock.loc[j,'up_down'] = os_label
                    for pca in pca_list:
                        position_stock.loc[j, pca] = float(-leverage * weight_pd.loc[j][i + ' ' + pca[3]])
            ## if flag =2 and stock_balance =0 open short, short stock
            elif (up_down_stock.loc[i][j] == co_label and position_stock.loc[j]['up_down']>0 ):
                #tag+=1
                cash += abs(position_stock.loc[j]['s1':'s'+str(stocks_number)]).sum()
                # add transaction cost
                cash = cash - tran_cost * cash
                position_stock.loc[j, :] = 0
            # if flag =-2 and stock_balance <0 clse short, sell stock
            elif (up_down_stock.loc[i][j] == cs_label and position_stock.loc[j]['up_down'] < 0):
                #tag+=1
                cash += abs(position_stock.loc[j]['s1':'s'+str(stocks_number)]).sum()
                cash = cash - tran_cost * cash
                position_stock.loc[j, :] = 0
            else:
                position_stock.loc[j, 'tag'] = 2
                

        # if tag>0 means we have open position this iteration.
        print("we have %s times trading at %s time" % (tag, i))
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

    return  pnl, position_stock, cash_pd, holding_pd

def plot_pnl(pnl, save_path):
    '''
    plot pnl
    
    '''
    fig = plt.figure(figsize = (50,20))
    ax = fig.add_subplot(1, 1, 1)
    pnl.plot(ax=ax, style='k-', label='pnl')  # 用dataframe 数据plot ax属性可以用subplot的ax传进去
    ax.legend(loc='best', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    # plt.show()
    plt.savefig(save_path)
    
def statistics(path_of_pnl, path_to_save):
    '''
    Parameters
    ----------
    path_of_pnl : the path of pnl file

    Returns
    -------
    annual_return : annual return
    sharpe_ratio : sharpe ratio
    
    '''
    pnl = pd.read_csv(path_of_pnl, names=['time', 'pnl'])
    pnl = pnl.drop([0])
    pnl['time'] = pd.to_datetime(pnl['time'])
    last_list = pnl.groupby([pd.Grouper(key='time', freq='Y')])['time'].last()
    first_list = pnl.groupby([pd.Grouper(key='time', freq='Y')])['time'].first()

    annual_return = []

    for first, last in zip(first_list, last_list):
        last_day = float(pnl.loc[(pnl['time'] == last, ['pnl'])].values)
        #     print(last_day)
        first_day = float(pnl.loc[(pnl['time'] == first, ['pnl'])].values)
        #     print(first_day)
        annual_return.append(((last_day - first_day) / first_day) * 100)

    pnl = pnl.set_index('time')
    pnl['return'] = get_Return(pnl)
    pnl['return'] = pnl['return'].fillna(value=0)
    sharpe_ratio = []
    for first, last in zip(first_list, last_list):
        days = np.shape(pnl[first:last])[0]
        average_return = pnl[first:last]['return'].mean() * days
        std_return = pnl[first:last]['return'].std() * ((days) ** 0.5)
        sharpe_ratio.append(average_return / std_return)

    print("annual_return: ", annual_return, flush=True)
    print("sharpe_ratio: ", sharpe_ratio, flush=True)

    with open(path_to_save, 'w') as f:
        print('annual_return: \n', annual_return, file=f)
        print('sharpe_ratio: \n', sharpe_ratio, file=f)
    f.close()


    return annual_return, sharpe_ratio

# Load data
df = pd.read_csv('./Data/sp500_2012_2022_1.csv', index_col=['symbol'])
timeline = pd.read_csv('./Data/timeline_modified.csv')

# return_data is return_data_transpose
# return_data contains daily precentage return of all stocks
data = df.T
return_data = data.apply(lambda x: get_Return(x)).T

# Loop
# initial setting
pnl_initial = 10000.0
dollar_amount = 2000
saved_model = None
pnl_all = pd.Series(dtype='float64')
cash_pd_all = pd.Series(dtype='float64')
holding_pd_all = pd.Series(dtype='float64')

# start time
for i in range(252*4+8, len(return_data.columns), window):

    # extract 4 years of data to get 3 years of training data and last 60 days as testing data

    hist_data = select_stocks(return_data, timeline, return_data.columns[i], return_data.columns[i + window - 1], 4)

    # X_pd, weight_pd = pca_residual(hist_data, delay, factor)

    # hist_data = hist_data.iloc[:5, :]

    X_pd, weight_pd = pca_residual(hist_data, delay, factor)

    stocks_number = len(hist_data.index)

    X_training, Y_training, X_testing, Y_testing = get_train_test_data(X_pd, hist_data, training_number, delay, window)
#
    number = float(Y_training.quantile(.8))

    for index, row in Y_training.iterrows():
        # if value > 0.8Y, tag is 1 means up, otherwise 0
        if row.values > number:
            Y_training.loc[index] = 1
        else:
            Y_training.loc[index] = 0

    for index, row in Y_testing.iterrows():
        # if value > 0.8Y, tag is 1 means up, otherwise 0
        if row.values > number:
            Y_testing.loc[index] = 1
        else:
            Y_testing.loc[index] = 0
#

    pnl, holding_pd, cash, cash_pd, up_down_stock, holding_stock, position_stock, ensemble_stock = back_testing_tables(hist_data, pnl_initial, factor, stocks_number)

    up_down_stock, saved_model, ensemble_stock = training_predict(X_training, Y_training, X_testing, Y_testing, 
                                                    saved_model, pnl, up_down_stock, ensemble_stock, stocks_number, 
                                                    ol, co, cs, ol_label, cs_label, os_label, co_label)

    pnl, position_stock, cash_pd, holding_pd = results(hist_data, pnl, stocks_number, up_down_stock, 
                                                        factor, position_stock, tran_cost, cash, cash_pd, 
                                                        holding_pd, weight_pd, leverage, ol_label, cs_label, os_label, co_label)
    
    pnl_all = pd.concat((pnl_all,pnl),axis=0)
    cash_pd_all = pd.concat((cash_pd_all,cash_pd),axis=0)
    holding_pd_all = pd.concat((holding_pd_all,holding_pd),axis=0)
    ensemble_stock.to_csv('./results/ada_ensemble_data_'+str(return_data.columns[i])+'.csv')
    pnl_initial = pnl[-1]


    
# 46 min 30 sec
cash_pd_all.to_csv("./results/Ada_cash_pd.csv")
holding_pd_all.to_csv("./results/Ada_holding_pd.csv")
pnl_all.to_csv("./results/Ada_pnl.csv")

plot_pnl(pnl_all, "./results/Ada_pnl.png")
statistics("./results/Ada_pnl.csv", './results/Ada_statistics.txt')