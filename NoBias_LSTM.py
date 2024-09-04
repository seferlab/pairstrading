import numpy as np
import pandas as pd
import os

# ?
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# ?
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
# might need to add tensorflow in front of keras
import tensorflow as tf
# if only use keras.models, lots of problems happens on cluster
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LSTM
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

warnings.filterwarnings('ignore')

# Global settings
delay = 252
window = 60
factor = 20
# K = 8.4
sbo = 1.25
sso = 1.25
sbc = 0.75
ssc = 0.5
r = 0.02
tran_cost = 0.0002
leverage = 1.0
start_val = 100
bo = 1
so = -1
ol = 0.5
co = 0.8
cs = 0.2
ol_label = 1
os_label = -1
cs_label = -2
co_label = 2
training_number = 3

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
    X_training = pd.Series(np.zeros(window))
    Y_training = []
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

def trainLSTMModel(shape, neurons, d, saved_model):

    if saved_model == None:
        model = Sequential()

        model.add(LSTM(neurons[0], input_shape=(shape[1], shape[2]), return_sequences=True, activation='relu'))
        # model.add(Dropout(d))
        model.add(BatchNormalization())
        model.add(LSTM(neurons[1], input_shape=(shape[1], shape[2]), return_sequences=True, activation='relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(d))

        model.add(LSTM(neurons[2], input_shape=(shape[1], shape[2]), return_sequences=False, activation='relu'))
        model.add(BatchNormalization())
        # model.add(Dense(neurons[3],activation='relu'))
        # predict up_down
        model.add(Dense(neurons[3], activation='sigmoid'))
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001)
        loss = 'binary_crossentropy'
       
        # adam = Adam(decay=0.2)
        # predict up and down
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        # model.compile(loss='mse', optimizer=optimizer)
        model.summary()

    else:
        model = load_model('nobias_best_model_ls.h5')

    return model

def get_trainingModel(shape, neurons, d, saved_model, X_train, Y_train, epochs, batch_size):
    model = trainLSTMModel(shape, neurons, d, saved_model)
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
    mc = ModelCheckpoint('nobias_best_model_ls.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    # fit model
    gpu_no = 0
    with tf.device('/gpu:' + str(gpu_no)):
        print('model_manager: running tensorflow version: ' + tf.__version__)

        print('model_manager: will attempt to run on ' + '/gpu:' + str(gpu_no))
        history = model.fit(X_train, Y_train, validation_split=0.3, epochs=epochs, verbose=0, callbacks=[es, mc], batch_size=batch_size)
        # load the saved model
        saved_model = load_model('./nobias_best_model_ls.h5')

        # evaluate the model
        _, train_acc = saved_model.evaluate(X_train, Y_train, verbose=0)
        
        print('Train: %.3f' % (train_acc), flush=True)

    return saved_model

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
    pnl_without_tc = pd.Series(index=return_data.columns[num:], dtype='float64')
    pnl.iloc[0] = pnl_initial   
    pnl_without_tc.iloc[0] = pnl_initial

    holding_pd = pd.Series(index=return_data.columns[num:], dtype='float64')
    holding_pd_without_tc = pd.Series(index=return_data.columns[num:], dtype='float64')
    # cash = pnl_initial
    cash = pnl_initial  
    cash_without_tc = pnl_initial
    cash_pd = pd.Series(index=return_data.columns[num:], dtype='float64')
    cash_pd_without_tc = pd.Series(index=return_data.columns[num:], dtype='float64')
    
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
    position_stock_without_tc = pd.DataFrame(0, index=return_data.index, columns=position_columns)
    
    return  pnl, pnl_without_tc, holding_pd, holding_pd_without_tc, cash, cash_without_tc, cash_pd, cash_pd_without_tc, up_down_stock, holding_stock, position_stock, position_stock_without_tc, ensemble_stock

def training_predict(X_training, Y_training, X_testing, Y_testing, saved_model, shape, neurons, d, epochs, batch_size, pnl, up_down_stock, ensemble_stock, stocks_number, ol, co, cs, ol_label, cs_label, os_label, co_label):
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
        X_test = X_test.append(X_testing[X_testing.index.str.startswith(j)])
        Y_test = Y_test.append(Y_testing[Y_testing.index.str.startswith(j)])
        
    Y_tag = np.array(Y_training.values)
    X_train = np.array(X_training)
    Y_train = np.array(Y_tag)
    # shape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    
    X_test_1 = np.array(X_test)
    X_test_1 = X_test_1.reshape((X_test_1.shape[0], X_test_1.shape[1], n_features))
    Y_test_tag = np.array(Y_test)

    saved_model = get_trainingModel(shape, neurons, d, saved_model, X_train, Y_train, epochs, batch_size)
    print("this is %s - %s time adboost"%(pnl.index[0], pnl.index[-1]),flush=True)
    test_loss, test_acc = saved_model.evaluate(X_test_1, Y_test_tag, batch_size=batch_size, verbose=0)
    print("test acc:{0:.3f}% loss:{1:.3f}".format(test_acc * 100, test_loss))
    predict_Y = saved_model.predict(X_test_1, batch_size=batch_size, verbose=0)

    # ensemble
    ensemble_stock.loc[time_list[0]:time_list[-1], :] = predict_Y.reshape(len(time_list), stocks_number)
    
    # if predict>1 means they will up, else go down
    # ol=0.5 co=0.8 cs=0.2
    # 1. "open long" if number is in (0.5,0.8) +1
    # 2. "close long" if the number is in (0.8, ) -1
    # 3."open short" if the number is in (0.2,0.5) 0
    # 4. "close short" if the number is in (,0.2) +2

    for x in range(0, len(predict_Y)):
        if predict_Y[x] >= ol and predict_Y[x] < co:
            predict_Y[x] = ol_label
        elif predict_Y[x] >= co:
            predict_Y[x] = cs_label
        elif predict_Y[x] < ol and predict_Y[x] > cs:
            predict_Y[x] = os_label
        else:
            predict_Y[x] = co_label
            
    predict_Y = predict_Y.reshape(len(time_list), stocks_number)
    up_down_stock.loc[time_list[0]:time_list[-1], :] = predict_Y
    K.clear_session()

    return  up_down_stock, saved_model, ensemble_stock

def call_dollar_amounts(position_stock, stocks_number, i, weight_pd, cash, tran_cost):
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

def results(return_data, pnl, pnl_without_tc, stocks_number, up_down_stock, factor, position_stock, position_stock_without_tc, tran_cost, cash, cash_without_tc, cash_pd, cash_pd_without_tc, holding_pd, holding_pd_without_tc, weight_pd, leverage, ol_label, cs_label, os_label, co_label):
    '''
    Parameters
    ----------
    
    Returns
    -------
    
    '''

    pnl_before = 0
    # tag the first, updating the pnl from the second day.
    count = 0
    trading_count = 0
    count_buy = 0
    count_sell = 0

    for i in pnl.index:
        s_list = ['s' + str(x) for x in range(1, stocks_number+1)]
        if (count > 0):
            # updating the holding accorind to the stock return
            for name, values in position_stock.iteritems():
                # check if name in the s1, s2, s3, s4 .... s20 if name is s1, update [0][i]
                if (name in s_list):
                    position_stock.loc[:, name] += values * return_data.iloc[int(name[1]) - 1][i]
            
            for name, values in position_stock_without_tc.iteritems():
            # check if name in the s1, s2, s3, s4 .... s20 if name is s1, update [0][i]
                if (name in s_list):
                    position_stock_without_tc.loc[:, name] += values * return_data.iloc[int(name[1]) - 1][i]


        tag = 0  # if we have bo or so tag++ then we will calculate the call_dollar_amounts

        # buy to open or sell to open

        for j in up_down_stock.columns:  # s1 s2 s3
            # holding =0 and up_down=1 and cash >0
            pca_list = ['pca' + str(x) for x in range(1, factor+1)]
            if(position_stock.loc[j]['up_down']==0):
                #open long
                if (up_down_stock.loc[i][j] == ol_label and cash > 0):
                    trading_count+=1
                    tag += 1
                    count_buy += 1
                    position_stock.loc[j, 'tag'] = 1
                    position_stock_without_tc.loc[j, 'tag'] = 1
                    position_stock.loc[j, 'up_down'] = ol_label
                    position_stock_without_tc.loc[j, 'up_down'] = ol_label
                    for pca in pca_list:
                        position_stock.loc[j, pca] = float(leverage * weight_pd.loc[j][i + ' ' + pca[3]])
                        position_stock_without_tc.loc[j, pca] = float(leverage * weight_pd.loc[j][i + ' ' + pca[3]])
    
                # if flag =-1 open_short
                elif (up_down_stock.loc[i][j] == os_label and cash > 0):
                    trading_count+=1
                    count_buy += 1
                    tag += 1
                    position_stock.loc[j, 'tag'] = 1
                    position_stock_without_tc.loc[j, 'tag'] = 1
                    position_stock.loc[j, 'up_down'] = os_label
                    position_stock_without_tc.loc[j, 'up_down'] = os_label
                    for pca in pca_list:
                        position_stock.loc[j, pca] = float(-leverage * weight_pd.loc[j][i + ' ' + pca[3]])
                        position_stock_without_tc.loc[j, pca] = float(-leverage * weight_pd.loc[j][i + ' ' + pca[3]])

            # if flag =2 and stock_balance =0 open short, short stock
            elif (up_down_stock.loc[i][j] == co_label and position_stock.loc[j]['up_down'] > 0):
                trading_count+=1
                count_sell += 1
                # tag+=1
                cash_without_tc += abs(position_stock_without_tc.loc[j]['s1':'s' + str(stocks_number)]).sum()
                # add transaction cost
                cash += abs(position_stock.loc[j]['s1':'s' + str(stocks_number)]).sum()

                cash = cash - tran_cost * cash
                position_stock.loc[j, :] = 0
                position_stock_without_tc.loc[j, :] = 0

            # if flag =-2 and stock_balance <0 clse short, sell stock
            elif (up_down_stock.loc[i][j] == cs_label and position_stock.loc[j]['up_down'] < 0):
                trading_count+=1
                count_sell += 1
                # tag+=1
                cash_without_tc += abs(position_stock_without_tc.loc[j]['s1':'s' + str(stocks_number)]).sum()
                cash += abs(position_stock.loc[j]['s1':'s' + str(stocks_number)]).sum()
                cash = cash - tran_cost * cash
                position_stock.loc[j, :] = 0
                position_stock_without_tc.loc[j, :] = 0

            else:
                position_stock.loc[j, 'tag'] = 2
                position_stock_without_tc.loc[j, 'tag'] = 2
                

        # if tag>0 means we have open position this iteration.
        print("we have %s times trading at %s time" % (trading_count, i))
        print("we have %.4f$ cash_without_tc at %s time" % (cash_without_tc, i))
        print("we have %.4f$ cash at %s time" % (cash, i))

        if (tag > 0):
            cash, position_stock = call_dollar_amounts(position_stock, stocks_number, i, weight_pd, cash, tran_cost)
            cash_without_tc, position_stock_without_tc = call_dollar_amounts(position_stock_without_tc, stocks_number, i, weight_pd,
                                                                            cash_without_tc, 0)

        holding = position_stock.loc[:, 's1':'s' + str(stocks_number)].abs().sum().sum()

        holding_without_tc = position_stock_without_tc.loc[:, 's1':'s' + str(stocks_number)].abs().sum().sum()

        holding_pd[i] = holding
        holding_pd_without_tc[i] = holding_without_tc
        cash_pd[i] = cash
        cash_pd_without_tc[i] = cash_without_tc
        print("we have %.4f$ holding at %s time" % (holding, i))
        print("we have %.4f$ holding_pd_without_tc at %s time" % (holding_without_tc, i))

        pnl_before = pnl.loc[i]

        pnl_before_without_tc = pnl_without_tc.loc[i]

        pnl[i] = holding + cash
        pnl_without_tc[i] = holding_without_tc + cash_without_tc
        print("we have %.4f$ total amount at %s" % (pnl[i], i))
        print("we have %.4f$ without_tc total amount at %s" % (pnl_without_tc[i], i))
        count += 1

    print('we have %f times buying, and %f selling' % (count_buy, count_sell))

    return  pnl, pnl_without_tc, holding_pd, holding_pd_without_tc, cash_pd, cash_pd_without_tc, position_stock, position_stock_without_tc

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
    # pnl['return'] = get_Return(pnl.iloc[:, 1])
    # pnl['return'] = pnl['return'].fillna(value=0)
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

    return


# Load data
df = pd.read_csv('./Data/sp500_2012_2022_1.csv', index_col=['symbol'])
timeline = pd.read_csv('./Data/timeline_modified.csv')
# df = pd.read_csv('/Users/yikai/Desktop/Statistical Arbitrage/RemoveBias/Data/sp500_2012_2022_1.csv', index_col=['symbol'])
# timeline = pd.read_csv('/Users/yikai/Desktop/Statistical Arbitrage/RemoveBias/Data/timeline_modified.csv')


# return_data is return_data_transpose
# return_data contains daily precentage return of all stocks
data = df.T
return_data = data.apply(lambda x: get_Return(x)).T
# LOOP
# initial setting
pnl_initial = 10000.0
dollar_amount = 2000
saved_model = None
pnl_all = pd.Series(dtype='float64')
pnl_all_without_tc = pd.Series(dtype='float64')
cash_pd_all = pd.Series(dtype='float64')
cash_pd_all_without_tc = pd.Series(dtype='float64')
holding_pd_all = pd.Series(dtype='float64')
holding_pd_all_without_tc = pd.Series(dtype='float64')

# start time
for i in range(252*4+8, len(return_data.columns), window):

    # extract 4 years of data to get 3 years of training data and last 60 days as testing data

    hist_data = select_stocks(return_data, timeline, return_data.columns[i], return_data.columns[i + window - 1], 4)

    # hist_data = hist_data.iloc[:5, :]

    stocks_number = len(hist_data.index)

    X_pd, weight_pd = pca_residual(hist_data, delay, factor)

    X_training, Y_training, X_testing, Y_testing = get_train_test_data(X_pd, hist_data, training_number, delay, window)

    pnl, pnl_without_tc, holding_pd, holding_pd_without_tc, cash, cash_without_tc, cash_pd, cash_pd_without_tc, up_down_stock, holding_stock, position_stock, position_stock_without_tc, ensemble_stock = back_testing_tables(hist_data, pnl_initial, factor, stocks_number)

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
#
    time_step = 60
    d = 0.3
    length = X_training.shape[0]
    output = 1
    shape = [length, time_step, output]  # feature, window, output
    neurons = [128, 64, 32, 1]
    epochs = 500
    batch_size = 10000
#
    up_down_stock, saved_model, ensemble_stock = training_predict(X_training, Y_training, X_testing, Y_testing, 
                                                    saved_model, shape, neurons, d, epochs, batch_size, 
                                                    pnl, up_down_stock, ensemble_stock, stocks_number, ol, co, cs, ol_label, cs_label, os_label, co_label)


    pnl, pnl_without_tc, holding_pd, holding_pd_without_tc, cash_pd, cash_pd_without_tc, position_stock, position_stock_without_tc = results(hist_data, pnl, pnl_without_tc, stocks_number, 
                                                                                                                                                up_down_stock, factor, position_stock, position_stock_without_tc, 
                                                                                                                                                tran_cost, cash, cash_without_tc, cash_pd, cash_pd_without_tc, 
                                                                                                                                                holding_pd, holding_pd_without_tc, weight_pd, leverage, 
                                                                                                                                                ol_label, cs_label, os_label, co_label)

    pnl_all = pd.concat((pnl_all,pnl),axis=0)
    pnl_all_without_tc = pd.concat((pnl_all_without_tc,pnl_without_tc),axis=0)
    cash_pd_all = pd.concat((cash_pd_all,cash_pd),axis=0)
    cash_pd_all_without_tc = pd.concat((cash_pd_all_without_tc,cash_pd_without_tc),axis=0)
    holding_pd_all = pd.concat((holding_pd_all,holding_pd),axis=0)
    holding_pd_all_without_tc = pd.concat((holding_pd_all_without_tc,holding_pd_without_tc),axis=0)
    ensemble_stock.to_csv('./results/lstm_ensemble_data_'+str(return_data.columns[i])+'.csv')
    pnl_initial = pnl[-1]

pnl_all.to_csv("./results/LSTM_pnl.csv")
pnl_all_without_tc.to_csv("./results/LSTM_pnl_without_tc.csv")
cash_pd_all.to_csv("./results/LSTM_cash.csv")
cash_pd_all_without_tc.to_csv("./results/LSTM_cash_without_tc.csv")
holding_pd_all.to_csv("./results/LSTM_holding.csv")
holding_pd_all_without_tc.to_csv("./results/LSTM_holding_without_tc.csv")

plot_pnl(pnl_all, "./results/LSTM_pnl.jpg")
plot_pnl(pnl_all_without_tc, "./results/LSTM_pnl_without_tc.jpg")

statistics('./results/LSTM_pnl.csv', './results/LSTM_statistics.txt')
statistics('./results/LSTM_pnl_without_tc.csv', './results/LSTM_statistics_without_tc.txt')