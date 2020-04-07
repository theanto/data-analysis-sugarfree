#!/usr/bin/python
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import read_csv
import datetime
#from pmdarima import auto_arima
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from statsmodels.tools.eval_measures import rmse

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
import os
from PIL import Image

import warnings
warnings.filterwarnings('ignore')



def boxplot(df, name, fold):
    

    sns.set(font_scale=2)

    g = sns.catplot(kind='box', data=df, x='Prediction Horizon (minutes)', y='Glycemia prediction RMSE (mg/dl)', col='PSW', height=7, aspect=0.5, palette='Greys', showfliers = False)

    g.fig.subplots_adjust(wspace=0)

    plt.subplots_adjust(top=0.8)
    g.fig.suptitle('Past sliding window (PSW)') 

    # remove the spines of the axes (except the leftmost one)
    # and replace with dasehd line
    for ax  in g.axes.flatten()[0:]:
        ax.spines['left'].set_visible(False)
        [tick.set_visible(True) for tick in ax.yaxis.get_major_ticks()]
        xmin,xmax = ax.get_xlim()
        ax.axvline(xmin, ls='--', color='k')

        
        lines = ax.get_lines()
        categories = ax.get_xticks()


        for cat in categories:
            
            # every 4th line at the interval of 6 is median line
            # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
            y = round(lines[4+cat*5].get_ydata()[0],3) 
            
            ax.text(
                cat, 
                y, 
                f'{y}', 
                ha='center', 
                va='center', 
                fontweight='bold', 
                size=15,
                color='white',
                bbox=dict(facecolor='#445A64'))
            
        for i, ax in enumerate(g.axes.flat):
            g.axes[0,i].set_xlabel('')

        g.axes[0,2].set_xlabel('Predictive Horizon (minutes)')  
        
    plt.savefig("uid/" + str(fold)+"/"+ str(name) +".jpg")




def loaddataset(file):
    
    series = read_csv(file, header=0, index_col=0, parse_dates=True, squeeze=True)
    dfusers = pd.DataFrame()
    dfusers = series.reset_index() 
    uid= dfusers.uid.unique()
    
    dataf = {}
    ind= 0
    for x in uid: 
        if (os.path.isdir('uid/'+str(ind)+'/')  == False) :
            os.mkdir('uid/'+str(ind))
        print("User "+str(ind),  x)
        dataf[ind] = pd.DataFrame()
        dataf[ind]= series.loc[x]
        dataf[ind] = dataf[ind].reset_index()
        dataf[ind] = dataf[ind].drop(dataf[ind][dataf[ind].sugarValue > 300].index, inplace=False) 
        dataf[ind] = dataf[ind].fillna(0)
        dataf[ind] = dataf[ind].drop(dataf[ind][dataf[ind].sugarValue < 1].index, inplace=False)
        ind = ind+1

    return dataf

def preprocess(data, fun):
    if(fun ==0):
        data = data[['sugarValue', 'time']]
        data['time']=pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        data.set_index(['time'], inplace=True)

    if(fun ==1):
        data = data.drop(['uid','time'], axis=1)
        float_col = data.select_dtypes(include=['float64']) # This will select float columns only
        #list(float_col.columns.values)
        for col in float_col.columns.values:
                data[col] = data[col].astype('int64')


    return data  


def app(window,  train, test, pred, interval, windo):
    window = window.append({'Current test': test.values, 
                                        'Current prediction': pred, 
                                        'MSE': np.square(np.subtract(test , pred)).mean(),
                                        'Glycemia prediction RMSE (mg/dl)': rmse(test, pred),
                                        'PSW': int(round(windo)) ,
                                         'Prediction Horizon (minutes)': interval },ignore_index=True)
    return window


def ARIMA(window, df ,x, n,v, interval, windo):

   
    train = df.iloc[x:n]
    test = df.iloc[n:n+v+1]

    #indces for the train test split
    start = len(train)
    end = start + len(test)-1

    model = SARIMAX(df['sugarValue'], order=(0, 1, 3), seasonal_order=(0, 0, 0, 12), enforce_invertibility=False).fit()
    pred = model.predict(n, n+v)
    
    window = app(window, train['sugarValue'], test['sugarValue'] ,pred, interval, windo)

    return window

def RFSVM(window, df, X, y ,x, n,v, interval, windo,fun):

   
    X_train = X.iloc[x:n]
    X_test = X.iloc[n:n+v+1]
    y_train = y.iloc[x:n]
    y_test = y.iloc[n:n+v+1]
   

   
    if (fun ==2):
        rfc = RandomForestClassifier()
        rfc.fit(X_train,y_train)
        # predictions
        pred = rfc.predict(X_test)
    if (fun ==3):
        

        svclassifier = SVR()
        svclassifier.fit(X_train, y_train)

        pred = svclassifier.predict(X_test)
        
    window = app(window, y_train, y_test ,pred, interval, windo)

    return window


def prediction(df, fun,*args, **kwargs):
    for ar in args:
        pass
    
    X= args[0]
    y= args[1]
    j= args[2] ##number of dataset

    start_time = time.time()

    list = [12,24,48,96,144] 
    window = pd.DataFrame(columns=['Current test', 'Current prediction','MSE', 'Glycemia prediction RMSE (mg/dl)', 'PSW', 'Prediction Horizon (minutes)'])
 
    graph=0

    for z in list:
       
        for v in range(4):
            n=z ##48 hour
            
            interval = (v+1)*15
            windo= n/4
            
            
            for x in range((len(df)-n-v)):
                

                if fun == 1:                   
                    window = ARIMA(window, df, x, n, v, interval, windo)
                if fun == 2:
                    
                    window = RFSVM(window, df, X,y,x, n, v, interval, windo,fun)
                if fun == 3:
                    
                    window = RFSVM(window, df, X,y,x, n, v, interval, windo,fun)
                n= n+1
                
                
            v= v+1

    print("--- %s Seconds for computation ---" % (time.time() - start_time))
    return window

    
    

def joinI(x):
    
    im1 = Image.open("uid/"+str(x)+'/Arima.jpg')
    im2 = Image.open("uid/"+str(x)+'/RF.jpg')
    im3 = Image.open("uid/"+str(x)+'/SVM.jpg')

    imageL = [im1, im2, im3]
    min_width = min(im.width for im in imageL)
    im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)),resample=Image.BICUBIC)
                      for im in imageL]
    total_height = sum(im.height for im in im_list_resize)
    dst = Image.new('RGB', (min_width, total_height))
    pos_y = 0
    for im in im_list_resize:
        dst.paste(im, (0, pos_y))
        pos_y += im.height
    
    dst.save("uid/"+str(x)+"/result.jpg")

    os.remove("uid/"+str(x)+"/Arima.jpg")
    os.remove("uid/"+str(x)+"/RF.jpg")
    os.remove("uid/"+str(x)+"/SVM.jpg")


def joinO(name,x):
    

    images = [Image.open(x) for x in [ "uid/"+str(x)+'/'+str(name)+" 12.jpg", "uid/"+str(x)+'/'+str(name)+" 24.jpg", "uid/"+str(x)+'/'+str(name)+" 48.jpg", "uid/"+str(x)+'/'+str(name)+" 96.jpg",
     "uid/"+str(x)+'/'+str(name)+" 144.jpg"]]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
        

    new_im.save("uid/"+str(x)+"/Result " + str(name) +".jpg")
   
    imgs= [ "uid/"+str(x)+'/'+str(name)+" 12.jpg", "uid/"+str(x)+'/'+str(name)+" 24.jpg", "uid/"+str(x)+'/'+str(name)+" 48.jpg", "uid/"+str(x)+'/'+str(name)+" 96.jpg",
     "uid/"+str(x)+'/'+str(name)+" 144.jpg"]
    for ims in imgs:

        os.remove(ims)
    #os.remove("uid/"+str(x)+'/RF.jpg')
    #os.remove("uid/"+str(x)+'/SVM.jpg')