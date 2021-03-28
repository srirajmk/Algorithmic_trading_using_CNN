import pandas as pd
import numpy as np
import math
import csv 
import pandas_ta as ta
import time
import matplotlib.pyplot as plt

#Functions for image creation
def mat_gen(df,):
  #GAF Matrix generation
  val_2d=df.values
  val_2d=np.around(val_2d,2)
  gaf_mat=np.array([[x for x in range(0,227)]])
  for i in range(val_2d.shape[0]):
    gaf_row=gaf(val_2d[i,2:17])
    #print(i)
    gaf_row=np.append(val_2d[i,0:2],gaf_row)
    gaf_row=np.reshape(gaf_row,(1,227))
    gaf_mat=np.append(gaf_mat,gaf_row,axis=0)
  return gaf_mat[1:,:]


def gaf_operation(x,y):
  t1=x*y
  t2=math.sqrt(1-math.pow(x,2))
  t3=math.sqrt(1-math.pow(y,2))
  t4=t2*t3
  f=t1-t4
  #f=(x*y)-((math.sqrt(1-(x**2)))*(math.sqrt(1-(y**2))))
  return f

def gaf(num_array):
  x_l=num_array.tolist()
  n=len(x_l)
  result=np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      #print("val@j=",x_l[j])
      result[i,j]=gaf_operation(x_l[i],x_l[j])
  return np.reshape(result,(1,225))

def normalize(df,feature_name):
  max_value = df[feature_name].max()
  min_value = df[feature_name].min()
  df[feature_name] = ((2*(df[feature_name] - min_value)) / (max_value - min_value))-1
  return df

etf_name=['1WMT','2GS','3MSFT','4JNJ','5AAPL']
for etf in etf_name:
  ti=int(time.time())

  df = pd.read_csv('/content/drive/My Drive/ATuGAFCNN/Phase2Labelling/{}_phase2.csv'.format(etf), sep=',')
  #Find values of 15 TA
  df.ta.rsi(df['Close'],cumulative=True,append=True)
  df.ta.willr(df['High'],df['Low'],df['Close'],cumulative=True, append=True)
  df.ta.cci(df['High'],df['Low'],df['Close'],cumulative=True, append=True)
  df.ta.cmo(df['Close'],cumulative=True, append=True)
  df.ta.cmf(df['High'],df['Low'],df['Close'],df['Volume'],cumulative=True, append=True)
  df.ta.adx(df['High'],df['Low'],df['Close'],cumulative=True, append=True)
  df.ta.stoch(df['High'],df['Low'],df['Close'],cumulative=True, append=True)
  df.ta.uo(df['High'],df['Low'],df['Close'],cumulative=True, append=True)
  df.ta.aroon(df['Close'],cumulative=True, append=True)
  df.ta.ppo(df['Close'],cumulative=True, append=True)
  df.ta.roc(df['Close'],cumulative=True, append=True)
  df.ta.coppock(df['Close'],cumulative=True, append=True)
  df.ta.macd(df['Close'],cumulative=True, append=True)
  df.ta.dpo(df['Close'],cumulative=True, append=True)
  df.ta.tsi(df['Close'],cumulative=True, append=True)
  df=df.dropna()
  df.drop(['Open','High','Low','Adj Close','Volume'], axis = 1, inplace = True)
  cols = df.columns.tolist()
  t=cols[2],cols[1]
  cols[1],cols[2]=t
  df=df[cols]

  #Normalizing in range[-1,1]
  df['RSI_14']=(df['RSI_14']/50)-1
  df['WILLR_14']=(df['WILLR_14']/50)+1
  #df['CCI_20_0.015']=df['CCI_20_0.015']/100
  df['CMO_14']=df['CMO_14']/100
  df['ADX_14']=(df['ADX_14']/50)-1
  df['STOCHF_14']=(df['STOCHF_14']/50)-1
  df['UO_7_14_28']=(df['UO_7_14_28']/50)-1
  df['AROON_14']=(df['AROONU_14']-df['AROOND_14'])/100
  norm_columns=['CCI_20_0.015','CMF_20','PPO_12_26_9','ROC_10','COPC_11_14_10','MACD_12_26_9','DPO_1','TSI_13_25']
  for col in norm_columns:
    df=normalize(df,col)
  df.drop([ 'DMP_14', 'DMN_14','STOCHF_3','STOCH_5', 'STOCH_3','AROOND_14', 'AROONU_14', 'PPOH_12_26_9', 'PPOS_12_26_9', 'MACDH_12_26_9', 'MACDS_12_26_9'], axis = 1, inplace = True)

  df['Date'] = pd.to_datetime(df['Date'])
  split_date = pd.datetime(2014,12,31)

  df_training = df.loc[df['Date'] <= split_date]
  df_test = df.loc[df['Date'] > split_date]
  
  dttest=df_test['Date']
  df_training.drop(['Date'], axis = 1, inplace = True)
  df_test.drop(['Date'], axis = 1, inplace = True)

  mat_tr=mat_gen(df_training)
  mat_te=mat_gen(df_test)
  to=int(time.time())
  toc=to-ti
  
  dttest.to_csv('/content/drive/My Drive/test_datset/{}_testdt.csv'.format(etf), index=False)

  np.savetxt('/content/drive/My Drive/ATuGAFCNN/Phase3ImgCreation/{}_train.csv'.format(etf),mat_tr, delimiter=',')
  np.savetxt('/content/drive/My Drive/ATuGAFCNN/Phase3ImgCreation/{}_test.csv'.format(etf), mat_te, delimiter=',')
  print("ETF: {} Train: {} Test: {} Time: {}".format(etf,len(mat_tr),len(mat_te),toc))
