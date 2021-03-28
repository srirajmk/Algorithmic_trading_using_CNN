import pandas as pd
import numpy as np
import math
import csv 
import pandas_ta as ta
import time
import matplotlib.pyplot as plt

#MAIN
#etf_name=['HDFC.BO','HDFCBANK.BO','HINDUNILVR.BO','RELIANCE.BO','TCS.BO','1SPY','2DIA','3QQQ','4XLF','5XLU','6EWZ','1WMT','2GS','3MSFT','4JNJ','5AAPL']
etf_name=['1WMT','2GS','3MSFT','4JNJ','5AAPL']
for etf in etf_name:
  #PHASE 1
  #Read csv file
  data=[]
  with open('/content/drive/My Drive/at_dataset/etfs_stocks/{}.csv'.format(etf),'r') as f1:
    reader=csv.reader(f1)
    for row in reader:
      data.append(row)
  #Write csv file excluding Null values
  null_str='null'
  with open('/content/drive/My Drive/ATuGAFCNN/Phase1Extraction/{}_phase1.csv'.format(etf), 'w', newline='') as f2:
    writer=csv.writer(f2)
    for row in data:
      if null_str not in row:
        writer.writerow(row)