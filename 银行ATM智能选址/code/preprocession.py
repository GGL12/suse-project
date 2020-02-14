'''
数据预处理
'''
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime

data_part1_ori = pd.read_csv('./data/part1/original.csv')
data_part1_loc = pd.read_csv('./data/part1/lon_and_lat.csv')
data_part1_trip = pd.read_csv('./data/part1/trip_mode.csv', encoding='gbk')
data_part1_loc.head()
data_part1_ori.head()
data_part1_trip.head()

print(data_part1_ori.info())
print(data_part1_ori.describe())
print(f'原始数据表中各列缺失值   缺失数量    总计:{len(data_part1_ori)} \n',data_part1_ori.isna().sum())

#此列完全没有数据
data_part1_ori.drop(['tmp0'], axis=1, inplace=True)
#考虑到只有少数的数据存在缺失值 删除 20(10169)
data_part1_ori.dropna(inplace=True)

def print_unique_col(df):
    cols = df.columns
    for col in cols:
        print(f"column:{col}, unique count:{len(df[col].unique())} \n")

def print_imsi_and_phone(df):
    df = df[['imsi', 'phone']].drop_duplicates()
    for index, data in df.groupby('phone'):
        if len(data) != 1:
            # for i in data.values:
            #     print(i)
            # print(f'phone:{index}, imsi:{data.imsi.values[0]} \n')
            print(data)

print_unique_col(data_part1_ori)#1. tmp1为空值。 2.有效用户122 or 114 imsi 和phone 数据去重后不一致
print_imsi_and_phone(data_part1_ori)#imsi字段中有些缺失，使用非数字填充，是和phone对应

#处理非数字集合的imsi字段，然后删除
data_part1_ori['imsi'] = data_part1_ori.imsi.apply(lambda x: x if x.isdigit() else np.nan)
data_part1_ori.dropna(inplace=True)
#打印查看
print_unique_col(data_part1_ori)#imsi和phone相同

#组合格式 laci
data_part1_ori['laci'] = data_part1_ori.lac_id.astype(int).astype(str) + '-' + data_part1_ori.cell_id.astype(int).astype(str)

#删除phone lac_id cell_id
data_part1_ori.drop(['phone', 'lac_id', 'cell_id'], axis=1, inplace=True)

data_part1_ori.npid = data_part1_ori.npid.apply(lambda x: x.replace('#', '').replace('*', ''))

loc_data = pd.merge(data_part1_ori, data_part1_loc)
loc_data.isna().sum()


def match_digit(string):
    try:
        return eval(re.search('[^0](.*)', string).group(0))
    except:
        return 0
def convert_time(df):
    '''
    去除了不是3号的数据
    '''
    add_cols = ['hour_s', 'minute_s', 'second_s','hour_e', 'minute_e', 'second_e']
    add_data = []
    index = []
    for i in range(len(df)):
        time_local_s = time.localtime(df.timestamp[i] / 1000)
        time_local_e = time.localtime(df.timestamp1[i] / 1000)
        if time_local_s.tm_mday == 3:
            index.append(i)
            add_data.append([time_local_s.tm_hour, time_local_s.tm_min, time_local_s.tm_sec, \
                             time_local_e.tm_hour, time_local_e.tm_min, time_local_e.tm_sec])         
    return pd.concat([df.loc[index].reset_index(drop=True), pd.DataFrame(columns=add_cols, data=add_data)], axis=1)
df = loc_data
part1_data = convert_time(loc_data)
def map_location(df):
    '''
        laci----map
    '''
    return dict(zip(df.laci.unique(),range(len(df.laci.unique()))))
def get_part1_data(df):
    df = part1_data
    location_map_dict = map_location(df)
    df['location'] = df.laci.apply(lambda x: location_map_dict.get(x))
    for imsi,data in df.groupby('imsi'):
        data.sort_values('timestamp', inplace=True)
        flag = 0
        save_index = [0]
        wait_time = []
        for i in range(len(data)-1):
            if data.laci.iloc[i] != data.laci.iloc[i+1]:
                save_index.append(i+1)
                
                flag = i
   
    (datetime.utcfromtimestamp(data.timestamp.values[1]/1000) - datetime.utcfromtimestamp(data.timestamp.values[0]/1000)).seconds/3600
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


data.to_csv('./data/part1/data_sample.csv')