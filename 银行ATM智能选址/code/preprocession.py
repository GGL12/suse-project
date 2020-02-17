'''
数据预处理
'''
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime
import geopy.distance
from geopy.distance import geodesic
import requests
import time

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
    '''
        Print unique data for data'columns
    '''
    cols = df.columns
    for col in cols:
        print(f"column:{col}, unique count:{len(df[col].unique())} \n")

def print_imsi_and_phone(df):
    '''
        1. Drop duplicates
        2. Print number data > 1
    '''
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

'''
def match_digit(string):
        #match some string
    try:
        return eval(re.search('[^0](.*)', string).group(0))
    except:
        return 0
'''
def convert_time(df):
    '''
        1. drop day=3
        2. timestamp ---> time
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

def map_location(df):
    '''
        laci---->> map
    '''
    return dict(zip(df.laci.unique(),range(len(df.laci.unique()))))

def get_wait_hour(t1, t2):
    '''
        timestamp ---->> hours
    '''
    dt1 = datetime.utcfromtimestamp(t1 / 1000)
    dt2 = datetime.utcfromtimestamp(t2 / 1000)
    return [(dt2 - dt1).seconds / 3600]

def get_part1_data(data):
    '''
        1. Confirming the data'columns what I need
        2. Group by imsi
        3. Get the amount of time that different people spend in different location
    '''
    part1_data = []
    df = convert_time(data)
    cols = ['start_time', 'imsi', 'laci', 'longitude', 'latitude', 'hour_s',
       'minute_s', 'second_s', 'location', 'end_time', 'hour_e', 'minute_e', 
       'second_e', 'wait_hour']
    location_map_dict = map_location(df)
    df['location'] = df.laci.apply(lambda x: location_map_dict.get(x))
    for imsi,data in df.groupby('imsi'):
        data.sort_values('timestamp', inplace=True)
        data.drop(['tmp1','nid','npid', 'timestamp1', 'hour_e', 'minute_e', 'second_e'], axis=1, inplace=True)
        flag = 0
        for i in range(len(data)-1):
            if data.location.iloc[i] != data.location.iloc[i+1]:
                #part1_data.append(data.iloc[flag].values.tolist() + 
                                  #data.iloc[i+1][['timestamp', 'hour_s','minute_s', 'second_s']].values.tolist() +
                                  #get_wait_hour(data.iloc[flag]['timestamp'],data.iloc[i+1]['timestamp']))
                
                part1_data.append(data.iloc[flag].values.tolist() + 
                                  data.iloc[i+1][['timestamp', 'hour_s','minute_s', 'second_s']].values.tolist() +
                                  get_wait_hour(data.iloc[flag]['timestamp'],data.iloc[i+1]['timestamp']))
                flag = i
        part1_data.append(data.iloc[flag].values.tolist() + 
                          data.iloc[-1][['timestamp', 'hour_s','minute_s', 'second_s']].values.tolist() +
                          get_wait_hour(data.iloc[flag]['timestamp'], data.iloc[-1]['timestamp']))
    return pd.DataFrame(columns=cols, data=part1_data)

part1_data = get_part1_data(loc_data)


def get_distance(p1, p2):
    '''
    
    '''
    print(f"两点相差{int(geodesic(p1,p2).m)}米!\n")
    return int(geodesic(p1,p2).m)

#p1 = (39.910925,116.413384)
#p2 = (39.915526,116.403847)
#get_distance(p1, p2)

def get_distance_point(center, distance, direction):
    """
    根据经纬度，距离，方向获得一个地点
    :param lat: 纬度
    :param lon: 经度
    :param center: 中心点
    :param distance: 距离（千米）
    :param direction: 方向（北：0，东：90，南：180，西：270）
    :return:
    """
    start = geopy.Point(center[0], center[1])
    d = geopy.distance.distance(kilometers=distance)
    end = d.destination(point=start, bearing=direction)
    lat = end.latitude
    lon = end.longitude
    return [lat, lon]


lat, lon = get_distance_point(p1, 1, 270)

#lat, lon = get_distance_point(39.910925,116.413384, 0.5, 180)
#print(f'{lon},{lat}')


def is_point_in_city(point, key):
    time.sleep(2)
    url = f'https://restapi.amap.com/v3/geocode/\
    regeo?output=json&location={point[1]},{point[0]}&key={key}'
    if "自贡" in requests.get(url).text:
        return True
    else:
        return False

def get_grid_point(center, ori_dis=0.5):
    
    dis = ori_dis * (2 ** 0.5)
    latmin, lonmin = get_distance_point(center, dis, 315)
    latmax, lonmax = get_distance_point(center, dis, 135)
    return[latmin, lonmin, latmax, lonmax]

def get_city_grid(city_center, ori_dis=0.5, key):
    '''
        1. 城市经纬度范围（轮廓？）
        2. 以城市原点划分网格
        3. 获得每个网格左上右下的点
        4. 根据轮廓经纬排除不符合的
        5. girds [[center_x, center_y, xmin, ymin, xmax, ymax],]
    '''
    grids = []
    city_center = (29.34537,104.785532)
    for i in range(50):
        for j in range(0, 360, 90):
            center = get_distance_point(city_center, i*ori_dis*2, j)
            if is_point_in_city(center, key):
                grids.append(center + get_grid_point(center, ori_dis))
    return grids



    
    
    
    
    
    
    
    
    
    