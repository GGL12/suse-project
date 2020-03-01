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
    if "北京" in requests.get(url).text:
        return True
    else:
        return False

def get_grid_point(center, ori_dis=0.5):
    
    dis = ori_dis * (2 ** 0.5)
    latmin, lonmin = get_distance_point(center, dis, 315)
    latmax, lonmax = get_distance_point(center, dis, 135)
    return[latmin, lonmin, latmax, lonmax]

def get_city_grids(other_point, key, count=50, ori_dis=0.5):
    '''
        1. 城市经纬度范围（轮廓？）
        2. 以城市原点划分网格
        3. 获得每个网格左上右下的点
        4. 根据轮廓经纬排除不符合的
        5. girds [[center_x, center_y, xmin, ymin, xmax, ymax],.....]
    '''
#city_center为该城市中的经纬度
#    grids = []
#    for i in range(1, count+1):
#        print(f"正在探索第{i}条十字网格,总计有{count}条需探索.")
#        for j in range(0, 360, 90):
#            center = get_distance_point(city_center, i*ori_dis*2, j)
#            #print(center)
#            if is_point_in_city(center, key):
#                grids.append(center + get_grid_point(center, ori_dis))
#            center =  get_distance_point(city_center, (i*ori_dis*2)*2**0.5, j+45)
#            if is_point_in_city(center, key):
#                grids.append(center + get_grid_point(center, ori_dis))
#    print(f"有效的城市网格有{len(grids)}个.")
 
#other_point为该城市之外左上角的经纬度
    grids = []
    for i in range(count+1):
        x_center = get_distance_point(other_point, i*ori_dis*2, 90)
        if is_point_in_city(x_center, key):
            grids.append(x_center + get_grid_point(x_center, ori_dis))
        for j in range(1, count+1):
            print(f'正在探测城市网格坐标(x:{i},y:{j}),待(x:{count},y:{count})时结束探测...')
            y_center = get_distance_point(x_center, j*ori_dis*2, 180)
            if is_point_in_city(y_center, key):
                grids.append(y_center + get_grid_point(y_center, ori_dis))
    pd.DataFrame(data=grids, 
                 columns=['center_x','center_y',
                          'latmin','lonmin',
                          'latmax','lonmax']
                 ).to_csv('./data/part1/grids.csv', index=False)
    print(f"有效的城市网格有{len(grids)}个.")
    return grids


def plot_city_grids(girds,city='北京'):
    '''
        根据切割出来的城市网格，绘画出来。
    '''
    data_pair = []
    city_map = Geo().add_schema(maptype=city)
    for i in range(len(grids)):
        city_map.add_coordinate(f'city grid{i+1}',grids[i][1],grids[i][0])
        data_pair.append((f'city grid{i+1}',1))
    # 画图
    city_map.add('',data_pair, type_=GeoType.EFFECT_SCATTER, symbol_size=2)
    city_map.set_series_opts(label_opts=options.LabelOpts(is_show=False))
    city_map.set_global_opts(title_opts=options.TitleOpts(title=f"{city}城市网格图"))

    webbrowser.open_new_tab(city_map.render())


def get_poi_data():
    
    '''
    得到poi数据
    '''
    display_x = []
    display_y = []
    name = []
    poi_file = ['./data/part2/' + file for file in os.listdir('./data/\
                part2') if 'xlsx' in file]
    for file in poi_file:
        temp = pd.read_excel(file)
        try:
            display_x += list(temp.display_x)
            display_y += list(temp.display_y)
            name += [file.split("_")[-1].split(".")[0] for i in temp.display_x]
        except:
            print(f"文件名'{file}'中，没有包含经纬度信息.")
    lat_and_lon = [str(display_y[i]) + ',' + str(display_x[i]) for i in range(len(display_x))]
    poi_data = pd.DataFrame({'poi_name':name,'lat_and_lon':lat_and_lon})
    poi_data.to_csv('./data/part2/poi_data.csv', index=False)
    return dict(zip(lat_and_lon, name))

def grids_data(grids, poi_data):
    
    '''
    在城市网格中统计各类poi个数
    '''
    for index in range(len(grids)):
        #for index in range(2):
            point_poi_count = dict(zip(set(poi_data.values()),[0 for i in set(poi_data.values())]))
            j = 1
            for location in poi_data.keys():
                try:
                    lat = eval(location.split(',')[0])
                    lon = eval(location.split(',')[1])
                    temp_sort = grids[index]
                    if lat >= temp_sort[0] and lat <= temp_sort[2] and \
                        lon >= temp_sort[3] and lon <= temp_sort[5]:
                            point_poi_count[location] += 1
                    print(f'正在处理第{index+1}个grid的第{j}个poi数据点计数,总计需完成{len(grids)}的{len(poi_data)}的核对.....')
                except:
                    print(f"这个经纬度数据有误{location}")
                j += 1 
            grids[index] = grids[index] + list(point_poi_count.values())
    return grids



    
    
    
    
    
    
    
    
    
    