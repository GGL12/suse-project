import numpy as np
from sklearn.metrics import r2_score

ori_data = [279.96,276.28,272.25,271.50,268.90,261.67,262.04,262.82,245.18,242.30]

def JudgeRatio(data):
    """
    判断此数列是否满足GM建模要求:
        数列级比检验属于[0.8338, 1.1994]范围为符合
        input:
            data:list
        return:
            tag:bool 是否符合建模要求的标记
    """
    tag = False
    for index in range(0, len(ori_data) -1):
        ratio = data[index] / data[index+1]
        #ratio_data.append(data[index] / data[index+1])
        if (ratio >= 0.8338) & (ratio <= 1.1994):
            tag = True
    return tag

def AccumulateData(data):
    """
    计算累加的数列数据:
        input:
            data:list
        return accu_list:list
    """
    data_temp = 0 
    accu_list = []
    for i in data:
        data_temp += i
        accu_list.append(data_temp)
    return accu_list
        
def GM(data):
    """
    搭建GM(1,1)模型:
        input:
            data:list
        return:
            paras:array shape(2,1) 返回正规方程的参数(a,b)
    """
    X_list = []
    Y_list = []
    accu_list = AccumulateData(data)
    #if JudgeRatio(data):
    for index in range(0, len(data) -1):
        X_list.append([-0.5 * (accu_list[index] + accu_list[index + 1]), 1])
        Y_list.append([data[index + 1]])
    X = np.array(X_list)
    Y = np.array(Y_list) 
    paras = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    return paras
    #else:
        #print("不符合GM建模规则")
        #return None

def Predict(curr_data,data,k):
    """
    预测距离当前数据到k年后的数据序列:
        input:
            curr_data:int
            data:list
            k:int
        return:
            predict_data:list
    """
    predict_data = [curr_data]
    paras = GM(data)
    a = paras[0][0]
    b = paras[1][0]
    predict_accumulate_data = []
    for i in range(k):
        predict_accumulate_data.append((curr_data - (b / a)) * np.exp(-a*i) + (b / a))
       
    for i in range(0,len(predict_accumulate_data)-1):
        predict_data.append(predict_accumulate_data[i+1] - predict_accumulate_data[i])
        
    return predict_data

def LossFunction(y_true, y_pred):
    
    """
    衡量模型好坏的损失函数:
        input:
            y_true:list
            y_pred:list
        return:
            MSE_loss:float
    """
    MSE_loss = 0
    assert len(y_true) == len(y_pred),"数据长度不一致！"
    for index in range(len(y_true)):
        MSE_loss += (y_true[index] - y_pred[index]) ** 2
    return MSE_loss


if __name__ == '__main__':
    k = 10
    pred_data = Predict(ori_data[0],ori_data,k)
    mse_score = LossFunction(ori_data, pred_data) / len(ori_data)
    #R平方介于0~1之间，越接近1，回归拟合效果越好，一般认为超过0.8的模型拟合优度比较高
    r2 = r2_score(ori_data, pred_data)
    print(f'MSE_score:{mse_score:.2f},R2_score:{r2:.2f}')
        
        
        
    