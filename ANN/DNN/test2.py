import errno
import os
import timeit
import torch
from datetime import datetime

import numpy as np
import pymysql
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from torch.nn import MSELoss
from shse.mlearn.neural_network import DNN
from shse.mlearn.utils import save_model, load_model
from sklearn.externals import joblib

import Data_File


def modelTrain(x_data,y_data,name):
    start = timeit.default_timer()
    before = 0

    x_data = torch.Tensor(x_data).float()
    y_data = torch.Tensor(y_data).float()

    input = len(x_data[0])
    hidden = 200
    hidden2 = 300
    hidden3 = 300
    hidden4 = 200
    hidden5 = 100
    output = 4

    epoch = 10000
    learning_rate = 0.01
    batch_size = 10

    dnn = DNN(input, output, [hidden,hidden2,hidden3,hidden4,hidden5],softmax=True)

    for step, loss in dnn.learn(x_data, y_data, epoch=epoch, learning_rate=learning_rate, batch_size=batch_size, loss_func=MSELoss):

        if step % 10 == 0:
            gap = loss.tolist() - before
            before = loss.tolist()
            pick = timeit.default_timer()
            time_gap = pick-start
            print('step : %d, loss : %.10f, loss gap = %.10f, seconds = %d' %(step,loss.tolist(),gap,time_gap))
        # if loss ==0 or gap==0:
        #     gap = loss.tolist() - before
        #     pick = timeit.default_timer()
        #     time_gap = pick - start
        #     print('step : %d, loss : %.10f, loss gap = %.10f, seconds = %d' % (step, loss.tolist(),gap,time_gap))
        #     break

    print('step : %d, loss : %.10f, loss gap = %.10f, seconds = %d' % (step, loss.tolist(), gap, time_gap))
    save_model(dnn, name+'.pkl')

def modelValid(x_data,y_data,name):

    dnn = load_model(name+'.pkl')
    dnn.train(False)

    for i in range(len(x_data)):
        x = torch.Tensor(x_data[i]).float()
        x = x.unsqueeze(dim=0)
        result = dnn.forward(x.cuda())
        print(np.round(result.tolist(), 2), y_data[i])
        # classify = torch.argmax(result, dim=-1)
        # classify = classify.squeeze().tolist()
        # print(classify, y_data[i])

    # percent = (count/num)*100
    # print(str(percent)+"%")
    #
    # plot_confusion_matrix(np.asarray(y_data),np.asarray(pred),classes=class_names,normalize=False,path=cm_path,percent=percent)
    # plot_confusion_matrix(np.asarray(y_data), np.asarray(pred), classes=class_names,normalize=True,path=cm_norm_path,percent=percent)
    #
    # plt.show()

def Db_seelct(start,end):
    print('Data_number select')
    conn = pymysql.connect(host='203.250.78.169',port=3307, database='gas', user='root', password='offset01')
    with conn.cursor() as cursor:
        sql = 'SELECT `H2`,  `VOC`,  `Methyl`,  `LP`,  `Solvent`,  `NH3` FROM `gas`.`gas_log` ' \
              'WHERE '+str(start)+'<=`idx` AND `idx`<='+str(end)+';'
        cursor.execute(sql)
        data = cursor.fetchall()
    conn.close()
    return data

def minmax_scale(X,model_dir,train=False, feature_range=(0, 1), axis=0, copy=True,type=type):
    # Unlike the scaler object, this function allows 1d input.
    # If copy is required, it will be done inside the scaler object.
    X = check_array(X, copy=False, ensure_2d=False, warn_on_dtype=True,
                    dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
    original_ndim = X.ndim

    if original_ndim == 1:
        X = X.reshape(X.shape[0], 1)
    if train==True:
        s = MinMaxScaler(feature_range=feature_range, copy=copy)
        s.fit(X)
        joblib.dump(s, model_dir + 'mm.pkl')  # 모델 세이브
    else :
        s = joblib.load(model_dir + 'mm.pkl')
    if axis == 0:
        X = s.transform(X)
    else:
        X = s.transform(X.T).T
    if original_ndim == 1:
        X = X.ravel()

    return X

if __name__ == '__main__':

    y_train = []
    reader = Data_File.Data_File()

    now = datetime.utcnow().strftime("%Y%m%d")
    model_dir = "model/{}/".format(now)
    try:
        if not (os.path.isdir(model_dir)):
            os.makedirs(os.path.join(model_dir))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    save_name = model_dir+"Clf_ratio_only"

    # normal = 'mm'
    normal = 'std'
    # normal = None

    normal_start = [63000, 36980, 38300, 64479]
    normal_end = [63100, 37080, 38400, 64579]

    h2s_start = [63280, 63434, 63585, 63724]
    h2s_end = [63380, 63534, 63685, 63824]

    nh3_start = [64017, 64170, 64332]
    nh3_end = [64117, 64270, 64432]

    co_start = [64771, 64940, 65121, 65300]
    co_end = [64871, 65040, 65221, 65400]

    h2s_nh3_start = [44150]
    h2s_nh3_end = [44360]

    h2s_co_start = [94327]
    h2s_co_end = [94516]

    nh3_co_start = [94600]
    nh3_co_end = [94763]

    class_bin = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]]

    start = normal_start
    end = normal_end
    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        if i == 0:
            data = tmp_data
        else:
            data = np.vstack((data, tmp_data))
        for i in range(len(tmp_data)):
            y_train.append(class_bin[0])

    start = h2s_start
    end = h2s_end
    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        data = np.vstack((data, tmp_data))
        for i in range(len(tmp_data)):
            y_train.append(class_bin[1])

    start = nh3_start
    end = nh3_end
    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        data = np.vstack((data, tmp_data))
        for i in range(len(tmp_data)):
            y_train.append(class_bin[2])

    start = co_start
    end = co_end
    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        data = np.vstack((data, tmp_data))
        for i in range(len(tmp_data)):
            y_train.append(class_bin[3])

    start = h2s_nh3_start
    end = h2s_nh3_end
    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        data = np.vstack((data, tmp_data))
        for i in range(len(tmp_data)):
            y_train.append(class_bin[4])

    start = h2s_co_start
    end = h2s_co_end
    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        data = np.vstack((data, tmp_data))
        for i in range(len(tmp_data)):
            y_train.append(class_bin[5])

    start = nh3_co_start
    end = nh3_co_end
    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        data = np.vstack((data, tmp_data))
        for i in range(len(tmp_data)):
            y_train.append(class_bin[6])

    x_train = reader.get_Ratio(data)
    y_train = np.array(y_train)

    # x_train = minmax_scale(x_train,model_dir,train=True)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    joblib.dump(scaler, model_dir+"scaler.pkl")

    modelTrain(x_train, y_train, save_name)

#-----------------------------------------------------------------------------------------------------------

    y_valid=[]

    normal_start = [65600]
    normal_end = [65700]

    h2s_start = [63868]
    h2s_end = [63968]

    nh3_start = [64610]
    nh3_end = [64710]

    co_start = [65455]
    co_end = [65555]

    h2s_nh3_start = [100314,44495]
    h2s_nh3_end = [100482,44635]

    nh3_co_start = [100593]
    nh3_co_end = [100773]

    h2s_co_start = [44709]
    h2s_co_end = [44796]

    start = normal_start
    end = normal_end
    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        if i == 0:
            data = tmp_data
        else:
            data = np.vstack((data, tmp_data))
        for i in range(len(tmp_data)):
            y_valid.append(class_bin[0])

    start = h2s_start
    end = h2s_end
    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        data = np.vstack((data, tmp_data))
        for i in range(len(tmp_data)):
            y_valid.append(class_bin[1])

    start = nh3_start
    end = nh3_end
    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        data = np.vstack((data, tmp_data))
        for i in range(len(tmp_data)):
            y_valid.append(class_bin[2])

    start = co_start
    end = co_end
    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        data = np.vstack((data, tmp_data))
        for i in range(len(tmp_data)):
            y_valid.append(class_bin[3])

    start = h2s_nh3_start
    end = h2s_nh3_end
    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        data = np.vstack((data, tmp_data))
        for i in range(len(tmp_data)):
            y_valid.append(class_bin[4])

    start = h2s_co_start
    end = h2s_co_end
    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        data = np.vstack((data, tmp_data))
        for i in range(len(tmp_data)):
            y_valid.append(class_bin[5])

    start = nh3_co_start
    end = nh3_co_end
    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        data = np.vstack((data, tmp_data))
        for i in range(len(tmp_data)):
            y_valid.append(class_bin[6])

    x_valid = reader.get_Ratio(data)
    y_valid = np.array(y_valid)

    # x_train = minmax_scale(x_valid, model_dir)

    scaler = joblib.load(model_dir+"scaler.pkl")
    x_valid = scaler.transform(x_valid)

    modelValid(x_valid, y_valid, save_name)