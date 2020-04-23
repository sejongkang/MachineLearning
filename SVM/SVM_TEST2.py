import errno
import os
from datetime import datetime

import numpy as np
import pymysql
from sklearn import preprocessing
from sklearn.externals import joblib

import Data_File
import SVM

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

if __name__ == '__main__':
    y_train = []
    # data = 'raw'
    data = 'ratio'
    # data = 'all'

    # normal = 'mm'
    normal = 'std'
    # normal = None

    normal_start = [63000, 36980, 38300, 64479, 65600]
    normal_end = [63100, 37080, 38400, 64579, 65700]

    h2s_start = [63280, 63434, 63585, 63724, 63868]
    h2s_end = [63380, 63534, 63685, 63824, 63968]

    nh3_start = [64017, 64170, 64332, 64610]
    nh3_end = [64117, 64270, 64432, 64710]

    co_start = [64771, 64940, 65121, 65300, 65455]
    co_end = [64871, 65040, 65221, 65400, 65555]

    h2s_nh3_start = [44150, 44495]
    h2s_nh3_end = [44360, 44635]

    h2s_co_start = [44709,94327]
    h2s_co_end = [44796,94516]

    nh3_co_start = [94600]
    nh3_co_end = [94763]

    start = normal_start
    end = normal_end

    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        if i == 0:
            data = tmp_data
        else:
            data = np.vstack((data, tmp_data))
        for i in range(len(tmp_data)):
            y_train.append(0)

    start = h2s_start
    end = h2s_end
    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        data = np.vstack((data, tmp_data))
        for i in range(len(tmp_data)):
            y_train.append(0)

    start = nh3_start
    end = nh3_end
    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        data = np.vstack((data, tmp_data))
        for i in range(len(tmp_data)):
            y_train.append(0)

    start = co_start
    end = co_end
    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        data = np.vstack((data, tmp_data))
        for i in range(len(tmp_data)):
            y_train.append(1)

    y_train = np.array(y_train)
    reader = Data_File.Data_File()

    data_ratio = reader.get_Ratio(data)

    # H2S+NH3
    # ratio_num = [2, 23]  #H2S
    # ratio_num = [7, 17, 26, 28] #NH3

    # H2S+CO
    # ratio_num = [2]  #H2S
    # ratio_num = [4,14,15,18,21,22] #CO

    # NH3+CO
    # ratio_num = [15]  # NH3
    ratio_num = [14,16] #CO

    for i in range(len(ratio_num)):
        tmp_data = data_ratio[:, ratio_num[i]]
        if i == 0:
            x_train = np.transpose([tmp_data])
        else:
            x_train = np.hstack((x_train, np.transpose([tmp_data])))

    svm = SVM.SVM()

    now = datetime.utcnow().strftime("%Y%m%d")
    model_dir = "Model/{}/".format(now)
    try:
        if not (os.path.isdir(model_dir)):
            os.makedirs(os.path.join(model_dir))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    path = model_dir+"SVM_model.pkl"

    svm.train(x_train, y_train, path=path)

    # start = co_start
    # end = co_end
    # start = h2s_nh3_start
    # end = h2s_nh3_end
    start = h2s_co_start
    end = h2s_co_end
    start = nh3_co_start
    end = nh3_co_end

    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        if i == 0:
            x_valid = tmp_data
        else:
            x_valid = np.vstack((x_valid, tmp_data))

    data_ratio = reader.get_Ratio(x_valid)

    for i in range(len(ratio_num)):
        tmp_data = data_ratio[:, ratio_num[i]]
        if i == 0:
            x_valid = np.transpose([tmp_data])
        else:
            x_valid = np.hstack((x_valid, np.transpose([tmp_data])))

    svc = joblib.load(path)

    results=[]
    probas = svc.predict_proba(x_valid)
    for x in probas:
        if x[0]<x[1]:
            results.append(1)
        else:
            results.append(0)
    percents = results.count(1)/len(results)*100

    # results = svc.predict(x_valid)
    print(probas)
    print(percents)
