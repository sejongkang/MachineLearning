import math
from datetime import datetime
import errno
import os
import timeit
import pymysql
from sklearn.model_selection import train_test_split
from shse.mlearn.neural_network import DNN, CNN
import torch
from shse.mlearn.utils import save_model, load_model
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def modelTrain(x_data,y_data):

    # now = datetime.utcnow().strftime("%Y%m%d")
    # model_dir = "model/{}/".format(now)
    model_dir = "Model/Regression/"+save_date+"/"

    name = model_dir + gas_type + "_reg"

    try:
        if not (os.path.isdir(model_dir)):
            os.makedirs(os.path.join(model_dir))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    start = timeit.default_timer()
    before = 0

    x_data = torch.Tensor(x_data).float()
    y_data = torch.Tensor(y_data).float()

    input = len(x_data[0])
    hidden = 10
    hidden2 = 50
    hidden3 = 10
    hidden4 = 30
    hidden5 = 50

    output = 1

    epoch = 20000
    learning_rate = 0.001
    batch_size = 20

    dnn = DNN(input, output, [hidden,hidden2,hidden3,hidden4,hidden5],softmax=False)

    for step, loss in dnn.learn(x_data, y_data, epoch=epoch, learning_rate=learning_rate, batch_size=batch_size,
                                loss_func=MSELoss):

        if step % 100 == 0:
            gap = loss.tolist() - before
            before = loss.tolist()
            pick = timeit.default_timer()
            time_gap = pick-start
            print('step : %d, loss : %.10f, loss gap = %.10f, seconds = %d' %(step,loss.tolist(),gap,time_gap))
        if loss==0:
            gap = loss.tolist() - before
            pick = timeit.default_timer()
            time_gap = pick - start
            break
    print('step : %d, loss : %.10f, loss gap = %.10f, seconds = %d' % (step, loss.tolist(), gap, time_gap))
    save_model(dnn, name+'.pkl')


def modelValid(x_data,y_data):

    error = 0
    err_list =[]
    y_list = np.unique(y_data)
    y_count =[]
    for j in range(len(y_list)):
        err_list.append(0)
        y_count.append(0)

    model_dir = "Model/Regression/"+load_date+"/"
    name = model_dir + gas_type + "_reg"

    dnn = load_model(name+'.pkl')
    dnn.train(False)

    for i in range(len(x_data)):
        x = torch.Tensor(x_data[i]).float()
        x = x.unsqueeze(dim=0)
        result = dnn.forward(x.cuda())

        result_int = int(result.tolist()[0][0])

        err = round(abs(y_data[i] - result_int)/y_data[i] * 100,1)
        error = error + err

        for j in range(len(y_list)):
            if y_data[i] == y_list[j]:
                err_list[j] = err_list[j]+err
                y_count[j] = y_count[j] + 1
        print(result_int, "-", y_data[i], ":", err)  # 인트

    for j in range(len(y_list)):
        print(y_list[j], ":", round(err_list[j] / y_count[j], 1))
    print(gas_type, "Error :", round(sum(err_list) / sum(y_count), 1))
    print("")

def Db_seelct(start,end):
    conn = pymysql.connect(host='203.250.78.169',port=3307, database='gas', user='root', password='offset01')
    with conn.cursor() as cursor:
        sql = 'SELECT `H2`,  `VOC`,  `Methyl`,  `LP`,  `Solvent`,  `NH3` FROM `gas`.`gas_log` ' \
              'WHERE '+str(start)+'<=`idx` AND `idx`<='+str(end)+';'
        cursor.execute(sql)
        data = cursor.fetchall()
    conn.close()
    return data

def Do():
    y_data = []

    if gas_type == type[0]:
        start = [600333, 600475, 626005]
        end = [600433, 600575, 626133]
        classes = [30]
    elif gas_type == type[1]:
        start = [601940, 603240, 626218]
        end = [602030, 603346, 626318]
        classes = [35]
    elif gas_type == type[2]:
        start = [604770, 604910, 626480]
        end = [604870, 605010, 626580]
        classes = [25]
    elif gas_type == type[3]:
        start = [605740, 605890, 626670]
        end = [605840, 605990, 626800]
        classes = [300]
    elif gas_type == type[4]:
        start = [607261, 607410, 626860]
        end = [607361, 607510, 626980]
        classes = [8000]

    for i in range(len(start)):
        tmp_data = Db_seelct(start[i], end[i])
        if i == 0:
            x_data = tmp_data
        else:
            x_data = np.vstack((x_data, tmp_data))
        print(int(len(tmp_data) * 2 / 3), int(len(tmp_data) * 1 / 3))
        for j in range(len(tmp_data)):
            y_data.append(classes[0])

    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.2, random_state=321)

    modelTrain(x_train, y_train)

    modelValid(x_valid, y_valid)

if __name__ == '__main__':

    save_date = '20191202'
    load_date = '20191202'

    type = ['H2S', 'NH3', 'CH3SH', 'CO', 'CH4']

    for i in range(len(type)):
        gas_type = type[i]
        Do()

# Valid ----------------------------------------------------------------------------------------------------------------------
#
#     y_valid=[]
#
#     if gas_type == 'H2S':
#         x_valid_num = [63280,63434,63585,63724,63868]
#         y_valid_num = np.asarray([5,10,15,20,25])
#     elif gas_type == 'NH3':
#         x_valid_num = [64017,64170,64332,64610]
#         y_valid_num = np.asarray([10,15,25,40])
#     elif gas_type == 'CO':
#         x_valid_num = [64771, 64940, 65121, 65300, 65455]
#         y_valid_num = np.asarray([30, 70, 100, 150, 200])
#
#     for i in range(len(x_valid_num)):
#         if i == 0:
#             x_valid = GetData(x_valid_num[i])
#         else:
#             x_valid = np.vstack((x_valid, GetData(x_valid_num[i])))
#         for j in range(len(GetData(x_valid_num[i]))):
#             y_valid.append([y_valid_num[i]])
#
#     y_train=np.array(y_train)



