from numpy import randomfrom datetime import datetimeimport errnoimport osimport matplotlib.pyplot as pltimport timeitimport pymysqlfrom shse.mlearn.neural_network import DNNimport torchfrom shse.mlearn.utils import save_model, load_modelfrom sklearn.externals import joblibfrom sklearn.model_selection import train_test_splitimport numpy as npfrom sklearn.metrics import confusion_matrixfrom sklearn.utils.multiclass import unique_labelsfrom sklearn.preprocessing import StandardScalerdef get_Ratio(sensorValue):    sensorValue = np.array(sensorValue)    featlist = []    for i in range(6):        for j in range(6):            if i != j :                featlist.append(sensorValue[:, j] / sensorValue[:, i])    return np.transpose(featlist)def modelTrain(x_data,y_data):    save_date = '20191118'    # now = datetime.utcnow().strftime("%Y%m%d")    # model_dir = "model/{}/".format(now)    model_dir = "model/"+save_date+"/dnn/"    name = model_dir + "Clf_" + gas_type    try:        if not (os.path.isdir(model_dir)):            os.makedirs(os.path.join(model_dir))    except OSError as e:        if e.errno != errno.EEXIST:            raise    start = timeit.default_timer()    before = 0    scaler = StandardScaler()    scaler.fit(x_data)    x_data = scaler.transform(x_data)    joblib.dump(scaler, 'model/'+save_date+'/dnn/scaler.pkl')    x_data = torch.Tensor(x_data).float()    y_data = torch.Tensor(y_data).long()    y_data = y_data.squeeze_()    input = len(x_data[0])    hidden = 200    hidden2 = 300    hidden3 = 300    hidden4 = 500    hidden5 = 400    output = 2    epoch = 20000    learning_rate = 0.001    batch_size = 10    dnn = DNN(input, output, [hidden,hidden2,hidden3,hidden4,hidden5])    for step, loss in dnn.learn(x_data, y_data, epoch=epoch, learning_rate=learning_rate, batch_size=batch_size):        if step % 1000 == 0:            gap = loss.tolist() - before            before = loss.tolist()            pick = timeit.default_timer()            time_gap = pick-start            print('step : %d, loss : %.10f, loss gap = %.10f, seconds = %d' %(step,loss.tolist(),gap,time_gap))        if loss == 0:            gap = loss.tolist() - before            pick = timeit.default_timer()            time_gap = pick - start            break    print('step : %d, loss : %.10f, loss gap = %.10f, seconds = %d' % (step, loss.tolist(), gap, time_gap))    save_model(dnn, name+'.pkl')def modelValid(x_data,y_data):    load_date = '20191118'    pred=[]    count=0    model_dir = "model/"+load_date+"/dnn/"    name = model_dir + 'Clf_' + gas_type    scaler = joblib.load('model/'+load_date+'/dnn/scaler.pkl')    x_data = scaler.transform(x_data)    dnn = load_model(name+'.pkl')    dnn.train(False)    for i in range(len(x_data)):        x = torch.Tensor(x_data[i]).float()        x = x.unsqueeze(dim=0)        result = dnn.forward(x.cuda())[0]        print(np.round(result.tolist(), 2))    #    #     result_label = torch.argmax(result, dim=-1)    #     result_label = result_label.squeeze().tolist()    #     if result_label==y_data[i][0]:    #         count=count+1    #     pred.append(result_label)    #    #     print(result_label, "-", y_data[i][0])    # percent = (count/len(x_data))*100    # print(gas_type+":"+str(percent)+"%")    #    # name = model_dir + gas_type + "_cm.png"    #    # try:    #     if not (os.path.isdir(model_dir)):    #         os.makedirs(os.path.join(model_dir))    # except OSError as e:    #     if e.errno != errno.EEXIST:    #         raise    #    # plot_confusion_matrix(np.asarray(y_data),np.asarray(pred), normalize=False,path=name,percent=percent)def plot_confusion_matrix(y_true, y_pred,                          normalize=False,                          title=None,                          cmap=plt.cm.Blues,                          path=False,                          percent=100):    """    This function prints and plots the confusion matrix.    Normalization can be applied by setting `normalize=True`.    """    if not title:        if normalize:            title = 'Normalized confusion matrix : ' + str(percent) + '%'        else:            title = 'Confusion matrix, without normalization : ' + str(percent) +'%'    # Compute confusion matrix    cm = confusion_matrix(y_true, y_pred)    # Only use the labels that appear in the data    # classes = unique_labels(y_true, y_pred)    classes = ['Normal', 'H2S', 'NH3', 'CH3SH', 'CO', 'CH4']    if normalize:        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    fig, ax = plt.subplots()    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)    ax.figure.colorbar(im, ax=ax)    # We want to show all ticks...    ax.set(xticks=np.arange(cm.shape[1]),           yticks=np.arange(cm.shape[0]),           # ... and label them with the respective list entries           xticklabels=classes, yticklabels=classes,           title=title,           ylabel='True label',           xlabel='Predicted label')    # Rotate the tick labels and set their alignment.    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",             rotation_mode="anchor")    # Loop over data dimensions and create text annotations.    fmt = '.2f' if normalize else 'd'    thresh = cm.max() / 2.    for i in range(cm.shape[0]):        for j in range(cm.shape[1]):            ax.text(j, i, format(cm[i, j], fmt),                    ha="center", va="center",                    color="white" if cm[i, j] > thresh else "black")    fig.tight_layout()    plt.savefig(path)    return axdef Db_seelct(start,end):    # print('Data_number select')    conn = pymysql.connect(host='203.250.78.169',port=3307, database='gas', user='root', password='offset01')    with conn.cursor() as cursor:        sql = 'SELECT `H2`,  `VOC`,  `Methyl`,  `LP`,  `Solvent`,  `NH3` FROM `gas`.`gas_log` ' \              'WHERE '+str(start)+'<=`idx` AND `idx`<='+str(end)+';'        cursor.execute(sql)        data = cursor.fetchall()    conn.close()    return datadef Do(num):    global gas_type    gas_type = class_name[num]    classes = []    features = []    start = []    end = []    # Class 0    normal_start = [63000, 36980, 38300, 64479, 65600]    normal_end = [63100, 37080, 38400, 64579, 65700]    start.append(normal_start)    end.append(normal_end)    # Class 1    h2s_start = [63280, 599560, 63434, 599680, 63585, 599856, 63724, 600000, 63883, 600180, 600333, 600475 ]    h2s_end = [63380, 599660, 63534, 599795, 63685, 599966, 63824, 600100, 63968, 600280, 600433, 600575]    start.append(h2s_start)    end.append(h2s_end)    # Class 2    nh3_start = [600619, 600960, 64017, 600760, 601650, 601110, 601280, 601485, 64332, 601800, 601940, 603240, 64611, 603465]    nh3_end = [600719, 601060, 64117, 600860, 601750, 601210, 601436, 601585, 64432, 601900, 602030, 603346, 64710, 603565]    start.append(nh3_start)    end.append(nh3_end)    # Class 3    ch3sh_start = [603640, 603760, 603870, 604010, 604160, 604310, 604450, 604610, 604770, 604910]    ch3sh_end = [603740, 603860, 603970, 604110, 604260, 604410, 604550, 604720, 604870, 605010]    start.append(ch3sh_start)    end.append(ch3sh_end)    # Class 4    co_start = [605110, 605270, 605410, 605555, 605740, 605890]    co_end = [605210, 605370, 605510, 605710, 605840, 605990]    start.append(co_start)    end.append(co_end)    # Class 5    ch4_start = [606030, 606170, 606610, 606815, 606965, 607125, 607261, 607410]    ch4_end = [606130, 606340, 606730, 606940, 607075, 607240, 607361, 607510]    start.append(ch4_start)    end.append(ch4_end)    class_num = [[0], [1]]    # for i in range(len(start)):    #     start_tmp = start[i]    #     end_tmp = end[i]    #     for j in range(len(start_tmp)):    #         tmp_data = Db_seelct(start_tmp[j], end_tmp[j])    #         if j == 0:    #             data = tmp_data    #         else:    #             data = np.vstack((data, tmp_data))    #         for k in range(len(tmp_data)):    #             if gas_type == 'H2S':    #                 if i == 1:    #                     classes.append(class_num[1])    #                 else:    #                     classes.append(class_num[0])    #             elif gas_type == 'NH3':    #                 if i == 2:    #                     classes.append(class_num[1])    #                 else:    #                     classes.append(class_num[0])    #             elif gas_type == 'CH3SH':    #                 if i == 3:    #                     classes.append(class_num[1])    #                 else:    #                     classes.append(class_num[0])    #             elif gas_type == 'CO':    #                 if i == 4:    #                     classes.append(class_num[1])    #                 else:    #                     classes.append(class_num[0])    #             elif gas_type == 'CH4':    #                 if i == 5:    #                     classes.append(class_num[1])    #                 else:    #                     classes.append(class_num[0])    #     if i == 0:    #         features = data    #     else:    #         features = np.vstack((features, data))    #    # classes = np.array(classes)    #    # classes_0 = classes[(classes == 0).reshape(-1)]    # features_0 = features[(classes == 0).reshape(-1)]    # classes_1 = classes[(classes == 1).reshape(-1)]    # features_1 = features[(classes == 1).reshape(-1)]    #    # rand_idx = random.choice(len(features_0), len(features_1),replace=False)    # features_0 = features_0[rand_idx]    # classes_0 = classes_0[rand_idx]    #    # features = np.vstack((features_0, features_1))    # classes = np.vstack((classes_0, classes_1))    #    # features = get_Ratio(features)    #    # x_train, x_valid, y_train, y_valid = train_test_split(features, classes, test_size=0.33, random_state=321)    #    # modelTrain(x_train, y_train)    # modelValid(features, classes)if __name__ == '__main__':    class_name = ['H2S', 'NH3', 'CH3SH', 'CO', 'CH4']    # for i in range(5):    #     Do(i)    gas_type = class_name[1]    class_num = [[0], [1]]    classes = []    valid_start = []    valid_end = []    h2s_nh3_start = [44150, 100314, 44495]    h2s_nh3_end = [44360, 100482, 44635]    valid_start.append(h2s_nh3_start)    valid_end.append(h2s_nh3_end)    for i in range(len(valid_start)):        start_tmp = valid_start[i]        end_tmp = valid_end[i]        for j in range(len(start_tmp)):            tmp_data = Db_seelct(start_tmp[j], end_tmp[j])            if j == 0:                data = tmp_data            else:                data = np.vstack((data, tmp_data))            for k in range(len(tmp_data)):                if gas_type == 'H2S' or gas_type == 'NH3':                    classes.append(class_num[1])                else:                    classes.append(class_num[0])    features = get_Ratio(data)    modelValid(features, classes)