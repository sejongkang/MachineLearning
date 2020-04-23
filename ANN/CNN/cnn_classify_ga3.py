from datetime import datetimeimport errnoimport osimport matplotlib.pyplot as pltimport timeitimport pymysqlfrom sklearn.decomposition import PCAfrom shse.mlearn.neural_network import DNN, CNNimport torchfrom shse.mlearn.utils import save_model, load_modelfrom sklearn.externals import joblibfrom sklearn.model_selection import train_test_splitimport numpy as npfrom sklearn.metrics import confusion_matrixfrom sklearn.preprocessing import StandardScalerdef get_Ratio(sensorValue):    sensorValue = np.array(sensorValue)    featlist = []    for i in range(6):        for j in range(6):            if i != j :                featlist.append(sensorValue[:, j] / sensorValue[:, i])    return np.transpose(featlist)def modelTrain(x_data,y_data):    # now = datetime.utcnow().strftime("%Y%m%d")    # model_dir = "model/{}/".format(now)    name = model_dir + "/Clf_" + data_type    try:        if not (os.path.isdir(model_dir)):            os.makedirs(os.path.join(model_dir))    except OSError as e:        if e.errno != errno.EEXIST:            raise    start = timeit.default_timer()    before = 0    scaler = StandardScaler()    scaler.fit(x_data)    x_data = scaler.transform(x_data)    joblib.dump(scaler, model_dir + "/" + data_type + "_scaler.pkl")    x_data = np.reshape(x_data, (-1, 1, 3))    y_data = np.reshape(y_data, (-1, 1))    x_data = torch.Tensor(x_data).float()    y_data = torch.Tensor(y_data).long()    y_data = y_data.squeeze_()    input = len(x_data[0][0])    hidden = 200    hidden2 = 300    hidden3 = 300    hidden4 = 500    hidden5 = 400    output = 5    epoch = 1000    learning_rate = 0.001    batch_size = 10    cnn = CNN(input, output, [1, 30, 30, -1, 10], [hidden, hidden2, hidden3, hidden4, hidden5], kernel_size=3, stride=1,              padding=1, softmax=True)    for step, loss in cnn.learn(x_data, y_data, epoch=epoch, learning_rate=learning_rate, batch_size=batch_size):        if step % 100 == 0:            gap = loss.tolist() - before            before = loss.tolist()            pick = timeit.default_timer()            time_gap = pick - start            print('step : %d, loss : %.10f, loss gap = %.10f, seconds = %d' % (step, loss.tolist(), gap, time_gap))        if loss == 0:            gap = loss.tolist() - before            pick = timeit.default_timer()            time_gap = pick - start            break    print('step : %d, loss : %.10f, loss gap = %.10f, seconds = %d' % (step, loss.tolist(), gap, time_gap))    save_model(cnn, name + '.pkl')def modelValid(x_data,y_data):    pred=[]    count=0    model_dir = "Model/Classify/"+load_date    name = model_dir + '/Clf_' + data_type    scaler = joblib.load(model_dir+"/"+data_type+"_scaler.pkl")    x_data = scaler.transform(x_data)    x_data = np.reshape(x_data, (-1, 1, 3))    cnn = load_model(name+'.pkl')    for i in range(len(x_data)):        result = cnn.test(torch.unsqueeze(torch.Tensor(x_data[i]).float(), dim=0))        result_label = torch.argmax(result, dim=-1)        result = result_label.squeeze().tolist()        if result==y_data[i][0]:            count=count+1        pred.append(result)        print(result, "-", y_data[i][0])    percent = (count/len(x_data))*100    print(str(percent)+"%")    model_dir = model_dir + "/cm/"    try:        if not (os.path.isdir(model_dir)):            os.makedirs(os.path.join(model_dir))    except OSError as e:        if e.errno != errno.EEXIST:            raise    name = model_dir+data_type+"_cm.png"    norm_name = model_dir+data_type+"_norm_cm.png"    plot_confusion_matrix(np.asarray(y_data),np.asarray(pred), normalize=False,path=name,percent=percent)    plot_confusion_matrix(np.asarray(y_data), np.asarray(pred), normalize=True,path=norm_name,percent=percent)    #    # plt.show()def plot_confusion_matrix(y_true, y_pred,                          normalize=False,                          title=None,                          cmap=plt.cm.Blues,                          path=False,                          percent=100):    """    This function prints and plots the confusion matrix.    Normalization can be applied by setting `normalize=True`.    """    if not title:        if normalize:            title = 'Normalized confusion matrix : ' + str(percent) + '%'        else:            title = 'Confusion matrix, without normalization : ' + str(percent) +'%'    # Compute confusion matrix    cm = confusion_matrix(y_true, y_pred)    # Only use the labels that appear in the data    # classes = unique_labels(y_true, y_pred)    classes = ['H2S', 'NH3', 'CH3SH', 'CO', 'CH4']    if normalize:        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    fig, ax = plt.subplots()    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)    ax.figure.colorbar(im, ax=ax)    # We want to show all ticks...    ax.set(xticks=np.arange(cm.shape[1]),           yticks=np.arange(cm.shape[0]),           # ... and label them with the respective list entries           xticklabels=classes, yticklabels=classes,           title=title,           ylabel='True label',           xlabel='Predicted label')    # Rotate the tick labels and set their alignment.    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",             rotation_mode="anchor")    # Loop over data dimensions and create text annotations.    fmt = '.2f' if normalize else 'd'    thresh = cm.max() / 2.    for i in range(cm.shape[0]):        for j in range(cm.shape[1]):            ax.text(j, i, format(cm[i, j], fmt),                    ha="center", va="center",                    color="white" if cm[i, j] > thresh else "black")    fig.tight_layout()    plt.savefig(path)    return axdef Db_seelct(start,end):    # print('Data_number select')    conn = pymysql.connect(host='203.250.78.169',port=3307, database='gas', user='root', password='offset01')    with conn.cursor() as cursor:        sql = 'SELECT `H2`,  `VOC`,  `Methyl`,  `LP`,  `Solvent`,  `NH3` FROM `gas`.`gas_log` ' \              'WHERE '+str(start)+'<=`idx` AND `idx`<='+str(end)+';'        cursor.execute(sql)        data = cursor.fetchall()    conn.close()    return dataif __name__ == '__main__':    data_type = 'single_ga'    print(data_type)    save_date = '20191123'    load_date = '20191123'    model_dir = "Model/Classify/"+save_date    class_num = [0, 1, 2, 3, 4]    classes = []    features = []    start = []    end = []    # Class 0    h2s_start = [63280, 599560, 63585, 599856, 63883, 600180, 600333, 600475]    h2s_end = [63380, 599660, 63685, 599966, 63968, 600280, 600433, 600575]    start.append(h2s_start)    end.append(h2s_end)    # Class 1    nh3_start = [600619, 600960, 601650, 601110, 64332, 601800, 601940, 603240, 64611, 603465]    nh3_end = [600719, 601060, 601750, 601210, 64432, 601900, 602030, 603346, 64710, 603565]    start.append(nh3_start)    end.append(nh3_end)    # Class 2    ch3sh_start = [603640, 603760, 603870, 604010, 604450, 604610, 604770, 604910]    ch3sh_end = [603740, 603860, 603970, 604110, 604550, 604720, 604870, 605010]    start.append(ch3sh_start)    end.append(ch3sh_end)    # Class 3    co_start = [605110, 605270, 605740, 605890]    co_end = [605210, 605370, 605840, 605990]    start.append(co_start)    end.append(co_end)    # Class 4    ch4_start = [606030, 606170, 606610, 606815, 607261, 607410]    ch4_end = [606130, 606340, 606730, 606940, 607361, 607510]    start.append(ch4_start)    end.append(ch4_end)    for i in range(len(start)):        start_tmp = start[i]        end_tmp = end[i]        for j in range(len(start_tmp)):            tmp_data = Db_seelct(start_tmp[j], end_tmp[j])            if j == 0:                data = tmp_data            else:                data = np.vstack((data, tmp_data))            for k in range(len(tmp_data)):                classes.append(class_num[i])        if i == 0:            features = data        else:            features = np.vstack((features, data))    features = get_Ratio(features)    features = features[:, (0, 14, 29)]    # x_train, x_valid, y_train, y_valid = train_test_split(features, classes, test_size=0.99, random_state=321)    modelTrain(features, classes)    # -----------------------------------------------------    classes = []    features = []    start = []    end = []    # Class 0    h2s_start = [63434, 599680, 63724, 600000]    h2s_end = [63534, 599795, 63824, 600100]    start.append(h2s_start)    end.append(h2s_end)    # Class 1    nh3_start = [64017, 600760, 601280, 601485]    nh3_end = [64117, 600860, 601436, 601585]    start.append(nh3_start)    end.append(nh3_end)    # Class 2    ch3sh_start = [604160, 604310]    ch3sh_end = [604260, 604410]    start.append(ch3sh_start)    end.append(ch3sh_end)    # Class 3    co_start = [605410, 605555]    co_end = [605510, 605710]    start.append(co_start)    end.append(co_end)    # Class 4    ch4_start = [606965, 607125]    ch4_end = [607075, 607240]    start.append(ch4_start)    end.append(ch4_end)    for i in range(len(start)):        start_tmp = start[i]        end_tmp = end[i]        for j in range(len(start_tmp)):            tmp_data = Db_seelct(start_tmp[j], end_tmp[j])            if j == 0:                data = tmp_data            else:                data = np.vstack((data, tmp_data))            for k in range(len(tmp_data)):                classes.append(class_num[i])        if i == 0:            features = data        else:            features = np.vstack((features, data))    features = get_Ratio(features)    features = features[:, (0, 14, 29)]    modelValid(features, classes)