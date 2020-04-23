from datetime import datetimeimport errnoimport osimport matplotlib.pyplot as pltimport timeitimport Data_Filefrom shse.mlearn.neural_network import DNNimport torchfrom shse.mlearn.utils import save_model, load_modelfrom sklearn.externals import joblibfrom sklearn.utils import check_arrayfrom sklearn.utils.validation import FLOAT_DTYPESfrom torch.nn import CrossEntropyLoss, L1Loss, MSELossimport numpy as npfrom sklearn.metrics import confusion_matrixfrom sklearn.utils.multiclass import unique_labelsfrom sklearn.preprocessing import MinMaxScalerdef modelTrain(x_data,y_data,name,isminmax=False,type=type):    start = timeit.default_timer()    before = 0    if isminmax == True:        x_data = minmax_scale(x_data,train=True,type=type)    x_data = torch.Tensor(x_data).float()    y_data = torch.Tensor(y_data).float()    input = len(x_data[0])    hidden = 200    hidden2 = 300    hidden3 = 300    hidden4 = 500    hidden5 = 400    output = 4    epoch = 500    learning_rate = 0.001    batch_size = 5    dnn = DNN(input, output, [hidden,hidden2,hidden3,hidden4,hidden5],softmax=True)    for step, loss in dnn.learn(x_data, y_data, epoch=epoch, learning_rate=learning_rate, batch_size=batch_size,                                loss_func=MSELoss):        if step % 10 == 0:            gap = loss.tolist() - before            before = loss.tolist()            pick = timeit.default_timer()            time_gap = pick-start            print('step : %d, loss : %.10f, loss gap = %.10f, seconds = %d' %(step,loss.tolist(),gap,time_gap))        if loss ==0 or gap==0:            gap = loss.tolist() - before            pick = timeit.default_timer()            time_gap = pick - start            print('step : %d, loss : %.10f, loss gap = %.10f, seconds = %d' % (step, loss.tolist(),gap,time_gap))            break    print('step : %d, loss : %.10f, loss gap = %.10f, seconds = %d' % (step, loss.tolist(), gap, time_gap))    save_model(dnn, name+"_"+type+'.pkl')def modelValid(x_data,y_data,name,isminmax=False,type=type):    if isminmax == True :        x_data = minmax_scale(x_data,type=type)    dnn = load_model(name+"_"+type+'.pkl')    dnn.train(False)    for i in range(len(x_data)):        x = torch.Tensor(x_data[i]).float()        x = x.unsqueeze(dim=0)        result = dnn.forward(x.cuda())        array = np.asarray(result.tolist()[0])        temp = array.argsort()        # ranks = np.arange(len(array))[temp.argsort()]        # print(ranks, y_Data[i])        # print(np.round(np.multiply(result.tolist(), 100000),2) / 100000, y_data[i])        print(np.round(result.tolist(),2), y_data[i])        # result_label = torch.argmax(result, dim=-1)        # result_label = result_label.squeeze().tolist()                # if result_label==y_Data[i]:        #     count=count+1        # pred.append(result_label)    # percent = (count/num)*100    # print(str(percent)+"%")    #    # plot_confusion_matrix(np.asarray(y_data),np.asarray(pred),classes=class_names,normalize=False,path=cm_path,percent=percent)    # plot_confusion_matrix(np.asarray(y_data), np.asarray(pred), classes=class_names,normalize=True,path=cm_norm_path,percent=percent)    #    # plt.show()def plot_confusion_matrix(y_true, y_pred, classes,                          normalize=False,                          title=None,                          cmap=plt.cm.Blues,                          path=False,                          percent=100):    """    This function prints and plots the confusion matrix.    Normalization can be applied by setting `normalize=True`.    """    if not title:        if normalize:            title = 'Normalized confusion matrix : ' + str(percent) + '%'        else:            title = 'Confusion matrix, without normalization : ' + str(percent) +'%'    # Compute confusion matrix    cm = confusion_matrix(y_true, y_pred)    # Only use the labels that appear in the data    classes = classes[unique_labels(y_true, y_pred)]    if normalize:        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    fig, ax = plt.subplots()    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)    ax.figure.colorbar(im, ax=ax)    # We want to show all ticks...    ax.set(xticks=np.arange(cm.shape[1]),           yticks=np.arange(cm.shape[0]),           # ... and label them with the respective list entries           xticklabels=classes, yticklabels=classes,           title=title,           ylabel='True label',           xlabel='Predicted label')    # Rotate the tick labels and set their alignment.    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",             rotation_mode="anchor")    # Loop over data dimensions and create text annotations.    fmt = '.2f' if normalize else 'd'    thresh = cm.max() / 2.    for i in range(cm.shape[0]):        for j in range(cm.shape[1]):            ax.text(j, i, format(cm[i, j], fmt),                    ha="center", va="center",                    color="white" if cm[i, j] > thresh else "black")    fig.tight_layout()    plt.savefig(path)    return axdef minmax_scale(X,train=False, feature_range=(0, 1), axis=0, copy=True,type=type):    # Unlike the scaler object, this function allows 1d input.    # If copy is required, it will be done inside the scaler object.    X = check_array(X, copy=False, ensure_2d=False, warn_on_dtype=True,                    dtype=FLOAT_DTYPES, force_all_finite='allow-nan')    original_ndim = X.ndim    if original_ndim == 1:        X = X.reshape(X.shape[0], 1)    if train==True:        s = MinMaxScaler(feature_range=feature_range, copy=copy)        s.fit(X)        joblib.dump(s, 'model/minmax_model_'+type+'.pkl')  # 모델 세이브    else :        s = joblib.load('model/minmax_model_'+type+'.pkl')    if axis == 0:        X = s.transform(X)    else:        X = s.transform(X.T).T    if original_ndim == 1:        X = X.ravel()    return Xif __name__ == '__main__':    # data = 'raw'    data = 'raw+'    # data = 'ratio'    # data = 'ratio+'    # data = 'all'    # data = 'all+'    now = datetime.utcnow().strftime("%Y%m%d")    save_name = 'Clf'    load_name = 'Clf'    # cm_number = 2    load_date = '20190713'    data_date = '20190712'    path = '../Data/acquisition/'+data_date    minmax = True    model_dir = "model/{}/".format(now)    # valid_dir = "valid/{}/{}".format(now,cm_number)    save_name = model_dir+save_name    load_name = 'model/' + load_date + '/' + load_name    try:        if not (os.path.isdir(model_dir)):            os.makedirs(os.path.join(model_dir))        # if not (os.path.isdir(valid_dir)):        #     os.makedirs(os.path.join(valid_dir))    except OSError as e:        if e.errno != errno.EEXIST:            raise    # cm_path = valid_dir+'/Confusion matrix.png'    # cm_norm_path = valid_dir+'/Confusion matrix_normalize.png'    class_name = np.asarray(['NO','H2S_5','H2S_25','NH3_10','NH3_40','H2S_10_NH3_10','H2S_25_NH3_40','H2S_20','NH3_25','H2S_15_NH3_25'])    class_bin = [[0, 0, 0, 0],[0, 1, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 1, 0],[0, 1, 1, 0],[0, 1, 1, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 1, 1, 0]]    for i in range(6):        reader = Data_File.Data_File(path, class_name[i], class_bin[i])        if i == 0:            if data == 'raw':                x_data, y_data = reader.raw_read(issplit=False)                minmax = False            elif data == 'raw+':                x_data, y_data = reader.raw_read(issplit=False)                x_data = reader.add_feature(x_data)            elif data == 'ratio':                x_data, y_data = reader.raw_read(issplit=False)                x_data = reader.get_Ratio(x_data)            elif data == 'ratio+':                x_data, y_data = reader.raw_read(issplit=False)                x_data = reader.get_Ratio(x_data)                x_data = reader.add_feature(x_data)            elif data == 'all':                x_data, y_data = reader.raw_read(issplit=False)                x_data = reader.get_Ratio(x_data,all=True)            else:                x_data, y_data = reader.raw_read(issplit=False)                x_data = reader.get_Ratio(x_data, all=True)                x_data = reader.add_feature(x_data)            x_train = x_data            y_train = y_data        else:            if data == 'raw':                x_data, y_data = reader.raw_read(issplit=False)            elif data == 'raw+':                x_data, y_data = reader.raw_read(issplit=False)                x_data = reader.add_feature(x_data)            elif data == 'ratio':                x_data = reader.get_Ratio(x_data)            elif data == 'ratio+':                x_data = reader.get_Ratio(x_data)                x_data = reader.add_feature(x_data)            elif data == 'all':                x_data = reader.get_Ratio(x_data,all=True)            else:                x_data = reader.get_Ratio(x_data, all=True)                x_data = reader.add_feature(x_data)            x_train = np.vstack((x_train, x_data))            y_train = np.vstack((y_train, y_data))    print("data :",data, ", minmax :",minmax)    modelTrain(x_train,y_train,save_name,isminmax=minmax,type=data)    for i in range(6,10):        reader = Data_File.Data_File(path, class_name[i], class_bin[i])        if data == 'raw':            x_valid, y_valid = reader.raw_read(issplit=False)        elif data == 'raw+':            x_valid, y_valid = reader.raw_read(issplit=False)            x_valid = reader.add_feature(x_valid)        elif data == 'ratio':            x_valid, y_valid = reader.raw_read(issplit=False)            x_valid = reader.get_Ratio(x_valid)        elif data == 'ratio+':            x_valid, y_valid = reader.raw_read(issplit=False)            x_valid = reader.get_Ratio(x_valid)            x_valid = reader.add_feature(x_valid)        elif data == 'all':            x_valid, y_valid = reader.raw_read(issplit=False)            x_valid = reader.get_Ratio(x_valid, all=True)        else:            x_valid, y_valid = reader.raw_read(issplit=False)            x_valid = reader.get_Ratio(x_valid, all=True)            x_valid = reader.add_feature(x_valid)        modelValid(x_valid, y_valid, load_name, isminmax=minmax,type=data)