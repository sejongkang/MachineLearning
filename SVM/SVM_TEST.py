import numpy as np
import Data_File
import SVM

if __name__ == '__main__':
    data = 'raw'
    # data = 'raw+'
    # data = 'ratio'
    # data = 'ratio+'
    # data = 'all'
    # data = 'all+'

    data_date = '20190712'
    path = '../Data/acquisition/'+data_date

    normal = 'mm'
    # normal = 'std'
    # normal = None

    class_name = np.asarray(['NO', 'H2S_5','H2S_25','NH3_10','NH3_40','H2S_10_NH3_10','H2S_25_NH3_40','H2S_20','NH3_25','H2S_15_NH3_25'])
    class_bin = [[0, 0, 0, 0],[0, 1, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 1, 0],[0, 1, 1, 0],[0, 1, 1, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 1, 1, 0]]

    for i in range(len(class_name)):
        reader = Data_File.Data_File(path, class_name[i], class_bin[i])
        if i == 0:
            if data == 'raw':
                x_data, y_data = reader.raw_read(issplit=False)
                normal = None
            elif data == 'raw+':
                x_data, y_data = reader.raw_read(issplit=False)
                x_data = reader.add_feature(x_data)
            elif data == 'ratio':
                x_data, y_data = reader.raw_read(issplit=False)
                x_data = reader.get_Ratio(x_data)
            elif data == 'ratio+':
                x_data, y_data = reader.raw_read(issplit=False)
                x_data = reader.get_Ratio(x_data)
                x_data = reader.add_feature(x_data)
            elif data == 'all':
                x_data, y_data = reader.raw_read(issplit=False)
                x_data = reader.get_Ratio(x_data,all=True)
            else:
                x_data, y_data = reader.raw_read(issplit=False)
                x_data = reader.get_Ratio(x_data, all=True)
                x_data = reader.add_feature(x_data)

            x_train = x_data
            y_train = y_data
        else:
            if data == 'raw':
                x_data, y_data = reader.raw_read(issplit=False)
            elif data == 'raw+':
                x_data, y_data = reader.raw_read(issplit=False)
                x_data = reader.add_feature(x_data)
            elif data == 'ratio':
                x_data = reader.get_Ratio(x_data)
            elif data == 'ratio+':
                x_data = reader.get_Ratio(x_data)
                x_data = reader.add_feature(x_data)
            elif data == 'all':
                x_data = reader.get_Ratio(x_data,all=True)
            else:
                x_data = reader.get_Ratio(x_data, all=True)
                x_data = reader.add_feature(x_data)

            x_train = np.vstack((x_train, x_data))
            y_train = np.vstack((y_train, y_data))

    print("data :",data, ", normal :",normal)

    svm = SVM.SVM()

    svm.train(x_train, y_train, multiLabel=True, normal=normal, path='SVM/Model/')

    for i in range(7,10):
        reader = Data_File.Data_File(path, class_name[i], class_bin[i])
        if data == 'raw':
            x_valid, y_valid = reader.raw_read(issplit=False)
        elif data == 'raw+':
            x_valid, y_valid = reader.raw_read(issplit=False)
            x_valid = reader.add_feature(x_valid)
        elif data == 'ratio':
            x_valid, y_valid = reader.raw_read(issplit=False)
            x_valid = reader.get_Ratio(x_valid)
        elif data == 'ratio+':
            x_valid, y_valid = reader.raw_read(issplit=False)
            x_valid = reader.get_Ratio(x_valid)
            x_valid = reader.add_feature(x_valid)
        elif data == 'all':
            x_valid, y_valid = reader.raw_read(issplit=False)
            x_valid = reader.get_Ratio(x_valid, all=True)
        else:
            x_valid, y_valid = reader.raw_read(issplit=False)
            x_valid = reader.get_Ratio(x_valid, all=True)
            x_valid = reader.add_feature(x_valid)
        print(svm.classify(x_valid, normal=normal, path='SVM/Model/'))

    # point = svm.score(x_valid, y_valid, multiLabel=True, normal=None, path='SVM/Model/')
    # percent = str(point * 100)+'%'
    # print("Score : ",percent)

    collect = 0

    # for index, d in enumerate(x_valid):
    #     match = y_valid[index] == svm.classify([d], path='SVM/Model/')[0]
    #     if match.all():
    #         collect += 1
    #     # if y_valid[index] == svm.classify([d], path='SVM/Model/')[0]:
    #     #     collect += 1
    #     else:
    #         print("분류 실패")
    #         print("index :", index, " 클래스 :", y_valid[index], " 분류된 클래스 :", svm.classify([d],path='SVM/Model/')[0])
    #
    # print("\n분류 결과 :", collect / len(x_valid) * 100, "%")