import errno
import os
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import make_scorer
import numpy as np

def my_score(y_true, y_pred):   # 이진분류일 때, 아닐 때 분리
    a = y_pred.shape # 차원 추출
    if len(a) == 1: # 1차원이면 = 이진분류이면
        y_class = y_pred > 0.5  # 확률 0.5 이상이면 클래스 1
        match = y_class == y_true   # 분류 결과가 올바른 인덱스 추출
        score = np.sum(match)   # 분류가 올바른 갯수 총합을 점수로
    else :
        y_class = np.argmax(y_pred, 1)  # 확률 가장 높은 클래스 추출
        match = y_class == y_true   # 분류 결과가 올바른 인덱스 추출
        score = np.sum(np.max(y_pred[match,:],1))   # 분류가 올바른 인덱스의 가장 높은 확률들을 총합해 점수로

    return score

class SVM:

    """ SVM """

    def train(self, x_data_o, y_data_o, multiLabel=False, normal=None, grid=True, path=''):

        """
        SVM에 해당 데이터 집합을 바탕으로 분류를 수행할 수 있도록 한다 (학습과 유사)
        :param x_data_o: 2D list , Training Feature
        :param y_data: 1D list(Multiclass) or 2D list(Multilabel) , Training Class
        :param multiLabel: Boolean , Multilabel Option
        :param normal : None or mm(MinMax) or std(StandardScaler) , Normalization Option
        :param grid : Boolean , GridSearch Option
        :param path: str(path/), Save Path
        :return: None
        """

        user_scorer = make_scorer(my_score)

        if multiLabel :
            y_data = []
            for i in range(len(y_data_o)):
                self.tmp = []
                self.tmp.append(y_data_o[i])
                y_data.append(self.tmp)
            y_data = MultiLabelBinarizer().fit_transform(y_data)
        else :
            y_data = y_data_o

        if normal == 'mm':
            self.min=np.min(x_data_o,0)
            self.max=np.max(x_data_o,0)
            x_data=(x_data_o-self.min)/(self.max-self.min)
            joblib.dump(np.asarray([self.min, self.max]), path+'SVM_mm.pkl')
        elif normal == 'std' :
            self.scaler = preprocessing.StandardScaler().fit(x_data_o)
            x_data = self.scaler.transform(x_data_o)
            joblib.dump(self.scaler, path+'SVM_std.pkl')
        elif normal == 'normal':
            x_data = preprocessing.normalize(x_data_o)
        else :
            x_data = x_data_o

        self.svc = OneVsRestClassifier(SVC(kernel='rbf', probability=True, gamma='auto'))

        if grid:
            c_range = np.arange(-5, 16, 2, dtype='float')
            c_array = np.power(2, c_range)
            sigma_range = np.arange(-2, 8, 1, dtype='float')
            sigma_array = np.power(2, sigma_range)
            g_array = np.power(np.multiply(np.power(sigma_array, 2), 2), -1)
            # self.svc = GridSearchCV(self.svc, {'estimator__C': c_array, 'estimator__gamma': g_array}, scoring='accuracy')
            self.svc = GridSearchCV(self.svc, {'estimator__C': c_array, 'estimator__gamma': g_array},
                                    scoring=user_scorer)

        self.svc.fit(x_data, y_data)

        if grid:
            print(self.svc.best_params_)

        joblib.dump(self.svc, path)       #모델 세이브

    # ==========================================================

    def classify(self, x_data_o, normal=None, path=''):
        """
        해당 데이터 분류를 수행함

        :param x_data_o: list, 분류할 데이터
        :param normal : None or mm(MinMax) or std(StandardScaler) , Normalization Option
        :param path: str(path/), Load Path
        :return: list, 분류 결과
        """

        if normal == 'mm':
            mm = joblib.load(path+'SVM_mm.pkl')
            self.min=mm[0]
            self.max=mm[1]
            x_data=(x_data_o-self.min)/(self.max-self.min)
        elif normal == 'std' :
            self.scaler = joblib.load(path+'SVM_std.pkl')
            x_data = self.scaler.transform(x_data_o)
        else:
            x_data = x_data_o

        self.svc = joblib.load(path+'SVM_model.pkl')
        results = self.svc.predict(x_data)

        return results

    # ==========================================================

    def score(self, x_data_o, y_data_o, multiLabel = False, normal=None ,path=''):
        """
        해당 데이터를 분류한 후, 그에 따른 정확도를 계산하여 반환한다.

        :param x_data_o: list, 검증 데이터
        :param y_data: list, 검증 데이터
        :param normal : None or mm(MinMax) or std(StandardScaler) , Normalization Option
        :param path: str(path/), Load Path
        :return: Float, 분류 정확도
        """

        if multiLabel :
            y_data=y_data_o
        else :
            y_data = []
            for i in range(len(y_data_o)):
                self.tmp = []
                self.tmp.append(y_data_o[i])
                y_data.append(self.tmp)
                y_data = MultiLabelBinarizer().fit_transform(y_data)

        if normal == 'mm':
            mm = joblib.load(path+'SVM_mm.pkl')
            self.min=mm[0]
            self.max=mm[1]
            x_data=(x_data_o-self.min)/(self.max-self.min)
        elif normal == 'std' :
            self.scaler = joblib.load(path+'SVM_std.pkl')
            x_data = self.scaler.transform(x_data_o)
        else:
            x_data = x_data_o

        results = self.svc.score(x_data, y_data)
        return results