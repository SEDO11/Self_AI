import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import SGDRegressor as sgd
from sklearn.model_selection import train_test_split as tts

dataset = pd.read_csv('C:/Users/sehoon/Desktop/git/Self_AI/ScikitLearn/LinearRegressionData.csv')

# 데이터 추출 iloc[row, col]
X = dataset.iloc[:, :-1].values #hour
y = dataset.iloc[:, -1].values #score

# 훈련 세트, 테스트 세트 4개로 데이터를 나눔
# 훈련 세트를 80% 테스트 세트를 20% 로 함
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=0)

sr = sgd(max_iter=10, eta0=0.001, random_state=0)
sr.fit(X_train, y_train)