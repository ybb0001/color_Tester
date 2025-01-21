import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import joblib
from sklearn.feature_extraction import DictVectorizer
import logging

filename = "{}.log".format(__file__)
fmt = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"

logging.basicConfig(
    level=logging.DEBUG,
    filename=filename,
    filemode="w",
    format=fmt
)

def color_check(a,b,c):
    x1 = [{'R': a, 'G': b, 'B': c}]
    logging.debug("R:%d, G:%d, B:%d", a, b, c)
    # 创建 DictVectorizer 对象
    transfer = DictVectorizer(sparse=False)

    # 将特征转换为数组并获取特征名称
    x = transfer.fit_transform(x1)

    # 加载模型
    estimator = joblib.load("./test_D2.pkl")

    # 预测
    predicted_y = estimator.predict(x)
    logging.debug("predict_Result: %d", predicted_y[0])

    return predicted_y[0]
	
