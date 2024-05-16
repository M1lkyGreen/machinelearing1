import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
warnings.filterwarnings("ignore")

# 读取数据集
df = pd.read_csv(r'C:\Users\94506\Desktop\代码\美元股票价格和信息数据集.csv')

# 检测缺失值
# missing_values = df.isnull().sum()
# print("缺失值统计：\n", missing_values)

# 填充缺失值
df.fillna(method='ffill', inplace=True)

X = df[['Volume']]  # 使用成交量作为特征
y = df['Close']  # 收盘价作为目标变量

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.model_selection import RandomizedSearchCV

param_space = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 实例化随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100,random_state=42)

# 实例化 RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_space, n_iter=10, cv=5, scoring='r2', random_state=42)

# 在训练数据上执行随机搜索
random_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数：", random_search.best_params_)

# 输出最佳交叉验证得分
print("最佳交叉验证得分（R²）：", random_search.best_score_)

# 输出测试集得分
test_score = random_search.score(X_test, y_test)
print("测试集得分（R²）：", test_score)

