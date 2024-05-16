import pandas as pd
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QTableWidgetItem

# 在QApplication之前先实例化
uiLoader = QUiLoader()


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
def predict():
    # 读取数据集
    df = pd.read_csv(r'E:\ml\xxx\美元股票价格和信息数据集.csv',nrows=100)

    # 检测缺失值并填充
    df.fillna(method='ffill', inplace=True)

    # 将时间戳列转换为 datetime 类型
    df['Date'] = pd.to_datetime(df['Date'], format='%y-%m-%d %H:%M', errors='coerce')

    # 将时间戳列设置为索引
    df.set_index('Date', inplace=True)

    # 重新采样数据并填充缺失值
    df_resampled = df.resample('15T').mean().fillna(method='ffill')

    # 创建滞后特征（lag features）
    for i in range(1, 6):  # 使用过去5个时间点的数据作为特征
        df_resampled[f'Close_Lag_{i}'] = df_resampled['Close'].shift(i)

    # 去除包含 NaN 值的行
    df_resampled.dropna(inplace=True)

    # 准备特征和目标变量
    X = df_resampled[['Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5']]
    y = df_resampled['Close']

    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 实例化随机森林回归模型
    rf_model = RandomForestRegressor(random_state=42)

    # 训练模型
    rf_model.fit(X_train, y_train)

    # 进行预测
    y_pred = rf_model.predict(X_test)

    # # 计算均方误差（MSE）
    # mse = mean_squared_error(y_test, y_pred)
    # print("测试集均方误差（MSE）:", mse)

    # 使用最新的历史数据进行未来15分钟内的价格预测
    latest_data = X_test.iloc[-1].values.reshape(1, -1)
    future_price = rf_model.predict(latest_data)
    s = "未来15分钟内的价格预测:"+str(future_price[0])
    print(s)
    stats.ui.textEdit.setText(s)
class Stats:

    def __init__(self):
        # 再加载界面
        self.ui = uiLoader.load(r'E:\ml\xxx\dollar.ui')
        self.ui.pushButton.clicked.connect(predict)
    # 其它代码 ...

app = QApplication([])
stats = Stats()
stats.ui.show()
app.exec()