# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
#
# # 读取数据集
# df = pd.read_csv(r'C:\Users\94506\Desktop\代码\美元股票价格和信息数据集.csv')
#
# # 准备特征和目标变量
# X = df[['Volume']]  # 使用成交量作为特征
# y = df['Close']  # 收盘价作为目标变量
#
# # 拆分数据集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # 实例化随机森林回归模型
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
#
# # 训练模型
# rf_model.fit(X_train, y_train)
#
# # 模型评估
# y_train_pred = rf_model.predict(X_train)
# mse_train = mean_squared_error(y_train, y_train_pred)
# print("训练集均方误差（MSE）：", mse_train)
#
# y_test_pred = rf_model.predict(X_test)
# mse_test = mean_squared_error(y_test, y_test_pred)
# print("测试集均方误差（MSE）：", mse_test)
# # import pandas as pd
# #
# # # 读取数据集
# # df = pd.read_csv(r'C:\Users\94506\Desktop\代码\美元股票价格和信息数据集.csv')
# #
# # # 将时间戳列转换为 datetime 类型
# # df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
# #
# # # 找出超出范围的分钟值
# # invalid_minutes = df[df['Date'].dt.minute >= 60]['Date']
# # print("超出范围的分钟值：\n", invalid_minutes)
#
# import pandas as pd
# # 读取数据集
# df = pd.read_csv(r'C:\Users\94506\Desktop\代码\美元股票价格和信息数据集.csv')
# # 将时间戳列转换为 datetime 类型
# df['Date'] = pd.to_datetime(df['Date'], format='%y-%m-%d %H:%M', errors='coerce')
# # 输出转换后的时间戳列
# print("转换后的时间戳：\n", df['Date'].head())
# # 检查是否还有转换错误的时间戳
# invalid_timestamps = df[df['Date'].isna()]['Date']
# print("转换错误的时间戳：\n", invalid_timestamps)
# # 删除转换错误的时间戳所在的行
# df_cleaned = df.dropna(subset=['Date'])
# # 输出删除转换错误的时间戳后的数据
# print(df_cleaned.head())
#
#
