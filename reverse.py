# %%
# 本反转策略采用量价数据，通过样本筛选、样本标记、模型训练得到有效的反转信号预测模型
import quool
import datetime
import numpy as np
import pandas as pd
import akshare as ak
import seaborn as sns
import lightgbm as lgb
import mplfinance as mpf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix

# %%
# 首先，获取上证指数[000001.XSHG]2024年1月的成分股信息
# TODO: 这里的股票池是否可以切换，换成：
#   1. 沪深300[000300.XSHG]
#   2. 上证50[?]
#   3. 中证100[?]
#   4. 创业板[399101.XSHE]
#   5. 中小板[399102.XHSE]
idxwgts = quool.PanelTable("/home/data/index-weights")
weights = idxwgts.read("000001.XSHG", start="20240102", 
    stop="20240102").dropna().droplevel(1).squeeze()

# %%
# 根据成分股的代码，获取出最近三个月的分钟线
# TODO: 这里的时间周期长一点，或许模型有更多的可用样本，效果会更好？
qtm = quool.PanelTable("/home/data/quotes-min")
df = qtm.read("open, high, low, close, volume", code=weights.index, start="20231227")

# %%
# 数据预处理部份
# 本质上，这个策略是在用日内开盘的数据预测N天的收益率，因此，仍然是一个日频率的择时策略
# 1. 合成日线
logic = {"open": "first", "close": "last", "high": "max", "low": "min", "volume": "sum"}
daily = df.groupby(level=0).apply(lambda x: x.groupby(x.droplevel(0).index.date).agg(logic))
daily.index = pd.MultiIndex.from_arrays([
    daily.index.get_level_values(0), pd.to_datetime(daily.index.get_level_values(1))
], names=["order_book_id", "datetime"])
# 2. 合成每日开盘1hK线
# TODO: 半小时线的效果怎么样呢？注意这里不能再拉长时间，我们用的就是开盘时间交易量的信息
def _select_opening(x):
    x = x.droplevel(0)
    x = x.between_time("9:30", "10:30")
    x = x.groupby(x.index.date).agg(logic)
    x.index = pd.to_datetime(x.index)
    x.index.name = "datetime"
    return x
    
opening = df.groupby(level=0).apply(_select_opening)

# %%
# 加入每天实时的数据
# WARNING: 如果你的数据已经是最新的，不要运行这个单元格
def _add_suffix(code):
    if code.startswith('6'):
        return code + ".XSHG"
    elif code.startswith('3') or code.startswith('0'):
        return code + ".XSHE"
    else:
        return np.nan

def add_spot_data(opening):
    spot = ak.stock_zh_a_spot_em()
    spot = spot[["代码", "最高", "最低", "今开", "最新价", "成交量"]]
    spot["代码"] = spot["代码"].map(_add_suffix)
    spot = spot.dropna(axis=0)
    spot.columns = ["code", "high", "low", "open", "close", "volume"]
    spot.index = pd.MultiIndex.from_arrays([
        spot["code"], pd.to_datetime([datetime.datetime.today().strftime("%Y-%m-%d")] * len(spot))
    ], names=["order_book_id", "datetime"])
    spot = spot.drop(["code"], axis=1)
    opening = pd.concat([opening, spot.loc[spot.index.get_level_values(0).isin(opening.index.get_level_values(0))]]).sort_index()
    return opening

opening = add_spot_data(opening)

# %%
# 样本筛选
# TODO: 如下的参数都可以进行调整，属于模型的超参数部份
lookback = 10 # 回看多少个交易日计算均值、标准差等
dev = 0.5 # 成交量需要偏离滚动均值多少倍倍才算放量

# 放量标准
volma = opening["volume"].groupby(level=0).rolling(lookback).mean().droplevel(0)
volstd = opening["volume"].groupby(level=0).rolling(lookback).std().droplevel(0)
volup = opening["volume"] >= volma + dev * volstd

# 下跌趋势判断
# TODO: 下跌趋势还可以有别的判断方法，
#   例如连续在lookback日均线下方N天确认下跌趋势、
#   通过adf单位根检验，当显著不平稳，且当前低价小于期初低价时确认下跌趋势...
llow = opening["low"].groupby(level=0).rolling(lookback).min().droplevel(0)
dntrd = opening["low"] <= llow

# 样本筛选
sample = opening[volup & dntrd]
sample

# %%
# 其他指标的计算
# TODO: 思考哪些指标还可能预测未来的收益情况
shadow = 1 - (opening["close"] - opening["open"]).abs() / (opening["high"] - opening["low"])
opmom = opening.groupby(level=0).apply(lambda x: 1 / x["high"].shift(periods=range(1, lookback + 1)).div(x["low"], axis=0)).droplevel(0)
opmom.columns = [f"opmom_{i}" for i in range(1, lookback + 1)]

# %%
# 现在要给每一个筛选出来的样本打上label，我们把label定义成可获利空间
# 可获利空间是买入后设置跟踪止损，当达到最大持有期或跟踪止损线后平仓的收益率
# TODO: 这里的打label方式也可以改改，用简单的持有一定时间试试
#   当然，还要测试不同崔大持有时期、以及不同跟踪止损线
stoploss = 0.8
holding = 20
lookback = 10

# WARNING: 这里必须要先把sample过滤掉最后holding天的，因为打不出label
dt = daily.index.get_level_values(1).unique()[-holding]
unlabel_sample = sample.loc[sample.index.get_level_values(1) > dt]
sample = sample.loc[sample.index.get_level_values(1) <= dt]

def _get_label(x, sample, stoploss, holding):
    code = x.index.get_level_values(0)[0]
    x = x.droplevel(0)
    if not code in sample.index.get_level_values(0):
        return pd.Series(np.nan)

    subsample = sample.loc[code]
    start_idx = x.index.get_indexer_for(subsample.index)
    stop_idx = start_idx + holding
    
    labs = pd.Series()
    for start, stop in zip(start_idx, stop_idx):
        hdp = x.iloc[start:stop]
        stopped = hdp["high"].cummax() * stoploss >= hdp["low"]
        if stopped.any():
            stopday = stopped[stopped].index[0]
            lab = hdp["high"].cummax().loc[stopday] * stoploss / hdp["open"].iloc[1] - 1
        else:
            lab = hdp["close"].iloc[-1] / hdp["open"].iloc[1] - 1
        labs.loc[x.index[start]] = lab

    return labs

def _get_feature(x, sample, lookback):
    code = x.index.get_level_values(0)[0]
    x = x.droplevel(0)
    if not code in sample.index.get_level_values(0):
        return pd.DataFrame(np.full((16 * lookback + 2, 1), np.nan)).T

    subsample = sample.loc[code]
    start_idx = x.index.get_indexer_for(subsample.index)
    prestart_idx = start_idx - lookback + 1
    
    feats = []
    for prestart, start in zip(prestart_idx, start_idx):
        feat = pd.Series(np.concatenate([
            x.iloc[prestart:start + 1].values.ravel(),
            np.nan_to_num(opmom.iloc[prestart:start + 1].values.ravel()),
            [volma.loc[code].iloc[start]], 
            [volstd.loc[code].iloc[start]], 
            shadow.loc[code].iloc[prestart:start + 1]
        ]), name=x.index[start])
        feats.append(feat)

    return pd.concat(feats, axis=1).T

features = daily.groupby(level=0).apply(
    _get_feature,
    sample=sample, 
    lookback=lookback,
)
label = daily.groupby(level=0).apply(
    _get_label,
    sample=sample,
    stoploss=stoploss,
    holding=holding,
)
features = features.dropna(axis=0, how='all').add_prefix("feature")
label = label.loc[features.index]

# %%
# 模型训练
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test, reference=train_data)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'early_stopping_rounds': 10,
    'verbose': 1,
}

gbm = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, test_data],
)

y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)

mse = mean_squared_error(y_test, y_pred)
print(f"测试集误差：{mse}")
print(f"Feature重要性: {gbm.feature_importance()}")

# %%
# 将模型由回归模型预测结果转化为分类结果
# 这里的转化阈值定义为正训练集样本预测值均值
pos_train_pred = gbm.predict(x_train[y_train > 0], num_iteration=gbm.best_iteration)
thresh = max(pos_train_pred.mean(), 0)
test_class = y_test.copy()
pred_class = y_pred.copy()
test_class[test_class > thresh] = 1
test_class[test_class <= thresh] = 0
pred_class[pred_class > thresh] = 1
pred_class[pred_class <= thresh] = 0

# %%
# 绘制混淆矩阵
cm = confusion_matrix(test_class, pred_class)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# 计算性能指标
pos_pred_label = y_pred[pred_class == 1]
print(f"胜率：{cm[1, 1] / cm[:, 1].sum()}")
print(f"赔率: {pos_pred_label.mean() + 1}")

# %%
# 可视化部份
index = x_test[pred_class == 1].index[1]
test_label = y_test.loc[index]
subsample = sample.loc[index[0]].loc[[index[1]]]
subdaily = daily.loc[index[0]]
buy_price = subdaily.loc[subdaily.index[subdaily.index.get_indexer_for([subsample.index[0]]) + 1][0], "open"]
addplot = [mpf.make_addplot(subsample["close"].reindex(subdaily.index), type='scatter', marker="x", markersize=1200),
    mpf.make_addplot(pd.Series(np.ones(len(subdaily)) * buy_price * (1 + test_label), index=subdaily.index), linestyle='--', color='red')]
mpf.plot(subdaily, type='candle', volume=True, figsize=(20, 10), addplot=addplot)

# %%
# 最后，用训练好的模型对没有label的数据进行计算
unlabel_features = daily.groupby(level=0).apply(_get_feature, sample=unlabel_sample, lookback=lookback)
unlabel_features = unlabel_features.dropna(axis=0).add_prefix('feature')
unlabel_features.index = pd.MultiIndex.from_arrays([
    unlabel_features.index.get_level_values(0), pd.to_datetime(unlabel_features.index.get_level_values(1))
], names=['order_book_id', 'datetime'])
unlabel_pred = gbm.predict(unlabel_features, num_iteration=gbm.best_iteration)
unlabel_pred = pd.Series(unlabel_pred, index=unlabel_features.index)
unlabel_class = unlabel_pred[unlabel_pred > thresh]
signals = unlabel_class.loc[unlabel_class.index.get_level_values(1) == datetime.datetime.today().strftime("%Y-%m-%d")]
