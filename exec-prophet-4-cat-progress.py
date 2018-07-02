import pandas as pd
import numpy as np
from datetime import datetime as dt
from matplotlib import pyplot as plt

#from fbprophet import Prophet
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../catprophet')
from catprophet import CatProphet

# 入力データ
# END_DATE = '2017-02-20'
# CURRENT_DATE = '2017-01-20'
# CSV_FILE = 'cat-data/progress-pj59-pc138.csv'

END_DATE = '2017-07-20'
CURRENT_DATE = '2017-07-06'
CSV_FILE = 'cat-data/progress-pj59-pc205.csv'

# プロジェクト進捗データ読込
#df = pd.read_csv(CSV_FILE, index_col='ds', parse_dates=True)
df = pd.read_csv(CSV_FILE)
df['ds'] = pd.to_datetime(df['ds'])
df['x'] = df['ds'] - df['ds'][0]  # 経過時間の算出
df.set_index(['ds'], inplace=True)
df = df.loc[:END_DATE]  # ★期間指定★

# 元データを保存
df_actual = df.copy()
df_actual = df_actual.rename(columns={'x':'time', 'y':'actual'})
print(df_actual.info())

# 教師データ（途中経過データ）の抽出
df = df.loc[:CURRENT_DATE]  # ★期間指定★
df['y'][df['y']==0] = 0.001  # 0を微小値に変換（logistic曲線はy>0）
df.reset_index(inplace=True)  # 'ds'のインデックス指定を解除（カラムになっていないとフィッティングできないため）
print(df.info())
print(df.tail())

# 予測モデルの作成（教師データのフィッティング）
model = CatProphet(
    growth='logistic',  # default:'linear'
    changepoints=None,
    n_changepoints=25,
    yearly_seasonality=None,  # default:'auto'
    weekly_seasonality=True,
    holidays=None,
    seasonality_prior_scale=10.0,
    holidays_prior_scale=10.0,
    changepoint_prior_scale=0.05,
    mcmc_samples=0,
    interval_width=0.80,
    uncertainty_samples=1000,
    daily_seasonality=True,
    )
df['cap'] = 100  # 上限値(capacity): 100%
model.fit(df)

# 予測結果用の空データフレームを用意
forecast_term = (dt.strptime(END_DATE,'%Y-%m-%d') - dt.strptime(CURRENT_DATE,'%Y-%m-%d'))
df_future = model.make_future_dataframe(
    periods=forecast_term.days,
    freq='D',
    include_history=True
    )
df_future['cap'] = 100  # 上限値(capacity): 100%

# 予測
df_forecast = model.predict(df_future)
print(df_forecast.info())
model.plot(df_forecast)
#model.plot_components(df_forecast)

# 実際の進捗とマージ
df_merge = df_forecast.set_index(['ds'])
df_merge = df_merge.join(df_actual, how='outer')
df_merge = df_merge.rename(columns={'yhat':'forecast'})
print(df_merge.info())

# 描画
df_merge[['forecast','actual']].plot(legend=True)
plt.show()
