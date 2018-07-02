import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../catprophet')

import pandas as pd
import numpy as np
from datetime import datetime as dt
from matplotlib import pyplot as plt
from catprophet import CatProphet


MINUTE_VALUE = 0.001
PROGRESS_CAP_VALUE = 100  # 進捗率上限値：100%
GROWTH = 'logistic'  # default:'linear'


def forecast_progress(csv_path, current_date, end_date=None):

    # プロジェクト進捗データ読込
    #df = pd.read_csv(csv_path, index_col='ds', parse_dates=True)
    df = pd.read_csv(csv_path)
    df['ds'] = pd.to_datetime(df['ds'])
    df['x'] = df['ds'] - df['ds'][0]  # 経過時間の算出
    df.set_index(['ds'], inplace=True)
    if end_date is not None:
        df = df.loc[:end_date]  # ★期間指定★

    # 元データを保存
    df_actual = df.copy()
    df_actual = df_actual.rename(columns={'x':'time', 'y':'actual'})
    print(df_actual.info())

    # 教師データ（途中経過データ）の抽出
    df = df.loc[:current_date]  # 期間指定
    df['y'][df['y']<=0] = MINUTE_VALUE  # 0を微小値に変換（logistic曲線はy>0）
    df.reset_index(inplace=True)  # 'ds'のインデックス指定を解除（カラムになっていないとフィッティングできないため）
    print(df.info())
    print(df.tail())

    # 予測モデルの作成（教師データのフィッティング）
    model = CatProphet(
        growth=GROWTH,
        changepoints=None,
        n_changepoints=25,
        yearly_seasonality=None,  # default:'auto'
        weekly_seasonality='auto',
        daily_seasonality=True,
        holidays=None,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        mcmc_samples=0,
        interval_width=0.80,
        uncertainty_samples=1000,
        )
    df['cap'] = PROGRESS_CAP_VALUE
    model.fit(df)

    # 予測結果用の空データフレームを用意
    forecast_term = (dt.strptime(df.index.max(),'%Y-%m-%d') - dt.strptime(current_date,'%Y-%m-%d'))
    df_future = model.make_future_dataframe(
        periods=forecast_term.days,
        freq='D',
        include_history=True
        )
    df_future['cap'] = PROGRESS_CAP_VALUE

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
