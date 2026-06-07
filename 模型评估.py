#!/usr/bin/env python3
"""
evaluate_predictions.py
比较实际数据与模型输出的预测(近3天),计算 MAE/RMSE/MAPE,并生成交互式 HTML 可视化。

用法示例：
  python evaluate_predictions.py --data "AK47抽象派1337(崭新出厂)_悠悠有品_近3个月数据.csv"
可选参数：
  --forecast 指定预测 CSV 文件路径（默认自动查找 forecast_<base>_h3*.csv)
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.io as pio


def find_forecast_file(data_path, horizon=3):
    base = os.path.splitext(os.path.basename(data_path))[0]
    pattern = f"forecast_{base}_h{horizon}*.csv"
    matches = glob.glob(pattern)
    if matches:
        # prefer tuned if present
        for m in matches:
            if '_tuned' in m:
                return m
        return matches[0]
    return None


def load_data(data_path):
    df = pd.read_csv(data_path, parse_dates=['date'])
    return df


def load_forecast(forecast_path):
    df = pd.read_csv(forecast_path, parse_dates=['date'])
    return df


def compute_metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((actual - pred) / (actual + 1e-9))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE(%)': mape}


def make_plot(df_actual_full, df_forecast, out_html):
    # plot actual price and forecasted price; show abs_error as bar on secondary y-axis
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_actual_full['date'], y=df_actual_full['yyyp_sell_price'], mode='lines+markers', name='实际价格', line=dict(color='#1f77b4')))

    # overlay forecast predictions on same timeline
    fig.add_trace(go.Scatter(x=df_forecast['date'], y=df_forecast['pred_price'], mode='lines+markers', name='预测价格', marker=dict(symbol='diamond', size=10), line=dict(color='#ff7f0e')))

    # add abs_error bar on secondary y axis if present
    if 'abs_error' in df_forecast.columns:
        fig.add_trace(go.Bar(x=df_forecast['date'], y=df_forecast['abs_error'], name='绝对误差', marker_color='rgba(200,30,30,0.6)', yaxis='y2'))

    # highlight forecasted dates as vertical shaded region
    if not df_forecast.empty:
        start = df_forecast['date'].min()
        end = df_forecast['date'].max()
        fig.add_vrect(x0=start, x1=end, fillcolor='LightSalmon', opacity=0.08, layer='below', line_width=0)

    fig.update_layout(title='实际价格 vs 预测价格', xaxis_title='date', yaxis_title='price', hovermode='x unified')
    # secondary y axis for error
    fig.update_layout(yaxis2=dict(title='绝对误差', overlaying='y', side='right'))
    pio.write_html(fig, out_html, include_plotlyjs='cdn', auto_open=False)
    return out_html


def main():
    parser = argparse.ArgumentParser()
    default_data = 'AK47抽象派1337(崭新出厂)_悠悠有品_近3个月数据.csv'
    parser.add_argument('--data', '-d', type=str, required=False, default=None, help='实际数据 CSV 路径')
    parser.add_argument('--forecast', '-f', type=str, default=None, help='预测 CSV 路径（可选）')
    parser.add_argument('--horizon', type=int, default=3, help='预测 horizon 天数（默认 3）')
    args = parser.parse_args()

    data_path = args.data
    # if no data path provided, try default common filename
    if data_path is None:
        if os.path.exists(default_data):
            data_path = default_data
            print(f'未指定 --data，使用默认文件: {data_path}')
        else:
            print('未指定 --data，且默认文件不存在。请使用 --data 指定实际数据文件路径。')
            return

    if not os.path.exists(data_path):
        print('未找到实际数据文件:', data_path)
        return

    forecast_path = args.forecast
    if forecast_path is None:
        forecast_path = find_forecast_file(data_path, horizon=args.horizon)
        if forecast_path is None:
            print('未找到预测文件，尝试指定 --forecast 参数。')
            return

    print('使用实际数据:', data_path)
    print('使用预测文件:', forecast_path)

    df_actual = load_data(data_path)
    df_fore = load_forecast(forecast_path)

    # merge on date to align actual vs pred
    df_merge = pd.merge(df_fore, df_actual[['date','yyyp_sell_price','yyyp_sell_num']], on='date', how='left')
    if df_merge['yyyp_sell_price'].isna().any():
        print('警告：部分预测日期在实际数据中找不到对应实际值，将仅评估有重合的日期。')
        df_merge = df_merge.dropna(subset=['yyyp_sell_price']).reset_index(drop=True)

    # add per-day error columns
    df_merge['abs_error'] = (df_merge['yyyp_sell_price'] - df_merge['pred_price']).abs()
    with np.errstate(divide='ignore', invalid='ignore'):
        df_merge['pct_error'] = (df_merge['yyyp_sell_price'] - df_merge['pred_price']) / (df_merge['yyyp_sell_price'] + 1e-9) * 100

    # print per-day comparison
    print('\n逐日对比 (实际 / 预测 / 绝对误差 / 百分比误差):')
    print(df_merge[['date','yyyp_sell_price','pred_price','abs_error','pct_error']].to_string(index=False, formatters={
        'yyyp_sell_price': '{:,.2f}'.format,
        'pred_price': '{:,.2f}'.format,
        'abs_error': '{:,.2f}'.format,
        'pct_error': lambda x: f"{x:.2f}%"
    }))

    if df_merge.empty:
        print('没有可用于评估的重合日期。')
        return

    metrics = compute_metrics(df_merge['yyyp_sell_price'].values, df_merge['pred_price'].values)
    print('\n评估指标（基于重合日期）:')
    for k,v in metrics.items():
        if isinstance(v, float):
            print(f' - {k}: {v:.4f}')
        else:
            print(f' - {k}: {v}')

    # 保存比较表
    out_compare = os.path.splitext(forecast_path)[0] + '_compare.csv'
    df_merge.to_csv(out_compare, index=False)
    print('已保存对比文件:', out_compare)

    # 可视化
    out_html = os.path.splitext(forecast_path)[0] + '_compare.html'
    make_plot(df_actual, df_fore.merge(df_merge[['date','abs_error']], on='date', how='left'), out_html)
    print('已保存交互式图表:', out_html)

if __name__ == '__main__':
    main()
