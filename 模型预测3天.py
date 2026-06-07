#!/usr/bin/env python3
"""
predict_AK47_3days.py
调用模型脚本生成未来 3 天预测（基于现有模型.py），并展示预测结果与简单趋势判断。

用法:
    python predict_AK47_3days.py --file "AK47抽象派1337(崭新出厂)_悠悠有品_近3个月数据.csv"

输出:
 - 在工作目录生成 forecast_<base>_h3.csv（如果模型.py 正常运行）
 - 打印预测表格；可视化以交互式 HTML 为主
"""

import argparse
import os
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt


def run_model(csv_path, horizon=3):
    # 调用 模型.py 生成预测
    py = sys.executable
    # 优先尝试调用同目录下的 模型.py；若不存在则调用 模型7天.py
    dirpath = os.path.dirname(__file__)
    candidates = ['模型.py', '模型7天.py']
    script = None
    for c in candidates:
        p = os.path.join(dirpath, c)
        if os.path.exists(p):
            script = p
            break
    if script is None:
        script = os.path.join(dirpath, '模型.py')

    cmd = [py, script, '--file', csv_path, '--horizon', str(horizon)]
    print('运行命令:', ' '.join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print('运行模型脚本失败，错误信息：')
        print(res.stderr)
        raise SystemExit(1)


def load_forecast(csv_path, horizon=3):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    forecast_name = f'forecast_{base}_h{horizon}.csv'
    if not os.path.exists(forecast_name):
        # 有时调参版本会生成 _h{horizon}_tuned.csv
        alt = f'forecast_{base}_h{horizon}_tuned.csv'
        if os.path.exists(alt):
            forecast_name = alt
        else:
            raise FileNotFoundError(f'未找到预测文件: {forecast_name} 或 {alt}')
    df = pd.read_csv(forecast_name, parse_dates=['date'])
    return df, forecast_name


def plot_and_report(df, out_png):
    # 简单折线图：预测价格与预测在售数量（如果存在）
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(df['date'], df['pred_price'], marker='o', color='tab:blue', label='pred_price')
    ax1.set_ylabel('price', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    if 'pred_vol' in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(df['date'], df['pred_vol'], marker='s', color='tab:orange', label='pred_vol')
        ax2.set_ylabel('vol', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title('未来3天预测')
    plt.grid(alpha=0.2)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.close()


def simple_trend(df):
    # 根据首尾价格变化判断趋势
    p0 = df['pred_price'].iloc[0]
    p1 = df['pred_price'].iloc[-1]
    change = (p1 - p0) / (p0 + 1e-9)
    pct = change * 100
    if abs(pct) < 0.5:
        trend = '基本持平'
    elif pct > 0:
        trend = f'上升 ({pct:.2f}% )'
    else:
        trend = f'下降 ({pct:.2f}% )'
    return trend, p0, p1, pct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, required=False, default='AK47抽象派1337(崭新出厂)_悠悠有品_近3个月数据.csv')
    args = parser.parse_args()

    csv_path = args.file
    if not os.path.exists(csv_path):
        print('未找到数据文件:', csv_path)
        raise SystemExit(1)

    # 运行模型脚本生成预测
    run_model(csv_path, horizon=3)

    # 读取预测结果
    df, fname = load_forecast(csv_path, horizon=3)
    print('\n预测文件:', fname)
    print(df)

    # 保存并显示简单图表与趋势描述
    plot_and_report(df, None)
    trend, p0, p1, pct = simple_trend(df)
    print(f"\n简单趋势判断: {trend} (从 {p0:.2f} 到 {p1:.2f}, 变化 {pct:.2f}%)")
    print('请打开生成的交互式 HTML 查看可视化。')

if __name__ == '__main__':
    main()
