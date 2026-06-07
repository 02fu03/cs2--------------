#!/usr/bin/env python3
"""
evaluate_predictions_interactive.py
交互版：对比未来预测结果与真实历史数据，计算 MAE/RMSE/MAPE，生成对比表与交互式可视化
"""
import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.io as pio


def find_forecast_file(data_path, horizon=3):
    """根据原始数据自动匹配对应的预测文件"""
    base = os.path.splitext(os.path.basename(data_path))[0]
    pattern = f"forecast_{base}_h{horizon}*.csv"
    matches = glob.glob(pattern)
    if matches:
        # 优先选择调优版预测文件
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
    """生成交互式对比图表"""
    fig = go.Figure()
    # 真实价格曲线（改回 yyyp_sell_price）
    fig.add_trace(go.Scatter(
        x=df_actual_full['date'],
        y=df_actual_full['yyyp_sell_price'],
        mode='lines+markers',
        name='实际价格',
        line=dict(color='#1f77b4')
    ))
    # 模型预测曲线
    fig.add_trace(go.Scatter(
        x=df_forecast['date'],
        y=df_forecast['pred_price'],
        mode='lines+markers',
        name='预测价格',
        marker=dict(symbol='diamond', size=10),
        line=dict(color='#ff7f0e')
    ))
    # 绝对误差柱状图
    if 'abs_error' in df_forecast.columns:
        fig.add_trace(go.Bar(
            x=df_forecast['date'],
            y=df_forecast['abs_error'],
            name='绝对误差',
            marker_color='rgba(200,30,30,0.6)',
            yaxis='y2'
        ))
    # 标注预测区间
    if not df_forecast.empty:
        start = df_forecast['date'].min()
        end = df_forecast['date'].max()
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor='LightSalmon', opacity=0.08,
            layer='below', line_width=0
        )
    fig.update_layout(
        title='模型预测价格 vs 实际真实价格',
        xaxis_title='date',
        yaxis_title='price',
        hovermode='x unified'
    )
    fig.update_layout(yaxis2=dict(title='绝对误差', overlaying='y', side='right'))
    pio.write_html(fig, out_html, include_plotlyjs='cdn', auto_open=False)
    return out_html


def input_int(prompt, default, min_val=None, max_val=None):
    """通用整数输入工具（和主模型脚本保持一致）"""
    while True:
        res = input(f"{prompt} (默认 {default}): ").strip()
        if not res:
            return default
        try:
            num = int(res)
            if min_val is not None and num < min_val:
                print(f"请输入不小于 {min_val} 的整数")
                continue
            if max_val is not None and num > max_val:
                print(f"请输入不大于 {max_val} 的整数")
                continue
            return num
        except ValueError:
            print("输入格式错误，请输入整数")


def main():
    print("=" * 60)
    print("        模型预测结果评估工具（交互版）")
    print("功能：将【未来预测数据】与【真实数据】比对，计算误差与评估指标")
    print("=" * 60)
    print("1. 请输入【真实历史数据CSV】路径（拖拽文件到窗口即可）")
    data_path = input("真实数据 CSV 路径：").strip()

    # 自动去除拖拽路径自带的单/双引号
    if (data_path.startswith("'") and data_path.endswith("'")) or (data_path.startswith('"') and data_path.endswith('"')):
        data_path = data_path[1:-1]

    # 校验真实数据文件
    if not os.path.isfile(data_path):
        print(f"错误：文件不存在 -> {data_path}")
        return
    if not data_path.lower().endswith(".csv"):
        print("错误：请选择 .csv 格式文件")
        return
    print(f"\n✅ 已加载真实数据：{os.path.basename(data_path)}")

    # 输入预测天数
    horizon = input_int("预测天数(horizon)", default=3, min_val=1, max_val=7)

    # 自动查找预测文件
    forecast_path = find_forecast_file(data_path, horizon=horizon)
    if forecast_path is None:
        print("\n⚠️ 未自动匹配到预测文件，请手动输入【模型预测结果CSV】路径")
        forecast_path = input("预测结果 CSV 路径：").strip()
        # 去除引号
        if (forecast_path.startswith("'") and forecast_path.endswith("'")) or (forecast_path.startswith('"') and forecast_path.endswith('"')):
            forecast_path = forecast_path[1:-1]
        if not os.path.isfile(forecast_path) or not forecast_path.lower().endswith(".csv"):
            print("错误：预测文件不存在或格式不正确")
            return

    print(f"✅ 已加载模型预测数据：{os.path.basename(forecast_path)}")
    print("\n" + "-" * 40 + " 开始比对评估 " + "-" * 40)

    # 加载两份数据
    df_real = load_data(data_path)
    df_pred = load_forecast(forecast_path)

    # 前置列名校验（改回 yyyp_sell_price）
    required_real_cols = ["date", "yyyp_sell_price", "yyyp_sell_num"]
    required_pred_cols = ["date", "pred_price"]
    for col in required_real_cols:
        if col not in df_real.columns:
            print(f"❌ 真实数据缺少必要字段：{col}")
            return
    for col in required_pred_cols:
        if col not in df_pred.columns:
            print(f"❌ 预测数据缺少必要字段：{col}")
            return

    # 按日期内连接：只保留「预测日期同时存在真实值」的样本（标准评估）
    df_merge = pd.merge(
        df_pred,
        df_real[["date", "yyyp_sell_price", "yyyp_sell_num"]],
        on="date",
        how="inner"
    )

    if df_merge.empty:
        print("❌ 预测日期与真实数据无重合日期，无法评估，程序终止")
        return

    print(f"📊 找到可比对样本数量：{len(df_merge)} 条")

    # 计算单日误差（改回 yyyp_sell_price）
    df_merge["abs_error"] = (df_merge["yyyp_sell_price"] - df_merge["pred_price"]).abs()
    with np.errstate(divide='ignore', invalid='ignore'):
        df_merge["pct_error"] = (df_merge["yyyp_sell_price"] - df_merge["pred_price"]) / (df_merge["yyyp_sell_price"] + 1e-9) * 100

    # 打印逐日对比表（改回 yyyp_sell_price）
    print("\n📋 逐日对比表（日期 | 实际价格 | 预测价格 | 绝对误差 | 百分比误差）：")
    print(df_merge[["date", "yyyp_sell_price", "pred_price", "abs_error", "pct_error"]].to_string(
        index=False,
        formatters={
            "yyyp_sell_price": "{:,.2f}".format,
            "pred_price": "{:,.2f}".format,
            "abs_error": "{:,.2f}".format,
            "pct_error": lambda x: f"{x:.2f}%"
        }
    ))

    # 计算整体评估指标（改回 yyyp_sell_price）
    actual_vals = df_merge["yyyp_sell_price"].values
    pred_vals = df_merge["pred_price"].values
    metrics = compute_metrics(actual_vals, pred_vals)

    print("\n📈 模型综合评估指标：")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")

    # 保存对比结果 CSV
    out_compare = os.path.splitext(forecast_path)[0] + "_compare_result.csv"
    df_merge.to_csv(out_compare, index=False)
    print(f"\n✅ 对比结果表已保存：{os.path.basename(out_compare)}")

    # 生成交互式可视化图表
    out_html = os.path.splitext(forecast_path)[0] + "_compare_chart.html"
    df_fore_plot = df_pred.merge(df_merge[["date", "abs_error"]], on="date", how="left")
    make_plot(df_real, df_fore_plot, out_html)
    print(f"✅ 交互式对比图表已保存：{os.path.basename(out_html)}")

    print("\n" + "=" * 40)
    print("🎉 模型预测评估全部完成！")
    print("=" * 40)


if __name__ == '__main__':
    main()