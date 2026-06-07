import os
import glob
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
from matplotlib import font_manager as fm
from matplotlib import rcParams
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# 滑动窗口缓存天数
CACHE_WINDOW_DAYS = 365


def save_feature_cache(df, cache_path):
    try:
        df.to_csv(cache_path, index=False)
    except Exception:
        try:
            pq_path = os.path.splitext(cache_path)[0] + '.parquet'
            df.to_parquet(pq_path, index=False)
        except Exception:
            raise


def load_feature_cache(cache_path):
    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path, parse_dates=['date'])
        except Exception:
            pass
    pq_path = os.path.splitext(cache_path)[0] + '.parquet'
    if os.path.exists(pq_path):
        try:
            return pd.read_parquet(pq_path)
        except Exception:
            pass
    return None


def find_csv_files(folder):
    pattern = os.path.join(folder, "*.csv")
    return sorted(glob.glob(pattern))


def load_csv(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def prepare_features(df, lags=7):
    df = df.copy()
    df["price"] = df["yyyp_sell_price"]
    df["vol"] = df["yyyp_sell_num"]
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["price"].shift(lag)
        df[f"vol_lag_{lag}"] = df["vol"].shift(lag)

    df["pct_change_1"] = df["price"].pct_change(1)
    df["rmean_3"] = df["price"].rolling(3).mean()
    df["rmean_7"] = df["price"].rolling(7).mean()
    df["rstd_7"] = df["price"].rolling(7).std()
    df["rmean_14"] = df["price"].rolling(14).mean()
    df["rmean_30"] = df["price"].rolling(30).mean()
    df["rstd_14"] = df["price"].rolling(14).std()
    df["ewm_7"] = df["price"].ewm(span=7, adjust=False).mean()

    df["price_log1p"] = np.log1p(df["price"])
    df["price_diff_1"] = df["price"].diff(1)

    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

    df["target_t+1"] = df["price"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df


def train_and_evaluate(df, test_days=14, n_estimators=200, init_model=None):
    if len(df) <= test_days + 10:
        test_days = max(1, int(len(df) * 0.2))

    train_df = df.iloc[:-test_days]
    test_df = df.iloc[-test_days:]

    feature_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("vol_") or
                    c.startswith("rmean_") or c.startswith("rstd_") or
                    c in ["pct_change_1", "is_weekend"]]

    X_train = train_df[feature_cols].values
    y_train = train_df["target_t+1"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target_t+1"].values

    if init_model is not None:
        model = init_model
    else:
        model = HistGradientBoostingRegressor(max_iter=n_estimators, random_state=42, warm_start=True)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape}
    return model, test_df, y_pred, metrics, feature_cols


def train_and_evaluate_multi(df, horizon=3, test_days=14, n_estimators=200, init_model=None):
    df = df.copy()
    price_cols = []
    vol_cols = []
    for h in range(1, horizon + 1):
        pc = f"target_price_t+{h}"
        vc = f"target_vol_t+{h}"
        df[pc] = df["price"].shift(-h)
        df[vc] = df["vol"].shift(-h)
        price_cols.append(pc)
        vol_cols.append(vc)

    df = df.dropna().reset_index(drop=True)

    if len(df) <= test_days + 10:
        test_days = max(1, int(len(df) * 0.2))

    train_df = df.iloc[:-test_days]
    test_df = df.iloc[-test_days:]

    feature_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("vol_") or
                    c.startswith("rmean_") or c.startswith("rstd_") or
                    c in ["pct_change_1", "is_weekend"]]

    X_train = train_df[feature_cols].values
    y_train = train_df[price_cols + vol_cols].values
    X_test = test_df[feature_cols].values
    y_test = test_df[price_cols + vol_cols].values

    if init_model is not None:
        model = init_model
    else:
        base = HistGradientBoostingRegressor(max_iter=n_estimators, random_state=42, warm_start=True)
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {"price": {}, "vol": {}}
    for h in range(horizon):
        y_true_p = y_test[:, h]
        y_pred_p = y_pred[:, h]
        mae_p = mean_absolute_error(y_true_p, y_pred_p)
        rmse_p = np.sqrt(mean_squared_error(y_true_p, y_pred_p))
        mape_p = np.mean(np.abs((y_true_p - y_pred_p) / (y_true_p + 1e-9))) * 100
        metrics["price"][f"t+{h+1}"] = {"MAE": mae_p, "RMSE": rmse_p, "MAPE(%)": mape_p}

        y_true_v = y_test[:, horizon + h]
        y_pred_v = y_pred[:, horizon + h]
        mae_v = mean_absolute_error(y_true_v, y_pred_v)
        rmse_v = np.sqrt(mean_squared_error(y_true_v, y_pred_v))
        mape_v = np.mean(np.abs((y_true_v - y_pred_v) / (y_true_v + 1e-9))) * 100
        metrics["vol"][f"t+{h+1}"] = {"MAE": mae_v, "RMSE": rmse_v, "MAPE(%)": mape_v}

    return model, test_df, y_pred, metrics, feature_cols, price_cols, vol_cols


def predict_future_from_last(model, df, feature_cols, horizon=3):
    last = df.copy().iloc[-1:]
    X = last[feature_cols].values
    yhat = model.predict(X)[0]
    prices = yhat[:horizon]
    vols = yhat[horizon: horizon * 2]
    last_date = df["date"].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
    out = pd.DataFrame({"date": future_dates, "pred_price": prices, "pred_vol": vols})
    return out


def perform_time_series_tuning(df, horizon=3, n_splits=5, n_iter=20, param_dist=None, n_jobs=-1, random_state=42):
    df = df.copy()
    price_cols = []
    vol_cols = []
    for h in range(1, horizon + 1):
        pc = f"target_price_t+{h}"
        vc = f"target_vol_t+{h}"
        df[pc] = df["price"].shift(-h)
        df[vc] = df["vol"].shift(-h)
        price_cols.append(pc)
        vol_cols.append(vc)

    df = df.dropna().reset_index(drop=True)

    feature_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("vol_") or
                    c.startswith("rmean_") or c.startswith("rstd_") or
                    c in ["pct_change_1", "is_weekend"]]

    X = df[feature_cols].values
    y = df[price_cols + vol_cols].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    base_est = HistGradientBoostingRegressor(random_state=random_state, warm_start=True)
    search_est = MultiOutputRegressor(base_est)

    if param_dist is None:
        param_dist = {
            'estimator__max_iter': [100, 200, 400],
            'estimator__learning_rate': [0.01, 0.05, 0.1],
            'estimator__max_leaf_nodes': [15, 31, 63, None],
            'estimator__max_depth': [3, 5, 10, None]
        }
    search = RandomizedSearchCV(search_est, param_distributions=param_dist, n_iter=n_iter,
                               cv=tscv, scoring='neg_mean_absolute_error',
                               n_jobs=n_jobs, random_state=random_state,
                               verbose=1, return_train_score=True)

    t0 = time.time()
    search.fit(X, y)
    t1 = time.time()

    res = {
        "search": search,
        "feature_cols": feature_cols,
        "price_cols": price_cols,
        "vol_cols": vol_cols,
        "fit_time": t1 - t0
    }
    return res


def plot_results(test_df, y_pred, out_path):
    plt.figure(figsize=(10, 5))
    plt.plot(test_df["date"], test_df["target_t+1"], label="实际价格")
    plt.plot(test_df["date"], y_pred, label="预测价格")
    plt.xlabel("date")
    plt.ylabel("price")
    plt.title("短期价格预测：实际 vs 预测")
    plt.legend()
    plt.tight_layout()
    plt.close()


def plot_results_interactive(test_df, y_pred, out_path_html):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df["date"], y=test_df["target_t+1"],
                             mode='lines+markers', name='实际价格'))
    fig.add_trace(go.Scatter(x=test_df["date"], y=y_pred,
                             mode='lines+markers', name='预测价格'))
    fig.update_layout(title='短期价格预测：实际 vs 预测', xaxis_title='date', yaxis_title='price')
    fig.update_xaxes(rangeslider_visible=True)
    fig.write_html(out_path_html, include_plotlyjs='cdn')


def plot_interactive_full(df, test_df=None, test_price_pred=None, test_vol_pred=None,
                          future_df=None, out_html="interactive.html"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["price"], mode="lines+markers",
                             name="实际价格", marker=dict(size=6)))
    if test_df is not None and test_price_pred is not None:
        fig.add_trace(go.Scatter(x=test_df["date"], y=test_price_pred, mode="lines+markers",
                                 name="测试集预测价格(t+1)", marker=dict(size=6)))
    if future_df is not None and "pred_price" in future_df.columns:
        fig.add_trace(go.Scatter(x=future_df["date"], y=future_df["pred_price"], mode="lines+markers",
                                 name="未来价格预测", marker=dict(symbol='diamond', size=8)))

    if "vol" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["vol"], mode="lines+markers",
                                 name="实际在售数量", marker=dict(size=6),
                                 line=dict(dash='dot'), yaxis='y2'))
    if test_df is not None and test_vol_pred is not None:
        fig.add_trace(go.Scatter(x=test_df["date"], y=test_vol_pred, mode="lines+markers",
                                 name="测试集在售数量预测(t+1)", marker=dict(size=6),
                                 line=dict(dash='dot'), yaxis='y2'))
    if future_df is not None and "pred_vol" in future_df.columns:
        fig.add_trace(go.Scatter(x=future_df["date"], y=future_df["pred_vol"], mode="lines+markers",
                                 name="未来在售数量预测", marker=dict(symbol='diamond', size=8),
                                 line=dict(dash='dot'), yaxis='y2'))

    fig.update_layout(
        title_text="短期价格与在售数量：实际 vs 预测",
        hovermode="x unified",
        height=600,
        xaxis=dict(rangeslider=dict(visible=True), dtick="D1", tickformat="%Y-%m-%d"),
        yaxis=dict(title="价格"),
        yaxis2=dict(title="在售数量", overlaying='y', side='right')
    )
    pio.write_html(fig, out_html, include_plotlyjs='cdn', auto_open=False)
    return out_html


def set_chinese_font():
    candidates = ["Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "Noto Sans CJK JP", "Arial Unicode MS"]
    available = {f.name: f.fname for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams["font.family"] = "sans-serif"
            rcParams["font.sans-serif"] = [name]
            rcParams["axes.unicode_minus"] = False
            return name
    for f in fm.fontManager.ttflist:
        if any(x in f.name for x in ["SimHei", "Hei", "Noto", "Microsoft", "WenQuanYi"]):
            rcParams["font.family"] = "sans-serif"
            rcParams["font.sans-serif"] = [f.name]
            rcParams["axes.unicode_minus"] = False
            return f.name
    rcParams["axes.unicode_minus"] = False
    return None


def input_int(prompt, default, min_val=None, max_val=None):
    """整数输入工具，带默认值和范围限制"""
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
    print("        CS2 饰品价量时序预测工具（交互版）")
    print("=" * 60)
    print("请输入 CSV 数据文件路径（可拖拽文件到窗口自动补全路径）")
    csv_path = input("CSV 文件路径：").strip()

    # 校验文件
    if not os.path.isfile(csv_path):
        print(f"错误：文件不存在 -> {csv_path}")
        return
    if not csv_path.lower().endswith(".csv"):
        print("错误：请选择 .csv 格式文件")
        return

    folder = os.path.dirname(csv_path) or "."
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n✅ 已选中文件：{os.path.basename(csv_path)}")

    # 交互式参数配置
    lags = input_int("滞后天数", default=14, min_val=1)
    test_days = input_int("测试集天数", default=14, min_val=1)
    n_estimators = input_int("模型迭代数", default=200, min_val=10)
    horizon = input_int("预测未来天数(1-7)", default=7, min_val=1, max_val=7)

    tune_choice = input("是否开启超参搜索调优？(y/n，默认 n)：").strip().lower()
    do_tune = True if tune_choice == "y" else False

    n_iter = 20
    n_splits = 5
    if do_tune:
        n_iter = input_int("随机搜索迭代次数", default=20, min_val=5)
        n_splits = input_int("时序交叉验证折数", default=5, min_val=2)

    print("\n" + "-" * 40 + " 开始运行 " + "-" * 40)

    # 读取数据
    df = load_csv(csv_path)
    cache_path = os.path.join(folder, f"cache_{base_name}.csv")
    cache_raw = load_feature_cache(cache_path)

    # 合并缓存数据
    if cache_raw is not None:
        df_all = pd.concat([cache_raw, df], ignore_index=True)
        df_all = df_all.sort_values('date').drop_duplicates(subset=['date'], keep='last')
        df_all = df_all.tail(CACHE_WINDOW_DAYS).reset_index(drop=True)
        df = df_all

    # 特征工程
    df_feat = prepare_features(df, lags=lags)

    # 更新原始数据缓存
    try:
        save_feature_cache(df[['date', 'yyyp_sell_price', 'yyyp_sell_num']], cache_path)
    except Exception:
        pass

    # 设置中文字体
    used_font = set_chinese_font()
    if used_font:
        print(f"字体加载成功：{used_font}")
    else:
        print("警告：未找到中文字体，图表可能乱码")

    # 加载历史模型
    model_path = os.path.join(folder, f"model_{base_name}_h{horizon}.joblib")
    old_model = None
    if os.path.exists(model_path) and not do_tune:
        try:
            old_dict = joblib.load(model_path)
            old_model = old_dict.get('model') if isinstance(old_dict, dict) else old_dict
            print(f"✅ 加载历史模型：{model_path}")
        except Exception:
            old_model = None

    # 超参调优分支
    if do_tune:
        print("\n🔍 开始时序交叉验证 + 超参数搜索...")
        res = perform_time_series_tuning(df_feat, horizon=horizon, n_splits=n_splits, n_iter=n_iter)
        search = res['search']
        feature_cols = res['feature_cols']
        price_cols = res['price_cols']
        vol_cols = res['vol_cols']

        print("\n最佳超参数：")
        print(search.best_params_)

        # 保存CV结果
        cv_df = pd.DataFrame(search.cv_results_)
        cv_path = os.path.join(folder, f"cv_results_{base_name}_h{horizon}.csv")
        cv_df.to_csv(cv_path, index=False)
        print(f"CV 结果已保存：{cv_path}")

        # 重建多目标数据集并评估
        df2 = df_feat.copy()
        for h in range(1, horizon + 1):
            df2[f"target_price_t+{h}"] = df2["price"].shift(-h)
            df2[f"target_vol_t+{h}"] = df2["vol"].shift(-h)
        df2 = df2.dropna().reset_index(drop=True)

        feature_cols = [c for c in df2.columns if c.startswith("lag_") or c.startswith("vol_") or
                        c.startswith("rmean_") or c.startswith("rstd_") or
                        c in ["pct_change_1", "is_weekend"]]

        if len(df2) <= test_days + 10:
            test_days = max(1, int(len(df2) * 0.2))

        train_df = df2.iloc[:-test_days]
        test_df = df2.iloc[-test_days:]
        X_train = train_df[feature_cols].values
        y_train = train_df[price_cols + vol_cols].values
        X_test = test_df[feature_cols].values
        y_test = test_df[price_cols + vol_cols].values

        best = search.best_estimator_
        best.fit(X_train, y_train)
        y_pred = best.predict(X_test)

        # 计算指标
        metrics = {"price": {}, "vol": {}}
        for h in range(horizon):
            y_true_p = y_test[:, h]
            y_pred_p = y_pred[:, h]
            mae_p = mean_absolute_error(y_true_p, y_pred_p)
            rmse_p = np.sqrt(mean_squared_error(y_true_p, y_pred_p))
            mape_p = np.mean(np.abs((y_true_p - y_pred_p) / (y_true_p + 1e-9))) * 100
            metrics["price"][f"t+{h+1}"] = {"MAE": mae_p, "RMSE": rmse_p, "MAPE(%)": mape_p}

            y_true_v = y_test[:, horizon + h]
            y_pred_v = y_pred[:, horizon + h]
            mae_v = mean_absolute_error(y_true_v, y_pred_v)
            rmse_v = np.sqrt(mean_squared_error(y_true_v, y_pred_v))
            mape_v = np.mean(np.abs((y_true_v - y_pred_v) / (y_true_v + 1e-9))) * 100
            metrics["vol"][f"t+{h+1}"] = {"MAE": mae_v, "RMSE": rmse_v, "MAPE(%)": mape_v}

        print("\n📊 模型评估指标：")
        for typ in ["price", "vol"]:
            print(f"【{typ}】")
            for h, met in metrics[typ].items():
                print(f"  {h}: MAE={met['MAE']:.4f}, RMSE={met['RMSE']:.4f}, MAPE={met['MAPE(%)']:.2f}%")

        # 保存调优后模型
        model_save_path = os.path.join(folder, f"model_{base_name}_h{horizon}_tuned.joblib")
        joblib.dump({"model": best, "features": feature_cols, "horizon": horizon}, model_save_path)
        print(f"✅ 调优模型已保存：{model_save_path}")

        # 未来预测
        future_df = predict_future_from_last(best, df_feat, feature_cols, horizon=horizon)
        forecast_path = os.path.join(folder, f"forecast_{base_name}_h{horizon}_tuned.csv")
        future_df.to_csv(forecast_path, index=False)
        print(f"✅ 未来{horizon}天预测结果已保存：{forecast_path}")

        # 交互式图表
        interactive_path = os.path.join(folder, f"pred_{base_name}_h{horizon}_tuned.html")
        test_price_pred_t1 = y_pred[:, 0]
        test_vol_pred_t1 = y_pred[:, horizon]
        plot_interactive_full(df_feat, test_df=test_df, test_price_pred=test_price_pred_t1,
                              test_vol_pred=test_vol_pred_t1, future_df=future_df, out_html=interactive_path)
        print(f"✅ 交互式预测图表已保存：{interactive_path}")

    else:
        # 常规训练 + 增量训练
        df2 = df_feat.copy()
        price_cols = []
        vol_cols = []
        for h in range(1, horizon + 1):
            pc = f"target_price_t+{h}"
            vc = f"target_vol_t+{h}"
            df2[pc] = df2["price"].shift(-h)
            df2[vc] = df2["vol"].shift(-h)
            price_cols.append(pc)
            vol_cols.append(vc)
        df2 = df2.dropna().reset_index(drop=True)

        feature_cols = [c for c in df2.columns if c.startswith("lag_") or c.startswith("vol_") or
                        c.startswith("rmean_") or c.startswith("rstd_") or
                        c in ["pct_change_1", "is_weekend"]]

        new_rows = df2
        if cache_raw is not None and not cache_raw.empty:
            prev_max = pd.to_datetime(cache_raw['date']).max()
            new_rows = df2[pd.to_datetime(df2['date']) > prev_max]

        init_model = None
        # 增量续训
        if old_model is not None and not new_rows.empty:
            X_new = new_rows[feature_cols].values
            y_new = new_rows[price_cols + vol_cols].values
            try:
                if hasattr(old_model, 'estimators_'):
                    for i, est in enumerate(old_model.estimators_):
                        est.set_params(warm_start=True)
                        est.fit(X_new, y_new[:, i])
                    init_model = old_model
                else:
                    old_model.set_params(warm_start=True)
                    old_model.fit(X_new, y_new[:, 0])
                    init_model = old_model
                joblib.dump({"model": init_model, "features": feature_cols, "horizon": horizon}, model_path)
                print("✅ 增量训练完成，模型已更新")
            except Exception as e:
                print(f"⚠️ 增量训练失败，使用全量训练: {e}")
                init_model = None

        # 训练评估
        model, test_df, y_pred, metrics, feature_cols, price_cols, vol_cols = train_and_evaluate_multi(
            df_feat, horizon=horizon, test_days=test_days, n_estimators=n_estimators, init_model=init_model
        )

        print("\n📊 模型评估指标：")
        for typ in ["price", "vol"]:
            print(f"【{typ}】")
            for h, met in metrics[typ].items():
                print(f"  {h}: MAE={met['MAE']:.4f}, RMSE={met['RMSE']:.4f}, MAPE={met['MAPE(%)']:.2f}%")

        # 保存模型
        joblib.dump({
            "model": model,
            "features": feature_cols,
            "horizon": horizon,
            "price_cols": price_cols,
            "vol_cols": vol_cols
        }, model_path)
        print(f"✅ 模型已保存：{model_path}")

        # 未来预测
        future_df = predict_future_from_last(model, df_feat, feature_cols, horizon=horizon)
        forecast_path = os.path.join(folder, f"forecast_{base_name}_h{horizon}.csv")
        future_df.to_csv(forecast_path, index=False)
        print(f"✅ 未来{horizon}天预测结果已保存：{forecast_path}")

        # 交互式图表
        interactive_path = os.path.join(folder, f"pred_{base_name}_h{horizon}.html")
        test_price_pred_t1 = y_pred[:, 0]
        test_vol_pred_t1 = y_pred[:, horizon]
        plot_interactive_full(df_feat, test_df=test_df, test_price_pred=test_price_pred_t1,
                              test_vol_pred=test_vol_pred_t1, future_df=future_df, out_html=interactive_path)
        print(f"✅ 交互式预测图表已保存：{interactive_path}")

    print("\n" + "=" * 40)
    print("🎉 全部任务执行完成！")
    print("=" * 40)


if __name__ == "__main__":
    main()