import os
import glob
import argparse
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

# cache window days for sliding window of historical features
CACHE_WINDOW_DAYS = 365


def save_feature_cache(df, cache_path):
    # save as CSV (preferred). Keep parquet fallback if requested by using .parquet path.
    try:
        df.to_csv(cache_path, index=False)
    except Exception:
        # fallback to parquet if CSV write fails
        try:
            pq_path = os.path.splitext(cache_path)[0] + '.parquet'
            df.to_parquet(pq_path, index=False)
        except Exception:
            raise


def load_feature_cache(cache_path):
    # prefer CSV cache; fallback to parquet
    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path, parse_dates=['date'])
        except Exception:
            pass
    # fallback: try parquet
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
    # additional rolling stats
    df["rmean_14"] = df["price"].rolling(14).mean()
    df["rmean_30"] = df["price"].rolling(30).mean()
    df["rstd_14"] = df["price"].rolling(14).std()
    df["ewm_7"] = df["price"].ewm(span=7, adjust=False).mean()

    # log transform and diff features
    df["price_log1p"] = np.log1p(df["price"])
    df["price_diff_1"] = df["price"].diff(1)

    # date features
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

    # target: next-day price (shift -1)
    df["target_t+1"] = df["price"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df


def train_and_evaluate(df, test_days=14, n_estimators=200, init_model=None):
    # time-based split: last `test_days` rows as test
    if len(df) <= test_days + 10:
        test_days = max(1, int(len(df) * 0.2))

    train_df = df.iloc[:-test_days]
    test_df = df.iloc[-test_days:]

    feature_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("vol_") or c.startswith("rmean_") or c.startswith("rstd_") or c in ["pct_change_1", "is_weekend"]]

    X_train = train_df[feature_cols].values
    y_train = train_df["target_t+1"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target_t+1"].values

    if init_model is not None:
        # 已有模型，直接使用并跳过全量重训
        model = init_model
    else:
        # 使用 HistGradientBoostingRegressor（支持 warm_start 增量训练）作为默认单输出树模型
        model = HistGradientBoostingRegressor(max_iter=n_estimators, random_state=42, warm_start=True)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape}

    return model, test_df, y_pred, metrics, feature_cols


def train_and_evaluate_multi(df, horizon=3, test_days=14, n_estimators=200, init_model=None):
    """Train a multi-output model predicting price and vol for next `horizon` days.
    Returns model, test_df, y_pred, metrics, feature_cols, y_test_cols
    """
    df = df.copy()
    # create multi-step targets
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

    feature_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("vol_") or c.startswith("rmean_") or c.startswith("rstd_") or c in ["pct_change_1", "is_weekend"]]

    X_train = train_df[feature_cols].values
    y_train = train_df[price_cols + vol_cols].values
    X_test = test_df[feature_cols].values
    y_test = test_df[price_cols + vol_cols].values

    if init_model is not None:
        model = init_model
    else:
        # 使用 MultiOutputRegressor 包装 HistGradientBoostingRegressor 支持多输出预测
        base = HistGradientBoostingRegressor(max_iter=n_estimators, random_state=42, warm_start=True)
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # metrics per horizon
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
    """Use the last available row in df to predict next `horizon` days (direct multi-output)."""
    last = df.copy().iloc[-1:]
    X = last[feature_cols].values
    yhat = model.predict(X)[0]
    # yhat contains [price_t+1..price_t+h, vol_t+1..vol_t+h]
    prices = yhat[:horizon]
    vols = yhat[horizon: horizon * 2]
    last_date = df["date"].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
    out = pd.DataFrame({"date": future_dates, "pred_price": prices, "pred_vol": vols})
    return out


def perform_time_series_tuning(df, horizon=3, n_splits=5, n_iter=20, param_dist=None, n_jobs=-1, random_state=42):
    """Perform time-series cross-validated randomized search for multi-output model."""
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

    feature_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("vol_") or c.startswith("rmean_") or c.startswith("rstd_") or c in ["pct_change_1", "is_weekend"]]

    X = df[feature_cols].values
    y = df[price_cols + vol_cols].values

    # remove RF-style param grid; we'll set HGB-specific grid below when needed

    tscv = TimeSeriesSplit(n_splits=n_splits)
    # 使用 HistGradientBoostingRegressor 作为搜索基础；包装为 MultiOutputRegressor 以支持多输出 y
    base_est = HistGradientBoostingRegressor(random_state=random_state, warm_start=True)
    search_est = MultiOutputRegressor(base_est)
    # 构造可搜索的参数字典（注意 MultiOutputRegressor 下需使用 estimator__ 前缀）
    if param_dist is None:
        param_dist = {
            'estimator__max_iter': [100, 200, 400],
            'estimator__learning_rate': [0.01, 0.05, 0.1],
            'estimator__max_leaf_nodes': [15, 31, 63, None],
            'estimator__max_depth': [3, 5, 10, None]
        }
    search = RandomizedSearchCV(search_est, param_distributions=param_dist, n_iter=n_iter, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=n_jobs, random_state=random_state, verbose=1, return_train_score=True)

    t0 = time.time()
    search.fit(X, y)
    t1 = time.time()

    res = {"search": search, "feature_cols": feature_cols, "price_cols": price_cols, "vol_cols": vol_cols, "fit_time": t1 - t0}
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
    """Save an interactive HTML plot with a range slider (supports mouse wheel zoom)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df["date"], y=test_df["target_t+1"], mode='lines+markers', name='实际价格'))
    fig.add_trace(go.Scatter(x=test_df["date"], y=y_pred, mode='lines+markers', name='预测价格'))
    fig.update_layout(title='短期价格预测：实际 vs 预测', xaxis_title='date', yaxis_title='price')
    fig.update_xaxes(rangeslider_visible=True)
    # write interactive html
    fig.write_html(out_path_html, include_plotlyjs='cdn')


def plot_interactive_full(df, test_df=None, test_price_pred=None, test_vol_pred=None, future_df=None, out_html="interactive.html"):
    """Create interactive Plotly HTML with a single plot showing price and volume.
    - Price series on left y-axis.
    - Volume (在售数量) on right secondary y-axis.
    All series include actual, test-set t+1 prediction, and future forecasts (if provided).
    x 轴间隔为 1 天。
    """
    fig = go.Figure()

    # Price traces (primary y)
    fig.add_trace(go.Scatter(x=df["date"], y=df["price"], mode="lines+markers", name="实际价格", marker=dict(size=6)))
    if test_df is not None and test_price_pred is not None:
        fig.add_trace(go.Scatter(x=test_df["date"], y=test_price_pred, mode="lines+markers", name="测试集预测价格(t+1)", marker=dict(size=6)))
    if future_df is not None and "pred_price" in future_df.columns:
        fig.add_trace(go.Scatter(x=future_df["date"], y=future_df["pred_price"], mode="lines+markers", name="未来价格预测", marker=dict(symbol='diamond', size=8)))

    # Volume traces (secondary y axis)
    if "vol" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["vol"], mode="lines+markers", name="实际在售数量", marker=dict(size=6), line=dict(dash='dot'), yaxis='y2'))
    if test_df is not None and test_vol_pred is not None:
        fig.add_trace(go.Scatter(x=test_df["date"], y=test_vol_pred, mode="lines+markers", name="测试集在售数量预测(t+1)", marker=dict(size=6), line=dict(dash='dot'), yaxis='y2'))
    if future_df is not None and "pred_vol" in future_df.columns:
        fig.add_trace(go.Scatter(x=future_df["date"], y=future_df["pred_vol"], mode="lines+markers", name="未来在售数量预测", marker=dict(symbol='diamond', size=8), line=dict(dash='dot'), yaxis='y2'))

    # Layout: primary y for price, secondary y for vol
    fig.update_layout(
        title_text="短期价格与在售数量：实际 vs 预测（单坐标系，右侧为在售数量）",
        hovermode="x unified",
        height=600,
        xaxis=dict(rangeslider=dict(visible=True), dtick="D1", tickformat="%Y-%m-%d"),
        yaxis=dict(title="价格"),
        yaxis2=dict(title="在售数量", overlaying='y', side='right')
    )

    pio.write_html(fig, out_html, include_plotlyjs='cdn', auto_open=False)
    return out_html


def set_chinese_font():
    """Try to set a Chinese-capable font for matplotlib to avoid missing glyphs."""
    candidates = ["Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "Noto Sans CJK JP", "Arial Unicode MS"]
    available = {f.name: f.fname for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams["font.family"] = "sans-serif"
            rcParams["font.sans-serif"] = [name]
            rcParams["axes.unicode_minus"] = False
            return name

    # fallback: pick first font with CJK in filename
    for f in fm.fontManager.ttflist:
        if any(x in f.name for x in ["SimHei", "Hei", "Noto", "Microsoft", "WenQuanYi"]):
            rcParams["font.family"] = "sans-serif"
            rcParams["font.sans-serif"] = [f.name]
            rcParams["axes.unicode_minus"] = False
            return f.name

    # as last resort, only ensure minus sign displays
    rcParams["axes.unicode_minus"] = False
    return None


def main():
    parser = argparse.ArgumentParser(description="基于历史价量的短期价格预测")
    parser.add_argument("--file", type=str, default=None, help="指定单个CSV路径，默认使用目录中第一个CSV")
    parser.add_argument("--folder", type=str, default=".", help="数据所在目录，默认当前目录")
    parser.add_argument("--lags", type=int, default=14, help="使用的滞后天数")
    parser.add_argument("--test_days", type=int, default=14, help="用于评估的天数")
    parser.add_argument("--n_estimators", type=int, default=200, help="随机森林树数量")
    parser.add_argument("--horizon", type=int, default=7, help="预测未来天数，支持 1-7")
    parser.add_argument("--tune", action='store_true', help="是否进行时间序列交叉验证与超参搜索")
    parser.add_argument("--n_iter", type=int, default=20, help="RandomizedSearchCV 的迭代次数")
    parser.add_argument("--n_splits", type=int, default=5, help="TimeSeriesSplit 的拆分数")
    args = parser.parse_args()

    folder = args.folder
    csv_path = args.file
    if csv_path is None:
        csvs = find_csv_files(folder)
        if not csvs:
            print("未找到CSV文件。请把数据放在指定目录或使用 --file 指定文件。")
            return
        csv_path = csvs[0]

    print(f"读取: {csv_path}")
    df = load_csv(csv_path)

    # 提前读取 horizon 与 base_name，用于缓存与模型命名
    horizon = int(args.horizon)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    # 缓存路径 (优先 parquet, fallback csv)
    cache_path = os.path.join(folder, f"cache_{base_name}.csv")
    cache_raw = load_feature_cache(cache_path)

    if cache_raw is not None:
        # 合并历史原始数据与新读取的数据，按日期去重并保留最近窗口
        df_all = pd.concat([cache_raw, df], ignore_index=True)
        df_all = df_all.sort_values('date').drop_duplicates(subset=['date'], keep='last')
        # 保留滑动窗口天数
        df_all = df_all.tail(CACHE_WINDOW_DAYS).reset_index(drop=True)
        df = df_all

    # 生成特征（基于合并后的原始 df）
    df_feat = prepare_features(df, lags=args.lags)

    # 更新原始数据缓存（保存原始 df，而非特征 df，以便后续合并）
    try:
        save_feature_cache(df[['date','yyyp_sell_price','yyyp_sell_num']], cache_path)
    except Exception:
        pass

    used_font = set_chinese_font()
    if used_font:
        print(f"设置 matplotlib 字体为: {used_font}")
    else:
        print("未找到合适中文字体，可能仍有缺字警告。")

    if horizon < 1 or horizon > 7:
        print("horizon 必须在 1 到 7 之间。")
        return

    # 尝试加载已有模型以支持增量续训（仅在非调参模式下）
    model_path = os.path.join(folder, f"model_{base_name}_h{horizon}.joblib")
    old_model = None
    if os.path.exists(model_path) and not args.tune:
        try:
            old_dict = joblib.load(model_path)
            old_model = old_dict.get('model') if isinstance(old_dict, dict) else old_dict
            print(f'已加载历史模型: {model_path}')
        except Exception:
            old_model = None

    if args.tune:
        print("开始时间序列交叉验证与超参数搜索...")
        res = perform_time_series_tuning(df_feat, horizon=horizon, n_splits=args.n_splits, n_iter=args.n_iter, n_jobs=-1)
        search = res['search']
        feature_cols = res['feature_cols']
        price_cols = res['price_cols']
        vol_cols = res['vol_cols']

        print("搜索完成，最佳参数:")
        print(search.best_params_)

        # save cv results
        cv_df = pd.DataFrame(search.cv_results_)
        cv_path = os.path.join(folder, f"cv_results_{base_name}_h{horizon}.csv")
        cv_df.to_csv(cv_path, index=False)
        print(f"CV 搜索结果已保存到: {cv_path}")

        # evaluate best estimator on hold-out test set
        # rebuild multi-target df
        df2 = df_feat.copy()
        for h in range(1, horizon + 1):
            df2[f"target_price_t+{h}"] = df2["price"].shift(-h)
            df2[f"target_vol_t+{h}"] = df2["vol"].shift(-h)
        df2 = df2.dropna().reset_index(drop=True)

        # prepare feature column list (same logic as in train functions)
        feature_cols = [c for c in df2.columns if c.startswith("lag_") or c.startswith("vol_") or c.startswith("rmean_") or c.startswith("rstd_") or c in ["pct_change_1", "is_weekend"]]
        if len(df2) <= args.test_days + 10:
            test_days = max(1, int(len(df2) * 0.2))
        else:
            test_days = args.test_days
        train_df = df2.iloc[:-test_days]
        test_df = df2.iloc[-test_days:]

        X_train = train_df[feature_cols].values
        y_train = train_df[price_cols + vol_cols].values
        X_test = test_df[feature_cols].values
        y_test = test_df[price_cols + vol_cols].values

        best = search.best_estimator_
        # re-fit on train set for final evaluation
        best.fit(X_train, y_train)
        y_pred = best.predict(X_test)

        # compute metrics per horizon
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

        print("调参后评估指标（分 horizon）:")
        for typ in ["price", "vol"]:
            print(f" - {typ}:")
            for h, met in metrics[typ].items():
                print(f"    {h}: MAE={met['MAE']:.4f}, RMSE={met['RMSE']:.4f}, MAPE={met['MAPE(%)']:.2f}%")

        # save tuned model
        model_path = os.path.join(folder, f"model_{base_name}_h{horizon}_tuned.joblib")
        joblib.dump({"model": best, "features": feature_cols, "horizon": horizon}, model_path)
        print(f"调参后模型已保存到: {model_path}")

        # save test plot (t+1)
        plot_results(test_df, y_pred[:, 0], None)
        print("已生成对比图，交互式 HTML 已保存或将被保存。")

        # save future forecast from last row
        future_df = predict_future_from_last(best, df_feat, feature_cols, horizon=horizon)
        forecast_path = os.path.join(folder, f"forecast_{base_name}_h{horizon}_tuned.csv")
        future_df.to_csv(forecast_path, index=False)
        print(f"未来 {horizon} 天预测已保存到: {forecast_path}")

        interactive_path = os.path.join(folder, f"pred_{base_name}_h{horizon}_tuned.html")
        # y_pred columns: [price_t+1..t+h, vol_t+1..t+h]
        test_price_pred_t1 = y_pred[:, 0]
        test_vol_pred_t1 = y_pred[:, horizon]
        plot_interactive_full(df_feat, test_df=test_df, test_price_pred=test_price_pred_t1, test_vol_pred=test_vol_pred_t1, future_df=future_df, out_html=interactive_path)
        print(f"交互式预测图已保存到: {interactive_path}")

    else:
        # 生成 multi-target df（与 train_and_evaluate_multi 中相同逻辑），用于确定新数据与训练目标
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

        # detect new rows since cache (use cached raw dates if available)
        new_rows = df2
        if cache_raw is not None and not cache_raw.empty:
            prev_max = pd.to_datetime(cache_raw['date']).max()
            new_rows = df2[pd.to_datetime(df2['date']) > prev_max]

        init_model = None
        # 如果存在历史模型并且有新增训练样本，尝试对历史模型进行增量续训
        if old_model is not None and not new_rows.empty:
            X_new = new_rows[feature_cols].values
            y_new = new_rows[price_cols + vol_cols].values
            try:
                if hasattr(old_model, 'estimators_'):
                    # MultiOutputRegressor: 每个子估计器独立 fit
                    for i, est in enumerate(old_model.estimators_):
                        if hasattr(est, 'set_params'):
                            try:
                                est.set_params(warm_start=True)
                            except Exception:
                                pass
                        # fit 对应输出列
                        est.fit(X_new, y_new[:, i])
                    init_model = old_model
                else:
                    # 单输出模型：直接设置 warm_start 并 fit
                    try:
                        old_model.set_params(warm_start=True)
                    except Exception:
                        pass
                    if y_new.ndim == 2:
                        old_model.fit(X_new, y_new[:, 0])
                    else:
                        old_model.fit(X_new, y_new)
                    init_model = old_model

                # 保存增量更新后的模型
                joblib.dump({"model": init_model, "features": feature_cols, "horizon": horizon}, model_path)
                print(f'已对历史模型进行增量续训并保存: {model_path}')
            except Exception as e:
                print('增量续训失败，回退为全量训练。错误：', e)
                init_model = None

        # 调用训练/评估函数，传入 init_model 以便跳过全量重训
        model, test_df, y_pred, metrics, feature_cols, price_cols, vol_cols = train_and_evaluate_multi(df_feat, horizon=horizon, test_days=args.test_days, n_estimators=args.n_estimators, init_model=init_model)

        print("评估指标（分 horizon）:")
        for typ in ["price", "vol"]:
            print(f" - {typ}:")
            for h, met in metrics[typ].items():
                print(f"    {h}: MAE={met['MAE']:.4f}, RMSE={met['RMSE']:.4f}, MAPE={met['MAPE(%)']:.2f}%")

        model_path = os.path.join(folder, f"model_{base_name}_h{horizon}.joblib")
        joblib.dump({"model": model, "features": feature_cols, "horizon": horizon, "price_cols": price_cols, "vol_cols": vol_cols}, model_path)
        print(f"模型已保存到: {model_path}")

        # save a short plot for t+1 actual vs predicted (for test set)
        plot_results(test_df, y_pred[:, 0], None)
        print("已生成对比图，交互式 HTML 已保存或将被保存。")

        # predict next horizon days from last available row and save CSV
        future_df = predict_future_from_last(model, df_feat, feature_cols, horizon=horizon)
        forecast_path = os.path.join(folder, f"forecast_{base_name}_h{horizon}.csv")
        future_df.to_csv(forecast_path, index=False)
        print(f"未来 {horizon} 天预测已保存到: {forecast_path}")

        interactive_path = os.path.join(folder, f"pred_{base_name}_h{horizon}.html")
        test_price_pred_t1 = y_pred[:, 0]
        test_vol_pred_t1 = y_pred[:, horizon]
        plot_interactive_full(df_feat, test_df=test_df, test_price_pred=test_price_pred_t1, test_vol_pred=test_vol_pred_t1, future_df=future_df, out_html=interactive_path)
        print(f"交互式预测图已保存到: {interactive_path}")


if __name__ == "__main__":
    main()
