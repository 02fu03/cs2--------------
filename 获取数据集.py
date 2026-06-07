import requests
import json
import pandas as pd
import time
import os

API_URL = "https://api.csqaq.com/api/v1/info/chart"
API_TOKEN = "CSLZ81C7U729F50169Z44687"
GOOD_ID = 13493
PLATFORM = 2
PERIOD = "90"
STYLE = "all_style"
SAVE_FILE = "AK47抽象派1337(崭新出厂)_悠悠有品_近3个月数据.csv"

headers = {
    "ApiToken": API_TOKEN,
    "Content-Type": "application/json"
}

def get_data(key_name):
    payload = json.dumps({
        "good_id": str(GOOD_ID),
        "key": key_name,
        "platform": PLATFORM,
        "period": PERIOD,
        "style": STYLE
    })
    try:
        res = requests.post(API_URL, data=payload, headers=headers, timeout=30)
        res.raise_for_status()
        ret = res.json()
        raw_dict = ret.get("data", {})
        ts_list = raw_dict.get("timestamp", [])
        val_list = raw_dict.get("main_data", [])
        zip_data = list(zip(ts_list, val_list))
        print(f"【{key_name}】获取数据：{len(zip_data)}条")
        return zip_data
    except Exception as e:
        print(f"{key_name} 请求异常：{e}")
        return []

# 售价：日均价
def build_price_df(data):
    if not data:
        return pd.DataFrame(columns=["date", "yyyp_sell_price"])
    df = pd.DataFrame(data, columns=["ts", "yyyp_sell_price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.date
    df = df.groupby("date", as_index=False)["yyyp_sell_price"].mean()
    df["date"] = pd.to_datetime(df["date"])
    # 强转为数值类型
    df["yyyp_sell_price"] = pd.to_numeric(df["yyyp_sell_price"], errors="coerce")
    return df

# 在售数量：当日分时求平均取整
def build_num_df(data):
    if not data:
        return pd.DataFrame(columns=["date", "yyyp_sell_num"])
    df = pd.DataFrame(data, columns=["ts", "yyyp_sell_num"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.date
    df = df.groupby("date", as_index=False)["yyyp_sell_num"].mean()
    # 强转数值，空值填充0，再取整
    df["yyyp_sell_num"] = pd.to_numeric(df["yyyp_sell_num"], errors="coerce").fillna(0)
    df["yyyp_sell_num"] = df["yyyp_sell_num"].round().astype(int)
    df["date"] = pd.to_datetime(df["date"])
    return df

if __name__ == "__main__":
    # 拉取接口数据
    price_list = get_data("sell_price")
    time.sleep(2)
    num_list = get_data("sell_num")

    df_p = build_price_df(price_list)
    df_n = build_num_df(num_list)
    df_new = pd.merge(df_p, df_n, on="date", how="outer").sort_values("date")
    print(f"本次接口返回新数据：{len(df_new)} 行")

    # 合并历史数据
    if os.path.exists(SAVE_FILE):
        df_old = pd.read_csv(SAVE_FILE, parse_dates=["date"], encoding="utf-8-sig")
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        # 去重：同一天保留最新数据
        df_all = df_all.drop_duplicates(subset=["date"], keep="last")
    else:
        df_all = df_new

    # 全空数据兜底（接口完全失效时不覆盖旧文件）
    if df_all.empty and os.path.exists(SAVE_FILE):
        print("⚠️  本次未获取到任何新数据，保留原有文件")
    else:
        df_all = df_all.sort_values("date").reset_index(drop=True)
        df_all.to_csv(SAVE_FILE, index=False, encoding="utf-8-sig")
        print(f"\n✅ 文件已保存：{SAVE_FILE}，全量有效数据行数：{len(df_all)}")
        print("最新5条数据：")
        print(df_all.tail())