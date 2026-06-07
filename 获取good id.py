import http.client
import json
from datetime import datetime

def get_good_ids():
    all_items = []
    page_index = 5
    page_size = 500
    search_keyword = "AK-47"

    while True:
        conn = http.client.HTTPSConnection("api.csqaq.com")
        payload = json.dumps({
            "page_index": page_index,
            "page_size": page_size,
            "search": search_keyword
        })
        headers = {
            'ApiToken': 'CSLZ81C7U729F50169Z44687',
            'Content-Type': 'application/json'
        }
        try:
            conn.request("POST", "/api/v1/info/get_good_id", payload, headers)
            res = conn.getresponse()
            raw = res.read()
            conn.close()

            print(f"第 {page_index} 页 | 状态码: {res.status}")
            if not raw.strip():
                print("接口返回空，结束")
                break
            resp = json.loads(raw.decode("utf-8"))
            print("接口返回：", resp)

            if res.status == 422:
                print("参数错误", resp)
                break

            outer_data = resp["data"]
            data_dict = outer_data["data"]

            if len(data_dict) == 0:
                print("✅ 全部数据获取完毕")
                break

            page_items = list(data_dict.values())
            all_items.extend(page_items)
            print(f"本页获取 {len(page_items)} 条，累计:{len(all_items)}")
            page_index += 1

        except Exception as e:
            print(f"第{page_index}页异常:{e}")
            conn.close()
            break

    # 仅生成txt
    if all_items:
        total = len(all_items)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"CS2_{search_keyword}_{ts}"

        with open(f"{base}.txt", "w", encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write(f"CS2 {search_keyword} 饰品数据\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据总量: {total} 条\n")
            f.write("="*70 + "\n\n")

            groups = {}
            for item in all_items:
                name = item["name"].split(" | ")[0] if " | " in item["name"] else item["name"]
                groups.setdefault(name, []).append(item)

            f.write("【分类统计】\n")
            for k, v in groups.items():
                f.write(f"{k}：{len(v)} 条\n")
            f.write("\n【明细列表】\n")
            for idx, item in enumerate(all_items, 1):
                f.write(f"\n序号：{idx}\nID：{item['id']}\n名称：{item['name']}\n哈希名：{item['market_hash_name']}\n")

            f.write("\n" + "="*70 + "\n原始JSON数据\n")
            f.write(json.dumps(all_items, ensure_ascii=False, indent=2))
        print(f"✅ 已保存 {base}.txt")
        print(f"\n🎉 爬取完成，总计 {total} 条")
    else:
        print("无数据保存")

if __name__ == "__main__":
    get_good_ids()