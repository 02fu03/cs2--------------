#!/usr/bin/env python3
"""
lookup_id.py
读取 AK-47 ID 文本文件，支持按中文名称或哈希名（部分匹配/不区分大小写）查询对应 ID。
用法示例：
  python lookup_id.py --file "AK-47 ID.txt" --name "精英之作"
或交互式运行：
  python lookup_id.py --file "AK-47 ID.txt"
然后输入要查询的名称。
"""

import argparse
import re
import sys
from difflib import get_close_matches


def parse_id_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 将文件按双换行或序号块分隔
    blocks = re.split(r"\n\s*\n", text)
    entries = []
    for block in blocks:
        # 只处理包含 "ID：" 和 "名称：" 的块
        if 'ID：' in block and '名称：' in block:
            id_match = re.search(r"ID：\s*(\d+)", block)
            name_match = re.search(r"名称：\s*(.+)", block)
            hash_match = re.search(r"哈希名：\s*(.+)", block)
            if id_match and name_match:
                eid = id_match.group(1).strip()
                name = name_match.group(1).strip()
                hname = hash_match.group(1).strip() if hash_match else ''
                entries.append({'id': eid, 'name': name, 'hash': hname})
    return entries


def search_entries(entries, query, max_results=20):
    q = query.strip().lower()
    exact = []
    partial = []
    hash_partial = []
    names = [e['name'] for e in entries]
    # first exact or full-name match
    for e in entries:
        if e['name'].lower() == q or e['hash'].lower() == q:
            exact.append(e)
    if exact:
        return exact

    # partial matches (substring)
    for e in entries:
        if q in e['name'].lower():
            partial.append(e)
        elif q in e['hash'].lower():
            hash_partial.append(e)

    results = partial + hash_partial
    if results:
        return results[:max_results]

    # fuzzy match on name using difflib
    close = get_close_matches(query, names, n=max_results, cutoff=0.6)
    fuzzy = [e for e in entries if e['name'] in close]
    return fuzzy


def print_results(results):
    if not results:
        print('未找到匹配项。')
        return
    for e in results:
        print(f"名称: {e['name']}")
        if e['hash']:
            print(f"哈希名: {e['hash']}")
        print(f"ID: {e['id']}")
        print('-' * 40)


def main():
    parser = argparse.ArgumentParser(description='按名称查询 AK-47 ID')
    parser.add_argument('--file', '-f', type=str, default='AK-47 ID.txt', help='ID 文本文件路径')
    parser.add_argument('--name', '-n', type=str, default=None, help='要查询的名称（支持部分匹配）')
    args = parser.parse_args()

    try:
        entries = parse_id_file(args.file)
    except FileNotFoundError:
        print(f"未找到文件: {args.file}")
        sys.exit(1)

    if args.name:
        res = search_entries(entries, args.name)
        print_results(res)
        return

    # 交互式查询
    print('已加载', len(entries), '条条目。输入要查询的名称（回车退出）。')
    while True:
        try:
            q = input('查询名称> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        res = search_entries(entries, q)
        print_results(res)


if __name__ == '__main__':
    main()
