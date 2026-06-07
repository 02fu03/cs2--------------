#!/usr/bin/env python3
"""
通用饰品ID查询工具，支持所有同格式txt文件
"""
import argparse
import re
import sys
from difflib import get_close_matches

def parse_id_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    blocks = re.split(r"\n\s*\n", text)
    entries = []
    for block in blocks:
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

    for e in entries:
        if e['name'].lower() == q or e['hash'].lower() == q:
            exact.append(e)
    if exact:
        return exact

    for e in entries:
        if q in e['name'].lower():
            partial.append(e)
        elif q in e['hash'].lower():
            hash_partial.append(e)

    results = partial + hash_partial
    if results:
        return results[:max_results]

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
    print("===== CS2 饰品ID通用查询工具 =====")
    file_path = input("请输入数据文件完整路径：").strip()
    try:
        entries = parse_id_file(file_path)
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
        sys.exit(1)

    print(f"\n已加载 {len(entries)} 条条目。输入查询内容，直接回车退出。")
    while True:
        q = input("\n查询名称> ").strip()
        if not q:
            break
        res = search_entries(entries, q)
        print_results(res)

if __name__ == '__main__':
    main()