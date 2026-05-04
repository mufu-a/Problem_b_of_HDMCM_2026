#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据可视化模块 — 生成论文所需的8张图表
图表1: 问题一成本构成饼图（可变成本 vs 固定成本）
图表2: 问题一各地区运输成本柱状图（前15城市 + 其他合并）
图表3: 问题二优化前后成本对比双柱图
图表4: 问题三熵权法权重分布饼图
图表5: 问题三仓库综合得分排名柱状图
图表6: 问题三三层优化成本减少瀑布图
图表7: 固定成本构成饼图（人工/保险/审验/违章/胎耗/维保/折旧）
图表8: 可变成本构成饼图（去程油费/回程油费/路桥费）
"""

import matplotlib
# Mac系统中文显示: 优先使用Arial Unicode MS (macOS自带，覆盖所有中日韩字符)
matplotlib.rcParams['font.sans-serif'] = [
    'Arial Unicode MS', 'Heiti TC', 'STHeiti', 'PingFang SC', 'SimHei'
]
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'stix'

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from config import VEHICLE_PATH, SHIPPING_PATH, NANTONG_COORDS, EMPTY_LOAD_FUEL_RATIO
from utils import (
    load_vehicle_data_full, load_shipping_data, build_detailed_cost_lookup,
    get_trips, compute_original_cost, load_waybill_data, load_dispatch_data,
    road_distance, get_coords, get_region_beta, tsp_order, route_distance,
    estimate_travel_days,
)
from first import compute_trip_costs
from second import optimize_all, calc_route_cost
from third import (
    compute_warehouse_indicators, compute_absolute_scores,
    entropy_weight_method, optimize_vrp_cw
)

OUTPUT_DIR = BASE_DIR


# ============================================================
# 数据加载（一次性加载所有问题所需数据）
# ============================================================
def load_all_data():
    """加载所有问题所需的数据"""
    print("正在加载车辆与排货数据...")
    vehicle_costs, cold_pallets, _ = load_vehicle_data_full(VEHICLE_PATH)
    shipping_df = load_shipping_data(SHIPPING_PATH)

    # 问题1: 趟次成本（详细版）
    print("  计算问题1趟次成本...")
    trips_df, total_cost, cost_lookup = compute_trip_costs(shipping_df, vehicle_costs)

    # 问题2: 优化数据
    print("  计算问题2拼车优化...")
    trips = get_trips(shipping_df)
    cost_lookup2 = build_detailed_cost_lookup(vehicle_costs)
    original_cost, original_var, original_fixed = compute_original_cost(
        trips, cost_lookup2, breakdown=True
    )
    optimized_routes = optimize_all(trips, cost_lookup2, cold_pallets)
    optimized_cost = sum(calc_route_cost(r, cost_lookup2) for r in optimized_routes)

    # 优化后的可变/固定成本
    opt_var = 0.0
    opt_fixed = 0.0
    for r in optimized_routes:
        cities = r['cities']
        vehicle = r['vehicle']
        c = cost_lookup2.get(vehicle)
        if c is None:
            base = vehicle.replace('M', '')
            c = cost_lookup2.get(base)
        if c:
            total_dist, travel_days, ordered = _calc_route_dist_and_time(cities)
            if ordered:
                last_city = ordered[-1]
                last_coords = get_coords(last_city)
                return_dist = road_distance(
                    last_coords[0], last_coords[1],
                    NANTONG_COORDS[0], NANTONG_COORDS[1],
                    beta=get_region_beta(last_city)
                )
            else:
                return_dist = 0
            forward_dist = total_dist - return_dist
            fuel_c = c.get('fuel_cost_per_km', c['var_cost_per_km'])
            toll_c = c.get('toll_per_km', 0)
            opt_var += c['var_cost_per_km'] * forward_dist + \
                       fuel_c * EMPTY_LOAD_FUEL_RATIO * return_dist + toll_c * return_dist
            opt_fixed += c['fixed_cost_per_day'] * travel_days

    # 问题3: 仓库评估与优化
    print("  计算问题3仓库评估与优化...")
    waybill_df = load_waybill_data()
    dispatch_df = load_dispatch_data()
    df_wh = compute_warehouse_indicators(waybill_df, dispatch_df)
    S, score_names = compute_absolute_scores(df_wh)
    weights, entropy, diff_coef = entropy_weight_method(S)
    optimization = optimize_vrp_cw(dispatch_df)

    # 仓库综合得分
    wh_scores = S @ weights

    print("  数据加载完成。")
    return {
        'trips_df': trips_df,
        'total_cost': total_cost,
        'original_cost': original_cost,
        'original_var': original_var,
        'original_fixed': original_fixed,
        'optimized_routes': optimized_routes,
        'optimized_cost': optimized_cost,
        'opt_var': opt_var,
        'opt_fixed': opt_fixed,
        'cost_lookup2': cost_lookup2,
        'df_wh': df_wh,
        'S': S,
        'score_names': score_names,
        'weights': weights,
        'wh_scores': wh_scores,
        'optimization': optimization,
    }


def _calc_route_dist_and_time(cities):
    """辅助: 计算路线距离和天数"""
    if len(cities) == 0:
        return 0, 0, []
    ordered = tsp_order(NANTONG_COORDS, cities)
    total_dist = route_distance(NANTONG_COORDS, ordered)
    travel_days = estimate_travel_days(total_dist, len(ordered))
    return total_dist, travel_days, ordered


# ============================================================
# 图表1: 问题一 成本构成饼图
# ============================================================
def figure1_cost_pie(data):
    """问题1: 运输成本构成饼图（可变成本 vs 固定成本）"""
    trips_df = data['trips_df']
    total_cost = data['total_cost']
    total_var = trips_df['可变成本_元'].sum()
    total_fixed = trips_df['固定成本_元'].sum()

    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = [total_var, total_fixed]
    labels = [
        f'可变成本（油费+路桥）\n{total_var:,.0f} 元 占比 {total_var/total_cost*100:.1f}%',
        f'固定成本（人工+摊销+折旧）\n{total_fixed:,.0f} 元 占比 {total_fixed/total_cost*100:.1f}%',
    ]
    colors = ['#4472C4', '#ED7D31']
    explode = (0.02, 0.02)

    wedges, _ = ax.pie(
        sizes, explode=explode, labels=None, colors=colors,
        startangle=90, pctdistance=0.6,
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'}
    )
    ax.legend(
        wedges, labels,
        loc='center', bbox_to_anchor=(0.5, -0.06),
        ncol=1, frameon=False, fontsize=12, handletextpad=0.8
    )
    ax.text(0, 0, f'总成本\n{total_cost:,.0f} 元',
            ha='center', va='center', fontsize=15, fontweight='bold')

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, '图1_问题一_成本构成饼图.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [1/8] 成本构成饼图 → {path}')


# ============================================================
# 图表2: 问题一 各地区运输成本柱状图
# ============================================================
def figure2_city_cost_bar(data):
    """问题1: 各地区运输成本柱状图（前15城市 + 其他合并，降序排列）"""
    trips_df = data['trips_df']
    city_stats = trips_df.groupby('城市').agg(
        趟次数=('趟次成本_元', 'count'),
        总成本=('趟次成本_元', 'sum'),
    ).sort_values('总成本', ascending=False)

    N_TOP = 15
    if len(city_stats) > N_TOP:
        top15 = city_stats.iloc[:N_TOP].copy()
        other_cities = city_stats.iloc[N_TOP:]
        other_cost = other_cities['总成本'].sum()
        other_count = int(other_cities['趟次数'].sum())
        other_names_list = other_cities.index.tolist()
        # 取前3个城市名作为示例
        sample = '、'.join(other_names_list[:3])
        other_label = f'其他{len(other_cities)}市\n（含{sample}等）'
        other_row = pd.DataFrame({
            '趟次数': [other_count],
            '总成本': [other_cost],
        }, index=[other_label])
        combined = pd.concat([top15, other_row])
    else:
        combined = city_stats

    # 按总成本降序排列，barh配合invert_yaxis使最大的在顶部
    combined = combined.sort_values('总成本', ascending=False)

    cities = combined.index.tolist()
    costs = combined['总成本'].values / 10000
    trips_n = combined['趟次数'].values

    fig, ax = plt.subplots(figsize=(14, 9))
    y_pos = range(len(cities))

    # 颜色: "其他"为灰色
    bar_colors = ['#4472C4'] * len(cities)
    bar_colors[0] = '#A5A5A5'  # "其他"在反转后的第0位（即底部），不对...

    # 实际: combined最后一行是"其他", iloc[::-1]后"其他"变为第一个（顶部）
    # 所以bar_colors[0]是"其他"
    # 不对...反转后index[0]是"其他", 所以最大的是"其他"
    # 需要修正: 标记"其他"行
    is_other = ['其他' in str(c) for c in cities]
    bar_colors = ['#A5A5A5' if io else '#4472C4' for io in is_other]

    bars = ax.barh(y_pos, costs, height=0.7,
                   color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cities, fontsize=12)
    ax.set_xlabel('运输总成本（万元）', fontsize=15)
    ax.invert_yaxis()

    # 数值标签
    for i, (cost, n, io) in enumerate(zip(costs, trips_n, is_other)):
        if io:
            label = f'{cost:.1f}万元 ({int(n)}趟)'
        else:
            label = f'{cost:.1f}万元 ({int(n)}趟)'
        ax.text(cost + max(costs) * 0.012, i, label,
                va='center', fontsize=12, color='#333333')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, '图2_问题一_各城市运输成本柱状图.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [2/8] 城市成本柱状图 → {path}')


# ============================================================
# 图表3: 问题二 优化前后成本对比双柱图
# ============================================================
def figure3_optimization_comparison(data):
    """问题2: 优化前后成本对比双柱图"""
    original = data['original_cost'] / 10000
    optimized = data['optimized_cost'] / 10000
    original_var = data['original_var'] / 10000
    original_fixed = data['original_fixed'] / 10000
    opt_var = data['opt_var'] / 10000
    opt_fixed = data['opt_fixed'] / 10000

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(3)
    width = 0.28

    before_vals = [original, original_var, original_fixed]
    after_vals = [optimized, opt_var, opt_fixed]

    bars1 = ax.bar(x - width/2, before_vals, width, label='优化前',
                   color='#4472C4', edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x + width/2, after_vals, width, label='优化后',
                   color='#ED7D31', edgecolor='white', linewidth=0.8)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + max(before_vals)*0.015,
                f'{h:.1f}', ha='center', va='bottom', fontsize=13, fontweight='bold',
                color='#4472C4')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + max(before_vals)*0.015,
                f'{h:.1f}', ha='center', va='bottom', fontsize=13, fontweight='bold',
                color='#ED7D31')

    ax.set_xticks(x)
    ax.set_xticklabels(['总成本', '可变成本', '固定成本'], fontsize=15)
    ax.set_ylabel('成本（万元）', fontsize=15)
    ax.legend(fontsize=14, loc='upper right', frameon=False)

    # 节约率标注
    savings_pct = (original - optimized) / original * 100
    ax.annotate(f'总成本节约 {savings_pct:.1f}%',
                xy=(0, original), xytext=(0.9, original * 1.06),
                fontsize=14, fontweight='bold', color='#C00000',
                arrowprops=dict(arrowstyle='->', color='#C00000', lw=1.8))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, '图3_问题二_优化前后成本对比.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [3/8] 优化对比柱状图 → {path}')


# ============================================================
# 图表4: 问题三 熵权法权重分布饼图
# ============================================================
def figure4_entropy_pie(data):
    """问题3: 熵权法权重分布饼图"""
    names = data['score_names']
    weights = data['weights'] * 100

    # 按权重降序排列
    sorted_idx = np.argsort(-weights)
    names = [names[i] for i in sorted_idx]
    weights = weights[sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ['#4472C4', '#ED7D31', '#FFC000', '#5B9BD5', '#70AD47', '#A5A5A5']
    explode = [0.06 if w > 30 else 0.03 for w in weights]

    wedges, _ = ax.pie(
        weights, explode=explode, labels=None, colors=colors,
        startangle=140, pctdistance=0.6,
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'}
    )

    # 手动标注每个扇区
    bbox_props = dict(boxstyle='round,pad=0.25', facecolor='white',
                      alpha=0.9, edgecolor='#CCCCCC', linewidth=0.5)
    for i, (wedge, name, w) in enumerate(zip(wedges, names, weights)):
        ang = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
        rad = np.deg2rad(ang)
        x = np.cos(rad)
        y = np.sin(rad)
        # 大扇区标签在内，小扇区在外
        if w > 20:
            r = 0.55
        elif w > 10:
            r = 0.70
        else:
            r = 1.20
        ax.annotate(f'{name}\n{w:.1f}%',
                    xy=(x * r * 0.95, y * r * 0.95),
                    fontsize=13, ha='center', va='center',
                    fontweight='bold', bbox=bbox_props)

    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, '图4_问题三_熵权法权重饼图.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [4/8] 熵权法权重饼图 → {path}')


# ============================================================
# 图表5: 问题三 仓库综合得分排名柱状图
# ============================================================
def figure5_warehouse_scores(data):
    """问题3: 仓库综合得分排名柱状图（前15名）"""
    df_wh = data['df_wh']
    wh_scores = data['wh_scores']
    sorted_idx = np.argsort(-wh_scores)[:15]

    warehouses = [df_wh.iloc[i]['仓库编码'] for i in sorted_idx]
    scores = [wh_scores[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 7))
    x_pos = range(len(warehouses))

    # 颜色: 前三名高亮
    bar_colors = ['#C00000', '#ED7D31', '#FFC000'] + ['#4472C4'] * (len(warehouses) - 3)

    bars = ax.bar(x_pos, scores, color=bar_colors,
                  edgecolor='white', linewidth=0.5, width=0.65)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(warehouses, fontsize=15)
    ax.set_ylabel('综合得分', fontsize=15)
    ax.set_ylim(0, max(scores) * 1.13)

    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{s:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # 系统均值参考线
    sys_avg = np.average(wh_scores, weights=df_wh['车辆日数'].values)
    ax.axhline(y=sys_avg, color='#888888', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(len(warehouses) - 0.5, sys_avg + 0.6,
            f'系统加权均值 {sys_avg:.1f}', fontsize=14, color='#555555', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='none'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, '图5_问题三_仓库综合得分排名.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [5/8] 仓库得分柱状图 → {path}')


# ============================================================
# 图表6: 问题三 三层优化成本减少瀑布图
# ============================================================
def figure6_waterfall(data):
    """问题3: 三层优化成本减少瀑布图"""
    opt = data['optimization']
    orig = opt['原始成本万元']
    tsp = opt['TSP优化成本万元']
    merged = opt['合并优化成本万元']

    step1_reduction = orig - tsp    # TSP节约
    step2_reduction = tsp - merged  # 合并节约

    fig, ax = plt.subplots(figsize=(11, 7))

    categories = ['原始派车成本', 'TSP路径排序\n优化', '同城合并+\n车型匹配', '优化后成本']
    # 瀑布图逻辑: [总量, 减少量, 减少量, 总量]
    plot_vals = [orig, step1_reduction, step2_reduction, merged]
    bottoms = [0, tsp, merged, 0]
    bar_colors = ['#4472C4', '#ED7D31', '#ED7D31', '#70AD47']
    edge_colors = ['#2F5597', '#C55A11', '#C55A11', '#548235']

    x_pos = np.arange(len(categories))

    for i in range(len(categories)):
        val = plot_vals[i]
        bot = bottoms[i]
        col = bar_colors[i]
        ec = edge_colors[i]

        ax.bar(i, val, 0.5, bottom=bot, color=col, edgecolor=ec, linewidth=1.2)

        if i in [0, 3]:
            # 总量柱: 标注在顶部
            ax.text(i, val + orig * 0.015, f'{val:.1f} 万元',
                    ha='center', va='bottom', fontsize=15, fontweight='bold', color=col)
        else:
            # 减少量: 小柱文字放柱外上方，大柱放柱内居中
            pct = val / orig * 100
            if val < orig * 0.08:
                # 柱太窄，文字放上方
                ax.text(i, bot + val + orig * 0.01, f'-{val:.1f} 万元\n(节约 {pct:.1f}%)',
                        ha='center', va='bottom', fontsize=14, fontweight='bold', color=col)
            else:
                ax.text(i, bot + val / 2, f'-{val:.1f} 万元\n(节约 {pct:.1f}%)',
                        ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    # 柱间连接虚线
    connector_y = [orig, tsp]
    for i, y in enumerate(connector_y):
        ax.plot([i + 0.25, i + 0.75], [y, y], 'gray',
                linestyle='--', linewidth=1.2, alpha=0.7)

    # 总节约标注
    total_save = orig - merged
    ax.annotate(f'总节约 {total_save:.1f} 万元\n节约率 {total_save/orig*100:.1f}%',
                xy=(len(categories) - 1, merged),
                xytext=(len(categories) - 0.7, orig * 0.72),
                fontsize=14, fontweight='bold', color='#C00000',
                arrowprops=dict(arrowstyle='->', color='#C00000', lw=1.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF2CC',
                         alpha=0.9, edgecolor='#C00000'))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=14)
    ax.set_ylabel('成本（万元）', fontsize=15)
    ax.set_ylim(0, orig * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, '图6_问题三_三层优化瀑布图.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [6/8] 三层优化瀑布图 → {path}')


# ============================================================
# 图表7: 固定成本构成饼图（以固定成本为100%）
# ============================================================
def figure7_fixed_cost_pie(data):
    """固定成本明细饼图：人工、保险、审验、违章、胎耗、维保、折旧"""
    trips_df = data['trips_df']
    items = ['人工', '保险', '审验', '违章', '胎耗', '维保', '折旧']
    vals = [trips_df[col].sum() for col in items]
    total = sum(vals)
    colors = ['#4472C4', '#ED7D31', '#FFC000', '#5B9BD5', '#70AD47',
              '#A5A5A5', '#C00000']

    fig, ax = plt.subplots(figsize=(8, 7))
    # 小于2%的合并为"其他"
    threshold = 0.02
    main_items, main_vals, main_colors = [], [], []
    other_val, other_names = 0.0, []
    for name, v, c in zip(items, vals, colors):
        if v / total >= threshold:
            main_items.append(name)
            main_vals.append(v)
            main_colors.append(c)
        else:
            other_val += v
            other_names.append(name)
    if other_val > 0:
        main_items.append(f'其他\n（{"+".join(other_names)}）')
        main_vals.append(other_val)
        main_colors.append('#CCCCCC')

    wedges, _ = ax.pie(
        main_vals, labels=None, colors=main_colors,
        startangle=90, pctdistance=0.6,
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'}
    )
    bbox_props = dict(boxstyle='round,pad=0.25', facecolor='white',
                      alpha=0.9, edgecolor='#CCCCCC', linewidth=0.5)
    for i, (wedge, name, v) in enumerate(zip(wedges, main_items, main_vals)):
        ang = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
        rad = np.deg2rad(ang)
        r = 0.55 if v / total > 0.15 else 0.75
        ax.annotate(f'{name}\n{v:,.0f} 元\n({v/total*100:.1f}%)',
                    xy=(r * np.cos(rad), r * np.sin(rad)),
                    fontsize=12, ha='center', va='center',
                    fontweight='bold', bbox=bbox_props)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, '图7_固定成本构成饼图.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [7/8] 固定成本构成饼图 → {path}')


# ============================================================
# 图表8: 可变成本构成饼图（以可变成本为100%）
# ============================================================
def figure8_var_cost_pie(data):
    """可变成本明细饼图：去程油费、回程油费、路桥费"""
    trips_df = data['trips_df']
    items = ['去程油费', '回程油费', '路桥费']
    vals = [
        trips_df['油费_去程'].sum(),
        trips_df['油费_回程'].sum(),
        trips_df['路桥费_总'].sum(),
    ]
    total = sum(vals)
    colors = ['#4472C4', '#5B9BD5', '#ED7D31']

    fig, ax = plt.subplots(figsize=(8, 7))
    wedges, _ = ax.pie(
        vals, labels=None, colors=colors,
        startangle=90, pctdistance=0.6, explode=(0.03, 0.03, 0.03),
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'}
    )
    bbox_props = dict(boxstyle='round,pad=0.25', facecolor='white',
                      alpha=0.9, edgecolor='#CCCCCC', linewidth=0.5)
    for i, (wedge, name, v) in enumerate(zip(wedges, items, vals)):
        ang = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
        rad = np.deg2rad(ang)
        ax.annotate(f'{name}\n{v:,.0f} 元\n({v/total*100:.1f}%)',
                    xy=(0.58 * np.cos(rad), 0.58 * np.sin(rad)),
                    fontsize=13, ha='center', va='center',
                    fontweight='bold', bbox=bbox_props)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, '图8_可变成本构成饼图.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [8/8] 可变成本构成饼图 → {path}')


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 60)
    print("  数据可视化 — 生成论文图表")
    print("=" * 60)

    data = load_all_data()

    print("\n正在生成图表...")
    figure1_cost_pie(data)
    figure2_city_cost_bar(data)
    figure3_optimization_comparison(data)
    figure4_entropy_pie(data)
    figure5_warehouse_scores(data)
    figure6_waterfall(data)
    figure7_fixed_cost_pie(data)
    figure8_var_cost_pie(data)

    print("\n" + "=" * 60)
    print("  全部8张图表生成完成！")
    print("  输出目录:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
