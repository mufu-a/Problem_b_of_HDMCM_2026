#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医药冷链物流拼车优化模型 - 问题2
基于Clarke-Wright节约算法与方向/距离可行性约束，对附件2排货数据进行拼单运输优化
"""

import math
import pandas as pd
import numpy as np
from collections import defaultdict

from config import (
    VEHICLE_PATH, SHIPPING_PATH, ROAD_FACTOR, NANTONG_COORDS, CITY_COORDS,
    AVG_SPEED_HIGHWAY, DRIVE_HOURS_PER_DAY, WORK_START, WORK_END, WORK_HOURS,
    LOAD_UNLOAD_HOURS, EMPTY_LOAD_FUEL_RATIO,
    CW_MAX_DEVIATION, CW_MAX_INTER_CITY_KM, CW_MAX_GROUP_BEARING_RANGE,
    CW_CROSS_WEEK_MERGE, CW_CROSS_WEEK_MAX_GAP,
)
from utils import (
    load_vehicle_data_full, build_detailed_cost_lookup, load_shipping_data,
    get_trips, compute_original_cost,
    compute_bearing, inter_city_distance, road_distance,
    road_distance_for_city, get_region_beta, get_coords,
    tsp_order, route_distance, estimate_travel_days,
    select_best_vehicle,
)


# ============================================================
# 1. Clarke-Wright 节约值计算
# ============================================================
def compute_savings(city_a, city_b, cost_lookup, pallets_a, pallets_b):
    """计算合并city_a和city_b的精确节约值（详细成本模型 + 空载回程系数）"""
    ca = CITY_COORDS[city_a]
    cb = CITY_COORDS[city_b]
    d_0a = road_distance_for_city(NANTONG_COORDS[0], NANTONG_COORDS[1],
                                   ca[0], ca[1], city_a)
    d_0b = road_distance_for_city(NANTONG_COORDS[0], NANTONG_COORDS[1],
                                   cb[0], cb[1], city_b)
    d_ab = inter_city_distance(city_a, city_b)

    def single_trip_cost(dist, pallets):
        v = select_best_vehicle(pallets, dist * 2, cost_lookup)
        c = cost_lookup.get(v, cost_lookup.get('7.6M'))
        if c is None:
            c = cost_lookup.get('7.6')
        travel_days = estimate_travel_days(dist * 2, 1)
        # 去程(满载) + 回程(空载, 油耗×空载系数, 路桥不变)
        fuel = c.get('fuel_cost_per_km', c['var_cost_per_km'])
        toll = c.get('toll_per_km', 0)
        var_out = c['var_cost_per_km'] * dist
        var_return = fuel * EMPTY_LOAD_FUEL_RATIO * dist + toll * dist
        var_cost = var_out + var_return
        fixed_cost = c['fixed_cost_per_day'] * travel_days
        return var_cost + fixed_cost

    orig_a = single_trip_cost(d_0a, pallets_a)
    orig_b = single_trip_cost(d_0b, pallets_b)

    # 合并后: 南通→a→b→南通
    merged_dist = d_0a + d_ab + d_0b
    merged_pallets = pallets_a + pallets_b
    vm = select_best_vehicle(merged_pallets, merged_dist, cost_lookup)
    cm = cost_lookup.get(vm, cost_lookup.get('9.6M'))
    if cm is None:
        cm = cost_lookup.get('9.6')
    travel_days = estimate_travel_days(merged_dist, 2)

    # 合并变量成本: forward (d_0a+d_ab) loaded + return (d_0b) empty
    fuel = cm.get('fuel_cost_per_km', cm['var_cost_per_km'])
    toll = cm.get('toll_per_km', 0)
    forward_dist = d_0a + d_ab
    return_dist = d_0b
    var_cost = cm['var_cost_per_km'] * forward_dist + \
               fuel * EMPTY_LOAD_FUEL_RATIO * return_dist + toll * return_dist
    fixed_cost = cm['fixed_cost_per_day'] * travel_days
    merged_cost = var_cost + fixed_cost

    return orig_a + orig_b - merged_cost


def check_direction_feasibility(city_a, city_b, max_deviation=90):
    """检查两城市是否在同一大方向上"""
    if city_a not in CITY_COORDS or city_b not in CITY_COORDS:
        return False
    bearing_a = compute_bearing(CITY_COORDS[city_a][0], CITY_COORDS[city_a][1])
    bearing_b = compute_bearing(CITY_COORDS[city_b][0], CITY_COORDS[city_b][1])
    diff = abs(bearing_a - bearing_b)
    if diff > 180:
        diff = 360 - diff
    return diff <= max_deviation


def check_distance_feasibility(city_a, city_b, max_inter_km=500):
    """城市间道路距离不超过max_inter_km，配合方向约束防止不合理的远距离合并"""
    d_ab = inter_city_distance(city_a, city_b)
    return d_ab <= max_inter_km


# ============================================================
# 2. 超容量拆分
# ============================================================
def _split_oversize(city, total_pallets, dist, cost_lookup):
    """超容量拆分: 用9.6M满载 + 余量"""
    routes = []
    n_full = int(total_pallets // 14)
    remainder = total_pallets % 14
    for _ in range(n_full):
        routes.append({'cities': [city], 'total_pallets': 14, 'vehicle': '9.6M'})
    if remainder > 0:
        v = select_best_vehicle(remainder, dist, cost_lookup)
        routes.append({'cities': [city], 'total_pallets': remainder, 'vehicle': v})
    return routes


# ============================================================
# 3. 单周 CW 节约优化
# ============================================================
def optimize_week_cw(week_cities, city_pallets, cost_lookup,
                      max_deviation=None, max_inter_km=None, max_group_range=None):
    """
    对一组城市执行CW节约算法优化。
    返回优化路线列表。
    """
    if max_deviation is None:
        max_deviation = CW_MAX_DEVIATION
    if max_inter_km is None:
        max_inter_km = CW_MAX_INTER_CITY_KM
    if max_group_range is None:
        max_group_range = CW_MAX_GROUP_BEARING_RANGE

    cities = [c for c in week_cities if c in city_pallets and c in CITY_COORDS]
    if len(cities) <= 1:
        routes = []
        for c in cities:
            pallets = city_pallets[c]
            cc = CITY_COORDS[c]
            # 单程距离（使用城市区域 β），往返距离 = 单程 × 2
            d_oneway = road_distance_for_city(NANTONG_COORDS[0], NANTONG_COORDS[1],
                                                cc[0], cc[1], c)
            d = d_oneway * 2
            v = select_best_vehicle(pallets, d, cost_lookup)
            if pallets > 14:
                routes.extend(_split_oversize(c, pallets, d, cost_lookup))
            else:
                routes.append({'cities': [c], 'total_pallets': pallets, 'vehicle': v})
        return routes

    # 计算所有城市对的节约值
    savings = []
    for i, ci in enumerate(cities):
        for cj in cities[i+1:]:
            if not check_direction_feasibility(ci, cj, max_deviation=max_deviation):
                continue
            if not check_distance_feasibility(ci, cj, max_inter_km=max_inter_km):
                continue
            s = compute_savings(ci, cj, cost_lookup,
                               city_pallets.get(ci, 0), city_pallets.get(cj, 0))
            if s > 0:
                savings.append((s, ci, cj))

    savings.sort(key=lambda x: -x[0])

    # Union-Find 合并
    parent = {c: c for c in cities}
    pallets = dict(city_pallets)
    bearings = {c: compute_bearing(CITY_COORDS[c][0], CITY_COORDS[c][1]) for c in cities}

    def find(c):
        while parent.get(c, c) != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    def get_merged_pallets(root):
        return sum(pallets.get(c, 0) for c in cities if find(c) == root)

    for s, ci, cj in savings:
        ri, rj = find(ci), find(cj)
        if ri == rj:
            continue
        total = get_merged_pallets(ri) + get_merged_pallets(rj)
        if total > 14:
            continue
        # 方位角跨度检查
        rj_bearings = [bearings[c] for c in cities if find(c) == rj]
        all_b = [bearings[c] for c in cities if find(c) == ri] + rj_bearings
        bs_sorted = sorted(all_b)
        max_gap = 0
        for i in range(len(bs_sorted)):
            gap = (bs_sorted[(i + 1) % len(bs_sorted)] - bs_sorted[i]) % 360
            if gap > max_gap:
                max_gap = gap
        combined_range = 360 - max_gap
        if combined_range > max_group_range:
            continue
        parent[rj] = ri

    # 收集结果
    components = defaultdict(list)
    for c in cities:
        components[find(c)].append(c)

    routes = []
    for root, comp_cities in components.items():
        total_p = sum(city_pallets.get(c, 0) for c in comp_cities)
        ordered = tsp_order(NANTONG_COORDS, comp_cities)
        route_dist = route_distance(NANTONG_COORDS, ordered)
        v = select_best_vehicle(total_p, route_dist, cost_lookup)
        routes.append({
            'cities': comp_cities,
            'total_pallets': total_p,
            'vehicle': v,
        })

    return routes


# ============================================================
# 4. 路线成本计算
# ============================================================
def calc_route_distance_and_time(route_cities):
    """计算多站路线的总距离和运输天数"""
    if len(route_cities) == 0:
        return 0, 0, []

    ordered = tsp_order(NANTONG_COORDS, route_cities)
    total_dist = route_distance(NANTONG_COORDS, ordered)
    travel_days = estimate_travel_days(total_dist, len(ordered))

    return total_dist, travel_days, ordered


def calc_route_cost(route, cost_lookup):
    """计算单条路线的总成本（详细成本模型 + 空载回程系数）"""
    cities = route['cities']
    total_dist, travel_days, ordered = calc_route_distance_and_time(cities)

    vehicle = route['vehicle']
    c = cost_lookup.get(vehicle)
    if c is None:
        base = vehicle.replace('M', '')
        c = cost_lookup.get(base)
    if c is None:
        var_c = 1.70 if '4.2' in str(vehicle) else (2.40 if '7.6' in str(vehicle) else 3.00)
        fixed_c = 1040.98 if '4.2' in str(vehicle) else (1090.32 if '7.6' in str(vehicle) else 1090.97)
        fuel_c = var_c
        toll_c = 0
    else:
        var_c = c['var_cost_per_km']
        fixed_c = c['fixed_cost_per_day']
        fuel_c = c.get('fuel_cost_per_km', var_c)
        toll_c = c.get('toll_per_km', 0)

    # 计算回程距离（最后城市→南通，空载段）
    if ordered:
        last_city = ordered[-1]
        last_coords = get_coords(last_city)
        return_dist = road_distance(last_coords[0], last_coords[1],
                                     NANTONG_COORDS[0], NANTONG_COORDS[1],
                                     beta=get_region_beta(last_city))
    else:
        return_dist = 0

    forward_dist = total_dist - return_dist
    # 可变成本: forward loaded + return empty
    var_cost = var_c * forward_dist + fuel_c * EMPTY_LOAD_FUEL_RATIO * return_dist + toll_c * return_dist
    fixed_cost = fixed_c * travel_days

    return var_cost + fixed_cost


# ============================================================
# 5. 主优化流程
# ============================================================
def optimize_all(trips, cost_lookup, cold_pallets,
                  max_deviation=None, max_inter_km=None, max_group_range=None,
                  cross_week=None, cross_week_gap=None):
    """按自然周分组 + CW节约优化，支持跨周合并"""
    if max_deviation is None:
        max_deviation = CW_MAX_DEVIATION
    if max_inter_km is None:
        max_inter_km = CW_MAX_INTER_CITY_KM
    if max_group_range is None:
        max_group_range = CW_MAX_GROUP_BEARING_RANGE
    if cross_week is None:
        cross_week = CW_CROSS_WEEK_MERGE
    if cross_week_gap is None:
        cross_week_gap = CW_CROSS_WEEK_MAX_GAP

    weeks = sorted(trips['提货周'].unique())
    all_routes = []

    if not cross_week:
        # 严格按周分组
        for week in weeks:
            week_trips = trips[trips['提货周'] == week]
            if len(week_trips) == 0:
                continue
            city_pallets = week_trips.groupby('城市')['总托盘数'].sum().to_dict()
            cities_with_data = [c for c in city_pallets if c in CITY_COORDS]
            if not cities_with_data:
                continue
            routes = optimize_week_cw(cities_with_data, city_pallets, cost_lookup,
                                      max_deviation, max_inter_km, max_group_range)
            for r in routes:
                r['周'] = week
            all_routes.extend(routes)
    else:
        # 跨周合并: 为每个(城市, 周)分配唯一标签, 加入临时坐标后执行CW
        used_entities = set()
        week_city_pallets = {}
        for week in weeks:
            wt = trips[trips['提货周'] == week]
            week_city_pallets[week] = wt.groupby('城市')['总托盘数'].sum().to_dict()

        for i, week in enumerate(weeks):
            # 收集当前周及相邻周的 (城市, 周) 实体
            entity_map = {}  # label -> (city, week)
            temp_pallets = {}
            for gap in range(cross_week_gap + 1):
                neighbor_week = week + gap
                if neighbor_week in week_city_pallets:
                    for city, pallets in week_city_pallets[neighbor_week].items():
                        ekey = (city, neighbor_week)
                        if city in CITY_COORDS and ekey not in used_entities:
                            label = f"{city}__W{neighbor_week}"
                            entity_map[label] = (city, neighbor_week)
                            temp_pallets[label] = pallets

            if not temp_pallets:
                continue

            # 临时扩充 CITY_COORDS 以支持跨周标签的坐标查找
            # 注意: 此处有意修改全局 CITY_COORDS，函数结束前会清理恢复
            saved = {}
            for label, (city, _) in entity_map.items():
                saved[label] = CITY_COORDS[city]
            CITY_COORDS.update(saved)

            routes = optimize_week_cw(list(temp_pallets.keys()), temp_pallets,
                                      cost_lookup,
                                      max_deviation, max_inter_km, max_group_range)

            for r in routes:
                # 还原城市名并标记已使用
                real_cities = []
                for c in r['cities']:
                    city, ew = entity_map.get(c, (c, week))
                    real_cities.append(city)
                    used_entities.add((city, ew))
                r['cities'] = real_cities
                r['周'] = week

            for label in saved:
                del CITY_COORDS[label]

            all_routes.extend(routes)

    return all_routes


# ============================================================
# 6. 输出结果
# ============================================================
def print_results(original_cost, optimized_routes, cost_lookup, trips,
                   original_var=None, original_fixed=None):
    """打印优化结果汇总"""
    optimized_cost = sum(calc_route_cost(r, cost_lookup) for r in optimized_routes)
    savings = original_cost - optimized_cost
    savings_pct = savings / original_cost * 100 if original_cost > 0 else 0

    print("=" * 70)
    print("          医药冷链物流拼车优化分析 (问题2)")
    print("=" * 70)

    print(f"\n{'—' * 50}")
    print("【优化前】")
    print(f"  运输趟次总数: {len(trips)}")
    print(f"  运输总费用:   {original_cost:,.2f} 元")
    if original_var is not None and original_fixed is not None:
        print(f"  可变成本:     {original_var:,.2f} 元 ({original_var/original_cost*100:.1f}%)")
        print(f"  固定成本:     {original_fixed:,.2f} 元 ({original_fixed/original_cost*100:.1f}%)")

    print(f"\n{'—' * 50}")
    print("【优化后】")
    original_trip_count = len(trips)
    print(f"  拼车路线数:   {len(optimized_routes)}")
    print(f"  运输总费用:   {optimized_cost:,.2f} 元")
    print(f"  趟次减少:     {original_trip_count - len(optimized_routes)} "
          f"({((original_trip_count - len(optimized_routes)) / original_trip_count * 100):.1f}%)")

    print(f"\n{'=' * 50}")
    print(f"  ★ 成本节约:   {savings:,.2f} 元 ({savings_pct:.1f}%)")
    print(f"{'=' * 50}")

    # 按周统计
    print(f"\n{'—' * 50}")
    print("【按周统计】")
    week_stats = defaultdict(lambda: {'routes': 0, 'cost': 0, 'cities': set()})
    for r in optimized_routes:
        w = r.get('周', '?')
        week_stats[w]['routes'] += 1
        week_stats[w]['cost'] += calc_route_cost(r, cost_lookup)
        week_stats[w]['cities'].update(r['cities'])

    print(f"  {'周':<8s} {'路线数':>6s} {'成本(元)':>16s} {'覆盖城市':>10s}")
    print(f"  {'-'*48}")
    for sname in sorted(week_stats.keys(), key=str):
        st = week_stats[sname]
        print(f"  {'周'+str(sname):<8s} {st['routes']:>6d} {st['cost']:>16,.2f} {len(st['cities']):>10d}")

    # 示例路线
    print(f"\n{'—' * 50}")
    print("【优化路线示例】(前10条)")
    example_routes = sorted(optimized_routes, key=lambda r: len(r['cities']), reverse=True)[:10]
    for i, r in enumerate(example_routes):
        _, days, ordered = calc_route_distance_and_time(r['cities'])
        cost = calc_route_cost(r, cost_lookup)
        print(f"  {i+1}. [周{r.get('周','?')}] 车辆:{r['vehicle']} 城市:{' → '.join(ordered)} "
              f"托盘:{r['total_pallets']:.1f} 天数:{days} 成本:{cost:,.2f}元")

    # 成本构成
    total_var = 0
    total_fixed = 0
    for r in optimized_routes:
        cities = r['cities']
        total_dist, travel_days, ordered = calc_route_distance_and_time(cities)
        vehicle = r['vehicle']
        c = cost_lookup.get(vehicle)
        if c is None:
            base = vehicle.replace('M', '')
            c = cost_lookup.get(base)
        if c:
            fuel_c = c.get('fuel_cost_per_km', c['var_cost_per_km'])
            toll_c = c.get('toll_per_km', 0)
            # 回程距离（最后城市→南通，空载段）
            if ordered:
                last_city = ordered[-1]
                last_coords = get_coords(last_city)
                return_dist = road_distance(last_coords[0], last_coords[1],
                                             NANTONG_COORDS[0], NANTONG_COORDS[1],
                                             beta=get_region_beta(last_city))
            else:
                return_dist = 0
            forward_dist = total_dist - return_dist
            total_var += c['var_cost_per_km'] * forward_dist + \
                         fuel_c * EMPTY_LOAD_FUEL_RATIO * return_dist + toll_c * return_dist
            total_fixed += c['fixed_cost_per_day'] * travel_days
        else:
            var_c = 1.70 if '4.2' in str(vehicle) else (2.40 if '7.6' in str(vehicle) else 3.00)
            fixed_c = 1040.98 if '4.2' in str(vehicle) else (1090.32 if '7.6' in str(vehicle) else 1090.97)
            total_var += var_c * total_dist
            total_fixed += fixed_c * travel_days

    print(f"\n{'—' * 50}")
    print("【优化后成本构成】")
    print(f"  可变成本: {total_var:,.2f} 元 ({total_var/optimized_cost*100:.1f}%)")
    print(f"  固定成本: {total_fixed:,.2f} 元 ({total_fixed/optimized_cost*100:.1f}%)")

    return optimized_cost, savings, savings_pct


# ============================================================
# 7. 网格搜索
# ============================================================
def grid_search(trips, cost_lookup, cold_pallets):
    """参数网格搜索，找出最优参数组合"""
    param_grid = {
        'max_deviation': [90, 120, 150],
        'max_inter_km': [200, 300, 400, 500],
        'max_group_range': [90, 120, 150],
        'cross_week': [False, True],
    }

    original_cost = compute_original_cost(trips, cost_lookup)
    results = []

    for max_dev in param_grid['max_deviation']:
        for max_km in param_grid['max_inter_km']:
            for max_range in param_grid['max_group_range']:
                for cross_w in param_grid['cross_week']:
                    routes = optimize_all(
                        trips, cost_lookup, cold_pallets,
                        max_deviation=max_dev,
                        max_inter_km=max_km,
                        max_group_range=max_range,
                        cross_week=cross_w,
                        cross_week_gap=1,
                    )
                    opt_cost = sum(calc_route_cost(r, cost_lookup) for r in routes)
                    savings = original_cost - opt_cost
                    savings_pct = savings / original_cost * 100
                    results.append({
                        'max_deviation': max_dev,
                        'max_inter_km': max_km,
                        'max_group_range': max_range,
                        'cross_week': cross_w,
                        'routes': len(routes),
                        'cost': opt_cost,
                        'savings': savings,
                        'savings_pct': savings_pct,
                    })

    results.sort(key=lambda x: -x['savings_pct'])

    print("=" * 80)
    print("  网格搜索结果 (按节约率降序, 前20)")
    print("=" * 80)
    print(f"  {'偏离':>5s} {'间距':>5s} {'跨度':>5s} {'跨周':>5s} "
          f"{'路线':>5s} {'成本(元)':>14s} {'节约(元)':>14s} {'节约率':>8s}")
    print(f"  {'-'*65}")
    for r in results[:20]:
        print(f"  {r['max_deviation']:>5d} {r['max_inter_km']:>5d} "
              f"{r['max_group_range']:>5d} {str(r['cross_week']):>5s} "
              f"{r['routes']:>5d} {r['cost']:>14,.2f} {r['savings']:>14,.2f} "
              f"{r['savings_pct']:>7.2f}%")

    best = results[0]
    print(f"\n{'=' * 50}")
    print(f"  ★ 最优参数组合:")
    print(f"    max_deviation      = {best['max_deviation']}")
    print(f"    max_inter_km       = {best['max_inter_km']}")
    print(f"    max_group_range    = {best['max_group_range']}")
    print(f"    cross_week         = {best['cross_week']}")
    print(f"    路线数: {best['routes']}")
    print(f"    优化成本: {best['cost']:,.2f} 元")
    print(f"    节约:     {best['savings']:,.2f} 元 ({best['savings_pct']:.2f}%)")
    print(f"{'=' * 50}")

    return best, results


# ============================================================
# 8. 主程序
# ============================================================
def main(run_grid=False):
    print("正在读取车辆成本与承载能力...")
    vehicle_costs, cold_pallets, _ = load_vehicle_data_full(VEHICLE_PATH)
    cost_lookup = build_detailed_cost_lookup(vehicle_costs)
    print(f"  冷藏车容量: 4.2M={cold_pallets['4.2']}托, 7.6M={cold_pallets['7.6']}托, "
          f"9.6M={cold_pallets['9.6']}托")

    print("正在读取排货数据...")
    shipping_df = load_shipping_data(SHIPPING_PATH)
    print(f"  读取到 {len(shipping_df)} 条有效记录")

    print("正在构建趟次数据...")
    trips = get_trips(shipping_df)
    print(f"  生成 {len(trips)} 个运输趟次, 覆盖 {trips['城市'].nunique()} 个城市")
    print(f"  时间跨度: {trips['提货dt'].min().date()} 至 {trips['提货dt'].max().date()}")
    print(f"  跨越 {trips['提货周'].nunique()} 个自然周")

    print("正在计算原始成本...")
    original_cost, original_var, original_fixed = compute_original_cost(trips, cost_lookup, breakdown=True)
    print(f"  原始总成本: {original_cost:,.2f} 元")
    print(f"  其中可变成本: {original_var:,.2f} 元, 固定成本: {original_fixed:,.2f} 元")

    if run_grid:
        print("\n正在执行参数网格搜索...")
        grid_search(trips, cost_lookup, cold_pallets)
        return trips, original_cost, [], original_cost

    print("正在执行方向聚类与拼车优化...")
    print(f"  CW参数: 偏离≤{CW_MAX_DEVIATION}° 间距≤{CW_MAX_INTER_CITY_KM}km "
          f"跨度≤{CW_MAX_GROUP_BEARING_RANGE}° 跨周={CW_CROSS_WEEK_MERGE}")
    optimized_routes = optimize_all(trips, cost_lookup, cold_pallets)

    print("\n")
    print_results(original_cost, optimized_routes, cost_lookup, trips,
                  original_var=original_var, original_fixed=original_fixed)
    optimized_cost = sum(calc_route_cost(r, cost_lookup) for r in optimized_routes)

    return trips, original_cost, optimized_routes, optimized_cost, original_var, original_fixed


if __name__ == '__main__':
    import sys
    run_grid = '--grid' in sys.argv
    trips, original_cost, optimized_routes, optimized_cost, original_var, original_fixed = main(run_grid=run_grid)
