#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医药冷链物流运输成本模型 - 问题1 (改进版)
改进内容:
  1. 往返运输 + 空载油耗系数 (非简单×2)
  2. 车辆折旧 (车价-报废补贴) / 4年 / 工作天数
  3. 工作天数区分: 人工/胎耗/维保/折旧使用年工作天数, 保险/审验/违章使用日历天
  4. 东/中/西部差异化道路绕行系数 β
"""

import pandas as pd
import numpy as np

from config import (
    VEHICLE_PATH, SHIPPING_PATH, NANTONG_COORDS, CITY_COORDS,
    EMPTY_LOAD_FUEL_RATIO, WORKING_DAYS_PER_YEAR, MAINTENANCE_DAYS_PER_YEAR,
    CALENDAR_DAYS_PER_YEAR, DEPRECIATION_YEARS,
    REGION_BETA,
)
from utils import (
    load_vehicle_costs, load_shipping_data,
    extract_city, haversine_distance,
    get_region_beta,
)


# ============================================================
# 1. 运输趟次分组与成本计算 (改进版)
# ============================================================
def compute_trip_costs(shipping_df, vehicle_costs):
    """对排货数据进行趟次分组，计算每趟往返成本，返回(趟次DataFrame, 总成本, 成本查找表)"""

    df = shipping_df.copy()

    # Step 1: 按运输交接单号分组，向前填充车型/时效/车辆数目
    for col in ['车型', '运输时效', '车辆数目']:
        df[col] = df.groupby('运输交接单号')[col].ffill()
    df['车辆数目'] = df['车辆数目'].fillna(1).astype(int)
    df['运输时效'] = df['运输时效'].fillna(1)

    # Step 2: 按 sub-trip 分组聚合
    trip_groups = df.groupby(
        ['运输交接单号', '车型', '车辆数目', '收货方地址', '运输时效'],
        dropna=False
    ).agg(
        总盒数=('盒数', 'sum'),
        总箱数=('箱数', 'sum'),
        总托盘数=('托盘数', 'sum'),
        记录数=('盒数', 'count')
    ).reset_index()

    # Step 3: 过滤无效车型
    trip_groups = trip_groups[trip_groups['车型'].notna()].copy()
    trip_groups = trip_groups[trip_groups['车型'].isin(['7.6M', '9.6M', '4.2M'])].copy()

    # Step 4: 提取城市、计算距离
    trip_groups['城市'] = trip_groups['收货方地址'].apply(extract_city)

    unmapped = trip_groups[trip_groups['城市'].isna()]['收货方地址'].unique()
    if len(unmapped) > 0:
        print(f"警告: 以下 {len(unmapped)} 个地址无法匹配城市:")
        for addr in unmapped:
            print(f"  - {addr}")

    trip_groups = trip_groups[trip_groups['城市'].notna()].copy()

    trip_groups['纬度'] = trip_groups['城市'].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
    trip_groups['经度'] = trip_groups['城市'].map(lambda c: CITY_COORDS.get(c, (None, None))[1])

    missing_coords = trip_groups[trip_groups['纬度'].isna()]['城市'].unique()
    if len(missing_coords) > 0:
        print(f"警告: 以下 {len(missing_coords)} 个城市缺少坐标:")
        for c in missing_coords:
            print(f"  - {c}")
    trip_groups = trip_groups[trip_groups['纬度'].notna()].copy()

    # Step 5: 计算距离 (直线距离 × 区域绕行系数 β)
    trip_groups['区域'] = trip_groups['城市'].apply(get_region_beta)  # 暂存 β 值

    def calc_distance(row):
        straight = haversine_distance(NANTONG_COORDS[0], NANTONG_COORDS[1],
                                      row['纬度'], row['经度'])
        return straight * row['区域']

    trip_groups['单程道路距离_km'] = trip_groups.apply(calc_distance, axis=1)
    trip_groups['直线距离_km'] = trip_groups.apply(
        lambda row: haversine_distance(NANTONG_COORDS[0], NANTONG_COORDS[1],
                                       row['纬度'], row['经度']), axis=1
    )

    # Step 6: 构建改进版成本查找表
    cost_lookup = {}
    for vtype, params in vehicle_costs.items():
        fuel_cost = params['fuel_cost_per_km']
        toll_cost = params['toll_per_km']
        var_cost_per_km = fuel_cost + toll_cost

        # 固定成本: 区分工作天与日历天
        labor = params['labor_daily']                                              # 已为每日
        insurance = params['insurance_annual'] / CALENDAR_DAYS_PER_YEAR           # 日历天
        inspection = (params['inspection_annual'] + params['road_inspection_annual']) / CALENDAR_DAYS_PER_YEAR
        violation = params['violation_monthly'] / 30.0                            # 日历月
        tire = params['tire_cost'] / WORKING_DAYS_PER_YEAR                        # 工作天
        maintenance = params['maintenance_monthly'] * 12 / MAINTENANCE_DAYS_PER_YEAR  # 工作天
        depreciation = ((params['vehicle_price'] - params['scrap_subsidy'])
                       / DEPRECIATION_YEARS / WORKING_DAYS_PER_YEAR)               # 工作天

        fixed_daily = labor + insurance + inspection + violation + tire + maintenance + depreciation

        cost_lookup[vtype] = {
            'var_cost_per_km':    var_cost_per_km,
            'fuel_cost_per_km':   fuel_cost,
            'toll_per_km':        toll_cost,
            'fixed_cost_per_day': fixed_daily,
            # 成本明细 (用于输出)
            'labor_daily':        labor,
            'insurance_daily':    insurance,
            'inspection_daily':   inspection,
            'violation_daily':    violation,
            'tire_daily':         tire,
            'maintenance_daily':  maintenance,
            'depreciation_daily': depreciation,
        }
    for base in ['4.2', '7.6', '9.6']:
        if base in cost_lookup:
            cost_lookup[base + 'M'] = cost_lookup[base]

    # Step 7: 计算每趟往返成本
    def calc_trip_cost(row):
        vtype = str(row['车型']).strip()
        if vtype not in cost_lookup:
            return 0.0
        c = cost_lookup[vtype]
        d = row['单程道路距离_km']
        n = row['车辆数目']
        t = row['运输时效']

        # 可变成本: 去程(满载) + 回程(空载, 油耗×空载系数, 路桥不变)
        var_out = c['var_cost_per_km'] * d
        var_return = c['fuel_cost_per_km'] * EMPTY_LOAD_FUEL_RATIO * d + c['toll_per_km'] * d
        var_part = (var_out + var_return) * n

        # 固定成本: 往返天数 ≈ 2 × 运输时效
        round_trip_days = t * 2
        fixed_part = c['fixed_cost_per_day'] * round_trip_days * n

        return var_part + fixed_part

    trip_groups['趟次成本_元'] = trip_groups.apply(calc_trip_cost, axis=1)

    # 可变成本明细
    def calc_var_cost(row):
        vtype = str(row['车型']).strip()
        if vtype not in cost_lookup:
            return 0.0
        c = cost_lookup[vtype]
        d = row['单程道路距离_km']
        n = row['车辆数目']
        var_out = c['var_cost_per_km'] * d
        var_return = c['fuel_cost_per_km'] * EMPTY_LOAD_FUEL_RATIO * d + c['toll_per_km'] * d
        return (var_out + var_return) * n

    trip_groups['可变成本_元'] = trip_groups.apply(calc_var_cost, axis=1)

    def calc_fixed_cost(row):
        vtype = str(row['车型']).strip()
        if vtype not in cost_lookup:
            return 0.0
        c = cost_lookup[vtype]
        return c['fixed_cost_per_day'] * row['运输时效'] * 2 * row['车辆数目']

    trip_groups['固定成本_元'] = trip_groups.apply(calc_fixed_cost, axis=1)

    # 成本子项明细（用于敏感性分析和可视化）
    def calc_cost_details(row):
        vtype = str(row['车型']).strip()
        if vtype not in cost_lookup:
            c = None
        else:
            c = cost_lookup[vtype]
        d = row['单程道路距离_km']
        n = row['车辆数目']
        t = row['运输时效']
        rt = t * 2  # 往返天数
        if c is None:
            return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return (
            c['fuel_cost_per_km'] * d * n,                                    # 油费_去程
            c['fuel_cost_per_km'] * EMPTY_LOAD_FUEL_RATIO * d * n,            # 油费_回程
            c['toll_per_km'] * d * n,                                         # 路桥费_去程
            c['toll_per_km'] * d * n,                                         # 路桥费_回程
            c['labor_daily'] * rt * n,                                        # 人工
            c['insurance_daily'] * rt * n,                                    # 保险
            c['inspection_daily'] * rt * n,                                   # 审验
            c['violation_daily'] * rt * n,                                    # 违章
            c['tire_daily'] * rt * n,                                         # 胎耗
            c['maintenance_daily'] * rt * n,                                  # 维保
            c['depreciation_daily'] * rt * n,                                 # 折旧
        )

    details = trip_groups.apply(calc_cost_details, axis=1, result_type='expand')
    details.columns = ['油费_去程', '油费_回程', '路桥费_去程', '路桥费_回程',
                       '人工', '保险', '审验', '违章', '胎耗', '维保', '折旧']
    for col in details.columns:
        trip_groups[col] = details[col].values
    trip_groups['油费_总'] = trip_groups['油费_去程'] + trip_groups['油费_回程']
    trip_groups['路桥费_总'] = trip_groups['路桥费_去程'] + trip_groups['路桥费_回程']

    total_cost = trip_groups['趟次成本_元'].sum()

    return trip_groups, total_cost, cost_lookup


# ============================================================
# 2. 输出结果 (改进版)
# ============================================================
def print_summary(trips_df, total_cost, cost_lookup, vehicle_costs):
    """打印成本分析汇总"""
    print("=" * 70)
    print("    医药冷链物流运输成本分析 (问题1 — 改进版)")
    print("    改进: 往返+空载系数 | 折旧 | 工作天区分 | 区域β")
    print("=" * 70)

    print(f"\n{'—' * 50}")
    print("【数据概览】")
    print(f"  有效运输趟次: {len(trips_df)}")
    print(f"  涉及城市数:   {trips_df['城市'].nunique()}")
    print(f"  使用车型:     {', '.join(sorted(trips_df['车型'].unique()))}")

    print(f"\n{'—' * 50}")
    print("【车辆成本参数 (改进版)】")
    print(f"  空载油耗系数: {EMPTY_LOAD_FUEL_RATIO}")
    print(f"  年工作天数:   {WORKING_DAYS_PER_YEAR} (人工/胎耗/折旧)")
    print(f"  维保摊消天数: {MAINTENANCE_DAYS_PER_YEAR}")
    print(f"  折旧年限:     {DEPRECIATION_YEARS} 年")
    print(f"  区域绕行系数: 东部{REGION_BETA['east']} / "
          f"中部{REGION_BETA['central']} / 西部{REGION_BETA['west']}")
    for vtype in sorted(cost_lookup.keys()):
        c = cost_lookup[vtype]
        print(f"  {vtype}: 可变 {c['var_cost_per_km']:.2f} 元/km, "
              f"固定 {c['fixed_cost_per_day']:.2f} 元/天")

    print(f"\n{'—' * 50}")
    print("【车辆固定成本明细 (改进版)】(元/天)")
    for vtype_key in sorted(vehicle_costs.keys()):
        c = cost_lookup.get(vtype_key)
        if c is None:
            continue
        print(f"  {vtype_key}: 人工={c['labor_daily']:.2f}, "
              f"保险={c['insurance_daily']:.2f}, 审验={c['inspection_daily']:.2f}, "
              f"违章={c['violation_daily']:.2f}, 胎耗={c['tire_daily']:.2f}, "
              f"维保={c['maintenance_daily']:.2f}, 折旧={c['depreciation_daily']:.2f} "
              f"→ 合计={c['fixed_cost_per_day']:.2f}")

    print(f"\n{'=' * 50}")
    print(f"  ★ 运输总费用: {total_cost:,.2f} 元")
    print(f"{'=' * 50}")

    print(f"\n{'—' * 50}")
    print("【按车型统计】")
    for vtype in sorted(trips_df['车型'].unique()):
        sub = trips_df[trips_df['车型'] == vtype]
        print(f"  {vtype}: {len(sub)} 趟次, "
              f"成本 {sub['趟次成本_元'].sum():,.2f} 元 "
              f"({sub['趟次成本_元'].sum()/total_cost*100:.1f}%)")

    total_var = trips_df['可变成本_元'].sum()
    total_fixed = trips_df['固定成本_元'].sum()
    print(f"\n{'—' * 50}")
    print("【成本构成】")
    print(f"  可变成本 (油费+路桥, 往返): {total_var:,.2f} 元 ({total_var/total_cost*100:.1f}%)")
    print(f"  固定成本 (人工+摊消+折旧, 往返): {total_fixed:,.2f} 元 ({total_fixed/total_cost*100:.1f}%)")

    # 可变成本明细
    fuel_out = trips_df['油费_去程'].sum()
    fuel_back = trips_df['油费_回程'].sum()
    toll_out = trips_df['路桥费_去程'].sum()
    toll_back = trips_df['路桥费_回程'].sum()
    fuel_total = fuel_out + fuel_back
    toll_total = toll_out + toll_back
    print(f"\n  可变成本明细 (以可变成本为100%):")
    print(f"    去程油费:    {fuel_out:>14,.2f} 元 ({fuel_out/total_var*100:5.1f}%)")
    print(f"    回程油费:    {fuel_back:>14,.2f} 元 ({fuel_back/total_var*100:5.1f}%)")
    print(f"    油费合计:    {fuel_total:>14,.2f} 元 ({fuel_total/total_var*100:5.1f}%)")
    print(f"    去程路桥费:  {toll_out:>14,.2f} 元 ({toll_out/total_var*100:5.1f}%)")
    print(f"    回程路桥费:  {toll_back:>14,.2f} 元 ({toll_back/total_var*100:5.1f}%)")
    print(f"    路桥费合计:  {toll_total:>14,.2f} 元 ({toll_total/total_var*100:5.1f}%)")

    # 固定成本明细
    fixed_items = {
        '人工': trips_df['人工'].sum(),
        '保险': trips_df['保险'].sum(),
        '审验': trips_df['审验'].sum(),
        '违章': trips_df['违章'].sum(),
        '胎耗': trips_df['胎耗'].sum(),
        '维保': trips_df['维保'].sum(),
        '折旧': trips_df['折旧'].sum(),
    }
    print(f"\n  固定成本明细 (以固定成本为100%):")
    for name, val in fixed_items.items():
        print(f"    {name:<10s} {val:>14,.2f} 元 ({val/total_fixed*100:5.1f}%)")

    # 区域统计
    print(f"\n{'—' * 50}")
    print("【按区域统计】")
    for region in ['east', 'central', 'west']:
        sub = trips_df[trips_df['区域'] == REGION_BETA[region]]
        beta_val = REGION_BETA[region]
        name = {'east': '东部', 'central': '中部', 'west': '西部'}[region]
        if len(sub) > 0:
            print(f"  {name} (β={beta_val}): {len(sub)} 趟次, "
                  f"成本 {sub['趟次成本_元'].sum():,.2f} 元 "
                  f"({sub['趟次成本_元'].sum()/total_cost*100:.1f}%), "
                  f"平均距离 {sub['单程道路距离_km'].mean():.0f} km")

    d = trips_df['单程道路距离_km']
    print(f"\n{'—' * 50}")
    print("【距离统计】")
    print(f"  最短单程道路距离: {d.min():.1f} km (南通 → {trips_df.loc[d.idxmin(), '城市']})")
    print(f"  最长单程道路距离: {d.max():.1f} km (南通 → {trips_df.loc[d.idxmax(), '城市']})")
    print(f"  平均单程道路距离: {d.mean():.1f} km")

    print(f"\n{'—' * 50}")
    print("【按城市统计】(前15, 按总成本降序)")
    city_stats = trips_df.groupby('城市').agg(
        趟次数=('趟次成本_元', 'count'),
        总成本=('趟次成本_元', 'sum'),
        平均距离=('单程道路距离_km', 'mean'),
        区域=('区域', 'first'),
    ).sort_values('总成本', ascending=False)
    region_names = {'east': '东', 'central': '中', 'west': '西'}
    print(f"  {'城市':<10s} {'区域':>4s} {'趟次数':>6s} {'总成本(元)':>16s} {'平均距离(km)':>14s}")
    print(f"  {'-'*52}")
    for city, row in city_stats.head(15).iterrows():
        rname = region_names.get(
            {v: k for k, v in REGION_BETA.items()}.get(row['区域'], ''), '?')
        print(f"  {city:<10s} {rname:>4s} {int(row['趟次数']):>6d} "
              f"{row['总成本']:>16,.2f} {row['平均距离']:>14.1f}")

    print(f"\n{'—' * 50}")
    print("【按运输时效统计】")
    time_stats = trips_df.groupby('运输时效').agg(
        趟次数=('趟次成本_元', 'count'),
        总成本=('趟次成本_元', 'sum'),
        平均趟次成本=('趟次成本_元', 'mean')
    )
    for t, row in time_stats.iterrows():
        print(f"  {int(t)}天: {int(row['趟次数'])} 趟次, "
              f"总成本 {row['总成本']:,.2f} 元, "
              f"平均 {row['平均趟次成本']:,.2f} 元/趟")


# ============================================================
# 3. 主程序
# ============================================================
def main():
    print("正在读取车辆成本数据...")
    vehicle_costs = load_vehicle_costs(VEHICLE_PATH)
    print(f"  读取到 {len(vehicle_costs)} 种车型: {list(vehicle_costs.keys())}")

    print("正在读取排货数据...")
    shipping_df = load_shipping_data(SHIPPING_PATH)
    print(f"  读取到 {len(shipping_df)} 条有效记录")

    print("正在计算运输成本 (改进版: 往返+空载系数+折旧+工作天+区域β)...\n")
    trips_df, total_cost, cost_lookup = compute_trip_costs(shipping_df, vehicle_costs)

    print_summary(trips_df, total_cost, cost_lookup, vehicle_costs)

    return trips_df, total_cost


if __name__ == '__main__':
    trips_df, total_cost = main()
