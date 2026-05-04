#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共享工具函数
- 城市名提取
- 距离计算
- 方位角
- 数据加载
- 成本计算
- TSP排序
"""

import math
import re
import os
import glob
import pandas as pd
import numpy as np
from config import (
    CITY_COORDS, SPECIAL_ADDRESS_MAP, ROAD_FACTOR, NANTONG_COORDS,
    AVG_SPEED_HIGHWAY, DRIVE_HOURS_PER_DAY, LOAD_UNLOAD_HOURS,
    WORK_HOURS, WAYBILL_DIR, DISPATCH_DIR, TIME_LIMIT_MAP, DEFAULT_TIME_LIMIT,
    CITY_REGION, REGION_BETA,
    WORKING_DAYS_PER_YEAR, MAINTENANCE_DAYS_PER_YEAR,
    CALENDAR_DAYS_PER_YEAR, DEPRECIATION_YEARS, EMPTY_LOAD_FUEL_RATIO,
)

# ============================================================
# 1. 城市名提取
# ============================================================
def extract_city(address):
    """从地址字符串中提取城市名，返回标准城市名或 None"""
    addr = str(address).strip()
    if not addr or addr == 'nan':
        return None

    # Layer 1: 处理 '|' 分隔符（问题3地址格式）
    if '|' in addr:
        parts = addr.split('|')
        for p in parts:
            p = p.strip()
            if p.endswith('市') and len(p) <= 6 and p in CITY_COORDS:
                return p
        if len(parts) >= 2 and parts[1].strip() in CITY_COORDS:
            return parts[1].strip()

    # Layer 2: 处理 '$' 分隔符和 '【返】' 标记
    if '$' in addr:
        addr = addr.split('$')[0].strip()
    addr = addr.replace('【返】', '').strip()

    # Layer 3: 精确匹配特殊地址映射
    if addr in SPECIAL_ADDRESS_MAP:
        return SPECIAL_ADDRESS_MAP[addr]

    # Layer 4: 特殊模式匹配（无"市"字的已知地址）
    if '天津滨海新区' in addr:
        return '天津市'
    if '山西综改示范区太原' in addr:
        return '太原市'
    if '瓯海区' in addr:
        return '温州市'
    if '萧山区' in addr and '杭州' not in addr:
        return '杭州市'

    # Layer 5: 去除省份前缀后匹配 "XX市"
    province_re = re.compile(
        r'^(?:新疆维吾尔自治区|广西壮族自治区|宁夏回族自治区|'
        r'内蒙古自治区|西藏自治区|'
        r'新疆|广西|宁夏|内蒙古|西藏|'
        r'[一-龥]{2,3}省)'
    )
    cleaned = province_re.sub('', addr)
    match = re.search(r'([一-龥]{2,4}市)', cleaned)
    if match:
        city = match.group(1)
        if city.endswith('市市'):
            city = city[:-1]
        if city in CITY_COORDS:
            return city

    # Layer 6: 回退——直接匹配原始地址中的 "XX市"
    match = re.search(r'([一-龥]{2,4}市)', addr)
    if match:
        city = match.group(1)
        if city.endswith('市市'):
            city = city[:-1]
        if city in CITY_COORDS:
            return city

    # Layer 7: 匹配 "XX地区/州/盟"
    match = re.search(r'([一-龥]{2,4}(?:地区|州|盟))', addr)
    if match and match.group(1) in CITY_COORDS:
        return match.group(1)

    # Layer 8: 在 CITY_COORDS 中做子串匹配（去除"市"后缀）
    for city in CITY_COORDS:
        base = city.replace('市', '').replace('地区', '')
        if len(base) >= 2 and base in addr:
            return city

    return None


# ============================================================
# 2. 距离计算
# ============================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两点间的球面直线距离 (km)"""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def road_distance(lat1, lon1, lat2, lon2, beta=None):
    """估算道路运输距离 (km) = 直线距离 × 绕行系数。
    若提供 beta 则使用该值，否则使用全局 ROAD_FACTOR（向后兼容）。"""
    if beta is None:
        beta = ROAD_FACTOR
    return haversine_distance(lat1, lon1, lat2, lon2) * beta


def get_region_beta(city):
    """返回城市对应的区域道路绕行系数 β，若未知则返回中部默认值"""
    region = CITY_REGION.get(city, 'central')
    return REGION_BETA.get(region, 1.40)


def road_distance_for_city(lat1, lon1, lat2, lon2, city):
    """从参考点(lat1,lon1)到城市 city 的道路距离，使用该城市的区域 β"""
    return road_distance(lat1, lon1, lat2, lon2, beta=get_region_beta(city))


def road_distance_between_cities(city_a, city_b):
    """两个城市间的道路距离，使用两城市区域 β 的平均值"""
    ca, cb = get_coords(city_a), get_coords(city_b)
    if ca[0] is None or cb[0] is None:
        return float('inf')
    beta_avg = (get_region_beta(city_a) + get_region_beta(city_b)) / 2
    return road_distance(ca[0], ca[1], cb[0], cb[1], beta=beta_avg)


# ============================================================
# 3. 方位角与坐标查询
# ============================================================
def compute_bearing(lat, lon, ref_lat=NANTONG_COORDS[0], ref_lon=NANTONG_COORDS[1]):
    """从参考点出发的方位角（度），正北为 0，顺时针"""
    dlon = math.radians(lon - ref_lon)
    lat_r = math.radians(lat)
    lat0_r = math.radians(ref_lat)
    x = math.sin(dlon) * math.cos(lat_r)
    y = (math.cos(lat0_r) * math.sin(lat_r) -
         math.sin(lat0_r) * math.cos(lat_r) * math.cos(dlon))
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def get_coords(city):
    """返回城市的 (纬度, 经度)，不存在则返回 (None, None)"""
    if city and city in CITY_COORDS:
        return CITY_COORDS[city]
    return None, None


def inter_city_distance(city_a, city_b):
    """计算两个城市间的道路距离（使用区域 β 平均值），无效则返回极大值"""
    return road_distance_between_cities(city_a, city_b)


# ============================================================
# 4. 车辆数据加载
# ============================================================
def load_vehicle_costs(path):
    """读取附件1车辆成本（第一sheet），返回 {车型: {cost_item: value}}"""
    df = pd.read_excel(path, header=0)
    costs = {}
    for _, row in df.iterrows():
        vtype_raw = str(row['车型']).strip()
        fuel_str = str(row['油耗成本（元/公里）']).replace('元/公里', '').strip()
        costs[vtype_raw] = {
            'vehicle_price':           float(row['车价（元）']),
            'insurance_annual':        float(row['年保险（元）']),
            'inspection_annual':       float(row['年审（元）']),
            'road_inspection_annual':  float(row['道路运输年审（元）']),
            'fuel_cost_per_km':        float(fuel_str),
            'fuel_per_100km':          float(str(row['百公里油耗']).replace('L', '').strip()),
            'toll_per_km':             float(row['平均每公里路桥费（元）']),
            'labor_daily':             float(row['每天2人工（元）']),
            'violation_monthly':       float(row['禁区违章每月计划（元）']),
            'tire_cost':               float(row['胎耗（元）']),
            'maintenance_monthly':     float(row['维保费用每月（元）']),
            'scrap_subsidy':           float(row['报废补贴']),
        }
    return costs


def load_vehicle_data_full(path):
    """读取附件1的两个Sheet，返回 (costs, cold_pallets, regular_pallets)"""
    # Sheet: 车辆成本
    df_cost = pd.read_excel(path, sheet_name='车辆成本', header=0)
    costs = {}
    for _, row in df_cost.iterrows():
        vtype_raw = str(row['车型']).strip()
        fuel_str = str(row['油耗成本（元/公里）']).replace('元/公里', '').strip()
        costs[vtype_raw] = {
            'vehicle_price':           float(row['车价（元）']),
            'insurance_annual':        float(row['年保险（元）']),
            'inspection_annual':       float(row['年审（元）']),
            'road_inspection_annual':  float(row['道路运输年审（元）']),
            'fuel_cost_per_km':        float(fuel_str),
            'toll_per_km':             float(row['平均每公里路桥费（元）']),
            'labor_daily':             float(row['每天2人工（元）']),
            'violation_monthly':       float(row['禁区违章每月计划（元）']),
            'tire_cost':               float(row['胎耗（元）']),
            'maintenance_monthly':     float(row['维保费用每月（元）']),
            'scrap_subsidy':           float(row['报废补贴']),
        }

    # Sheet: 车辆托数信息
    df_cap = pd.read_excel(path, sheet_name='车辆托数信息', header=None)
    cold_pallets = {
        '4.2': int(df_cap.iloc[3, 1]),
        '7.6': int(df_cap.iloc[3, 2]),
        '9.6': int(df_cap.iloc[3, 3]),
    }
    regular_pallets = {
        '4.2': int(df_cap.iloc[12, 1]),
        '7.6': int(df_cap.iloc[12, 2]),
        '9.6': int(df_cap.iloc[12, 3]),
    }
    return costs, cold_pallets, regular_pallets


def build_cost_lookup(vehicle_costs):
    """构建简化成本查找表（仅问题3使用）"""
    lookup = {}
    for vtype, params in vehicle_costs.items():
        var_cost = params['fuel_cost_per_km'] + params['toll_per_km']
        fixed_daily = (
            params['labor_daily']
            + params['insurance_annual'] / 365.0
            + (params['inspection_annual'] + params['road_inspection_annual']) / 365.0
            + params['violation_monthly'] / 30.0
            + params['tire_cost'] / 365.0
            + params['maintenance_monthly'] / 30.0
        )
        lookup[vtype] = {'var_cost_per_km': var_cost, 'fixed_cost_per_day': fixed_daily}
    for base in ['4.2', '7.6', '9.6']:
        if base in lookup:
            lookup[base + 'M'] = lookup[base]
    return lookup


def build_detailed_cost_lookup(vehicle_costs):
    """构建详细成本查找表（与问题1一致：工作天区分、折旧、区域β）
    返回 {车型: {var_cost_per_km, fuel_cost_per_km, toll_per_km, fixed_cost_per_day}}"""
    lookup = {}
    for vtype, params in vehicle_costs.items():
        fuel_cost = params['fuel_cost_per_km']
        toll_cost = params['toll_per_km']
        var_cost_per_km = fuel_cost + toll_cost

        labor = params['labor_daily']
        insurance = params['insurance_annual'] / CALENDAR_DAYS_PER_YEAR
        inspection = (params['inspection_annual'] + params['road_inspection_annual']) / CALENDAR_DAYS_PER_YEAR
        violation = params['violation_monthly'] / 30.0
        tire = params['tire_cost'] / WORKING_DAYS_PER_YEAR
        maintenance = params['maintenance_monthly'] * 12 / MAINTENANCE_DAYS_PER_YEAR
        depreciation = ((params['vehicle_price'] - params['scrap_subsidy'])
                       / DEPRECIATION_YEARS / WORKING_DAYS_PER_YEAR)

        fixed_daily = labor + insurance + inspection + violation + tire + maintenance + depreciation

        lookup[vtype] = {
            'var_cost_per_km':  var_cost_per_km,
            'fuel_cost_per_km': fuel_cost,
            'toll_per_km':      toll_cost,
            'fixed_cost_per_day': fixed_daily,
        }
    for base in ['4.2', '7.6', '9.6']:
        if base in lookup:
            lookup[base + 'M'] = lookup[base]
    return lookup


# ============================================================
# 5. 排货数据加载（问题1-2 共用）
# ============================================================
def load_shipping_data(path):
    """读取附件2排货数据，合并3个sheet，返回清洗后的DataFrame"""
    xls = pd.ExcelFile(path)
    sheet_names = xls.sheet_names

    MANUAL_COLS = [
        '月份', '填写日期', '随货通行单号', '温度计编号', '运输交接单号',
        '盒数', '箱数', '托装', '托盘数', '收货方地址',
        '运输时效', '预计提货日期', '预计到货日期', '车型', '运输方式',
        '车辆数目', '始发地天气', '目的地天气', '实际提货日期', '实际到货日期',
        '签收状态', '随货单据是否齐全', '备注'
    ]

    all_data = []
    for sheet in sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, header=None, skiprows=3)
        df.columns = MANUAL_COLS
        all_data.append(df)

    df = pd.concat(all_data, ignore_index=True)
    df = df[df['运输交接单号'].notna()].copy()
    df = df[df['运输交接单号'].astype(str).str.strip() != ''].copy()
    df['车辆数目'] = pd.to_numeric(df['车辆数目'], errors='coerce')
    df['车型'] = df['车型'].astype(str).str.strip().replace('nan', np.nan)
    df['收货方地址'] = df['收货方地址'].astype(str).str.strip()
    df['运输时效'] = pd.to_numeric(df['运输时效'], errors='coerce')
    for col in ['盒数', '箱数', '托盘数']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


def get_trips(shipping_df):
    """从排货数据提取趟次级别的DataFrame（问题2用）"""
    df = shipping_df.copy()
    for col in ['车型', '运输时效', '车辆数目']:
        df[col] = df.groupby('运输交接单号')[col].ffill()
    df['车辆数目'] = df['车辆数目'].fillna(1).astype(int)
    df['运输时效'] = df['运输时效'].fillna(1)

    trips = df.groupby(
        ['运输交接单号', '车型', '车辆数目', '收货方地址', '运输时效'],
        dropna=False
    ).agg(
        总托盘数=('托盘数', 'sum'),
        预计提货日期=('预计提货日期', 'first'),
        预计到货日期=('预计到货日期', 'first'),
    ).reset_index()

    trips = trips[trips['车型'].notna()].copy()
    trips = trips[trips['车型'].isin(['7.6M', '9.6M'])].copy()

    trips['城市'] = trips['收货方地址'].apply(extract_city)
    trips = trips[trips['城市'].notna()].copy()

    trips['纬度'] = trips['城市'].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
    trips['经度'] = trips['城市'].map(lambda c: CITY_COORDS.get(c, (None, None))[1])
    trips = trips[trips['纬度'].notna()].copy()

    trips['单程道路距离_km'] = trips.apply(
        lambda row: road_distance(NANTONG_COORDS[0], NANTONG_COORDS[1],
                                  row['纬度'], row['经度'],
                                  beta=get_region_beta(row['城市'])), axis=1
    )

    trips['提货dt'] = pd.to_datetime(trips['预计提货日期'], errors='coerce')
    trips['到货dt'] = pd.to_datetime(trips['预计到货日期'], errors='coerce')
    trips['提货周'] = trips['提货dt'].dt.isocalendar().week.astype(int)

    return trips


def compute_original_cost(trips, cost_lookup, breakdown=False):
    """计算原始（未优化）总成本 — 往返运输，使用详细成本模型（空载油耗系数）
    若 breakdown=True，返回 (total, var_total, fixed_total) 三元组"""
    def calc_var_cost(row):
        vtype = str(row['车型']).strip()
        if vtype not in cost_lookup:
            return 0.0
        c = cost_lookup[vtype]
        d = row['单程道路距离_km']
        n = row['车辆数目']
        var_out = c['var_cost_per_km'] * d
        var_return = c.get('fuel_cost_per_km', c['var_cost_per_km']) * EMPTY_LOAD_FUEL_RATIO * d \
                     + c.get('toll_per_km', 0) * d
        return (var_out + var_return) * n

    def calc_fixed_cost(row):
        vtype = str(row['车型']).strip()
        if vtype not in cost_lookup:
            return 0.0
        c = cost_lookup[vtype]
        return c['fixed_cost_per_day'] * row['运输时效'] * 2 * row['车辆数目']

    var_total = trips.apply(calc_var_cost, axis=1).sum()
    fixed_total = trips.apply(calc_fixed_cost, axis=1).sum()
    total = var_total + fixed_total
    if breakdown:
        return total, var_total, fixed_total
    return total


# ============================================================
# 6. 运单与派车单数据加载（问题3）
# ============================================================
def load_waybill_data():
    """读取附件3运单数据，返回清洗后的DataFrame"""
    files = sorted(glob.glob(os.path.join(WAYBILL_DIR, '*.xlsx')))

    cols = ['序号', '货主单号', '物流单号', '运单号', '运输方式', '业务类型',
            '调度时间', '市内或干线', '承运商', '收货地址', '运输时限', '线路号',
            '普通或冷链', '毒麻药品', '仓库编码', '物流单明细号', '运单明细号',
            '商品编码', '商品名', '规格', '批号', '数量']
    all_data = []
    for f in files:
        df = pd.read_excel(f, header=None, skiprows=1)
        df.columns = cols
        all_data.append(df)
    df = pd.concat(all_data, ignore_index=True)
    df['数量'] = pd.to_numeric(df['数量'], errors='coerce').fillna(0)
    df['城市'] = df['收货地址'].apply(extract_city)
    df['冷链需求'] = df['普通或冷链'].apply(
        lambda x: str(x).strip() in ['冷链', '冷藏', '冷特', '疫苗', '二类'])
    return df


def load_dispatch_data():
    """读取附件4派车单数据，返回清洗后的DataFrame"""
    files = sorted(glob.glob(os.path.join(DISPATCH_DIR, '*.xlsx')))

    cols = ['序号', '货主物流代码', '仓库编码', '发货地址', '收货地址', '物流单号',
            '商品件数', '运输方式', '车牌号', '运输工具', '启运时间', '签收时间']
    all_data = []
    for f in files:
        df = pd.read_excel(f, header=None, skiprows=1)
        df.columns = cols
        all_data.append(df)
    df = pd.concat(all_data, ignore_index=True)
    df['发货城市'] = df['发货地址'].apply(extract_city)
    df['收货城市'] = df['收货地址'].apply(extract_city)

    def parse_parcels(s):
        try:
            s = str(s).strip()
            if '+' in s:
                p = s.split('+')
                return int(p[0]), int(p[1]), int(p[0]) + int(p[1])
            v = int(s)
            return v, 0, v
        except Exception:
            return 0, 0, 0

    parsed = df['商品件数'].apply(parse_parcels)
    df['箱数'] = parsed.apply(lambda x: x[0])
    df['盒数'] = parsed.apply(lambda x: x[1])
    df['总件数'] = parsed.apply(lambda x: x[2])
    df['启运dt'] = pd.to_datetime(df['启运时间'], errors='coerce')
    df['签收dt'] = pd.to_datetime(df['签收时间'], errors='coerce')
    df['运输小时'] = (df['签收dt'] - df['启运dt']).dt.total_seconds() / 3600
    df['启运日期'] = df['启运dt'].dt.date

    def classify(v):
        v = str(v).strip()
        if any(k in v for k in ['冷藏', '冷']):
            return '冷藏车'
        if any(k in v for k in ['金杯', '依维柯', '全顺', '面包', '客车', '救护',
                                '尼桑', '日产', '五十铃', '江铃', '跃进', '福田',
                                '五菱', '皮卡', '电动', '小型', '微型', '封闭']):
            return '小型车'
        if any(k in v for k in ['中型', '6.2']):
            return '中型车'
        return '大型车'

    df['车辆分类'] = df['运输工具'].apply(classify)
    df['是冷藏车'] = df['运输工具'].apply(lambda v: '冷藏' in str(v) or '冷' in str(v))
    return df


# ============================================================
# 7. TSP排序与路线距离
# ============================================================
def tsp_order(start_coords, cities):
    """最近邻构造 + 2-opt 改进，返回最优访问顺序"""
    if len(cities) <= 1:
        return list(cities)

    remaining = set(cities)
    ordered = []
    current = start_coords
    while remaining:
        nearest = min(remaining, key=lambda c:
            haversine_distance(current[0], current[1],
                             CITY_COORDS[c][0], CITY_COORDS[c][1]))
        ordered.append(nearest)
        current = get_coords(nearest)
        remaining.remove(nearest)

    # 2-opt 局部优化（使用区域 β 距离）
    improved = True
    while improved:
        improved = False
        for i in range(len(ordered) - 1):
            for j in range(i + 2, len(ordered)):
                if i == 0 and j == len(ordered) - 1:
                    continue
                ci_name = ordered[i]
                ci_next = ordered[(i + 1) % len(ordered)]
                cj_name = ordered[j]
                cj_next = ordered[(j + 1) % len(ordered)]
                old_d = road_distance_between_cities(ci_name, ci_next) + \
                        road_distance_between_cities(cj_name, cj_next)
                new_d = road_distance_between_cities(ci_name, cj_name) + \
                        road_distance_between_cities(ci_next, cj_next)
                if new_d < old_d - 0.1:
                    ordered[i + 1:j + 1] = reversed(ordered[i + 1:j + 1])
                    improved = True
    return ordered


def route_distance(start_coords, ordered_cities):
    """计算从 start_coords 出发访问 ordered_cities 并返回的总距离（使用区域 β）"""
    if not ordered_cities:
        return 0.0
    d = 0.0
    prev = start_coords
    prev_city = None
    for city in ordered_cities:
        cc = get_coords(city)
        if cc[0] is not None:
            d += road_distance(prev[0], prev[1], cc[0], cc[1],
                              beta=get_region_beta(city))
            prev = cc
            prev_city = city
    # 返回段：使用最后一个城市的区域 β
    if prev_city is not None:
        d += road_distance(prev[0], prev[1], start_coords[0], start_coords[1],
                          beta=get_region_beta(prev_city))
    else:
        d += road_distance(prev[0], prev[1], start_coords[0], start_coords[1])
    return d


# ============================================================
# 8. 运输天数估计与车型选择
# ============================================================
def estimate_travel_days(route_distance_km, num_stops):
    """估计多站路线的运输天数"""
    drive_days = route_distance_km / (AVG_SPEED_HIGHWAY * DRIVE_HOURS_PER_DAY)
    load_days = num_stops * LOAD_UNLOAD_HOURS / WORK_HOURS
    return max(1, math.ceil(drive_days + load_days))


def select_best_vehicle(total_pallets, total_dist_km, cost_lookup):
    """选择满足约束的最经济车型（问题2用，托盘容量 + 距离约束）"""
    can_use_42 = total_dist_km <= 300 and total_pallets <= 5
    can_use_76 = total_pallets <= 11
    can_use_96 = total_pallets <= 14

    if can_use_42 and ('4.2' in cost_lookup or '4.2M' in cost_lookup):
        return '4.2M' if '4.2M' in cost_lookup else '4.2'
    elif can_use_76 and ('7.6' in cost_lookup or '7.6M' in cost_lookup):
        return '7.6M' if '7.6M' in cost_lookup else '7.6'
    elif can_use_96 and ('9.6' in cost_lookup or '9.6M' in cost_lookup):
        return '9.6M' if '9.6M' in cost_lookup else '9.6'
    else:
        return '9.6M' if '9.6M' in cost_lookup else '9.6'


def select_vehicle_by_pallets(total_pallets):
    """简单的容量匹配车型选择（问题2回退用）"""
    if total_pallets <= 5:
        return '4.2M'
    elif total_pallets <= 11:
        return '7.6M'
    else:
        return '9.6M'
