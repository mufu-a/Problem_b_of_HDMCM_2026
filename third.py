#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医药冷链物流派车合理性评估与优化模型 — 问题3
评估: 6维指标体系 + 熵权法客观赋权 + 行业基准绝对评分
优化: 仓库-日期分组 + CW节约算法 + 方向/距离约束
"""

import math
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

from config import (
    VEHICLE_PATH, ROAD_FACTOR, NANTONG_COORDS, CITY_COORDS,
    AVG_SPEED_CITY, DRIVE_HOURS_PER_DAY, TIME_LIMIT_MAP, DEFAULT_TIME_LIMIT,
    CW3_MAX_INTER_CITY_KM, CW3_MAX_TOTAL_ROUTE_KM, CW3_MAX_STOPS_PER_ROUTE,
    CW3_MAX_QTY_COLD, CW3_MAX_QTY_REGULAR,
    CW3_COLD_SMALL_MAX_QTY, CW3_COLD_SMALL_MAX_DIST, CW3_COLD_76_MAX_QTY,
    CW3_REG_SMALL_MAX_QTY, CW3_REG_MEDIUM_MAX_QTY, CW3_REG_76_MAX_QTY,
    VEHICLE_COST_PARAMS, VEHICLE_CAT_TO_TYPE_COLD, VEHICLE_CAT_TO_TYPE_REGULAR,
    CW3_SENSITIVITY_DISTANCES,
)
from utils import (
    extract_city, haversine_distance, road_distance,
    compute_bearing, get_coords, inter_city_distance,
    get_region_beta,
    load_waybill_data, load_dispatch_data,
    tsp_order, route_distance,
)

AVG_SPEED = AVG_SPEED_CITY
DRIVE_HOURS = DRIVE_HOURS_PER_DAY


# ============================================================
# 1. 仓库级指标计算
# ============================================================

def compute_warehouse_indicators(waybill_df, dispatch_df):
    """
    按仓库聚合计算6项原始指标值。
    返回: DataFrame, 每行一个仓库。
    """
    dp = dispatch_df.dropna(subset=['车牌号', '启运日期', '发货城市', '收货城市']).copy()
    dp['日期'] = dp['启运dt'].dt.date

    # ---- 指标1: 配送密度 (按市内/干线分层) ----

    # Step A: 逐车辆-日计算市内/干线停靠点
    detail = dp.dropna(subset=['车牌号', '日期', '收货地址', '发货城市', '收货城市']).copy()
    detail['是否市内'] = (detail['发货城市'] == detail['收货城市'])

    stops_records = []
    for (wh, plate, dt), g in detail.groupby(['仓库编码', '车牌号', '日期']):
        ship_city = g['发货城市'].iloc[0]
        all_addr = g['收货地址'].nunique()
        city_addr = g[g['是否市内']]['收货地址'].nunique()
        line_addr = g[~g['是否市内']]['收货地址'].nunique()
        # 分类: 市内停靠点 >= 干线停靠点 → 市内型, 否则干线型
        vd_type = '市内' if city_addr >= line_addr else '干线'
        stops_records.append({
            '仓库编码': wh, '车牌号': plate, '日期': dt,
            '发货城市': ship_city,
            '总停靠点': all_addr,
            '市内停靠点': city_addr,
            '干线停靠点': line_addr,
            '类型': vd_type,
        })
    stops_df = pd.DataFrame(stops_records)

    # Step B: 按仓库汇总 (整体 + 分层)
    def _high(x, thresh):
        return (x >= thresh).mean()

    def _low(x, thresh):
        return (x <= thresh).mean()

    density = stops_df.groupby('仓库编码').agg(
        车辆日数=('总停靠点', 'count'),
        平均停靠点=('总停靠点', 'mean'),
        中位停靠点=('总停靠点', 'median'),
        高效日25占比=('总停靠点', lambda x: _high(x, 25)),
        低效日3占比=('总停靠点', lambda x: _low(x, 3)),
    )

    # 分类型统计
    city_stats = stops_df[stops_df['类型'] == '市内'].groupby('仓库编码').agg(
        市内车辆日=('总停靠点', 'count'),
        市内平均停靠点=('总停靠点', 'mean'),
    )
    line_stats = stops_df[stops_df['类型'] == '干线'].groupby('仓库编码').agg(
        干线车辆日=('总停靠点', 'count'),
        干线平均停靠点=('总停靠点', 'mean'),
    )

    # ---- 指标2: 路线合并潜力 ----
    route_groups = dp.groupby(['仓库编码', '车牌号', '日期'])
    wh_savings = defaultdict(list)
    for (wh, plate, date), g in route_groups:
        sc = g['发货城市'].iloc[0]
        sc_c = get_coords(sc)
        if sc_c[0] is None:
            continue
        dests = list(dict.fromkeys(g['收货城市'].dropna().tolist()))
        dests = [d for d in dests if get_coords(d)[0] is not None]
        if len(dests) < 2:
            continue
        separate = sum(2 * road_distance(sc_c[0], sc_c[1],
                       get_coords(d)[0], get_coords(d)[1], beta=get_region_beta(d)) for d in dests)
        ordered = tsp_order(sc_c, dests)
        tsp_dist = route_distance(sc_c, ordered)
        if separate > 0:
            wh_savings[wh].append((separate - tsp_dist) / separate * 100)

    # ---- 指标3: 时效达标率 ----
    wb_time = waybill_df.groupby('物流单号').agg(
        时限=('运输时限', 'first'),
        仓库=('仓库编码', 'first')
    ).reset_index()
    dp_time = dp.groupby('物流单号').agg(运输h=('运输小时', 'max')).reset_index()
    m = wb_time.merge(dp_time, on='物流单号', how='inner')
    def get_lim(t):
        return TIME_LIMIT_MAP.get(str(t).strip(), DEFAULT_TIME_LIMIT)
    m['limit'] = m['时限'].apply(get_lim)
    m['达标'] = m['运输h'].fillna(0) <= m['limit']
    wh_time = m.groupby('仓库').agg(总订单=('达标', 'count'), 达标数=('达标', 'sum'))

    # ---- 指标5: 冷链合规率 ----
    wb_cold = waybill_df.groupby('物流单号').agg(
        冷链需求=('冷链需求', 'max'),
        仓库=('仓库编码', 'first')
    ).reset_index()
    dp_cold = dp.groupby('物流单号').agg(是冷藏车=('是冷藏车', 'max')).reset_index()
    mc = wb_cold.merge(dp_cold, on='物流单号', how='inner')
    cold_orders = mc[mc['冷链需求']]
    wh_cold = cold_orders.groupby('仓库').agg(冷链订单数=('冷链需求', 'count'),
                                               合规数=('是冷藏车', 'sum'))

    # ---- 指标4: 线路集中度 ----
    # 基于信息熵衡量各仓库运单在"线路号"维度上的集中程度。
    # 集中度高 → 订单集中在少数成熟线路上 → 易形成规模效应、便于优化;
    # 集中度低 → 订单分散在大量线路上 → 需求碎片化、难以合并。
    wh_line_conc = {}
    for wh, g in waybill_df.groupby('仓库编码'):
        line_counts = g['线路号'].value_counts()
        probs = line_counts.values / line_counts.values.sum()
        L = len(probs)
        if L <= 1:
            conc = 100.0
        else:
            entropy = -float(np.sum(probs * np.log(probs)))
            max_entropy = np.log(L)
            conc = (1.0 - entropy / max_entropy) * 100
        wh_line_conc[wh] = {'线路数': L, '线路集中度%': conc}

    # ---- 指标6: 车辆装载饱和度 ----
    # 计算每个车辆-日的总装载件数, 以各车型全局p75分位数为基准。
    # 饱和度 = 总件数 / 该车型p75基准, 超过100%截断。
    # 该指标衡量车辆运力利用效率: 装载过少→浪费运力, 接近或超过行业p75→利用充分。
    vd_load = dp.dropna(subset=['车辆分类', '总件数']).copy()
    vd_load_sum = vd_load.groupby(['仓库编码', '车辆分类', '车牌号', '日期'])['总件数'].sum().reset_index()
    # 全局各车型p75基准
    cat_p75 = vd_load_sum.groupby('车辆分类')['总件数'].apply(lambda x: np.percentile(x, 75)).to_dict()
    # 每个车辆-日的饱和度
    vd_load_sum['饱和度%'] = vd_load_sum.apply(
        lambda r: min(r['总件数'] / cat_p75[r['车辆分类']] * 100, 100)
        if cat_p75.get(r['车辆分类'], 0) > 0 else 100.0, axis=1)
    # 按仓库汇总, 同时保留各车型的车辆日数和平均饱和度
    wh_load_cat = vd_load_sum.groupby(['仓库编码', '车辆分类']).agg(
        车辆日=('总件数', 'count'),
        平均饱和度=('饱和度%', 'mean'),
    ).reset_index()
    wh_load = vd_load_sum.groupby('仓库编码').agg(
        装载车辆日=('总件数', 'count'),
        平均饱和度=('饱和度%', 'mean'),
    )

    # ---- 合并 ----
    records = []
    warehouses = sorted(dp['仓库编码'].dropna().unique())

    for wh in warehouses:
        if wh not in density.index:
            continue
        d = density.loc[wh]
        n_vd = int(d['车辆日数'])
        if n_vd < 1:
            continue

        avg_stops = float(d['平均停靠点'])

        # 分层: 市内
        if wh in city_stats.index:
            cs = city_stats.loc[wh]
            n_city_vd = int(cs['市内车辆日'])
            avg_city = float(cs['市内平均停靠点'])
        else:
            n_city_vd = 0
            avg_city = 0.0

        # 分层: 干线
        if wh in line_stats.index:
            ls = line_stats.loc[wh]
            n_line_vd = int(ls['干线车辆日'])
            avg_line = float(ls['干线平均停靠点'])
        else:
            n_line_vd = 0
            avg_line = 0.0

        # 路线潜力
        sv = wh_savings.get(wh, [])
        route_saving = np.mean(sv) if sv else 0.0

        # 时效达标
        if wh in wh_time.index:
            t = wh_time.loc[wh]
            time_ok_rate = (float(t['达标数']) / float(t['总订单']) * 100) if t['总订单'] > 0 else 100.0
            n_time = int(t['总订单'])
        else:
            time_ok_rate = 100.0
            n_time = 0

        # 冷链合规
        if wh in wh_cold.index:
            c = wh_cold.loc[wh]
            n_cold = int(c['冷链订单数'])
            cold_ok_rate = float(c['合规数']) / n_cold * 100 if n_cold > 0 else 100.0
        else:
            n_cold = 0
            cold_ok_rate = 100.0

        # 装载饱和度
        if wh in wh_load.index:
            load_sat = float(wh_load.loc[wh, '平均饱和度'])
        else:
            load_sat = 0.0

        # 线路集中度
        lc = wh_line_conc.get(wh, {'线路数': 0, '线路集中度%': 50.0})
        n_lines = lc['线路数']
        line_conc = lc['线路集中度%']

        records.append({
            '仓库编码': wh,
            '车辆日数': n_vd,
            '市内车辆日': n_city_vd,
            '干线车辆日': n_line_vd,
            '冷链订单数': n_cold,
            '时效订单数': n_time,
            '平均停靠点': avg_stops,
            '市内平均停靠点': avg_city,
            '干线平均停靠点': avg_line,
            '路线潜力_节约率%': route_saving,
            '时效达标率%': time_ok_rate,
            '线路集中度%': line_conc,
            '线路数': n_lines,
            '冷链合规率%': cold_ok_rate,
            '装载饱和度%': load_sat,
        })

    return pd.DataFrame(records)


# ============================================================
# 2. 行业基准绝对评分
# ============================================================

def compute_absolute_scores(df_wh):
    """
    对每项指标按行业基准映射为 0-100 的绝对合理性得分。
    配送密度按市内/干线分层评分后加权汇总。
    返回: (score_matrix (m×6), score_col_names)
    """
    m = len(df_wh)
    n_city = df_wh['市内车辆日'].values
    n_line = df_wh['干线车辆日'].values
    avg_city = df_wh['市内平均停靠点'].values
    avg_line = df_wh['干线平均停靠点'].values

    # ---- 指标1: 配送密度 (分层, 中间型指标) ----
    # 配送密度并非越高越好——停靠点过多意味着单个点服务时间不足,
    # 可能存在卸货草率、客户服务质量下降的风险。
    # 采用梯形隶属函数: 最优区间内满分, 两侧线性递减。
    #   市内: 最优区间 [12, 18] 点/日, 下界基准12, 上界基准18
    #   干线: 最优区间 [3, 5] 点/日,   下界基准3,  上界基准5
    def _middle_score(val, lo_opt, hi_opt):
        """中间型梯形得分: [lo_opt, hi_opt]内满分, 两侧线性递减"""
        if np.isnan(val) or val <= 0:
            return np.nan
        if lo_opt <= val <= hi_opt:
            return 100.0
        elif val < lo_opt:
            return val / lo_opt * 100.0
        else:
            return max(100.0 - (val - hi_opt) / hi_opt * 100.0, 0.0)

    city_density = np.array([_middle_score(v, 12, 18) for v in avg_city])
    line_density = np.array([_middle_score(v, 3, 5) for v in avg_line])
    s1 = np.zeros(m)
    for i in range(m):
        parts, wts = [], []
        if n_city[i] > 0 and not np.isnan(city_density[i]):
            parts.append(city_density[i]); wts.append(n_city[i])
        if n_line[i] > 0 and not np.isnan(line_density[i]):
            parts.append(line_density[i]); wts.append(n_line[i])
        s1[i] = np.average(parts, weights=wts) if wts else 0.0

    # ---- 指标2: 路线合并潜力 ----
    s2 = np.clip(50 + df_wh['路线潜力_节约率%'].values, 0, 100)

    # ---- 指标3: 时效达标率 ----
    s3 = df_wh['时效达标率%'].values

    # ---- 指标4: 线路集中度 ----
    s4 = df_wh['线路集中度%'].values

    # ---- 指标5: 冷链合规率 ----
    s5 = df_wh['冷链合规率%'].values

    # ---- 指标6: 车辆装载饱和度 ----
    # 饱和度 ∈ [0, 100+], 直接作为得分 (高于100%截断为满分)
    s6 = np.clip(df_wh['装载饱和度%'].values, 0, 100)

    names = ['配送密度', '路线潜力', '时效达标', '线路集中', '冷链合规', '装载饱和']
    return np.column_stack([s1, s2, s3, s4, s5, s6]), names


# ============================================================
# 3. 熵权法客观赋权
# ============================================================

def entropy_weight_method(S):
    """
    熵权法: 从得分矩阵 S (m×n, 0-100) 计算客观权重。
    返回: weights, entropy, diff_coef
    """
    m, n = S.shape
    # Min-Max 归一化到 [0.0001, 1]
    S_min = S.min(axis=0)
    S_max = S.max(axis=0)
    S_norm = (S - S_min) / (S_max - S_min + 1e-12)
    S_norm = np.clip(S_norm, 0.0001, 1.0)

    # 比重矩阵
    P = S_norm / S_norm.sum(axis=0)

    # 信息熵
    k = 1.0 / np.log(m)
    entropy = -k * np.sum(P * np.log(P), axis=0)
    entropy = np.clip(entropy, 0, 1)

    # 差异系数 → 权重
    diff_coef = 1.0 - entropy
    weights = diff_coef / diff_coef.sum()

    return weights, entropy, diff_coef


def entropy_sensitivity_analysis(S, score_names, df_wh):
    """
    熵权法敏感性分析: 剔除冷链合规得分为0的极端仓库后重新计算权重。
    返回: 剔除前后的权重对比信息
    """
    m, n = S.shape
    cold_idx = 4  # 冷链合规指标列

    # 找出冷链合规得分为0的仓库
    extreme_mask = S[:, cold_idx] < 0.01
    n_extreme = extreme_mask.sum()
    if n_extreme == 0:
        print("  无冷链合规0分仓库，跳过敏感性分析。")
        return None

    extreme_wh = df_wh.iloc[extreme_mask]['仓库编码'].tolist()

    # 剔除极端仓库后重新计算
    S_clean = S[~extreme_mask]
    w_orig, ent_orig, diff_orig = entropy_weight_method(S)
    w_clean, ent_clean, diff_clean = entropy_weight_method(S_clean)

    print(f"\n{'='*70}")
    print("  熵权法敏感性分析: 剔除冷链合规0分仓库")
    print(f"{'='*70}")
    print(f"  剔除仓库 ({n_extreme}个): {', '.join(extreme_wh)}")
    print(f"  剔除前有效仓库: {m} → 剔除后: {m - n_extreme}")
    print(f"\n  {'指标':<10s} {'剔除前权重':>10s} {'剔除后权重':>10s} {'变动':>10s}")
    print(f"  {'-'*44}")
    for i, name in enumerate(score_names):
        delta = (w_clean[i] - w_orig[i]) * 100
        direction = '↑' if delta > 0.5 else '↓' if delta < -0.5 else '→'
        print(f"  {name:<8s} {w_orig[i]*100:>9.2f}% {w_clean[i]*100:>9.2f}% "
              f"{delta:>+9.2f}% {direction}")
    print(f"\n  路线合并潜力保持最高权重: {w_clean[1]*100:.1f}% "
          f"({'是' if np.argmax(w_clean) == 1 else '否'})")

    return {
        'extreme_wh': extreme_wh,
        'n_extreme': n_extreme,
        'w_orig': w_orig,
        'w_clean': w_clean,
    }


# ============================================================
# 4. TOPSIS 辅助排名
# ============================================================

def topsis_ranking(S, weights):
    """
    TOPSIS 相对排名 (辅助分析，不作为主评分)。
    S: 正向化得分矩阵, weights: 熵权
    """
    m, n = S.shape
    # 向量归一化
    col_norms = np.sqrt(np.sum(S ** 2, axis=0))
    R = S / (col_norms + 1e-12)
    Z = R * weights

    V_plus = np.max(Z, axis=0)
    V_minus = np.min(Z, axis=0)

    D_pos = np.sqrt(np.sum((Z - V_plus) ** 2, axis=1))
    D_neg = np.sqrt(np.sum((Z - V_minus) ** 2, axis=1))
    C = D_neg / (D_pos + D_neg + 1e-12)
    return C, D_pos, D_neg


# ============================================================
# 5. 结果输出
# ============================================================

def print_evaluation(df_wh, S, score_names, weights, entropy, diff_coef,
                     C_topsis, D_pos, D_neg):
    """完整评估结果输出"""
    m, n = S.shape
    vd = df_wh['车辆日数'].values

    # 加权综合得分 (绝对合理性)
    wh_scores = S @ weights   # (m,)
    system_score = np.average(wh_scores, weights=vd) if vd.sum() > 0 else np.mean(wh_scores)

    # ---- 熵权法详细 ----
    print(f"\n{'='*70}")
    print("  熵权法 — 客观权重计算 (基于各仓库得分矩阵的差异程度)")
    print(f"{'='*70}")
    print(f"\n  {'指标':<10s} {'信息熵 eⱼ':>10s} {'差异系数 dⱼ':>12s} {'权重 wⱼ':>10s} {'权重%':>8s}")
    print(f"  {'-'*52}")
    for i, name in enumerate(score_names):
        print(f"  {name:<8s} {entropy[i]:>10.4f} {diff_coef[i]:>12.4f} "
              f"{weights[i]:>10.4f} {weights[i]*100:>7.2f}%")
    print(f"  {'-'*52}")
    print(f"  {'合计':<8s} {'':>10s} {'':>12s} {weights.sum():>10.4f} {weights.sum()*100:>7.2f}%")

    # ---- 权重解读 ----
    max_w_idx = np.argmax(weights)
    min_w_idx = np.argmin(weights)
    print(f"\n  权重解读:")
    print(f"  ★ 最具区分度: {score_names[max_w_idx]} ({weights[max_w_idx]*100:.1f}%)"
          f" — 仓库间差异最大, 信息熵仅{entropy[max_w_idx]:.4f}")
    print(f"  ☆ 最缺乏区分度: {score_names[min_w_idx]} ({weights[min_w_idx]*100:.1f}%)"
          f" — 仓库间表现趋同, 信息熵高达{entropy[min_w_idx]:.4f}")

    # ---- 综合合理性评分 ----
    print(f"\n{'='*70}")
    print("  派车单合理性综合评估 (熵权法客观赋权 + 行业基准绝对评分)")
    print(f"{'='*70}")
    print(f"  评估仓库数: {m} (覆盖全部有配送记录的仓库)")
    print(f"  总车辆日覆盖: {vd.sum():.0f}")
    print(f"")
    print(f"  系统加权综合得分: {system_score:.1f} / 100")
    rating = '优秀' if system_score >= 85 else '良好' if system_score >= 70 \
              else '一般' if system_score >= 55 else '需改进'
    print(f"  评级: {rating}")

    # 各指标系统平均得分
    print(f"\n  各指标系统加权平均得分:")
    print(f"  {'指标':<10s} {'加权均分':>8s} {'算术均分':>8s} {'最低分':>8s} {'最高分':>8s}")
    print(f"  {'-'*42}")
    for i, name in enumerate(score_names):
        w_avg = np.average(S[:, i], weights=vd)
        print(f"  {name:<8s} {w_avg:>8.1f} {S[:, i].mean():>8.1f} "
              f"{S[:, i].min():>8.1f} {S[:, i].max():>8.1f}")

    # 分层分布统计
    total_city = df_wh['市内车辆日'].sum()
    total_line = df_wh['干线车辆日'].sum()
    print(f"\n  配送密度分层统计:")
    print(f"    市内型车辆日: {total_city:.0f} ({total_city/(total_city+total_line)*100:.1f}%) "
          f"— 最优区间 [12, 18] 点/日")
    print(f"    干线型车辆日: {total_line:.0f} ({total_line/(total_city+total_line)*100:.1f}%) "
          f"— 最优区间 [3, 5] 点/日")
    print(f"    未分层均分: {np.average(df_wh['平均停靠点'].values, weights=vd):.1f} 点/日")
    print(f"    分层加权后配送密度得分: {np.average(S[:,0], weights=vd):.1f} 分")

    # 得分构成
    print(f"\n  系统综合得分构成 (加权求和):")
    print(f"  {'指标':<10s} {'权重%':>7s} {'加权均分':>8s} {'贡献':>8s}")
    print(f"  {'-'*36}")
    for i, name in enumerate(score_names):
        w_avg = np.average(S[:, i], weights=vd)
        contrib = w_avg * weights[i]
        print(f"  {name:<8s} {weights[i]*100:>6.1f}% {w_avg:>8.1f} {contrib:>8.1f}")
    print(f"  {'-'*36}")
    print(f"  {'合计':<8s} {'100.0%':>7s} {'':>8s} {system_score:>8.1f}")

    # ---- 全部仓库排名 ----
    sorted_idx = np.argsort(-wh_scores)
    print(f"\n{'='*70}")
    print(f"  全部 {m} 个仓库合理性得分排名")
    print(f"{'='*70}")
    print(f"  {'排名':<5s} {'仓库':<8s} {'综合分':>7s} {'配送密度':>8s} {'路线潜力':>8s} "
          f"{'时效达标':>8s} {'线路集中':>8s} {'冷链合规':>8s} {'装载饱和':>8s} {'车辆日':>6s}")
    print(f"  {'-'*85}")
    for rank, idx in enumerate(sorted_idx, 1):
        wh = df_wh.iloc[idx]['仓库编码']
        print(f"  {rank:<5d} {wh:<8s} {wh_scores[idx]:>7.1f} "
              f"{S[idx,0]:>8.1f} {S[idx,1]:>8.1f} {S[idx,2]:>8.1f} "
              f"{S[idx,3]:>8.1f} {S[idx,4]:>8.1f} {S[idx,5]:>8.1f} {vd[idx]:>6.0f}")

    # 冷链合规异常警示
    cold_scores = S[:, 4]
    bad_cold = cold_scores < 50
    if bad_cold.sum() > 0:
        print(f"\n  ⚠ 冷链合规得分 < 50 的仓库 ({bad_cold.sum()} 个):")
        for idx in np.where(bad_cold)[0]:
            wh = df_wh.iloc[idx]['仓库编码']
            n_cold = df_wh.iloc[idx]['冷链订单数']
            print(f"    {wh}: 合规率 {df_wh.iloc[idx]['冷链合规率%']:.1f}% "
                  f"({n_cold:.0f}单冷链) 得分 {cold_scores[idx]:.1f}")

    # ---- TOPSIS 辅助分析 ----
    print(f"\n{'='*70}")
    print("  TOPSIS 辅助排名 (识别距最优/最劣方案最近的仓库)")
    print(f"{'='*70}")
    topsis_sorted = np.argsort(-C_topsis)
    print(f"  Top 5: ", end="")
    for i, idx in enumerate(topsis_sorted[:5]):
        wh = df_wh.iloc[idx]['仓库编码']
        print(f"#{i+1} {wh}({C_topsis[idx]:.4f})  ", end="")
    print(f"\n  Bottom 5: ", end="")
    for i, idx in enumerate(topsis_sorted[-5:][::-1]):
        wh = df_wh.iloc[idx]['仓库编码']
        print(f"#{m-i} {wh}({C_topsis[idx]:.4f})  ", end="")
    print()

    # Spearman秩相关系数: TOPSIS vs 综合得分
    rho, pval = spearmanr(C_topsis, wh_scores)
    print(f"\n  秩相关验证:")
    print(f"  TOPSIS贴近度 vs 综合得分 Spearman ρ = {rho:.4f} (p = {pval:.4f})")

    return wh_scores, system_score


def print_subjective_comparison(weights, score_names, S, df_wh):
    """与原始主观权重对比"""
    subjective = {
        '配送密度': 0.20, '路线潜力': 0.15, '时效达标': 0.15,
        '线路集中': 0.15, '冷链合规': 0.25, '装载饱和': 0.10,
    }
    vd = df_wh['车辆日数'].values
    sub_scores = S @ np.array([subjective[n] for n in score_names])
    ent_scores = S @ weights
    sub_sys = np.average(sub_scores, weights=vd)
    ent_sys = np.average(ent_scores, weights=vd)

    print(f"\n{'='*70}")
    print("  熵权法(客观) vs 主观赋权 对比")
    print(f"{'='*70}")
    print(f"\n  {'指标':<10s} {'主观权重':>8s} {'熵权权重':>8s} {'差异':>8s}")
    print(f"  {'-'*38}")
    for i, name in enumerate(score_names):
        sw = subjective.get(name, 0)
        ew = weights[i]
        direction = '↑' if ew - sw > 0.01 else '↓' if ew - sw < -0.01 else '→'
        print(f"  {name:<8s} {sw*100:>7.1f}% {ew*100:>7.1f}% "
              f"{(ew-sw)*100:>+7.1f}% {direction}")
    print(f"\n  主观赋权系统得分: {sub_sys:.1f} / 100")
    print(f"  熵权赋权系统得分: {ent_sys:.1f} / 100")
    print(f"  差异: {ent_sys - sub_sys:+.1f} 分")

    return sub_sys, ent_sys


# ============================================================
# 6. CW节约算法优化 (保留原始优化逻辑)
# ============================================================

def cw_solve(wh_coords, cities, city_qty, must_cold, max_inter_city_km=None):
    """CW节约算法: 对给定仓库的目的地执行路线合并"""
    MAX_INTER_CITY_KM = max_inter_city_km if max_inter_city_km is not None else CW3_MAX_INTER_CITY_KM
    MAX_TOTAL_ROUTE_KM = CW3_MAX_TOTAL_ROUTE_KM
    MAX_STOPS_PER_ROUTE = CW3_MAX_STOPS_PER_ROUTE
    MAX_QTY = CW3_MAX_QTY_COLD if must_cold else CW3_MAX_QTY_REGULAR

    savings = []
    for i, ci in enumerate(cities):
        for j, cj in enumerate(cities):
            if i >= j:
                continue
            ci_c, cj_c = get_coords(ci), get_coords(cj)
            if ci_c[0] is None or cj_c[0] is None:
                continue
            d_ij = road_distance(ci_c[0], ci_c[1], cj_c[0], cj_c[1])
            if d_ij > MAX_INTER_CITY_KM:
                continue
            bi = compute_bearing(ci_c[0], ci_c[1], wh_coords[0], wh_coords[1])
            bj = compute_bearing(cj_c[0], cj_c[1], wh_coords[0], wh_coords[1])
            diff = abs(bi - bj)
            if diff > 180:
                diff = 360 - diff
            if diff > 90:
                continue
            d_0i = road_distance(wh_coords[0], wh_coords[1], ci_c[0], ci_c[1])
            d_0j = road_distance(wh_coords[0], wh_coords[1], cj_c[0], cj_c[1])
            saving = d_0i + d_0j - d_ij
            if saving > 0:
                savings.append((saving, ci, cj, d_ij))

    savings.sort(key=lambda x: -x[0])
    roots = {c: c for c in cities}
    members = {c: [c] for c in cities}
    total_qty = {c: city_qty.get(c, 0) for c in cities}

    def find(c):
        while roots.get(c, c) != c:
            roots[c] = roots.get(roots.get(c, roots[c]), roots[c])
            c = roots[c]
        return c

    def route_total_dist(city_list):
        ordered = tsp_order(wh_coords, city_list)
        return route_distance(wh_coords, ordered)

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if total_qty[ra] + total_qty[rb] > MAX_QTY:
            return False
        if len(members[ra]) + len(members[rb]) > MAX_STOPS_PER_ROUTE:
            return False
        if route_total_dist(members[ra] + members[rb]) > MAX_TOTAL_ROUTE_KM:
            return False
        roots[rb] = ra
        members[ra].extend(members[rb])
        total_qty[ra] += total_qty[rb]
        del members[rb], total_qty[rb]
        return True

    for s, ci, cj, dij in savings:
        union(ci, cj)

    seen = set()
    result = []
    for c in cities:
        r = find(c)
        if r not in seen:
            seen.add(r)
            result.append((members[r], total_qty[r]))
    return result


def _select_vehicle(total_qty, dist, must_cold):
    if must_cold:
        if total_qty <= CW3_COLD_SMALL_MAX_QTY and dist <= CW3_COLD_SMALL_MAX_DIST:
            return '小型冷藏车'
        elif total_qty <= CW3_COLD_76_MAX_QTY:
            return '7.6M冷藏车'
        else:
            return '9.6M冷藏车'
    else:
        if total_qty <= CW3_REG_SMALL_MAX_QTY:
            return '金杯/面包车'
        elif total_qty <= CW3_REG_MEDIUM_MAX_QTY:
            return '中型厢货'
        elif total_qty <= CW3_REG_76_MAX_QTY:
            return '7.6M厢货'
        else:
            return '9.6M厢货'


def _vehicle_cost(vehicle):
    return VEHICLE_COST_PARAMS.get(vehicle, {'var': 1.5, 'fixed': 700})


def _map_vcat_to_vtype(vcat, is_cold):
    if is_cold:
        return VEHICLE_CAT_TO_TYPE_COLD.get(vcat, '9.6M冷藏车')
    else:
        return VEHICLE_CAT_TO_TYPE_REGULAR.get(vcat, '7.6M厢货')


def optimize_vrp_cw(dispatch_df, max_inter_city_km=None):
    """CW节约算法优化: 第1层TSP排序 + 第2层同城合并 + 第3层车型+冷链"""
    verbose = max_inter_city_km is None
    if verbose:
        print("\n  正在执行三层优化...")
    df = dispatch_df.dropna(subset=['发货城市', '收货城市', '启运dt']).copy()
    df['日期'] = df['启运dt'].dt.date

    n_records = len(dispatch_df)
    n_vehicles = dispatch_df['车牌号'].nunique()
    vehicle_day_groups = df.groupby(['车牌号', '日期'])
    n_vehicle_days = vehicle_day_groups.ngroups

    # 收集车辆日信息
    vehicle_day_info = {}
    orig_cost = 0.0
    for (plate, date), g in vehicle_day_groups:
        sc = g['发货城市'].iloc[0]
        sc_c = get_coords(sc)
        if sc_c[0] is None:
            continue
        dests_valid = [d for d in list(dict.fromkeys(g['收货城市'].dropna().tolist()))
                       if get_coords(d)[0] is not None]
        if not dests_valid:
            continue
        ordered_actual = [d for d in list(dict.fromkeys(
            g.sort_values('启运dt')['收货城市'].dropna().tolist())) if get_coords(d)[0]]
        actual_dist = route_distance(sc_c, ordered_actual)
        vcat = g['车辆分类'].mode().iloc[0] if len(g['车辆分类'].mode()) > 0 else '中型车'
        is_cold = g['是冷藏车'].mean() > 0.5
        vname = _map_vcat_to_vtype(vcat, is_cold)
        vc = _vehicle_cost(vname)
        days = max(1, math.ceil(actual_dist / (AVG_SPEED * DRIVE_HOURS)))
        orig_cost += vc['var'] * actual_dist + vc['fixed'] * days
        vehicle_day_info[(plate, date)] = {
            'origin': sc, 'origin_coords': sc_c,
            'dests_valid': dests_valid,
            'actual_dist': actual_dist, 'vehicle_cat': vcat,
            'is_cold': is_cold, 'total_qty': g['总件数'].sum(),
        }

    # 第1层: TSP排序
    tsp_cost = 0.0
    for (plate, date), info in vehicle_day_info.items():
        sc_c = info['origin_coords']
        ordered_tsp = tsp_order(sc_c, info['dests_valid'])
        tsp_dist = route_distance(sc_c, ordered_tsp)
        vname = _map_vcat_to_vtype(info['vehicle_cat'], info['is_cold'])
        vc = _vehicle_cost(vname)
        days = max(1, math.ceil(tsp_dist / (AVG_SPEED * DRIVE_HOURS)))
        tsp_cost += vc['var'] * tsp_dist + vc['fixed'] * days

    # 第2层: 同城同日合并
    city_day_groups = defaultdict(list)
    for (plate, date), info in vehicle_day_info.items():
        city_day_groups[(info['origin'], date)].append((plate, info))

    merged_cost = 0.0
    merged_routes = 0
    merged_examples = []

    for (origin, date), vd_list in city_day_groups.items():
        sc_c = get_coords(origin)
        if sc_c[0] is None:
            continue
        cold_vds = [(p, i) for p, i in vd_list if i['is_cold']]
        reg_vds = [(p, i) for p, i in vd_list if not i['is_cold']]

        for vds, must_cold in [(cold_vds, True), (reg_vds, False)]:
            if not vds:
                continue
            all_dests = []
            for _, info in vds:
                all_dests.extend(info['dests_valid'])
            all_dests = list(dict.fromkeys(all_dests))

            if len(all_dests) <= 1:
                for _, info in vds:
                    ordered = tsp_order(sc_c, info['dests_valid'])
                    dist = route_distance(sc_c, ordered)
                    v = _select_vehicle(info['total_qty'], dist, must_cold)
                    vc = _vehicle_cost(v)
                    days = max(1, math.ceil(dist / (AVG_SPEED * DRIVE_HOURS)))
                    merged_cost += vc['var'] * dist + vc['fixed'] * days
                    merged_routes += 1
                continue

            dests_valid = [d for d in all_dests if get_coords(d)[0] is not None]
            if len(dests_valid) <= 1:
                for _, info in vds:
                    ordered = tsp_order(sc_c, info['dests_valid'])
                    dist = route_distance(sc_c, ordered)
                    v = _select_vehicle(info['total_qty'], dist, must_cold)
                    vc = _vehicle_cost(v)
                    days = max(1, math.ceil(dist / (AVG_SPEED * DRIVE_HOURS)))
                    merged_cost += vc['var'] * dist + vc['fixed'] * days
                    merged_routes += 1
                continue

            city_qty_map = {}
            for _, info in vds:
                n_d = max(len(info['dests_valid']), 1)
                for d in info['dests_valid']:
                    if get_coords(d)[0] is not None:
                        city_qty_map[d] = city_qty_map.get(d, 0) + info['total_qty'] / n_d

            cw_result = cw_solve(sc_c, dests_valid, city_qty_map, must_cold, max_inter_city_km)
            for route_cities, route_qty in cw_result:
                ordered = tsp_order(sc_c, route_cities)
                dist = route_distance(sc_c, ordered)
                v = _select_vehicle(route_qty, dist, must_cold)
                vc = _vehicle_cost(v)
                days = max(1, math.ceil(dist / (AVG_SPEED * DRIVE_HOURS)))
                merged_cost += vc['var'] * dist + vc['fixed'] * days
                merged_routes += 1
                if len(merged_examples) < 10 and len(route_cities) >= 3:
                    cities_str = ' → '.join(route_cities[:6])
                    if len(route_cities) > 6:
                        cities_str += ' ...'
                    merged_examples.append(
                        f"[{v}] {origin}→{cities_str} | "
                        f"{len(route_cities)}站 {dist:.0f}km ¥{dist*vc['var']+vc['fixed']*days:.0f} "
                        f"{'❄' if must_cold else '常'}")

    return {
        '原始配送记录': n_records,
        '原始车辆数': n_vehicles,
        '原始车辆日': n_vehicle_days,
        '原始成本万元': orig_cost / 10000,
        'TSP优化成本万元': tsp_cost / 10000,
        'TSP节约率%': (orig_cost - tsp_cost) / orig_cost * 100 if orig_cost > 0 else 0,
        '合并优化路线数': merged_routes,
        '合并优化成本万元': merged_cost / 10000,
        '合并优化节约率%': (orig_cost - merged_cost) / orig_cost * 100 if orig_cost > 0 else 0,
        '趟次减少': n_vehicle_days - merged_routes,
        '趟次减少率%': (n_vehicle_days - merged_routes) / n_vehicle_days * 100 if n_vehicle_days > 0 else 0,
        '成本节约万元': (orig_cost - merged_cost) / 10000,
        '成本节约率%': (orig_cost - merged_cost) / orig_cost * 100 if orig_cost > 0 else 0,
        '路线示例': merged_examples,
    }


def print_optimization(opt):
    print(f"\n{'—'*50}")
    print("【派车优化方案 — 三层递进优化】")
    print(f"  原始数据:")
    print(f"    配送记录: {opt['原始配送记录']:,} | 车辆数: {opt['原始车辆数']:,}")
    print(f"    车辆日: {opt['原始车辆日']:,} | 原始成本: {opt['原始成本万元']:,.1f} 万元")
    print(f"  第1层 — TSP路线排序:")
    print(f"    成本: {opt['TSP优化成本万元']:,.1f} 万元 | 节约: {opt['TSP节约率%']:.1f}%")
    print(f"  第2+3层 — 同城合并 + 车型匹配 + 冷链强制:")
    print(f"    路线数: {opt['合并优化路线数']:,} | 成本: {opt['合并优化成本万元']:,.1f} 万元")
    print(f"    总节约: {opt['合并优化节约率%']:.1f}% | 车辆日减少: {opt['趟次减少']:,} "
          f"({opt['趟次减少率%']:.1f}%)")
    if opt['成本节约万元'] >= 0:
        print(f"    成本节约: {opt['成本节约万元']:,.1f} 万元")
    if opt.get('路线示例'):
        print(f"  合并路线示例:")
        for ex in opt['路线示例'][:6]:
            print(f"    • {ex}")


# ============================================================
# 7. 主程序
# ============================================================

def main():
    print("=" * 70)
    print("  医药冷链物流派车合理性评估与优化 — 熵权法 + 行业基准")
    print("=" * 70)

    print("\n正在加载数据...")
    waybill_df = load_waybill_data()
    dispatch_df = load_dispatch_data()
    print(f"  运单: {len(waybill_df):,}行 | 派车单: {len(dispatch_df):,}行")
    print(f"  城市: {waybill_df['城市'].nunique()} | 仓库编码: {waybill_df['仓库编码'].nunique()}")
    print(f"  车辆: {dispatch_df['车牌号'].nunique()} | 冷链订单: {waybill_df['冷链需求'].sum():,}")

    # Step 1: 仓库级指标
    print(f"\n{'='*70}")
    print("  Step 1: 按仓库聚合计算6项原始指标")
    print(f"{'='*70}")
    df_wh = compute_warehouse_indicators(waybill_df, dispatch_df)
    print(f"  有效仓库数: {len(df_wh)} (覆盖全部有配送记录的仓库)")
    print(f"  总车辆日: {df_wh['车辆日数'].sum():.0f}")
    small = df_wh['冷链订单数'] < 3
    if small.sum() > 0:
        print(f"  冷链订单<3单的仓库: {small.sum()} 个 (合规率按实际计算, 在输出中标注)")

    # Step 2: 绝对基准评分
    print(f"\n{'='*70}")
    print("  Step 2: 行业基准绝对评分 (0-100)")
    print(f"{'='*70}")
    S, score_names = compute_absolute_scores(df_wh)
    print(f"  各指标得分描述性统计:")
    print(f"  {'指标':<10s} {'均值':>7s} {'标准差':>7s} {'最小值':>7s} {'最大值':>7s}")
    print(f"  {'-'*38}")
    for i, name in enumerate(score_names):
        print(f"  {name:<8s} {S[:, i].mean():>7.1f} {S[:, i].std():>7.1f} "
              f"{S[:, i].min():>7.1f} {S[:, i].max():>7.1f}")

    # Step 3: 熵权法
    print(f"\n{'='*70}")
    print("  Step 3: 熵权法客观赋权")
    print(f"{'='*70}")
    weights, entropy, diff_coef = entropy_weight_method(S)

    # 熵权法敏感性分析: 剔除冷链合规0分极端仓库
    entropy_sens = entropy_sensitivity_analysis(S, score_names, df_wh)

    # Step 4: TOPSIS 辅助
    C_topsis, D_pos, D_neg = topsis_ranking(S, weights)

    # Step 5: 综合评估
    wh_scores, system_score = print_evaluation(
        df_wh, S, score_names, weights, entropy, diff_coef, C_topsis, D_pos, D_neg)

    # Step 6: 与主观权重对比
    sub_sys, ent_sys = print_subjective_comparison(weights, score_names, S, df_wh)

    # Step 7: 优化
    print(f"\n{'='*70}")
    print("  Step 4: CW节约算法优化")
    print(f"{'='*70}")
    optimization = optimize_vrp_cw(dispatch_df)
    print_optimization(optimization)

    # 灵敏度分析: 城市间距离阈值
    print(f"\n{'='*70}")
    print("  灵敏度分析: 城市间距离阈值对优化结果的影响")
    print(f"{'='*70}")
    print(f"  {'距离阈值':<12s} {'路线数':>8s} {'成本(万元)':>10s} {'节约率':>8s}")
    print(f"  {'-'*42}")
    for dist in CW3_SENSITIVITY_DISTANCES:
        opt_s = optimize_vrp_cw(dispatch_df, max_inter_city_km=dist)
        print(f"  {dist} km{'':<7s} {opt_s['合并优化路线数']:>8,} "
              f"{opt_s['合并优化成本万元']:>10.1f} {opt_s['合并优化节约率%']:>8.1f}%")

    return df_wh, S, weights, wh_scores, system_score, score_names, optimization


if __name__ == '__main__':
    df_wh, S, weights, wh_scores, system_score, score_names, optimization = main()
