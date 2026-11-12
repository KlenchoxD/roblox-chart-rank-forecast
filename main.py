#!/usr/bin/env python3
"""Deterministic forecast for Roblox's hourly Up-and-Coming chart."""

import datetime as dt
import heapq
import json
import math
import os
import random
from pathlib import Path


UTC = dt.timezone.utc
# 640 trajectories per slot reduce rank-probability Monte Carlo noise at low cost.
SAMPLES_PER_HOUR = 640
UNKNOWN_POOL = 24


def parse_time(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except (TypeError, ValueError, OverflowError):
        return None


def rows_from_chart(chart):
    if isinstance(chart, list):
        return chart
    if isinstance(chart, dict):
        for key in ("games", "data", "items"):
            value = chart.get(key)
            if isinstance(value, list):
                return value
    return []


def detail_rows(details):
    if isinstance(details, list):
        return [(item.get("universe_id"), item) for item in details if isinstance(item, dict)]
    if isinstance(details, dict):
        if "universe_id" in details:
            return [(details.get("universe_id"), details)]
        result = []
        for key, value in details.items():
            if isinstance(value, dict):
                result.append((value.get("universe_id", key), value))
        return result
    return []


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def safe_id(value):
    try:
        result = int(value)
        return result if result > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def clamp(value, low, high):
    return min(high, max(low, value))


def log_count(value):
    value = number(value)
    return math.log1p(max(0.0, value)) if value is not None else None


def chart_observation(item, fallback_rank, age_hours, target):
    rank = number(item.get("rank"))
    if rank is None or rank <= 0.0:
        rank = float(fallback_rank)
    observation = {"age": age_hours, "rank": rank, "playing": number(item.get("playing"))}
    if target == "uc":
        observation["up"] = number(item.get("up_votes"))
        observation["down"] = number(item.get("down_votes"))
    return observation


def make_candidates(history, start):
    candidates = {}

    def get_candidate(uid):
        uid = safe_id(uid)
        if uid is None:
            return None
        if uid not in candidates:
            candidates[uid] = {"uid": uid, "uc": [], "trend": [], "playing_now": [], "details": [], "name": ""}
        return candidates[uid]

    for read in history:
        stamp = parse_time(read.get("at"))
        if stamp is None or stamp > start:
            continue
        age_hours = (start - stamp).total_seconds() / 3600.0
        for field, target in (("up_and_coming", "uc"), ("top_trending", "trend"), ("top_playing_now", "playing_now")):
            observed_ids = set()
            for fallback_rank, item in enumerate(rows_from_chart(read.get(field)), 1):
                if not isinstance(item, dict):
                    continue
                uid = safe_id(item.get("universe_id"))
                if uid is None or uid in observed_ids:
                    continue
                observed_ids.add(uid)
                candidate = get_candidate(uid)
                if candidate is None:
                    continue
                candidate[target].append(chart_observation(item, fallback_rank, age_hours, target))
                if item.get("name"):
                    candidate["name"] = str(item["name"])
        observed_details = set()
        for raw_uid, item in detail_rows(read.get("details")):
            uid = safe_id(raw_uid)
            if uid is None or uid in observed_details:
                continue
            observed_details.add(uid)
            candidate = get_candidate(uid)
            if candidate is not None:
                candidate["details"].append({"age": age_hours, "data": item})
                if item.get("name"):
                    candidate["name"] = str(item["name"])
    return candidates


def weighted_log_rank(observations, half_life):
    if not observations:
        return None
    decay = max(6.0, half_life)
    weighted = []
    for obs in observations:
        age = obs["age"]
        weight = math.exp(-age / decay)
        if age <= 2.0:
            weight *= 2.0
        weighted.append((math.log(max(1.0, obs["rank"])), weight))
    total = sum(weight for _, weight in weighted)
    return sum(value * weight for value, weight in weighted) / max(total, 1e-12)


def trend_from_ranks(observations):
    recent = [(math.log(o["rank"]), math.exp(-o["age"] / 18.0)) for o in observations if o["age"] <= 30.0]
    prior = [(math.log(o["rank"]), math.exp(-(o["age"] - 30.0) / 48.0)) for o in observations if 30.0 < o["age"] <= 120.0]
    if not recent or not prior:
        return 0.0
    rmean = sum(v * w for v, w in recent) / sum(w for _, w in recent)
    pmean = sum(v * w for v, w in prior) / sum(w for _, w in prior)
    return clamp(pmean - rmean, -1.4, 1.4)


def growth_signal(observations, key):
    usable = []
    for obs in observations:
        value = log_count(obs["data"].get(key))
        if value is not None:
            usable.append((obs["age"], value))
    usable.sort()
    if len(usable) < 2:
        return 0.0
    newest = usable[0]
    older = next((item for item in usable[1:] if item[0] - newest[0] >= 6.0), usable[-1])
    gap = max(1.0, older[0] - newest[0])
    return clamp((newest[1] - older[1]) / math.sqrt(gap / 24.0), -1.5, 1.5)


def utility_features(candidates, start, rank_half_life):
    latest_playing = []
    for candidate in candidates.values():
        current = [o for o in candidate["uc"] if o["age"] <= 36.0 and o["playing"] is not None]
        if current:
            newest = min(current, key=lambda o: o["age"])
            latest_playing.append(math.log1p(max(0.0, newest["playing"])))
    latest_playing.sort()
    median_playing = latest_playing[len(latest_playing) // 2] if latest_playing else 0.0

    features = {}
    for uid, c in candidates.items():
        rank_log = weighted_log_rank(c["uc"], rank_half_life)
        latest = min(c["uc"], key=lambda o: o["age"]) if c["uc"] else None
        trend_obs = [o for o in c["trend"] if o["age"] <= 12.0]
        play_now_obs = [o for o in c["playing_now"] if o["age"] <= 12.0]
        trend_rank = min((o["rank"] for o in trend_obs), default=None)
        playing_now_rank = min((o["rank"] for o in play_now_obs), default=None)
        rank_momentum = trend_from_ranks(c["uc"])

        if rank_log is None:
            # A current trending or popular game can be an entrant even if it
            # has not reached this chart yet.
            seed_rank = 125.0
            if trend_rank is not None:
                seed_rank = min(seed_rank, 48.0 + 1.15 * trend_rank)
            if playing_now_rank is not None:
                # Top Playing Now is a weaker discovery signal than either
                # this chart or Top Trending, so it seeds candidates gently.
                seed_rank = min(seed_rank, 165.0 + 0.8 * playing_now_rank)
            rank_log = math.log(seed_rank)
        last_uc_age = latest["age"] if latest else (0.0 if trend_rank is not None or playing_now_rank is not None else 999.0)
        stale_age = max(0.0, last_uc_age - 2.0)
        stale_scale = max(12.0, 1.6 * rank_half_life)
        rank_log += min(math.log(5.0), stale_age * math.log(2.0) / stale_scale)
        chart_tenure = 0.0
        if len(c["uc"]) > 1:
            chart_tenure = max(o["age"] for o in c["uc"]) - min(o["age"] for o in c["uc"])

        current_play = latest["playing"] if latest else None
        player_level = 0.0
        if current_play is not None:
            player_level = clamp((math.log1p(max(0.0, current_play)) - median_playing) * 0.10, -0.28, 0.28)
        old_play = next((o for o in c["uc"] if o["age"] >= 20.0 and o["age"] <= 100.0 and o["playing"] is not None), None)
        player_growth = 0.0
        if latest and current_play is not None and old_play and old_play["playing"] is not None:
            player_growth = clamp(math.log1p(max(0.0, current_play)) - math.log1p(max(0.0, old_play["playing"])), -1.5, 1.5)

        vote_signal = 0.0
        if latest and latest.get("up") is not None and latest.get("down") is not None:
            up = max(0.0, latest["up"])
            down = max(0.0, latest["down"])
            total = up + down
            if total >= 5.0:
                vote_signal = clamp(((up + 1.0) / (total + 2.0) - 0.82) * 1.3, -0.20, 0.20)

        detail = min(c["details"], key=lambda o: o["age"])["data"] if c["details"] else {}
        game_age_bonus = 0.0
        created = parse_time(detail.get("created")) if isinstance(detail, dict) else None
        if created is not None:
            age_days = max(0.0, (start - created).total_seconds() / 86400.0)
            game_age_bonus = 0.18 * math.exp(-age_days / 18.0)
        update_bonus = 0.0
        updated = parse_time(detail.get("updated")) if isinstance(detail, dict) else None
        if updated is not None:
            update_days = max(0.0, (start - updated).total_seconds() / 86400.0)
            update_bonus = 0.14 * math.exp(-update_days / 9.0)

        detail_growth = 0.055 * (growth_signal(c["details"], "favorites") + growth_signal(c["details"], "visits"))
        detail_growth = clamp(detail_growth, -0.16, 0.16)
        trend_bonus = 0.0 if trend_rank is None else 0.22 * math.exp(-max(0.0, trend_rank - 1.0) / 24.0)
        playing_now_bonus = 0.0 if playing_now_rank is None else 0.12 * math.exp(-max(0.0, playing_now_rank - 1.0) / 40.0)

        features[uid] = {
            "rank_log": rank_log,
            "momentum": rank_momentum,
            "trend_bonus": trend_bonus,
            "playing_now_bonus": playing_now_bonus,
            "player_level": player_level,
            "player_growth": player_growth,
            "vote_signal": vote_signal,
            "age_bonus": game_age_bonus,
            "update_bonus": update_bonus,
            "detail_growth": detail_growth,
            "last_age": last_uc_age,
            "chart_tenure": chart_tenure,
        }
    return features


def inferred_noise_growth(history, start):
    snapshots = []
    for read in history:
        stamp = parse_time(read.get("at"))
        if stamp is None:
            continue
        age = (start - stamp).total_seconds() / 3600.0
        if age < 0.0 or age > 96.0:
            continue
        top = []
        for item in rows_from_chart(read.get("up_and_coming"))[:10]:
            if isinstance(item, dict):
                uid = safe_id(item.get("universe_id"))
                if uid is not None:
                    top.append(uid)
        if len(top) == 10:
            snapshots.append((stamp, top))

    snapshots.sort(key=lambda item: item[0])
    daily_rates = []
    rank_motion = []
    for (previous_time, previous), (current_time, current) in zip(snapshots, snapshots[1:]):
        gap = (current_time - previous_time).total_seconds() / 3600.0
        if 0.5 <= gap <= 24.0:
            retention = max(0.1, len(set(previous) & set(current)) / 10.0)
            daily_rates.append(-math.log(retention) * 24.0 / gap)
            previous_rank = {uid: rank for rank, uid in enumerate(previous)}
            current_rank = {uid: rank for rank, uid in enumerate(current)}
            shared = previous_rank.keys() & current_rank.keys()
            if shared:
                displacement = sum(abs(previous_rank[uid] - current_rank[uid]) for uid in shared) / len(shared)
                rank_motion.append(displacement * 24.0 / (9.0 * gap))

    daily_rates.sort()
    rank_motion.sort()
    # Membership churn captures replacement; rank motion captures turbulence
    # that overlap alone misses when the same games trade positions.
    typical_turnover = daily_rates[len(daily_rates) // 2] if daily_rates else 1.0
    typical_motion = rank_motion[len(rank_motion) // 2] if rank_motion else 0.0
    turnover_noise = 0.04 * math.sqrt(max(0.0, typical_turnover))
    motion_noise = 0.025 * math.sqrt(max(0.0, typical_motion))
    return clamp(0.08 + turnover_noise + motion_noise, 0.08, 0.22)


def recent_chart_newness(history, start):
    first_seen_age = {}
    latest_stamp = None
    latest_ids = []
    for read in history:
        stamp = parse_time(read.get("at"))
        if stamp is None or stamp > start:
            continue
        age = max(0.0, (start - stamp).total_seconds() / 3600.0)
        ids = []
        # Measure churn on the scored top ten, not unseen lower chart ranks.
        for item in rows_from_chart(read.get("up_and_coming"))[:10]:
            if not isinstance(item, dict):
                continue
            uid = safe_id(item.get("universe_id"))
            if uid is None:
                continue
            ids.append(uid)
            first_seen_age[uid] = max(age, first_seen_age.get(uid, 0.0))
        if latest_stamp is None or stamp > latest_stamp:
            latest_stamp = stamp
            latest_ids = list(dict.fromkeys(ids))
    if not latest_ids:
        return 0.0
    newcomers = sum(first_seen_age.get(uid, float("inf")) <= 24.0 for uid in latest_ids)
    return newcomers / len(latest_ids)


def empirical_rank_volatility(history, start):
    snapshots = []
    for read in history:
        stamp = parse_time(read.get("at"))
        if stamp is None or stamp > start:
            continue
        age = (start - stamp).total_seconds() / 3600.0
        if age > 120.0:
            continue
        ranks = {}
        for fallback_rank, item in enumerate(rows_from_chart(read.get("up_and_coming")), 1):
            if not isinstance(item, dict):
                continue
            uid = safe_id(item.get("universe_id"))
            rank = number(item.get("rank")) or float(fallback_rank)
            if uid is not None and uid not in ranks:
                ranks[uid] = max(1.0, rank)
        if ranks:
            snapshots.append((stamp, ranks))

    snapshots.sort(key=lambda item: item[0])
    all_rates = []
    top_rates = []
    for (previous_time, previous), (current_time, current) in zip(snapshots, snapshots[1:]):
        gap = (current_time - previous_time).total_seconds() / 3600.0
        if not 0.5 <= gap <= 24.0:
            continue
        for uid in previous.keys() & current.keys():
            change = math.log(current[uid]) - math.log(previous[uid])
            rate = change / math.sqrt(gap)
            all_rates.append(rate)
            if min(previous[uid], current[uid]) <= 20.0:
                top_rates.append(rate)

    def rms(values):
        return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0

    broad = rms(all_rates)
    elite = rms(top_rates)
    if len(top_rates) >= 30:
        return clamp(0.35 * broad + 0.65 * elite, 0.06, 0.18)
    if all_rates:
        return clamp(broad, 0.06, 0.18)
    return 0.11


def normal_cdf(value):
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def rank_probability(log_mean, log_sigma, place):
    lower_rank = max(0.5, place - 0.5)
    upper_rank = place + 0.5
    upper = normal_cdf((math.log(upper_rank) - log_mean) / log_sigma)
    lower = normal_cdf((math.log(lower_rank) - log_mean) / log_sigma) if place > 1 else 0.0
    return max(0.0, upper - lower)


def age_survival(candidate, start, horizon):
    if not candidate["details"]:
        return 1.0
    detail = min(candidate["details"], key=lambda row: row["age"])["data"]
    created = parse_time(detail.get("created")) if isinstance(detail, dict) else None
    if created is None:
        return 1.0
    age_hours = max(0.0, (start - created).total_seconds() / 3600.0)
    remaining = 28.0 * 24.0 - age_hours
    if horizon <= remaining:
        return 1.0
    return math.exp(-(horizon - remaining) / 6.0)


def forecast(history, task):
    start = parse_time(task.get("window", {}).get("from")) or parse_time(task.get("history_until"))
    if start is None:
        start = parse_time(history[-1].get("at")) if history else None
    if start is None:
        start = dt.datetime(1970, 1, 1, tzinfo=UTC)
    slots = int(task.get("window", {}).get("slots", 72) or 72)
    candidates = make_candidates(history, start)
    noise_growth = inferred_noise_growth(history, start)
    rank_half_life = clamp(30.0 * 0.12 / max(0.08, noise_growth), 18.0, 48.0)
    features = utility_features(candidates, start, rank_half_life)
    newcomer_share = recent_chart_newness(history, start)
    volatility = empirical_rank_volatility(history, start)
    uids = sorted(uid for uid, c in candidates.items()
                  if features[uid]["last_age"] <= 168.0
                  or min((o["age"] for o in c["trend"] + c["playing_now"]), default=999.0) <= 36.0)
    if not uids:
        uids = sorted(features)

    # Forecast each game's place distribution from its rank trajectory. The
    # historical rank-difference scale widens the kernel with time; momentum
    # moves its center modestly, then decays rather than extrapolating linearly.
    accumulated = [[0.0] * 10 for _ in uids]
    turnover = clamp((noise_growth - 0.08) / 0.14, 0.0, 1.0)
    for hour in range(slots):
        progress = hour / max(1, slots - 1)
        known_mass = 1.0 - clamp(0.07 + 0.17 * progress + 0.08 * newcomer_share + 0.06 * turnover, 0.07, 0.42)
        by_place = [[] for _ in range(10)]
        for uid in uids:
            f = features[uid]
            candidate = candidates[uid]
            drift = clamp(0.18 * f["momentum"] + 0.05 * f["player_growth"], -0.25, 0.25)
            trend_distance = 1.0 - math.exp(-hour / 36.0)
            rank_signal = (
                0.65 * f["trend_bonus"]
                + 0.35 * f["playing_now_bonus"]
                + 0.15 * f["player_level"]
                + f["vote_signal"]
                + f["age_bonus"]
                + f["update_bonus"]
                + f["detail_growth"]
            )
            log_mean = f["rank_log"] - drift * trend_distance - clamp(rank_signal, -0.35, 0.45)
            last_age = f["last_age"]
            sigma = math.sqrt((0.16 + 0.004 * min(24.0, last_age)) ** 2 + volatility * volatility * hour)
            sigma = max(0.16, sigma)
            survival = age_survival(candidate, start, hour)
            stale_weight = math.exp(-max(0.0, last_age - 2.0) / 72.0)
            for place in range(1, 11):
                likelihood = rank_probability(log_mean, sigma, place) * survival * stale_weight
                by_place[place - 1].append(likelihood)

        hourly = [[0.0] * 10 for _ in uids]
        for place, likelihoods in enumerate(by_place):
            total = sum(likelihoods)
            if total <= 1e-15:
                continue
            for index, likelihood in enumerate(likelihoods):
                hourly[index][place] = known_mass * likelihood / total

        # A game's ten place marginals cannot claim more than one place.
        for index, values in enumerate(hourly):
            row_mass = sum(values)
            if row_mass > 1.0:
                scale = 1.0 / row_mass
                values = [value * scale for value in values]
                hourly[index] = values
            for place, probability in enumerate(values):
                accumulated[index][place] += probability / slots

    rows = []
    for place in range(10):
        order = sorted(range(len(uids)), key=lambda i: accumulated[i][place], reverse=True)
        for i in order[:100]:
            p = accumulated[i][place]
            if p >= 0.0001:
                rows.append({"universe_id": uids[i], "place": place + 1, "p": round(p, 8)})

    # Expected rank probability gives the requested ordered top-ten estimate.
    order_strength = []
    for i, uid in enumerate(uids):
        p = accumulated[i]
        hit_weight = sum((11 - place) * p[place - 1] for place in range(1, 11))
        expected_place = sum(place * p[place - 1] for place in range(1, 11)) / max(1e-12, sum(p))
        order_strength.append((uid, hit_weight, expected_place))
    order_strength.sort(key=lambda item: (-item[1], item[2], item[0]))
    top10 = [uid for uid, _, _ in order_strength[:10]]
    return {"probabilities": rows, "top10": top10}


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    task = json.loads((input_dir / "task.json").read_text(encoding="utf-8"))
    history = [json.loads(line) for line in (input_dir / "history.ndjson").read_text(encoding="utf-8").splitlines() if line.strip()]
    answer = forecast(history, task)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "forecast.json").write_text(json.dumps(answer, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
