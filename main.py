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
SAMPLES_PER_HOUR = 256
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
        if stamp is None:
            continue
        age_hours = max(0.0, (start - stamp).total_seconds() / 3600.0)
        for field, target in (("up_and_coming", "uc"), ("top_trending", "trend"), ("top_playing_now", "playing_now")):
            for fallback_rank, item in enumerate(rows_from_chart(read.get(field)), 1):
                if not isinstance(item, dict):
                    continue
                candidate = get_candidate(item.get("universe_id"))
                if candidate is None:
                    continue
                rank = number(item.get("rank")) or float(fallback_rank)
                obs = {"age": age_hours, "rank": max(1.0, rank), "playing": number(item.get("playing"))}
                if target == "uc":
                    obs["up"] = number(item.get("up_votes"))
                    obs["down"] = number(item.get("down_votes"))
                candidate[target].append(obs)
                if item.get("name"):
                    candidate["name"] = str(item["name"])
        for raw_uid, item in detail_rows(read.get("details")):
            candidate = get_candidate(raw_uid)
            if candidate is not None:
                candidate["details"].append({"age": age_hours, "data": item})
                if item.get("name"):
                    candidate["name"] = str(item["name"])
    return candidates


def weighted_log_rank(observations):
    if not observations:
        return None
    weighted = []
    for obs in observations:
        age = obs["age"]
        weight = math.exp(-age / 30.0)
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


def utility_features(candidates, start):
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
        rank_log = weighted_log_rank(c["uc"])
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
        rank_log += min(math.log(5.0), max(0.0, last_uc_age - 2.0) * math.log(2.0) / 48.0)
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
    for (previous_time, previous), (current_time, current) in zip(snapshots, snapshots[1:]):
        gap = (current_time - previous_time).total_seconds() / 3600.0
        if 0.5 <= gap <= 24.0:
            retention = max(0.1, len(set(previous) & set(current)) / 10.0)
            daily_rates.append(-math.log(retention) * 24.0 / gap)

    if not daily_rates:
        return 0.12
    daily_rates.sort()
    typical_daily_rate = daily_rates[len(daily_rates) // 2]
    return clamp(0.08 + 0.04 * math.sqrt(typical_daily_rate), 0.08, 0.20)


def forecast(history, task):
    start = parse_time(task.get("window", {}).get("from")) or parse_time(task.get("history_until"))
    if start is None:
        start = parse_time(history[-1].get("at")) if history else None
    if start is None:
        start = dt.datetime(1970, 1, 1, tzinfo=UTC)
    slots = int(task.get("window", {}).get("slots", 72) or 72)
    candidates = make_candidates(history, start)
    features = utility_features(candidates, start)
    noise_growth = inferred_noise_growth(history, start)

    # Recent chart ranks give a strong short-horizon prior. Gumbel noise grows
    # with the forecast horizon, letting the chart turn over without assuming
    # that the latest order persists for all three days.
    uids = sorted(features)
    unknown_start = len(uids)
    base = []
    for uid in uids:
        f = features[uid]
        base.append(-f["rank_log"] + f["trend_bonus"] + f["playing_now_bonus"] + f["player_level"] * 0.5 + f["vote_signal"] + f["age_bonus"] + f["update_bonus"] + f["detail_growth"])
    for _ in range(UNKNOWN_POOL):
        base.append(-math.log(170.0))

    counts = [[0] * 10 for _ in base]
    rng = random.Random(271828)
    for hour in range(slots):
        sigma = 0.20 + noise_growth * math.sqrt(hour)
        progress = hour / max(1, slots - 1)
        dummy_rank = max(36.0, 170.0 - 132.0 * progress)
        for draw in range(SAMPLES_PER_HOUR):
            scores = []
            for i, value in enumerate(base):
                if i >= unknown_start:
                    utility = -math.log(dummy_rank)
                else:
                    f = features[uids[i]]
                    utility = value + 0.28 * f["momentum"] * math.exp(-hour / 36.0)
                    utility += 0.11 * f["player_growth"] * math.exp(-hour / 48.0)
                    utility -= min(math.log(5.0), max(0.0, f["last_age"] - 2.0) / 48.0 * math.log(2.0)) * 0.28
                    utility -= progress * min(0.28, max(0.0, f["last_age"] - 10.0) / 240.0)
                    utility -= progress * min(0.22, max(0.0, f["chart_tenure"] - 48.0) / 360.0)
                uniform = min(1.0 - 1e-12, max(1e-12, rng.random()))
                noise = -math.log(-math.log(uniform))
                scores.append((utility + sigma * noise, i))
            best = heapq.nlargest(10, scores)
            for place, (_, index) in enumerate(best):
                if index < unknown_start:
                    counts[index][place] += 1

    total_draws = max(1, slots * SAMPLES_PER_HOUR)
    probabilities = [[counts[i][place] / total_draws for place in range(10)] for i in range(unknown_start)]
    rows = []
    for place in range(10):
        order = sorted(range(unknown_start), key=lambda i: probabilities[i][place], reverse=True)
        for i in order[:100]:
            p = probabilities[i][place]
            if p >= 0.0007:
                rows.append({"universe_id": uids[i], "place": place + 1, "p": round(p, 8)})

    # A separate expected-order summary drives the top10 tie-break field.
    order_strength = []
    for i, uid in enumerate(uids):
        p = probabilities[i]
        hit_weight = sum((11 - k) * p[k - 1] for k in range(1, 11))
        expected_place = sum(k * p[k - 1] for k in range(1, 11)) / max(1e-12, sum(p))
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
