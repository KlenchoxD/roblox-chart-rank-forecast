#!/usr/bin/env python3
"""Deterministic cohort forecast for Roblox's Up-and-Coming chart."""

import datetime as dt
import heapq
import json
import math
import os
import random
from pathlib import Path


UTC = dt.timezone.utc
PATHS = 384
MAX_CANDIDATES = 180
# Share of the empirical rank-transition table in the final probabilities. A rolling backtest on the
# kit's history (21 cutoffs, scored as RULES.txt scores) was flat from 0.4 to 0.7; 0.5 sits in the middle.
TRANSITION_WEIGHT = 0.5
HORIZON_BANDWIDTH = 8.0


def parse_time(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except (TypeError, ValueError, OverflowError):
        return None


def finite_number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def universe_id(value):
    try:
        result = int(value)
        return result if result > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def clamp(value, low, high):
    return min(high, max(low, value))


def chart_rows(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("games", "data", "items"):
            if isinstance(value.get(key), list):
                return value[key]
    return []


def detail_rows(value):
    if isinstance(value, list):
        return [(row.get("universe_id"), row) for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        if "universe_id" in value:
            return [(value.get("universe_id"), value)]
        return [
            (row.get("universe_id", key), row)
            for key, row in value.items()
            if isinstance(row, dict)
        ]
    return []


def weighted_median(pairs, default):
    usable = sorted((value, weight) for value, weight in pairs if weight > 0.0 and math.isfinite(value))
    if not usable:
        return default
    midpoint = sum(weight for _, weight in usable) / 2.0
    total = 0.0
    for value, weight in usable:
        total += weight
        if total >= midpoint:
            return value
    return usable[-1][0]


def weighted_quantile(pairs, quantile, default):
    usable = sorted((value, weight) for value, weight in pairs if weight > 0.0 and math.isfinite(value))
    if not usable:
        return default
    target = clamp(quantile, 0.0, 1.0) * sum(weight for _, weight in usable)
    total = 0.0
    for value, weight in usable:
        total += weight
        if total >= target:
            return value
    return usable[-1][0]


def new_game(uid):
    return {
        "uid": uid,
        "uc": [],
        "trending": [],
        "playing": [],
        "details": [],
    }


def collect_history(history, cutoff):
    games = {}
    snapshots = []

    def game_for(raw_uid):
        uid = universe_id(raw_uid)
        if uid is None:
            return None
        if uid not in games:
            games[uid] = new_game(uid)
        return games[uid]

    for read in history:
        stamp = parse_time(read.get("at"))
        if stamp is None or stamp > cutoff:
            continue
        age = max(0.0, (cutoff - stamp).total_seconds() / 3600.0)
        ranks = {}
        ordered = []
        for field, destination in (
            ("up_and_coming", "uc"),
            ("top_trending", "trending"),
            ("top_playing_now", "playing"),
        ):
            seen = set()
            for fallback_rank, row in enumerate(chart_rows(read.get(field)), 1):
                if not isinstance(row, dict):
                    continue
                uid = universe_id(row.get("universe_id"))
                if uid is None or uid in seen:
                    continue
                seen.add(uid)
                game = game_for(uid)
                rank = finite_number(row.get("rank")) or float(fallback_rank)
                observation = {
                    "time": stamp,
                    "age": age,
                    "rank": max(1.0, rank),
                    "playing": finite_number(row.get("playing")),
                    "up": finite_number(row.get("up_votes")),
                    "down": finite_number(row.get("down_votes")),
                }
                game[destination].append(observation)
                if destination == "uc":
                    ranks[uid] = observation["rank"]
                    ordered.append(uid)
        for raw_uid, row in detail_rows(read.get("details")):
            game = game_for(raw_uid)
            if game is not None:
                game["details"].append({"time": stamp, "age": age, "data": row})
        if ranks:
            # The rank field, not list order, defines the top 10.
            top = sorted(ordered, key=lambda uid: (ranks[uid], uid))[:10]
            snapshots.append({"time": stamp, "age": age, "ranks": ranks, "top": top})

    for game in games.values():
        for key in ("uc", "trending", "playing", "details"):
            game[key].sort(key=lambda row: row["time"])
    snapshots.sort(key=lambda row: row["time"])
    return games, snapshots


def estimate_dynamics(snapshots):
    exits = []
    motion = []
    arrivals = []
    for previous, current in zip(snapshots, snapshots[1:]):
        gap = (current["time"] - previous["time"]).total_seconds() / 3600.0
        if not 0.5 <= gap <= 18.0 or len(previous["top"]) < 8 or len(current["top"]) < 8:
            continue
        previous_top = set(previous["top"])
        current_top = set(current["top"])
        retention = clamp(len(previous_top & current_top) / 10.0, 0.05, 1.0)
        weight = math.exp(-current["age"] / 96.0)
        exits.append((-math.log(retention) / gap, weight))
        arrivals.append((len(current_top - previous_top) / gap, weight))
        shared = previous["ranks"].keys() & current["ranks"].keys()
        for uid in shared:
            old_rank = previous["ranks"][uid]
            new_rank = current["ranks"][uid]
            if min(old_rank, new_rank) <= 35.0:
                rate = abs(math.log(new_rank) - math.log(old_rank)) / math.sqrt(gap)
                motion.append((rate, weight))

    exit_hazard = clamp(weighted_median(exits, 0.018), 0.004, 0.075)
    busy_exit_hazard = clamp(weighted_quantile(exits, 0.75, exit_hazard * 1.5), exit_hazard, 0.11)
    entry_rate = clamp(weighted_median(arrivals, 0.12), 0.025, 0.30)
    rank_noise = clamp(weighted_median(motion, 0.10), 0.045, 0.24)
    newcomer_pressure = 0.0
    if snapshots and snapshots[-1]["top"]:
        first_seen = {}
        for snapshot in snapshots:
            for uid in snapshot["top"]:
                first_seen.setdefault(uid, snapshot["time"])
        newest_time = snapshots[-1]["time"]
        newcomer_pressure = sum(
            (newest_time - first_seen[uid]).total_seconds() <= 36.0 * 3600.0
            for uid in snapshots[-1]["top"]
        ) / len(snapshots[-1]["top"])
    return {
        "exit_hazard": exit_hazard,
        "busy_exit_hazard": busy_exit_hazard,
        "entry_rate": entry_rate,
        "rank_noise": rank_noise,
        "newcomer_pressure": newcomer_pressure,
    }


def latest(observations):
    return observations[-1] if observations else None


def robust_rank_velocity(observations):
    recent = [row for row in observations if row["age"] <= 120.0]
    rates = []
    for older, newer in zip(recent, recent[1:]):
        gap = (newer["time"] - older["time"]).total_seconds() / 3600.0
        if 0.5 <= gap <= 24.0:
            improvement = (math.log(older["rank"]) - math.log(newer["rank"])) / gap
            rates.append((improvement, math.exp(-newer["age"] / 48.0)))
    return clamp(weighted_median(rates, 0.0), -0.045, 0.045)


def log_growth(observations, accessor):
    usable = []
    for row in observations:
        value = accessor(row)
        if value is not None and value >= 0.0:
            usable.append((row, math.log1p(value)))
    if len(usable) < 2:
        return 0.0
    newer_row, newer_value = usable[-1]
    older = next(
        ((row, value) for row, value in reversed(usable[:-1]) if newer_row["age"] + 6.0 <= row["age"]),
        usable[0],
    )
    older_row, older_value = older
    gap = max(1.0, older_row["age"] - newer_row["age"])
    return clamp((newer_value - older_value) * 24.0 / gap, -1.8, 1.8)


def recent_rank(game, key, maximum_age):
    rows = [row for row in game[key] if row["age"] <= maximum_age]
    return min((row["rank"] for row in rows), default=None)


def build_features(games, snapshots, cutoff):
    current = snapshots[-1] if snapshots else {"time": cutoff, "age": 0.0, "ranks": {}, "top": []}
    recent_snapshots = [row for row in snapshots if row["age"] <= 72.0]
    features = {}
    for uid, game in games.items():
        newest_uc = latest(game["uc"])
        current_rank = current["ranks"].get(uid)
        trend_rank = recent_rank(game, "trending", 30.0)
        playing_rank = recent_rank(game, "playing", 30.0)
        last_age = newest_uc["age"] if newest_uc else 999.0
        presence = 0.0
        if recent_snapshots:
            presence = sum(uid in snapshot["ranks"] for snapshot in recent_snapshots) / len(recent_snapshots)

        if current_rank is not None:
            base_rank = current_rank
        else:
            base_rank = 140.0
            if newest_uc is not None:
                base_rank = min(base_rank, newest_uc["rank"] * math.exp(min(2.0, last_age / 36.0)))
            if trend_rank is not None:
                base_rank = min(base_rank, 13.0 + 1.15 * trend_rank)
            if playing_rank is not None:
                base_rank = min(base_rank, 38.0 + 1.7 * playing_rank)

        velocity = robust_rank_velocity(game["uc"])
        player_growth = log_growth(game["uc"], lambda row: row.get("playing"))
        detail_growth = 0.0
        if game["details"]:
            visits = log_growth(game["details"], lambda row: finite_number(row["data"].get("visits")))
            favorites = log_growth(game["details"], lambda row: finite_number(row["data"].get("favorites")))
            detail_growth = 0.5 * (visits + favorites)

        cross_signal = 0.0
        if trend_rank is not None:
            cross_signal += 0.34 * math.exp(-(trend_rank - 1.0) / 30.0)
        if playing_rank is not None:
            cross_signal += 0.16 * math.exp(-(playing_rank - 1.0) / 45.0)
        quality = 0.0
        if newest_uc and newest_uc.get("up") is not None and newest_uc.get("down") is not None:
            up = max(0.0, newest_uc["up"])
            down = max(0.0, newest_uc["down"])
            if up + down >= 10.0:
                quality = clamp(((up + 2.0) / (up + down + 4.0) - 0.78) * 0.9, -0.18, 0.18)

        created = None
        updated = None
        if game["details"]:
            data = game["details"][-1]["data"]
            created = parse_time(data.get("created")) if isinstance(data, dict) else None
            updated = parse_time(data.get("updated")) if isinstance(data, dict) else None
        remaining = None
        freshness = 0.0
        if created is not None:
            game_age = max(0.0, (cutoff - created).total_seconds() / 3600.0)
            remaining = 28.0 * 24.0 - game_age
            freshness += 0.16 * math.exp(-game_age / (18.0 * 24.0))
        if updated is not None:
            update_age = max(0.0, (cutoff - updated).total_seconds() / 3600.0)
            freshness += 0.12 * math.exp(-update_age / (8.0 * 24.0))

        signal = (
            cross_signal
            + quality
            + freshness
            + 0.08 * clamp(player_growth, -1.0, 1.0)
            + 0.05 * clamp(detail_growth, -1.0, 1.0)
        )
        entry_strength = clamp(
            0.04
            + (0.34 if trend_rank is not None else 0.0)
            + (0.12 if playing_rank is not None else 0.0)
            + 0.10 * max(0.0, player_growth),
            0.02,
            0.72,
        )
        features[uid] = {
            "uid": uid,
            "current": current_rank is not None,
            "base_log_rank": math.log(max(1.0, base_rank)),
            "velocity": velocity,
            "signal": clamp(signal, -0.35, 0.75),
            "entry_strength": entry_strength,
            "presence": presence,
            "last_age": last_age,
            "remaining": remaining,
        }

    ordered = sorted(
        features.values(),
        key=lambda row: (
            not row["current"],
            row["base_log_rank"] - row["signal"],
            row["uid"],
        ),
    )
    keep = [row for row in ordered if row["current"] or row["last_age"] <= 168.0 or row["entry_strength"] >= 0.15]
    return {row["uid"]: row for row in keep[:MAX_CANDIDATES]}


def sample_poisson(rng, expectation):
    expectation = max(0.0, expectation)
    threshold = math.exp(-expectation)
    product = 1.0
    count = 0
    while product > threshold and count < 12:
        count += 1
        product *= rng.random()
    return max(0, count - 1)


def expiry_hazard(feature, hour):
    remaining = feature["remaining"]
    if remaining is None:
        return 0.0
    distance = remaining - hour
    if distance <= 0.0:
        return 1.0
    if distance < 12.0:
        return 0.12 * (12.0 - distance) / 12.0
    return 0.0


def stable_seed(features, slots, snapshot_count):
    value = 1469598103934665603
    for item in (slots, snapshot_count, *sorted(features)):
        value ^= int(item) & ((1 << 64) - 1)
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return value


def simulate(features, dynamics, slots, snapshot_count):
    uids = sorted(features)
    accumulated = {uid: [0.0] * 10 for uid in uids}
    rng = random.Random(stable_seed(features, slots, snapshot_count))
    unseen_fraction = clamp(
        0.38 + 0.45 * dynamics["entry_rate"] + 0.28 * dynamics["newcomer_pressure"],
        0.38,
        0.72,
    )

    for _ in range(PATHS):
        busy_probability = 0.22 + 0.46 * dynamics["newcomer_pressure"]
        busy_path = rng.random() < busy_probability
        path_exit_hazard = (
            dynamics["busy_exit_hazard"]
            if busy_path
            else clamp(dynamics["exit_hazard"] * 0.68, 0.003, dynamics["exit_hazard"])
        )
        path_entry_rate = dynamics["entry_rate"] * (1.35 if busy_path else 0.72)
        active = {uid for uid, feature in features.items() if feature["current"]}
        inactive_cross = {
            uid
            for uid, feature in features.items()
            if not feature["current"] and feature["entry_strength"] >= 0.10
        }
        rank_noise = {uid: rng.gauss(0.0, 0.07) for uid in uids}
        path_bias = {uid: rng.gauss(0.0, 0.09) for uid in uids}
        anonymous = []

        for hour in range(slots):
            if hour > 0:
                for uid in tuple(active):
                    feature = features[uid]
                    instability = 1.22 - 0.70 * feature["presence"]
                    stale = 1.0 + min(1.0, max(0.0, feature["last_age"] - 2.0) / 48.0)
                    hazard = path_exit_hazard * instability * stale + expiry_hazard(feature, hour)
                    if rng.random() < clamp(hazard, 0.001, 1.0):
                        active.remove(uid)
                        if (
                            feature["entry_strength"] >= 0.10
                            and (feature["remaining"] is None or feature["remaining"] > hour)
                        ):
                            inactive_cross.add(uid)
                for uid in tuple(inactive_cross):
                    feature = features[uid]
                    if feature["remaining"] is not None and feature["remaining"] <= hour:
                        inactive_cross.remove(uid)
                        continue
                    hourly_entry = feature["entry_strength"] / 42.0
                    hourly_entry *= math.exp(-max(0.0, feature["last_age"] - 12.0) / 96.0)
                    if rng.random() < hourly_entry:
                        active.add(uid)
                        inactive_cross.remove(uid)

            unknown_arrivals = sample_poisson(rng, path_entry_rate * unseen_fraction)
            for _arrival in range(unknown_arrivals):
                anonymous.append(
                    {
                        "score": -math.log(math.exp(rng.uniform(math.log(9.0), math.log(38.0)))),
                        "drift": rng.uniform(-0.004, 0.028),
                        "noise": rng.gauss(0.0, 0.13),
                        "life": rng.randint(18, 96),
                    }
                )

            ranked = []
            for uid in active:
                feature = features[uid]
                innovation = rng.gauss(0.0, dynamics["rank_noise"])
                rank_noise[uid] = 0.96 * rank_noise[uid] + 0.50 * innovation
                momentum_shift = feature["velocity"] * 30.0 * (1.0 - math.exp(-hour / 30.0))
                signal_shift = feature["signal"] * (1.0 - math.exp(-(hour + 4.0) / 22.0))
                score = (
                    -feature["base_log_rank"]
                    + clamp(momentum_shift, -0.80, 0.80)
                    + signal_shift
                    + rank_noise[uid]
                    + path_bias[uid]
                )
                ranked.append((score, uid))

            surviving_anonymous = []
            for index, entrant in enumerate(anonymous):
                entrant["life"] -= 1
                if entrant["life"] <= 0:
                    continue
                entrant["noise"] = 0.94 * entrant["noise"] + rng.gauss(0.0, dynamics["rank_noise"] * 0.7)
                entrant["score"] += entrant["drift"]
                surviving_anonymous.append(entrant)
                ranked.append((entrant["score"] + entrant["noise"], -(index + 1)))
            anonymous = surviving_anonymous

            winners = heapq.nlargest(10, ranked, key=lambda item: (item[0], item[1]))
            for place, (_, uid) in enumerate(winners):
                if uid > 0:
                    accumulated[uid][place] += 1.0

    denominator = float(PATHS * max(1, slots))
    for values in accumulated.values():
        for place in range(10):
            values[place] /= denominator
    return accumulated


def rank_bucket(rank):
    rank = int(rank)
    if rank <= 10:
        return rank
    for bucket, upper in enumerate((15, 25, 40), 11):
        if rank <= upper:
            return bucket
    return 14


def transition_table(snapshots, max_hours):
    """Counts of (rank bucket, whole-hour gap) -> place 1..10 or off the top 10 (index 10), over every pair of reads."""
    counts = {}
    for index, older in enumerate(snapshots):
        for newer in snapshots[index + 1:]:
            hours = (newer["time"] - older["time"]).total_seconds() / 3600.0
            if hours > max_hours:
                break
            for uid, rank in older["ranks"].items():
                target = newer["ranks"].get(uid)
                place = int(target) - 1 if target is not None and target <= 10.0 else 10
                row = counts.setdefault((rank_bucket(rank), int(round(hours))), [0.0] * 11)
                row[place] += 1.0
    return counts


def blend_transitions(probabilities, snapshots, cutoff, slots, slot_hours):
    """Mix each charted game's simulated places with how games at its rank actually moved over the same horizons."""
    if len(snapshots) < 2:
        return probabilities
    lag = max(0.0, (cutoff - snapshots[-1]["time"]).total_seconds() / 3600.0)
    horizons = [lag + slot * slot_hours for slot in range(slots)]
    by_bucket = {}
    for (bucket, hour), row in transition_table(snapshots, horizons[-1] + 3.0 * HORIZON_BANDWIDTH).items():
        by_bucket.setdefault(bucket, []).append((hour, row, sum(row)))
    empirical = {}
    for bucket, cells in sorted(by_bucket.items()):
        accumulated = [0.0] * 10
        used = 0
        for horizon in horizons:
            weights = [math.exp(-0.5 * ((hour - horizon) / HORIZON_BANDWIDTH) ** 2) for hour, _, _ in cells]
            mass = sum(weight * size for weight, (_, _, size) in zip(weights, cells))
            if mass < 1.0:  # under one effective observed transition at this horizon: leave it to the simulation
                continue
            used += 1
            for weight, (_, row, _) in zip(weights, cells):
                for place in range(10):
                    accumulated[place] += weight * row[place] / mass
        if used:
            empirical[bucket] = [value / used for value in accumulated]

    blended = dict(probabilities)
    for uid, rank in snapshots[-1]["ranks"].items():
        row = empirical.get(rank_bucket(rank))
        if row is None or uid not in probabilities:
            continue
        blended[uid] = [
            (1.0 - TRANSITION_WEIGHT) * simulated + TRANSITION_WEIGHT * observed
            for simulated, observed in zip(probabilities[uid], row)
        ]
    return blended


def format_forecast(probabilities, features, dynamics, chart_size):
    turnover_uncertainty = clamp(
        1.7 * dynamics["entry_rate"] + 0.8 * dynamics["newcomer_pressure"],
        0.0,
        1.0,
    )
    rows = []
    for place in range(10):
        calibration = clamp(
            0.965 - 0.055 * turnover_uncertainty - 0.0065 * place,
            0.84,
            0.95,
        )
        # RULES.txt: a place's named probabilities add up to at most 1. The transition blend does not
        # guarantee that, so scale the whole column down when needed (1e-6 covers rounding to 8 decimals).
        column = sum(values[place] for values in probabilities.values())
        if column > 0.0:
            calibration = min(calibration, (1.0 - 1e-6) / column)
        candidates = sorted(
            ((uid, values[place] * calibration) for uid, values in probabilities.items()),
            key=lambda item: (-item[1], item[0]),
        )
        named = [(uid, probability) for uid, probability in candidates[:100] if probability > 0.0]
        # RULES.txt gives every unnamed charted game (1 - named sum) / (unnamed count) at this place;
        # a game named below that share would score better unnamed, so drop it and recompute.
        while named:
            share = (1.0 - sum(probability for _, probability in named)) / max(1, chart_size - len(named))
            kept = [(uid, probability) for uid, probability in named if probability >= share]
            if len(kept) == len(named):
                break
            named = kept
        for uid, probability in named:
            rows.append({"universe_id": uid, "place": place + 1, "p": round(probability, 8)})

    chosen = set()
    top10 = []
    inclusion = {uid: sum(values) for uid, values in probabilities.items()}
    for place in range(10):
        available = [uid for uid in probabilities if uid not in chosen]
        if not available:
            break
        uid = max(
            available,
            key=lambda candidate: (
                probabilities[candidate][place] + 0.035 * inclusion[candidate],
                -features[candidate]["base_log_rank"],
                -candidate,
            ),
        )
        chosen.add(uid)
        top10.append(uid)
    if len(top10) < 10:
        fallback = sorted(
            (uid for uid in features if uid not in chosen),
            key=lambda uid: (features[uid]["base_log_rank"], uid),
        )
        top10.extend(fallback[: 10 - len(top10)])
    return {"probabilities": rows, "top10": top10}


def forecast(history, task):
    cutoff = parse_time(task.get("history_until"))
    if cutoff is None:
        cutoff = parse_time(task.get("window", {}).get("from"))
    if cutoff is None and history:
        cutoff = parse_time(history[-1].get("at"))
    if cutoff is None:
        cutoff = dt.datetime(1970, 1, 1, tzinfo=UTC)
    window = task.get("window") if isinstance(task.get("window"), dict) else {}
    slots = max(1, int(window.get("slots", 72) or 72))
    slot_hours = clamp((finite_number(window.get("slot_minutes")) or 60.0) / 60.0, 0.25, 24.0)
    games, snapshots = collect_history(history, cutoff)
    dynamics = estimate_dynamics(snapshots)
    features = build_features(games, snapshots, cutoff)
    probabilities = simulate(features, dynamics, slots, len(snapshots))
    probabilities = blend_transitions(probabilities, snapshots, cutoff, slots, slot_hours)
    chart_size = len(snapshots[-1]["ranks"]) if snapshots else 100
    return format_forecast(probabilities, features, dynamics, chart_size)


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    task = json.loads((input_dir / "task.json").read_text(encoding="utf-8"))
    history = [
        json.loads(line)
        for line in (input_dir / "history.ndjson").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    answer = forecast(history, task)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "forecast.json").write_text(
        json.dumps(answer, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
