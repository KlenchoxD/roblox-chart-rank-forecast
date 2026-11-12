#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10 from the chart's own history.

How it works (every number below is either fixed by RULES.txt or picked by replaying the history):

1. Transitions. For every pair of history reads, an earlier one A and a later one B, and for each of B's
   places 1..10, record the rank the game holding it had in A (or None: it was not on A's chart).
   Counting these over all pairs, grouped by the gap between A and B in whole hours, says how often a
   place is held, that many hours later, by the game that was at rank r.
2. Window. Each scored slot sits a known number of hours after the last read. A Gaussian kernel on the
   gap turns the counts into, per slot, the share of places held by today's rank r; the forecast is the
   mean over the window's slots, because one forecast is scored against every hourly read. The share held
   by games not on the chart stays unnamed, which RULES.txt spreads over the games left unnamed.
   Every pair contributes exactly one occupant per place, so each place's named total is at most 1.
3. Adjustments. Source ranks are smoothed in log-rank (sparse far ranks borrow from neighbours), mapped
   onto today's chart, optionally tilted toward games that rose over the last day, and scaled down for
   games that reach the chart's 28-day limit inside the window.
4. Tuning. The gap bandwidth, rank width and tilt come from a replay of the contest on the history: each
   past read is a cutoff, the reads that follow it inside a window of the task's length are scored with
   RULES.txt's log score, and the setting with the best mean wins.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


# Fixed by RULES.txt.
TOP_SIZE = 10                 # "top10: exactly 10 distinct universe ids"
MAX_ROWS_PER_PLACE = 100      # "a place has at most 100 rows"
SCORE_FLOOR = 1e-4            # each place's probability is "floored at 0.0001"
LIFETIME_HOURS = 28.0 * 24.0  # "Only games published in the last 28 days are on it"
ROUNDING_GUARD = 1e-6         # keeps a place's printed sum <= 1 after flooring to 8 decimals

# Search grids. The replay picks one value from each; the grids only bracket the data's own scales:
# reads come 1 h (house recorder) to 11 h (archive) apart, rank moves run from a few percent to a third
# in log terms, and the tilt runs from none to strong.
GAP_BANDWIDTHS = (2.0, 4.0, 8.0, 16.0)
RANK_WIDTHS = (0.0, 0.15, 0.3)
DRIFT_TILTS = (0.0, 0.25, 0.5, 1.0)
DRIFT_LOOKBACK_HOURS = 24.0   # one day back, so the daily player cycle cancels out of a game's drift
MAX_REPLAYS = 120             # cutoffs replayed at most, keeping the run well inside the 300 s limit


def hours(text):
    """ISO 8601 time as hours since the epoch, or None."""
    if not isinstance(text, str) or not text:
        return None
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp() / 3600.0


def number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def game_id(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        text = value.strip()
        if text.isascii() and text.isdigit():
            result = int(text)
            return result if result > 0 else None
    return None


def detail_items(details):
    """(universe id, details) pairs from a read's `details`, keyed by universe id or listed."""
    if isinstance(details, dict):
        for key, info in details.items():
            if isinstance(info, dict):
                yield game_id(info.get("universe_id")) or game_id(key), info
    elif isinstance(details, list):
        for info in details:
            if isinstance(info, dict):
                yield game_id(info.get("universe_id")) or game_id(info.get("id")), info


def load_reads(lines, until):
    """Chart reads started before `until`, oldest first, and each game's creation time."""
    reads = []
    created = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("ok") is False:
            continue
        at = hours(record.get("at"))
        chart = record.get("up_and_coming")
        if at is None or not isinstance(chart, list) or (until is not None and at >= until):
            continue
        entries = []
        for position, row in enumerate(chart):
            if not isinstance(row, dict) or row.get("sponsored") is True:
                continue
            uid = game_id(row.get("universe_id"))
            if uid is None:
                continue
            rank = number(row.get("rank"))
            entries.append((rank if rank is not None else float(position + 1), position, uid))
        entries.sort()
        order = []
        seen = set()
        for _, _, uid in entries:
            if uid not in seen:
                seen.add(uid)
                order.append(uid)
        if not order:
            continue
        for uid, info in detail_items(record.get("details")):
            stamp = hours(info.get("created"))
            if uid is not None and stamp is not None:
                created.setdefault(uid, stamp)
        reads.append({
            "t": at,
            "order": order,  # a game's place is its position in this list, 1 first
            "rank": {uid: place for place, uid in enumerate(order, 1)},
            "members": frozenset(order),
        })
    reads.sort(key=lambda read: read["t"])
    # RULES.txt scores the first valid read of each UTC hour. Sampling the history the same way also keeps
    # bursts of near-identical reads (retries, reads seconds apart) from counting one moment several times.
    hourly = []
    for read in reads:
        if not hourly or math.floor(read["t"]) != math.floor(hourly[-1]["t"]):
            hourly.append(read)
    return hourly, created


def trusted_created(reads, created):
    """Creation times, unless the history shows a game charted past created + 28 days (+1 h for the
    chart's own refresh): then `created` is not the publish date the chart uses, and none is applied."""
    for read in reads:
        for uid in read["order"]:
            stamp = created.get(uid)
            if stamp is not None and read["t"] > stamp + LIFETIME_HOURS + 1.0:
                return {}
    return created


def build_pairs(reads, places, max_gap):
    """(index of the later read, gap in whole hours, rank in the earlier read of each top place's game)."""
    pairs = []
    for later in range(len(reads)):
        newer = reads[later]
        top = newer["order"][:places]
        for earlier in range(later - 1, -1, -1):
            older = reads[earlier]
            gap = newer["t"] - older["t"]
            if gap > max_gap:
                break
            if gap <= 0.0:
                continue
            pairs.append((later, int(round(gap)), tuple(older["rank"].get(uid) for uid in top)))
    return pairs


class Transitions:
    def __init__(self, places):
        self.places = places
        self.counts = {}  # gap -> per place: {source rank or None: count}
        self.totals = {}  # gap -> number of pairs

    def add(self, gap, sources):
        cells = self.counts.get(gap)
        if cells is None:
            cells = self.counts[gap] = [dict() for _ in range(self.places)]
            self.totals[gap] = 0
        self.totals[gap] += 1
        for place, source in enumerate(sources):
            cells[place][source] = cells[place].get(source, 0) + 1


def place_shares(table, horizons, bandwidth):
    """Per place, {source rank: share of the window's slots it holds that place}, averaged over slots."""
    gaps = sorted(table.totals)
    weight = dict.fromkeys(gaps, 0.0)
    for horizon in horizons:
        kernel = [math.exp(-0.5 * ((gap - horizon) / bandwidth) ** 2) for gap in gaps]
        norm = sum(value * table.totals[gap] for value, gap in zip(kernel, gaps))
        if norm > 0.0:
            for value, gap in zip(kernel, gaps):
                weight[gap] += value / norm
        else:  # every gap is far from this horizon: use the nearest one alone
            nearest = min(gaps, key=lambda gap: (abs(gap - horizon), gap))
            weight[nearest] += 1.0 / table.totals[nearest]
    shares = [dict() for _ in range(table.places)]
    for gap in gaps:
        factor = weight[gap] / len(horizons)
        if factor <= 0.0:
            continue
        for place, cell in enumerate(table.counts[gap]):
            target = shares[place]
            for source, count in cell.items():
                if source is not None:
                    target[source] = target.get(source, 0.0) + factor * count
    return shares


def rank_kernel(source, size, width, cache):
    """How a share from source rank `source` spreads over today's ranks 1..size (sums to at most 1)."""
    key = (source, size, width)
    row = cache.get(key)
    if row is None:
        if width <= 0.0:
            row = ((min(source, size), 1.0),)
        else:
            centre = math.log(source)
            raw = [(rank, math.exp(-0.5 * ((math.log(rank) - centre) / width) ** 2)) for rank in range(1, size + 1)]
            total = sum(value for _, value in raw)
            row = tuple((rank, value / total) for rank, value in raw if value > 1e-9 * total)
        cache[key] = row
    return row


def smooth(shares, size, width, cache):
    """Per place, a list indexed by today's rank (index 0 unused) of the share that rank holds."""
    columns = []
    for cell in shares:
        column = [0.0] * (size + 1)
        for source, mass in cell.items():
            for rank, part in rank_kernel(source, size, width, cache):
                column[rank] += mass * part
        columns.append(column)
    return columns


def drift(reads, index):
    """Each charted game's log-rank gain over about a day (absent then = one past that chart's end)."""
    now = reads[index]
    best = None
    for earlier in range(index - 1, -1, -1):
        gap = now["t"] - reads[earlier]["t"]
        if gap > 1.5 * DRIFT_LOOKBACK_HOURS:
            break
        miss = abs(gap - DRIFT_LOOKBACK_HOURS)
        if gap >= 0.5 * DRIFT_LOOKBACK_HOURS and (best is None or miss < best[0]):
            best = (miss, earlier)
    if best is None:
        return {}
    past = reads[best[1]]
    absent = len(past["order"]) + 1
    return {uid: math.log(past["rank"].get(uid, absent)) - math.log(rank) for uid, rank in now["rank"].items()}


def alive_shares(now, created, start, slots, slot_hours):
    """Share of the window's slots before each charted game turns 28 days old (games with a known date)."""
    alive = {}
    for uid in now["order"]:
        stamp = created.get(uid)
        if stamp is None or stamp + LIFETIME_HOURS <= now["t"]:
            continue
        end = stamp + LIFETIME_HOURS
        alive[uid] = sum(1 for slot in range(slots) if start + slot * slot_hours < end) / slots
    return alive


def place_columns(smoothed, now, tilt, drift_map, alive):
    """Per place, {universe id: probability} over today's chart."""
    columns = []
    for values in smoothed:
        column = {}
        for rank, uid in enumerate(now["order"], 1):
            if values[rank] > 0.0:
                column[uid] = values[rank]
        if tilt and drift_map and column:
            mass = sum(column.values())
            weighted = {uid: p * math.exp(tilt * drift_map.get(uid, 0.0)) for uid, p in column.items()}
            total = sum(weighted.values())
            column = {uid: mass * value / total for uid, value in weighted.items()}
        for uid, share in alive.items():
            if uid in column:
                column[uid] *= share
        columns.append(column)
    return columns


def place_rows(columns, chart_size):
    """Per place, the {universe id: p} actually written: at most 100 rows, sum <= 1, 8 decimals."""
    printed_by_place = []
    for column in columns:
        named = sorted(((p, uid) for uid, p in column.items() if p > 0.0), key=lambda item: (-item[0], item[1]))
        named = named[:MAX_ROWS_PER_PLACE]
        total = sum(p for p, _ in named)
        if total > 1.0 - ROUNDING_GUARD:
            named = [(p * (1.0 - ROUNDING_GUARD) / total, uid) for p, uid in named]
        # RULES.txt gives each unnamed charted game (1 - named sum) / (games left unnamed). A game named
        # below that share would score more unnamed, so leave it out and recompute until none is.
        while named:
            share = (1.0 - sum(p for p, _ in named)) / max(1, chart_size - len(named))
            kept = [item for item in named if item[0] >= share]
            if len(kept) == len(named):
                break
            named = kept
        printed = {}
        for p, uid in named:
            value = math.floor(p * 1e8) / 1e8
            if value > 0.0:
                printed[uid] = value
        printed_by_place.append(printed)
    return printed_by_place


def read_score(printed_by_place, read):
    """RULES.txt's score of one read: sum over places of ln(p of the game there), floored at 0.0001."""
    chart = read["order"]
    total = 0.0
    for place, named in enumerate(printed_by_place):
        if place >= len(chart):
            break
        p = named.get(chart[place])
        if p is None:
            unnamed = len(chart) - len(read["members"].intersection(named))
            p = (1.0 - sum(named.values())) / unnamed if unnamed > 0 else 0.0
        total += math.log(max(p, SCORE_FLOOR))
    return total


def window_targets(reads, first, start, slots, slot_hours):
    """The read that counts for each slot: the first one started in that slot (RULES.txt, The answer)."""
    chosen = {}
    for read in reads[first:]:
        offset = read["t"] - start
        if offset < 0.0:
            continue
        slot = int(offset // slot_hours)
        if slot >= slots:
            break
        chosen.setdefault(slot, read)
    return list(chosen.values())


def replay(reads, pairs, created, settings, lag, slots, slot_hours, places):
    """Mean log score of each (bandwidth, width, tilt) setting over past cutoffs, None if none replays.

    A cutoff sees only the pairs that end at or before it, and its window starts `lag` hours after it, as
    the real window starts after the last read. A cutoff whose window runs past the history counts in
    proportion to the part of a window it covers."""
    last = reads[-1]["t"]
    span = slots * slot_hours
    candidates = [index for index in range(1, len(reads)) if reads[index]["t"] + lag < last]
    if len(candidates) > MAX_REPLAYS:
        picks = sorted({round(k * (len(candidates) - 1) / (MAX_REPLAYS - 1)) for k in range(MAX_REPLAYS)})
        candidates = [candidates[pick] for pick in picks]
    table = Transitions(places)
    totals = [0.0] * len(settings)
    weight_sum = 0.0
    next_pair = 0
    kernels = {}
    for index in candidates:
        while next_pair < len(pairs) and pairs[next_pair][0] <= index:
            table.add(pairs[next_pair][1], pairs[next_pair][2])
            next_pair += 1
        if not table.totals:
            continue
        now = reads[index]
        start = now["t"] + lag
        targets = window_targets(reads, index + 1, start, slots, slot_hours)
        if not targets:
            continue
        weight = min(1.0, (last - start) / span)
        horizons = [lag + slot * slot_hours for slot in range(slots)]
        alive = alive_shares(now, created, start, slots, slot_hours)
        drift_map = None
        share_cache = {}
        smooth_cache = {}
        for position, (bandwidth, width, tilt) in enumerate(settings):
            if bandwidth not in share_cache:
                share_cache[bandwidth] = place_shares(table, horizons, bandwidth)
            key = (bandwidth, width)
            if key not in smooth_cache:
                smooth_cache[key] = smooth(share_cache[bandwidth], len(now["order"]), width, kernels)
            if tilt and drift_map is None:
                drift_map = drift(reads, index)
            columns = place_columns(smooth_cache[key], now, tilt, drift_map or {}, alive)
            printed = place_rows(columns, len(now["order"]))
            totals[position] += weight * sum(read_score(printed, read) for read in targets) / len(targets)
        weight_sum += weight
    if weight_sum <= 0.0:
        return None
    return [total / weight_sum for total in totals]


def choose_settings(reads, pairs, created, lag, slots, slot_hours, places):
    """Best bandwidth and rank width with no tilt, then the best tilt for them."""
    first = [(bandwidth, width, 0.0) for bandwidth in GAP_BANDWIDTHS for width in RANK_WIDTHS]
    scores = replay(reads, pairs, created, first, lag, slots, slot_hours, places)
    if scores is None:  # too little history to replay: the middle of each grid
        return GAP_BANDWIDTHS[len(GAP_BANDWIDTHS) // 2], RANK_WIDTHS[len(RANK_WIDTHS) // 2], 0.0
    bandwidth, width, _ = first[scores.index(max(scores))]
    second = [(bandwidth, width, tilt) for tilt in DRIFT_TILTS]
    scores = replay(reads, pairs, created, second, lag, slots, slot_hours, places)
    return second[scores.index(max(scores))]


def pick_top(columns, reads):
    """Most likely game per place, then today's chart order to fill the ten."""
    now = reads[-1]
    chosen = []
    for column in columns:
        best = max(
            ((p, -now["rank"].get(uid, len(now["order"]) + 1), uid) for uid, p in column.items() if uid not in chosen),
            default=None,
        )
        if best is not None:
            chosen.append(best[2])
    for read in reversed(reads):
        for uid in read["order"]:
            if len(chosen) >= TOP_SIZE:
                return chosen
            if uid not in chosen:
                chosen.append(uid)
    return chosen


def forecast(task, lines):
    window = task.get("window") if isinstance(task.get("window"), dict) else {}
    until = hours(task.get("history_until"))
    start = hours(window.get("from"))
    if start is None:
        start = until
    slots = max(1, int(number(window.get("slots")) or 72))
    slot_hours = max(1.0 / 60.0, (number(window.get("slot_minutes")) or 60.0) / 60.0)
    places = min(TOP_SIZE, max(1, int(number(task.get("places")) or TOP_SIZE)))

    reads, created = load_reads(lines, until if until is not None else start)
    if not reads:
        return {"probabilities": [], "top10": []}
    if start is None:
        start = reads[-1]["t"]
    lag = max(0.0, start - reads[-1]["t"])
    pairs = build_pairs(reads, places, lag + slots * slot_hours + 3.0 * max(GAP_BANDWIDTHS))
    created = trusted_created(reads, created)
    bandwidth, width, tilt = choose_settings(reads, pairs, created, lag, slots, slot_hours, places)

    table = Transitions(places)
    for _, gap, sources in pairs:
        table.add(gap, sources)
    now = reads[-1]
    if table.totals:
        horizons = [lag + slot * slot_hours for slot in range(slots)]
        smoothed = smooth(place_shares(table, horizons, bandwidth), len(now["order"]), width, {})
        drift_map = drift(reads, len(reads) - 1) if tilt else {}
        columns = place_columns(smoothed, now, tilt, drift_map, alive_shares(now, created, start, slots, slot_hours))
    else:
        columns = [dict() for _ in range(places)]
    printed = place_rows(columns, len(now["order"]))
    rows = [
        {"universe_id": uid, "place": place, "p": p}
        for place, named in enumerate(printed, 1)
        for uid, p in named.items()
    ]
    return {"probabilities": rows, "top10": pick_top(columns, reads)}


def main():
    source = Path(os.environ.get("MAKEFUN_INPUT_DIR") or "/input")
    target = Path(os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")
    task = json.loads((source / "task.json").read_text(encoding="utf-8"))
    lines = (source / "history.ndjson").read_text(encoding="utf-8").splitlines()
    answer = forecast(task if isinstance(task, dict) else {}, lines)
    target.mkdir(parents=True, exist_ok=True)
    (target / "forecast.json").write_text(json.dumps(answer, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
