"""Generate a synthetic image-access trace and daily forecasting features.

Standard library only. Run: python generate_data.py
Validate saved data again: python generate_data.py --check
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import random
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


GENERATOR_VERSION = "1.0.0"
HISTORY_DAYS = 7
VALIDATION_DAYS = 14
TEST_DAYS = 14
CSV_PATHS = (
    "raw/images.csv",
    "raw/access_log.csv",
    "processed/daily_features.csv",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    for field, minimum in (("n_images", 2), ("n_requests", 50000), ("days", 56)):
        value = config[field]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer")
    date.fromisoformat(config["start_date"])
    if not 0 < config["weekly_amplitude"] < 1:
        raise ValueError("weekly_amplitude must be between 0 and 1")
    if config["zipf_exponent"] <= 1:
        raise ValueError("zipf_exponent must be > 1")
    for field in ("day_noise_sigma", "image_noise_sigma"):
        if config[field] < 0:
            raise ValueError(f"{field} must be non-negative")
    for field, length in (("weekday_weights", 7), ("hour_weights", 24)):
        values = config[field]
        if len(values) != length or any(x <= 0 for x in values):
            raise ValueError(f"{field} must contain {length} positive values")
    return config


def split_name(day_index, days):
    if day_index < HISTORY_DAYS:
        return "warmup"
    if day_index < days - VALIDATION_DAYS - TEST_DAYS:
        return "train"
    if day_index < days - TEST_DAYS:
        return "validation"
    return "test"


def split_ranges(config):
    start = date.fromisoformat(config["start_date"])
    days = config["days"]
    bounds = (
        ("warmup", 0, HISTORY_DAYS),
        ("train", HISTORY_DAYS, days - VALIDATION_DAYS - TEST_DAYS),
        ("validation", days - VALIDATION_DAYS - TEST_DAYS, days - TEST_DAYS),
        ("test", days - TEST_DAYS, days),
    )
    return {
        name: {
            "start_date": (start + timedelta(days=left)).isoformat(),
            "end_date": (start + timedelta(days=right - 1)).isoformat(),
            "n_dates": right - left,
            "n_rows": (right - left) * config["n_images"],
        }
        for name, left, right in bounds
    }


def generate(config, output):
    rng = random.Random(config["seed"])
    n_images = config["n_images"]
    days = config["days"]
    start = datetime.combine(
        date.fromisoformat(config["start_date"]),
        datetime.min.time(), tzinfo=timezone.utc,
    )
    image_ids = list(range(1, n_images + 1))

    # Random assignment prevents image_id from encoding a popularity rank.
    ranks = image_ids.copy()
    rng.shuffle(ranks)
    base_weights = [rank ** -config["zipf_exponent"] for rank in ranks]
    preferred_weekdays = [rng.randrange(7) for _ in image_ids]
    sigma = config["day_noise_sigma"]
    day_weights = [
        config["weekday_weights"][(start + timedelta(days=d)).weekday()]
        * rng.lognormvariate(-sigma * sigma / 2, sigma)
        for d in range(days)
    ]
    # Exactly n_requests events, with unequal totals across calendar days.
    day_counts = Counter(rng.choices(
        range(days), weights=day_weights, k=config["n_requests"]
    ))
    events = []
    daily = [[0] * days for _ in image_ids]
    sigma = config["image_noise_sigma"]
    for d in range(days):
        weekday = (start + timedelta(days=d)).weekday()
        weights = [
            base * (
                1 + config["weekly_amplitude"]
                * math.cos(2 * math.pi * (weekday - preferred) / 7)
            ) * rng.lognormvariate(-sigma * sigma / 2, sigma)
            for base, preferred in zip(base_weights, preferred_weekdays)
        ]
        count = day_counts[d]
        selected = rng.choices(image_ids, weights=weights, k=count)
        hours = rng.choices(range(24), weights=config["hour_weights"], k=count)
        for image_id, hour in zip(selected, hours):
            stamp = start + timedelta(
                days=d, hours=hour, seconds=rng.randrange(3600)
            )
            events.append((stamp, image_id))
            daily[image_id - 1][d] += 1

    events.sort(key=lambda event: (event[0], event[1]))
    write_csv(output / "raw/images.csv", ["image_id"], ((i,) for i in image_ids))
    write_csv(
        output / "raw/access_log.csv",
        ["request_id", "timestamp", "image_id", "request_type", "status"],
        (
            (i, stamp.strftime("%Y-%m-%dT%H:%M:%SZ"), image_id, "user", "success")
            for i, (stamp, image_id) in enumerate(events, start=1)
        ),
    )

    def feature_rows():
        for d in range(days):
            target_date = (start + timedelta(days=d)).date()
            for image_id in image_ids:
                history = daily[image_id - 1]
                yield (
                    image_id, target_date.isoformat(), history[d],
                    history[d - 1] if d >= 1 else "",
                    history[d - 2] if d >= 2 else "",
                    history[d - 7] if d >= 7 else "",
                    f"{sum(history[d - 7:d]) / 7:.9f}" if d >= 7 else "",
                    target_date.weekday(), split_name(d, days),
                )

    write_csv(
        output / "processed/daily_features.csv",
        ["image_id", "date", "target_count", "lag1", "lag2", "lag7",
         "mean7", "weekday", "split"],
        feature_rows(),
    )
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": sha256_file(Path(__file__)),
        "python_version": platform.python_version(),
        "random_generator": "Python random.Random (Mersenne Twister)",
        "source": "Synthetic data generated by the supplied script; no external records",
        "data_license": "CC0-1.0",
        "time_zone": "UTC",
        "all_images_available_from_start": True,
        "event_scope": "Successful user downloads only; no prefetch or failed requests",
        "model_features": ["lag1", "lag2", "lag7", "mean7", "weekday"],
        "target": "target_count for the date column; past-only input features",
        "config": config,
        "splits": split_ranges(config),
        "files": {
            rel: {
                "sha256": sha256_file(output / rel),
                "size_bytes": (output / rel).stat().st_size,
            }
            for rel in CSV_PATHS
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def validate(output, metadata):
    config = metadata["config"]
    n_images, days = config["n_images"], config["days"]
    start_date = date.fromisoformat(config["start_date"])
    start = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=days)
    for rel, details in metadata["files"].items():
        if sha256_file(output / rel) != details["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {rel}")

    with (output / "raw/images.csv").open(encoding="utf-8", newline="") as f:
        ids = [int(r["image_id"]) for r in csv.DictReader(f)]
    if ids != list(range(1, n_images + 1)):
        raise ValueError("Image catalog is incomplete or contains duplicates")

    counts = Counter()
    seen_requests = set()
    previous = None
    with (output / "raw/access_log.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if any(value == "" for value in row.values()):
                raise ValueError("Missing required access-log field")
            request_id, image_id = int(row["request_id"]), int(row["image_id"])
            stamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            if request_id in seen_requests or not 1 <= image_id <= n_images:
                raise ValueError("Duplicate request_id or invalid image_id")
            if not start <= stamp < end or (previous is not None and stamp < previous):
                raise ValueError("Timestamp is outside the period or not sorted")
            if row["request_type"] != "user" or row["status"] != "success":
                raise ValueError("Unexpected event scope")
            seen_requests.add(request_id)
            counts[(image_id, stamp.date())] += 1
            previous = stamp
    if len(seen_requests) != config["n_requests"]:
        raise ValueError("Unexpected access-log length")

    seen_pairs, splits, missing = set(), Counter(), Counter()
    target_total = 0
    with (output / "processed/daily_features.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            image_id = int(row["image_id"])
            target_date = date.fromisoformat(row["date"])
            d = (target_date - start_date).days
            key = (image_id, target_date)
            if key in seen_pairs or not (1 <= image_id <= n_images and 0 <= d < days):
                raise ValueError("Invalid or duplicate image-date pair")
            seen_pairs.add(key)
            actual = int(row["target_count"])
            if actual != counts[key] or int(row["weekday"]) != target_date.weekday():
                raise ValueError("Daily target or weekday does not match the log")
            target_total += actual
            for lag in (1, 2, 7):
                value = row[f"lag{lag}"]
                if d < lag:
                    if value != "":
                        raise ValueError("Unknown history must remain missing")
                    missing[f"lag{lag}"] += 1
                elif value == "" or int(value) != counts[(image_id, target_date - timedelta(days=lag))]:
                    raise ValueError("Lag does not match the same image's past")
            if d < 7:
                if row["mean7"] != "":
                    raise ValueError("Incomplete weekly history must remain missing")
                missing["mean7"] += 1
            else:
                expected = sum(counts[(image_id, target_date - timedelta(days=k))] for k in range(1, 8)) / 7
                if row["mean7"] == "" or abs(float(row["mean7"]) - expected) > 1e-8:
                    raise ValueError("mean7 contains an incorrect historical window")
            expected_split = split_name(d, days)
            if row["split"] != expected_split:
                raise ValueError("Temporal split does not match the common date boundaries")
            splits[expected_split] += 1
    if len(seen_pairs) != n_images * days or target_total != config["n_requests"]:
        raise ValueError("Daily grid is incomplete or its counts do not match the log")
    return {
        "checks": "OK",
        "access_log_rows": len(seen_requests),
        "catalog_images": len(ids),
        "observed_images": len({key[0] for key in counts}),
        "daily_feature_rows": len(seen_pairs),
        "split_rows": dict(splits),
        "missing_feature_values_before_warmup_exclusion": dict(missing),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--check", action="store_true", help="Validate existing outputs without changing them")
    args = parser.parse_args()
    if args.check:
        metadata = json.loads((args.output_dir / "metadata.json").read_text(encoding="utf-8"))
    else:
        metadata = generate(load_config(args.config), args.output_dir)
    result = validate(args.output_dir, metadata)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nData directory:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
