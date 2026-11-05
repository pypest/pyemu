"""List the installed packages with the release date of the installed version.

Like `uv pip list`, plus when each version was first uploaded and how many days
ago that was (newest first; `*` marks anything under 7 days old, the Dependabot
cooldown), to show which brand-new releases a CI run picked up.

The dates come from the `upload-time` fields of a PEP 751 pylock.toml, e.g.
    uv export --frozen --format pylock.toml --all-extras --all-groups \\
        --no-emit-project -o pylock.toml
    python .github/scripts/list_release_dates.py pylock.toml
"""
import importlib.metadata as md
import sys
import tomllib
from datetime import datetime, timezone

from packaging.utils import canonicalize_name


def as_datetime(t):  # a TOML datetime in pylock.toml; tolerate an ISO string too
    return t if isinstance(t, datetime) else datetime.fromisoformat(t.replace("Z", "+00:00"))


def main(pylock_path):
    with open(pylock_path, "rb") as f:
        lock = tomllib.load(f)

    released = {}  # (name, version) -> first upload time of that version
    for p in lock["packages"]:
        files = p.get("wheels", []) + ([p["sdist"]] if p.get("sdist") else [])
        times = [as_datetime(f["upload-time"]) for f in files if f.get("upload-time")]
        if times:
            key = (canonicalize_name(p["name"]), p.get("version"))
            released[key] = min(times + ([released[key]] if key in released else []))

    now = datetime.now(timezone.utc)
    rows = {}
    for dist in md.distributions():
        if dist.metadata["Name"]:
            name = canonicalize_name(dist.metadata["Name"])
            rows[name] = (released.get((name, dist.version)), dist.version)

    # newest release first; packages missing from the lock (e.g. pyemu itself) last
    order = sorted(rows.items(), key=lambda kv: (kv[1][0] is None, -(kv[1][0].timestamp() if kv[1][0] else 0)))
    print(f"{'package':34s} {'version':16s} {'released':11s} age (days)")
    for name, (when, version) in order:
        if when is None:
            print(f"{name:34s} {version:16s} {'-':11s} -")
        else:
            age = (now - when).days
            print(f"{name:34s} {version:16s} {when:%Y-%m-%d}  {age:>5d}{'  *' if age < 7 else ''}")
    print("* released less than 7 days ago (the repo's Dependabot cooldown)")


if __name__ == "__main__":
    main(sys.argv[1])
