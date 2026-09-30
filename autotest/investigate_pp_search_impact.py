"""
Investigate the impact of the setup_pilotpoints_grid search-loop fix
(pyemu/utils/pp_utils.py, structured-grid branch) across several ibound/
zone scenarios.

Compares:
  - old_search(): a standalone re-implementation of the pre-fix,
    global-stride search loop (row/col loop anchored to the full array).
  - the new (current) pyemu.pp_utils.setup_pilotpoints_grid, called
    through the real public API - including its MIN_KRIGE_PPOINTS floor
    (raises if a zone yields < 3 points) and dense-fallback cap.

This is a throwaway investigation script (not part of the test suite) -
results are printed to stdout, and a figure comparing the zone rasters and
old-vs-new pilot point locations for each scenario is saved next to this
script as investigate_pp_search_impact.png.
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import matplotlib.pyplot as plt

REPO = "/Users/brioch/Projects/dev/pyemu"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "autotest"))

import pyemu


def old_search(ib, every_n_cell, use_ibound_zones):
    """Re-implementation of the pre-fix search loop (global-array-anchored
    stride) exactly as it existed at pp_utils.py:217-237 before the fix."""
    start = int(float(every_n_cell) / 2.0)
    start_row = 0 if ib.shape[0] == 1 else start
    start_col = 0 if ib.shape[1] == 1 else start
    hits = []
    for i in range(start_row, ib.shape[0] - start_row // 2, every_n_cell):
        for j in range(start_col, ib.shape[1] - start_col // 2, every_n_cell):
            if ib[i, j] <= 0:
                continue
            zone = ib[i, j] if use_ibound_zones else 1
            hits.append((i, j, zone))
    return hits


def make_sr(nrow, ncol, delr=100.0, delc=100.0):
    return pyemu.helpers.SpatialReference(
        delr=[delr] * ncol, delc=[delc] * nrow, rotation=0, epsg=3070,
        xul=0.0, yul=0.0, units="meters", lenuni=2,
    )


def new_search(ib, sr, every_n_cell, use_ibound_zones, tmp_path):
    """Run the real (current) setup_pilotpoints_grid. May raise if a zone
    can't produce MIN_KRIGE_PPOINTS - callers should catch that."""
    par_info = pyemu.pp_utils.setup_pilotpoints_grid(
        sr=sr, ibound=ib, prefix_dict={0: "hk1_"}, every_n_cell=every_n_cell,
        use_ibound_zones=use_ibound_zones,
        pp_dir=tmp_path, tpl_dir=tmp_path, shapename=None,
    )
    return [(int(r.i), int(r.j), r.zone) for r in par_info.itertuples()]


def run_scenario(ib, sr, every_n_cell, use_ibound_zones, tmp_path):
    """Run both the old and new search, catching any exception the new
    (current) implementation raises so one bad scenario doesn't kill the
    rest of the investigation."""
    old_hits = old_search(ib, every_n_cell, use_ibound_zones)
    new_err = None
    try:
        new_hits = new_search(ib, sr, every_n_cell, use_ibound_zones, tmp_path)
    except Exception as e:
        new_hits = []
        new_err = str(e)
    return old_hits, new_hits, new_err


def summarize(name, ib, old_hits, new_hits, new_err=None, by_zone=False):
    n_active = int((ib > 0).sum())
    print(f"\n=== {name} ===")
    print(f"grid shape={ib.shape}, active cells={n_active}")
    print(f"  old search: {len(old_hits)} pilot point(s)"
          + ("  <-- BUG: zone has active cells but old search found none" if len(old_hits) == 0 and n_active > 0 else ""))
    if new_err:
        print(f"  new search: RAISED -- {new_err}")
    else:
        print(f"  new search: {len(new_hits)} pilot point(s)")
    if by_zone and not new_err:
        for zone in sorted(set(z for *_, z in old_hits) | set(z for *_, z in new_hits)):
            n_old = sum(1 for *_, z in old_hits if z == zone)
            n_new = sum(1 for *_, z in new_hits if z == zone)
            print(f"    zone {zone}: old={n_old}, new={n_new}")


def nn_stats(hits):
    if len(hits) < 2:
        return None
    pts = np.array([(i, j) for i, j, z in hits])
    dmin = []
    for k in range(len(pts)):
        d = np.sqrt(((pts - pts[k]) ** 2).sum(axis=1))
        d[k] = np.inf
        dmin.append(d.min())
    return {"min_nn_dist": float(np.min(dmin)), "mean_nn_dist": float(np.mean(dmin)), "max_nn_dist": float(np.max(dmin))}


def plot_scenario(ax, name, ib, old_hits, new_hits, new_err=None, by_zone=False):
    """Show the zone/ibound raster with the old- vs new-search pilot point
    locations overlaid, so the two can be compared visually in one panel.
    If by_zone, color pilot points by their zone id (marker shape still
    distinguishes old 'x' vs new 'o') instead of a flat red/cyan, so
    multi-zone scenarios are easier to read."""
    ax.imshow(ib, origin="upper", cmap="tab20", interpolation="nearest")
    if by_zone:
        zones = sorted(set(z for *_, z in old_hits) | set(z for *_, z in new_hits))
        palette = plt.get_cmap("Dark2")
        zone_color = {z: palette(i % 8) for i, z in enumerate(zones)}
        for z in zones:
            pts = [(i, j) for i, j, zz in old_hits if zz == z]
            if pts:
                oi, oj = zip(*pts)
                ax.scatter(oj, oi, marker="x", color=zone_color[z], s=50,
                           linewidths=1.5, label=f"old zone {z} ({len(pts)})")
        for z in zones:
            pts = [(i, j) for i, j, zz in new_hits if zz == z]
            if pts:
                ni, nj = zip(*pts)
                ax.scatter(nj, ni, marker="o", facecolors="none",
                           edgecolors=[zone_color[z]], s=70, linewidths=1.5,
                           label=f"new zone {z} ({len(pts)})")
    else:
        if old_hits:
            oi, oj = zip(*[(i, j) for i, j, z in old_hits])
            ax.scatter(oj, oi, marker="x", c="red", s=50, linewidths=1.5,
                       label=f"old ({len(old_hits)})")
        if new_hits:
            ni, nj = zip(*[(i, j) for i, j, z in new_hits])
            ax.scatter(nj, ni, marker="o", facecolors="none",
                       edgecolors="cyan", s=70, linewidths=1.5,
                       label=f"new ({len(new_hits)})")
    if new_err:
        ax.text(0.5, 0.5, "new search RAISED\n(< MIN_KRIGE_PPOINTS)",
                transform=ax.transAxes, ha="center", va="center", fontsize=8,
                color="yellow", weight="bold",
                bbox=dict(facecolor="black", alpha=0.6, boxstyle="round"))
        for spine in ax.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(3)
    ax.set_title(name, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    if old_hits or new_hits:
        ax.legend(fontsize=6, loc="upper right", framealpha=0.7)


scenarios = []  # (name, ib, old_hits, new_hits, new_err, by_zone) collected for plotting

with tempfile.TemporaryDirectory() as tmp_path:
    every_n_cell = 4

    # 1) baseline: full active grid, single zone - should match exactly
    nrow, ncol = 20, 20
    ib = np.ones((nrow, ncol), dtype=int)
    sr = make_sr(nrow, ncol)
    old_hits, new_hits, err = run_scenario(ib, sr, every_n_cell, False, tmp_path)
    name = "1) full active grid (baseline equivalence check)"
    summarize(name, ib, old_hits, new_hits, err)
    assert err is None and sorted(old_hits) == sorted(new_hits), "REGRESSION: baseline full-grid case changed!"
    print("  -> IDENTICAL to old algorithm (as expected)")
    scenarios.append((name, ib, old_hits, new_hits, err, False))

    # 2) thin band off the stride grid (same as the new autotest case)
    nrow, ncol = 20, 20
    ib = np.zeros((nrow, ncol), dtype=int)
    ib[7:10, :] = 1
    sr = make_sr(nrow, ncol)
    old_hits, new_hits, err = run_scenario(ib, sr, every_n_cell, False, tmp_path)
    name = "2) thin band missed by global stride"
    summarize(name, ib, old_hits, new_hits, err)
    if not err:
        print(f"  new hit rows: {sorted(set(i for i, j, z in new_hits))}")
    scenarios.append((name, ib, old_hits, new_hits, err, False))

    # 3) small isolated blob, off-grid, inside a big mostly-inactive domain
    #    - small enough (3x3, thinner than every_n_cell in both axes) that
    #    the direct search only checks a single midpoint candidate, but the
    #    fallback (now triggered whenever hits < MIN_KRIGE_PPOINTS, not
    #    just when hits is empty) picks up the blob's other active cells
    #    instead of raising
    nrow, ncol = 50, 50
    ib = np.zeros((nrow, ncol), dtype=int)
    ib[23:26, 31:34] = 1  # 3x3 blob, deliberately not aligned to stride 4
    sr = make_sr(nrow, ncol)
    old_hits, new_hits, err = run_scenario(ib, sr, every_n_cell, False, tmp_path)
    name = "3) small isolated 3x3 blob, off-grid (fallback, not raise)"
    summarize(name, ib, old_hits, new_hits, err)
    scenarios.append((name, ib, old_hits, new_hits, err, False))

    # 4) irregular (non-rectangular / diagonal) zone shape - exercises the
    #    dense-fallback safety net
    nrow, ncol = 30, 30
    ib = np.zeros((nrow, ncol), dtype=int)
    for k in range(30):
        ib[k, (k * 2) % 30] = 1  # scattered diagonal-ish pattern
    sr = make_sr(nrow, ncol)
    old_hits, new_hits, err = run_scenario(ib, sr, every_n_cell, False, tmp_path)
    name = "4) scattered/irregular zone shape (fallback path)"
    summarize(name, ib, old_hits, new_hits, err)
    if not err:
        print(f"  fallback triggered: {len(new_hits) == int((ib > 0).sum())}")
    scenarios.append((name, ib, old_hits, new_hits, err, False))

    # 5) multi-zone map, use_ibound_zones=True, with one zone split into
    #    two disjoint patches sharing the same zone id
    nrow, ncol = 40, 40
    ib = np.ones((nrow, ncol), dtype=int)  # zone 1 = background
    ib[5:9, 5:35] = 2       # zone 2, patch A (thin band, off-stride)
    ib[31:35, 5:35] = 2     # zone 2, patch B (same zone id, disjoint)
    sr = make_sr(nrow, ncol)
    old_hits, new_hits, err = run_scenario(ib, sr, every_n_cell, True, tmp_path)
    name = "5) multi-zone, disjoint same-id zone 2 (use_ibound_zones=True)"
    summarize(name, ib, old_hits, new_hits, err, by_zone=True)
    if not err:
        zone2_old_rows = sorted(set(i for i, j, z in old_hits if z == 2))
        zone2_new_rows = sorted(set(i for i, j, z in new_hits if z == 2))
        print(f"  zone 2 rows hit -- old: {zone2_old_rows}  new: {zone2_new_rows}")
    scenarios.append((name, ib, old_hits, new_hits, err, True))

    # 6) same multi-zone map, nearest-neighbor spacing sanity check for the
    #    dominant zone (zone 1) to confirm per-zone anchoring doesn't
    #    degrade spacing/regularity for the common case
    if not err:
        old_zone1 = [(i, j, z) for i, j, z in old_hits if z == 1]
        new_zone1 = [(i, j, z) for i, j, z in new_hits if z == 1]
        print(f"\n  zone 1 nn-dist stats -- old: {nn_stats(old_zone1)}")
        print(f"  zone 1 nn-dist stats -- new: {nn_stats(new_zone1)}")

    # 7) real Freyberg extra_crispy model ibound, default (use_ibound_zones=False)
    try:
        import flopy
        o_model_ws = os.path.join(REPO, "examples", "Freyberg", "extra_crispy")
        model_ws = os.path.join(tmp_path, "extra_crispy")
        shutil.copytree(o_model_ws, model_ws)
        ml = flopy.modflow.Modflow.load("freyberg.nam", model_ws=model_ws, check=False)
        ib_real = ml.bas6.ibound.array[0]
        sr_real = pyemu.helpers.SpatialReference.from_namfile(
            os.path.join(ml.model_ws, ml.namefile), delc=ml.dis.delc, delr=ml.dis.delr)
        sr_real.rotation = 0.0
        old_hits, new_hits, err = run_scenario(ib_real, sr_real, 2, False, tmp_path)
        name = "7) real Freyberg extra_crispy ibound (mix of 1/0/-1)"
        summarize(name, ib_real, old_hits, new_hits, err)
        assert err is None and sorted(old_hits) == sorted(new_hits), "REGRESSION: real Freyberg ibound case changed!"
        print("  -> IDENTICAL to old algorithm (as expected, single contiguous active zone)")
        scenarios.append((name, ib_real, old_hits, new_hits, err, False))
    except Exception as e:
        print(f"\n=== 7) real Freyberg ibound check skipped: {e} ===")

    # 8) dense-fallback cap check: a large zone that fully aliases against
    #    the stride (forcing the empty-hits fallback) should be decimated
    #    back down, not explode into one pilot point per active cell
    nrow, ncol = 400, 400
    ib = np.zeros((nrow, ncol), dtype=int)
    col = 200
    for i in range(1, 399):
        if i % 4 != 3:
            ib[i, col] = 1
    sr = make_sr(nrow, ncol)
    old_hits, new_hits, err = run_scenario(ib, sr, every_n_cell, False, tmp_path)
    name = "8) dense aliased zone - fallback cap check"
    summarize(name, ib, old_hits, new_hits, err)
    if not err:
        print(f"  n_active={int((ib > 0).sum())}, capped new count={len(new_hits)}")
    scenarios.append((name, ib, old_hits, new_hits, err, False))

print("\nDone.")

# --- plotting: zone raster + old-vs-new pilot point locations, one panel
#     per scenario ---
ncols_fig = 3
nrows_fig = int(np.ceil(len(scenarios) / ncols_fig))
fig, axes = plt.subplots(nrows_fig, ncols_fig, figsize=(5 * ncols_fig, 4.5 * nrows_fig))
axes = np.atleast_1d(axes).flatten()
for ax, (name, ib, old_hits, new_hits, err, by_zone) in zip(axes, scenarios):
    plot_scenario(ax, name, ib, old_hits, new_hits, err, by_zone=by_zone)
for ax in axes[len(scenarios):]:
    ax.axis("off")
fig.suptitle("setup_pilotpoints_grid search fix: zones + old vs new pilot points", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_png = os.path.join(os.path.dirname(__file__), "investigate_pp_search_impact.png")
fig.savefig(out_png, dpi=130)
print(f"\nSaved comparison figure to: {out_png}")
