"""
Investigate the impact of the setup_pilotpoints_grid search fix
(pyemu/utils/pp_utils.py, structured-grid branch) across a single set of
ibound/zone scenarios.

For every scenario three results are shown side by side:
  - old             : standalone re-implementation of the pre-fix,
                      global-stride search (anchored to the full array), no
                      fallback.
  - new (stride)    : the per-zone-anchored stride search on its own, i.e.
                      the "direct" hits (mirrors the candidate generation in
                      pp_utils.py).
  - new + fallback  : what the real pyemu.pp_utils.setup_pilotpoints_grid
                      returns, including the MIN_KRIGE_PPOINTS floor (raises
                      if a zone yields < 3 points) and the greedy gap-filling
                      fallback (blue = direct hits kept, red x = added by the
                      fallback).

The thresholds being illustrated are MIN_KRIGE_PPOINTS (hard floor) and
PPOINT_UNDER_DELIVERY_FRACTION (fraction of the area-implied point count,
active cells / every_n_cell**2, below which the fallback engages), plus the
spread check (an active cell further than every_n_cell from every direct hit).

This is a throwaway investigation script (not part of the test suite) -
results are printed to stdout, and one figure (a row of three panels per
scenario) is saved next to this script as investigate_pp_search_impact.png.
"""
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "autotest"))

import pyemu
from pyemu.utils import pp_utils


# --------------------------------------------------------------------------
# the three searches
# --------------------------------------------------------------------------
def old_search(ib, every_n_cell, use_ibound_zones):
    """Re-implementation of the pre-fix search loop (global-array-anchored
    stride). Returns (candidates, hits); hits are (i, j, zone)."""
    start = int(float(every_n_cell) / 2.0)
    start_row = 0 if ib.shape[0] == 1 else start
    start_col = 0 if ib.shape[1] == 1 else start
    cands, hits = [], []
    for i in range(start_row, ib.shape[0] - start_row // 2, every_n_cell):
        for j in range(start_col, ib.shape[1] - start_col // 2, every_n_cell):
            cands.append((i, j))
            if ib[i, j] <= 0:
                continue
            hits.append((i, j, ib[i, j] if use_ibound_zones else 1))
    return cands, hits


def zone_list(ib, use_ibound_zones):
    if use_ibound_zones:
        return sorted(int(z) for z in np.unique(ib) if z > 0)
    return [1] if np.any(ib > 0) else []


def new_stride_search(ib, n, use_ibound_zones):
    """Mirror of the per-zone candidate generation in pp_utils.py (stride
    anchored to each zone's own bounding box, midpoint if the zone is no
    wider than one stride). Returns (candidates, hits); hits are (i, j, zone).
    """
    start = int(float(n) / 2.0)
    cands, hits = [], []
    for zone in zone_list(ib, use_ibound_zones):
        mask = (ib == zone) if use_ibound_zones else (ib > 0)
        rows, cols = np.where(mask)
        r0, c0 = int(rows.min()), int(cols.min())
        hr, hc = int(rows.max()) - r0 + 1, int(cols.max()) - c0 + 1
        rc = [r0 + hr // 2] if hr <= n else list(range(r0 + start, r0 + hr - start // 2, n))
        cc = [c0 + hc // 2] if hc <= n else list(range(c0 + start, c0 + hc - start // 2, n))
        for i in rc:
            for j in cc:
                cands.append((i, j))
                if mask[i, j]:
                    hits.append((i, j, zone))
    return cands, hits


def make_sr(nrow, ncol, delr=100.0, delc=100.0):
    return pyemu.helpers.SpatialReference(
        delr=[delr] * ncol, delc=[delc] * nrow, rotation=0, epsg=3070,
        xul=0.0, yul=0.0, units="meters", lenuni=2,
    )


def new_search(ib, sr, n, use_ibound_zones, tmp_path):
    """Run the real (current) setup_pilotpoints_grid, capturing warnings and
    any exception (e.g. the MIN_KRIGE_PPOINTS raise). Returns
    (hits, warning_messages, error_message)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            par_info = pp_utils.setup_pilotpoints_grid(
                sr=sr, ibound=ib, prefix_dict={0: "hk1_"}, every_n_cell=n,
                use_ibound_zones=use_ibound_zones,
                pp_dir=tmp_path, tpl_dir=tmp_path, shapename=None,
            )
            hits = [(int(r.i), int(r.j), r.zone) for r in par_info.itertuples()]
            err = None
        except Exception as e:  # the raise is one of the things being shown
            hits, err = [], str(e)
    return hits, [str(x.message) for x in w], err


def trigger_stats(ib, n, use_ibound_zones, direct_hits):
    """Per-zone numbers behind the fallback trigger: (zone, n_direct,
    threshold, max_gap) where threshold = max(MIN_KRIGE_PPOINTS, fraction *
    active cells / n**2) and max_gap is the furthest any active cell is from
    a direct hit."""
    out = []
    for zone in zone_list(ib, use_ibound_zones):
        mask = (ib == zone) if use_ibound_zones else (ib > 0)
        rows, cols = np.where(mask)
        zh = [(i, j) for i, j, z in direct_hits if z == zone]
        thr = max(pp_utils.MIN_KRIGE_PPOINTS,
                  len(rows) / float(n) ** 2 * pp_utils.PPOINT_UNDER_DELIVERY_FRACTION)
        gap = (cKDTree(np.array(zh)).query(np.column_stack([rows, cols]))[0].max()
               if zh else np.inf)
        out.append((zone, len(zh), thr, gap))
    return out


# --------------------------------------------------------------------------
# scenarios: (group, title, ib, every_n_cell, use_ibound_zones, sr, expect_identical)
# --------------------------------------------------------------------------
def block(shape, rs, cs):
    ib = np.zeros(shape, dtype=int)
    ib[rs, cs] = 1
    return ib


def diag_band(n, half=1):
    ib = np.zeros((n, n), dtype=int)
    for i in range(n):
        ib[i, max(0, i - half):min(n, i + half + 1)] = 1
    return ib


def ell(n=30):
    ib = np.zeros((n, n), dtype=int)
    ib[:, :6] = 1
    ib[24:, :] = 1
    return ib


def scenarios(tmp_path):
    sc = []

    def add(group, title, ib, n, zones=False, sr=None, identical=False):
        sc.append((group, title, ib, n, zones, sr if sr is not None else make_sr(*ib.shape), identical))

    # --- stride coverage -------------------------------------------------
    add("Baseline", "full active grid", np.ones((20, 20), dtype=int), 4, identical=True)
    add("Stride coverage", "thin band\nmissed by global stride",
        block((20, 20), slice(7, 10), slice(0, 20)), 4)

    ib = np.zeros((30, 30), dtype=int)
    for k in range(30):
        ib[k, (k * 2) % 30] = 1
    add("Stride coverage", "scattered zone\n(two diagonal segments)", ib, 4)

    add("Stride coverage", "L-shape\n(bbox mostly empty)", ell(), 5)
    add("Stride coverage", "diagonal band\n(stride aliases)", diag_band(30), 5)
    lat = np.fromfunction(lambda i, j: ((i % 5 != 2) & (j % 5 != 2)).astype(int), (30, 30), dtype=int)
    add("Stride coverage", "sparse lattice\n(stride falls in the gaps)", lat, 5)

    ib = np.zeros((400, 400), dtype=int)
    for i in range(1, 399):
        if i % 4 != 3:
            ib[i, 200] = 1
    add("Stride coverage", "dense aliased column\n(fallback is capped)", ib, 4)

    # --- 3-point floor -----------------------------------------------------
    ib = np.zeros((50, 50), dtype=int)
    ib[23:26, 31:34] = 1
    add("Floor (min 3 pts)", "3x3 blob off the stride\n(fallback, not raise)", ib, 4)
    add("Floor (min 3 pts)", "4-cell zone\n(below the floor)",
        block((20, 30), slice(5, 7), slice(5, 7)), 5)
    add("Floor (min 3 pts)", "2-cell zone\n(cannot reach the floor)",
        block((20, 30), slice(5, 6), slice(5, 7)), 5)

    # --- multi-zone / real model -------------------------------------------
    ib = np.ones((40, 40), dtype=int)
    ib[5:9, 5:35] = 2
    ib[31:35, 5:35] = 2
    add("Multi-zone", "disjoint same-id zone 2\n(use_ibound_zones=True)", ib, 4, zones=True)

    try:
        import flopy
        ws = os.path.join(tmp_path, "extra_crispy")
        shutil.copytree(os.path.join(REPO, "examples", "Freyberg", "extra_crispy"), ws)
        ml = flopy.modflow.Modflow.load("freyberg.nam", model_ws=ws, check=False)
        sr = pyemu.helpers.SpatialReference.from_namfile(
            os.path.join(ml.model_ws, ml.namefile), delc=ml.dis.delc, delr=ml.dis.delr)
        sr.rotation = 0.0
        add("Real model", "Freyberg extra_crispy ibound\n(1 / 0 / -1)", ml.bas6.ibound.array[0], 2,
            sr=sr, identical=True)
    except Exception as e:
        print(f"Freyberg scenario skipped: {e}")
    return sc


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
ZONE_COLORS = ["#ffffff", "#d9d9d9", "#f6dfb0", "#cfe6c0", "#d8c8e8", "#e6bcd4"]
# point colours used when scenario points are coloured by zone (multi-zone
# scenarios) - chosen to contrast with the zone shading above
ZONE_POINT_COLORS = {1: "#00796b", 2: "#c2185b", 3: "#1565c0", 4: "#6a1b9a", 5: "#e65100"}


def draw_zone(ax, ib):
    ax.imshow(np.clip(ib, 0, len(ZONE_COLORS) - 1), cmap=ListedColormap(ZONE_COLORS),
              vmin=0, vmax=len(ZONE_COLORS) - 1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def scatter(ax, pts, **kw):
    if pts:
        ax.scatter([p[1] for p in pts], [p[0] for p in pts], **kw)


def scatter_hits(ax, hits, by_zone, color, **kw):
    """hits are (i, j, zone). Flat ``color`` unless by_zone, in which case each
    zone's points get their own colour (see ZONE_POINT_COLORS)."""
    if not by_zone:
        scatter(ax, hits, color=color, **kw)
        return
    for z in sorted({h[2] for h in hits}):
        scatter(ax, [h for h in hits if h[2] == z], color=ZONE_POINT_COLORS[int(z)], **kw)


MAX_CANDIDATES_SHOWN = 150  # don't draw hollow candidate markers for huge grids


def plot_all(results, out):
    ncols_sc = 2  # scenarios per figure row (each is 3 panels wide)
    nrows = int(np.ceil(len(results) / ncols_sc))
    fig, axes = plt.subplots(nrows, 3 * ncols_sc, figsize=(7.2 * ncols_sc, 2.9 * nrows))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for k, r in enumerate(results):
        row, cblock = divmod(k, ncols_sc)
        a_old, a_new, a_fb = axes[row, 3 * cblock:3 * cblock + 3]
        for ax in (a_old, a_new, a_fb):
            ax.axis("on")
            draw_zone(ax, r["ib"])

        for ax, cands, hits, col, mk in ((a_old, r["old_c"], r["old"], "tab:orange", "s"),
                                         (a_new, r["new_c"], r["new"], "tab:blue", "o")):
            if len(cands) <= MAX_CANDIDATES_SHOWN:
                scatter(ax, cands, s=45, marker=mk, facecolors="none", edgecolors=col, linewidths=0.9)
            scatter_hits(ax, hits, r["zones"], col, s=16, marker=mk)

        direct = {(i, j) for i, j, z in r["new"]}
        kept = [h for h in r["final"] if (h[0], h[1]) in direct]
        added = [h for h in r["final"] if (h[0], h[1]) not in direct]
        scatter_hits(a_fb, kept, r["zones"], "tab:blue", s=22)
        scatter_hits(a_fb, added, r["zones"], "tab:red", s=26, marker="x", linewidths=1.2)
        if r["err"]:
            a_fb.text(0.5, 0.5, "RAISED\n(< 3 points)", transform=a_fb.transAxes, ha="center",
                      va="center", color="red", fontsize=9, weight="bold")

        a_old.set_xlabel(f"{len(r['old'])} pt" + (" (empty par_info)" if not r["old"] else ""), fontsize=7)
        a_new.set_xlabel(f"{len(r['new'])} hit(s)", fontsize=7)
        a_fb.set_xlabel(("RAISED" if r["err"] else f"{len(r['final'])} pt ({len(added)} from fallback)")
                        + f", {len(r['warns'])} warn", fontsize=7)
        a_old.set_ylabel(f"{r['group']}\n{r['title']}", fontsize=7)
        if r["zones"]:
            from matplotlib.lines import Line2D
            zs = sorted({int(z) for z in np.unique(r["ib"]) if z > 0})
            for ax, hits in ((a_old, r["old"]), (a_new, r["new"]), (a_fb, r["final"])):
                counts = ", ".join(f"z{z}: {sum(1 for h in hits if h[2] == z)}" for z in zs)
                ax.set_xlabel(ax.get_xlabel() + f"\n{counts}", fontsize=7)
            handles = [Line2D([], [], marker="o", ls="", color=ZONE_POINT_COLORS[z],
                              label=f"zone {z} points") for z in zs]
            handles += [Line2D([], [], marker="o", ls="", color="0.4", label="direct hit"),
                        Line2D([], [], marker="x", ls="", color="0.4", label="added by fallback")]
            a_new.legend(handles=handles, fontsize=6, loc="upper center",
                         bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
        if row == 0:
            a_old.set_title("old", fontsize=8)
            a_new.set_title("new (stride hits)", fontsize=8)
            a_fb.set_title("new + fallback", fontsize=8)
    fig.suptitle("setup_pilotpoints_grid: old vs new vs new + fallback "
                 f"(MIN_KRIGE_PPOINTS={pp_utils.MIN_KRIGE_PPOINTS}, "
                 f"PPOINT_UNDER_DELIVERY_FRACTION={pp_utils.PPOINT_UNDER_DELIVERY_FRACTION}; "
                 "hollow = stride candidate, filled = on an active cell, red x = added by fallback)",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=110)
    print(f"\nSaved comparison figure to: {out}")


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
results = []
with tempfile.TemporaryDirectory() as tmp_path:
    for group, title, ib, n, zones, sr, identical in scenarios(tmp_path):
        old_c, old = old_search(ib, n, zones)
        new_c, new = new_stride_search(ib, n, zones)
        final, warns, err = new_search(ib, sr, n, zones, tmp_path)
        results.append(dict(group=group, title=title, ib=ib, n=n, old_c=old_c, old=old,
                            new_c=new_c, new=new, final=final, warns=warns, err=err, zones=zones))

        name = f"{group}: {title.splitlines()[0]}"
        n_active = int((ib > 0).sum())
        print(f"\n=== {name} (every_n_cell={n}, active cells={n_active}) ===")
        print(f"  old: {len(old)}" + ("  <-- zone has active cells but old search found none" if not old and n_active else "")
              + f" | new stride hits: {len(new)} | final: " + ("RAISED" if err else str(len(final)))
              + f" | warnings: {len(warns)}")
        for zone, n_dir, thr, gap in trigger_stats(ib, n, zones, new):
            print(f"  zone {zone}: direct hits {n_dir} vs threshold {thr:.1f}; max gap {gap:.1f} (limit {n})")
        if err:
            print(f"  RAISED: {err[:160]}")
        if identical:
            assert not err and sorted(old) == sorted(final), f"REGRESSION: {name} changed!"
            print("  -> IDENTICAL to old algorithm (as expected)")

plot_all(results, os.path.join(os.path.dirname(os.path.abspath(__file__)), "investigate_pp_search_impact.png"))
