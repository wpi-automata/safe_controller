#!/usr/bin/env python3
"""
plot_cbf_debug.py

Plot the /lane_keeping/cbf_debug topic (published by clauded.py) from a
ROS 2 bag.

Record:
    ros2 bag record /lane_keeping/cbf_debug /lane_position /cmd_vel -o cbf_run

Plot:
    python3 plot_cbf_debug.py cbf_run                  # interactive window
    python3 plot_cbf_debug.py cbf_run -o cbf_run.png   # save to file
    python3 plot_cbf_debug.py cbf_run --csv cbf_run.csv
    python3 plot_cbf_debug.py cbf_run --t0 10 --t1 40  # time window [s]

Panels (shared time axis, seconds since first debug message):
    1. Lane position p with the safe band [p_min, p_max] and lane center
    2. Barrier margins h1 = p - p_min, h2 = p_max - p (must stay >= 0)
    3. omega_desired vs omega_safe inside the CBF bound [omega_lower, omega_upper],
       with the spans where the CBF clipped the command shaded
    4. Commanded forward speed
"""

import argparse
import os
import sys

import numpy as np

# Must match LABELS in clauded.py
LABELS = [
    "lane_position",
    "error",
    "h1_left_margin",
    "h2_right_margin",
    "omega_desired",
    "omega_lower",
    "omega_upper",
    "omega_safe",
    "cbf_active",
    "speed_cmd",
]
IDX = {name: i for i, name in enumerate(LABELS)}

# Colors (fixed order; same entity always gets the same color)
C_BLUE = "#2a78d6"     # p / omega_safe
C_ORANGE = "#eb6834"   # h1 (left) / omega_desired
C_AQUA = "#1baf7a"     # h2 (right)
C_BOUND = "#86b6ef"    # admissible omega band
C_SAFE_BAND = "#cde2fb"
C_ACTIVE = "#fab219"   # CBF clipping spans
C_VIOLATION = "#d03b3b"
C_INK = "#52514e"


# ----------------------------------------------------------------------------
# Bag reading
# ----------------------------------------------------------------------------
def detect_storage_id(bag_path):
    if os.path.isfile(bag_path):
        return "mcap" if bag_path.endswith(".mcap") else "sqlite3"
    files = os.listdir(bag_path)
    if any(f.endswith(".mcap") for f in files):
        return "mcap"
    return "sqlite3"


def read_cbf_debug(bag_path, topic):
    """Return (t [s, bag receive time, absolute], data [N x len(LABELS)])."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import Float64MultiArray

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=detect_storage_id(bag_path)),
        rosbag2_py.ConverterOptions("", ""),
    )

    available = {m.name: m.type for m in reader.get_all_topics_and_types()}
    if topic not in available:
        sys.exit(
            f"Topic '{topic}' not found in bag. Available topics:\n  "
            + "\n  ".join(f"{k} [{v}]" for k, v in sorted(available.items()))
        )

    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    times, rows, skipped = [], [], 0
    while reader.has_next():
        _, raw, t_ns = reader.read_next()
        msg = deserialize_message(raw, Float64MultiArray)
        if len(msg.data) != len(LABELS):
            skipped += 1
            continue
        times.append(t_ns * 1e-9)
        rows.append(list(msg.data))

    if skipped:
        print(f"Warning: skipped {skipped} messages with unexpected length "
              f"(expected {len(LABELS)}).")
    if not rows:
        sys.exit(f"No usable messages on '{topic}'.")

    t = np.asarray(times)
    data = np.asarray(rows, dtype=float)
    order = np.argsort(t)
    return t[order], data[order]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def break_gaps(t, data, factor=5.0):
    """Insert NaN rows where the debug stream paused (e.g. /lane_position went
    stale and the node only published stop commands), so lines don't bridge
    the gap."""
    if len(t) < 3:
        return t, data
    dt = np.diff(t)
    gap_thresh = factor * np.median(dt)
    gaps = np.where(dt > gap_thresh)[0]
    if len(gaps) == 0:
        return t, data
    t_out = np.insert(t, gaps + 1, t[gaps] + 0.5 * dt[gaps])
    nan_rows = np.full((len(gaps), data.shape[1]), np.nan)
    d_out = np.insert(data, gaps + 1, nan_rows, axis=0)
    print(f"Note: {len(gaps)} gap(s) > {gap_thresh:.3f}s in the debug stream "
          f"(stale /lane_position -> node stopped publishing).")
    return t_out, d_out


def true_spans(t, mask):
    """Contiguous [t_start, t_end] intervals where mask is True."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    edges = np.diff(mask.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return [(t[s], t[min(e, len(t) - 1)]) for s, e in zip(starts, ends)]


def summarize(t, d):
    col = lambda name: d[:, IDX[name]]
    p, h1, h2 = col("lane_position"), col("h1_left_margin"), col("h2_right_margin")
    active = col("cbf_active") > 0.5
    infeasible = col("omega_lower") > col("omega_upper") + 1e-9
    violation = (h1 < 0) | (h2 < 0)

    print("\n=== CBF debug summary ===")
    print(f"messages           : {len(t)}")
    print(f"duration           : {t[-1] - t[0]:.2f} s "
          f"(~{(len(t) - 1) / max(t[-1] - t[0], 1e-9):.1f} Hz)")
    print(f"p range            : [{np.min(p):.3f}, {np.max(p):.3f}]")
    print(f"min h1 (left)      : {np.min(h1):+.4f}")
    print(f"min h2 (right)     : {np.min(h2):+.4f}")
    print(f"safe-set violations: {violation.sum()} samples "
          f"({100 * violation.mean():.1f}%)")
    print(f"CBF active         : {active.sum()} samples ({100 * active.mean():.1f}%)")
    print(f"infeasible QP      : {infeasible.sum()} samples")
    print(f"|omega_safe - des| : max {np.max(np.abs(col('omega_safe') - col('omega_desired'))):.3f} rad/s")
    print()


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def style_axis(ax, ylabel):
    ax.set_ylabel(ylabel, color=C_INK)
    ax.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors=C_INK, labelsize=9)


def plot(t, d, title):
    import matplotlib.pyplot as plt

    col = lambda name: d[:, IDX[name]]
    p = col("lane_position")
    h1, h2 = col("h1_left_margin"), col("h2_right_margin")
    om_des, om_safe = col("omega_desired"), col("omega_safe")
    om_lo, om_hi = col("omega_lower"), col("omega_upper")
    speed = col("speed_cmd")
    active = np.nan_to_num(col("cbf_active")) > 0.5
    violation = (np.nan_to_num(h1, nan=1) < 0) | (np.nan_to_num(h2, nan=1) < 0)

    # p_min / p_max are recoverable from the margins: p_min = p - h1, p_max = p + h2
    p_min = np.nanmedian(p - h1)
    p_max = np.nanmedian(p + h2)

    fig, axes = plt.subplots(
        4, 1, sharex=True, figsize=(12, 10),
        gridspec_kw={"height_ratios": [1.3, 1, 1.3, 0.7]},
    )
    ax_p, ax_h, ax_w, ax_v = axes

    # Shade CBF-active and violation spans on every panel
    for ax in axes:
        for a, b in true_spans(t, active):
            ax.axvspan(a, b, color=C_ACTIVE, alpha=0.18, linewidth=0)
        for a, b in true_spans(t, violation):
            ax.axvspan(a, b, color=C_VIOLATION, alpha=0.15, linewidth=0)

    # 1. Lane position
    ax_p.axhspan(p_min, p_max, color=C_SAFE_BAND, alpha=0.6, linewidth=0,
                 label=f"safe set [{p_min:.2f}, {p_max:.2f}]")
    ax_p.axhline(0.5, color=C_INK, linestyle="--", linewidth=1, label="lane center")
    ax_p.plot(t, p, color=C_BLUE, linewidth=2, label="p (lane position)")
    ax_p.set_ylim(-0.02, 1.02)
    style_axis(ax_p, "p  (0=left, 1=right)")
    ax_p.legend(loc="lower right", bbox_to_anchor=(1, 1), fontsize=8, frameon=False, borderaxespad=0.2, ncol=3)

    # 2. Barrier margins
    ax_h.axhline(0.0, color=C_VIOLATION, linewidth=1.2)
    ax_h.plot(t, h1, color=C_ORANGE, linewidth=2, label="h1 = p - p_min (left)")
    ax_h.plot(t, h2, color=C_AQUA, linewidth=2, label="h2 = p_max - p (right)")
    style_axis(ax_h, "margin h")
    ax_h.legend(loc="lower right", bbox_to_anchor=(1, 1), fontsize=8, frameon=False, borderaxespad=0.2, ncol=2)

    # 3. Omega and CBF bounds
    ax_w.fill_between(t, om_lo, om_hi, color=C_BOUND, alpha=0.35, linewidth=0,
                      label="CBF admissible [ω_lower, ω_upper]")
    ax_w.plot(t, om_lo, color=C_BOUND, linewidth=1)
    ax_w.plot(t, om_hi, color=C_BOUND, linewidth=1)
    ax_w.plot(t, om_des, color=C_ORANGE, linewidth=1.5, linestyle="--",
              label="ω_desired (P ctrl)")
    ax_w.plot(t, om_safe, color=C_BLUE, linewidth=2, label="ω_safe (sent)")
    ax_w.axhline(0.0, color=C_INK, linewidth=0.8)
    style_axis(ax_w, "ω  [rad/s]")
    ax_w.legend(loc="lower right", bbox_to_anchor=(1, 1), fontsize=8, frameon=False, borderaxespad=0.2, ncol=3)

    # 4. Speed
    ax_v.plot(t, speed, color=C_BLUE, linewidth=2)
    ax_v.set_ylim(bottom=0)
    style_axis(ax_v, "speed  [m/s]")
    ax_v.set_xlabel("time  [s]", color=C_INK)

    # Legend entries for the shaded spans
    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(color=C_ACTIVE, alpha=0.4, label="CBF clipping ω"),
                 Patch(color=C_VIOLATION, alpha=0.3, label="safe-set violation (h < 0)")],
        loc="upper right", fontsize=9, frameon=False, ncol=2,
    )
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="path to the bag directory (or .mcap/.db3 file)")
    ap.add_argument("--topic", default="/lane_keeping/cbf_debug")
    ap.add_argument("-o", "--output", help="save figure to this file instead of showing it")
    ap.add_argument("--csv", help="also export the decoded data to this CSV file")
    ap.add_argument("--t0", type=float, help="start of time window [s, relative]")
    ap.add_argument("--t1", type=float, help="end of time window [s, relative]")
    args = ap.parse_args()

    if args.output:
        import matplotlib
        matplotlib.use("Agg")

    t_abs, data = read_cbf_debug(args.bag, args.topic)
    t = t_abs - t_abs[0]

    mask = np.ones_like(t, dtype=bool)
    if args.t0 is not None:
        mask &= t >= args.t0
    if args.t1 is not None:
        mask &= t <= args.t1
    t, data = t[mask], data[mask]
    if len(t) == 0:
        sys.exit("No samples in the requested time window.")

    summarize(t, data)

    if args.csv:
        np.savetxt(args.csv, np.column_stack([t, data]), delimiter=",",
                   header="t," + ",".join(LABELS), comments="", fmt="%.6f")
        print(f"Wrote {args.csv}")

    t_plot, d_plot = break_gaps(t, data)
    fig = plot(t_plot, d_plot, title=f"Lane-keeping CBF — {os.path.basename(os.path.normpath(args.bag))}")

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Wrote {args.output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
