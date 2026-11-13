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

Two debug layouts are supported, detected from the message length:
  * 16 values (current clauded.py: DubinsDynamics1D + EKF + HOCBF-QP)
      1. measured p and EKF estimate p_hat (+/- 2 sigma) with the safe band
      2. barrier margins h1 = p_hat - p_min, h2 = p_max - p_hat
      3. omega_desired vs omega_safe inside the admissible omega band (given v_safe)
      4. v_desired, v_safe and EKF v_hat
      5. EKF heading theta_hat
  * 10 values (older first-order clauded.py): p, margins, omega, speed

Spans where the CBF modified the command, and where a margin was negative,
are shaded on every panel.
"""

import argparse
import os
import sys

import numpy as np

# Must match LABELS in clauded.py (current layout)
LABELS_V2 = [
    "lane_position", "p_hat", "p_hat_std", "h1_left_margin", "h2_right_margin",
    "y_hat", "v_hat", "theta_hat", "v_desired", "omega_desired",
    "v_safe", "omega_safe", "omega_lower", "omega_upper", "cbf_active", "qp_status",
]
# Older first-order clauded.py layout
LABELS_V1 = [
    "lane_position", "error", "h1_left_margin", "h2_right_margin", "omega_desired",
    "omega_lower", "omega_upper", "omega_safe", "cbf_active", "speed_cmd",
]
LAYOUTS = {len(LABELS_V2): LABELS_V2, len(LABELS_V1): LABELS_V1}

# Colors (fixed order; same entity always gets the same color)
C_BLUE = "#2a78d6"     # p_hat / omega_safe / v_safe
C_ORANGE = "#eb6834"   # h1 (left) / desired commands
C_AQUA = "#1baf7a"     # h2 (right) / v_hat
C_BOUND = "#86b6ef"    # admissible omega band, p_hat sigma band
C_SAFE_BAND = "#cde2fb"
C_ACTIVE = "#fab219"   # CBF-active spans
C_VIOLATION = "#d03b3b"
C_INK = "#52514e"
C_MEAS = "#9a9992"     # raw measurement


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
    """Return (t [s, bag receive time, absolute], data [N x n], labels)."""
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

    times, rows, skipped, labels = [], [], 0, None
    while reader.has_next():
        _, raw, t_ns = reader.read_next()
        msg = deserialize_message(raw, Float64MultiArray)
        n = len(msg.data)
        if labels is None and n in LAYOUTS:
            labels = LAYOUTS[n]
        if labels is None or n != len(labels):
            skipped += 1
            continue
        times.append(t_ns * 1e-9)
        rows.append(list(msg.data))

    if skipped:
        print(f"Warning: skipped {skipped} messages with unexpected length "
              f"(known layouts: {sorted(LAYOUTS)} values).")
    if not rows:
        sys.exit(f"No usable messages on '{topic}'.")
    print(f"Detected {len(labels)}-value debug layout.")

    t = np.asarray(times)
    data = np.asarray(rows, dtype=float)
    order = np.argsort(t)
    return t[order], data[order], labels


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


def make_col(d, labels):
    idx = {name: i for i, name in enumerate(labels)}
    return lambda name: d[:, idx[name]]


def summarize(t, d, labels):
    col = make_col(d, labels)
    v2 = labels is LABELS_V2
    p, h1, h2 = col("lane_position"), col("h1_left_margin"), col("h2_right_margin")
    active = col("cbf_active") > 0.5
    violation = (h1 < 0) | (h2 < 0)

    print("\n=== CBF debug summary ===")
    print(f"messages             : {len(t)}")
    print(f"duration             : {t[-1] - t[0]:.2f} s "
          f"(~{(len(t) - 1) / max(t[-1] - t[0], 1e-9):.1f} Hz)")
    print(f"p (measured) range   : [{np.min(p):.3f}, {np.max(p):.3f}]")
    if v2:
        p_min = np.median(col("p_hat") - h1)
        p_max = np.median(col("p_hat") + h2)
        meas_viol = (p < p_min) | (p > p_max)
        err = p - col("p_hat")
        print(f"p_hat range          : [{np.min(col('p_hat')):.3f}, {np.max(col('p_hat')):.3f}]")
        print(f"p - p_hat            : rms {np.sqrt(np.mean(err ** 2)):.4f}, "
              f"max |.| {np.max(np.abs(err)):.4f}")
        print(f"theta_hat range      : [{np.min(col('theta_hat')):+.3f}, {np.max(col('theta_hat')):+.3f}] rad")
        print(f"measured p outside   : {meas_viol.sum()} samples ({100 * meas_viol.mean():.1f}%)")
    print(f"min h1 (left)        : {np.min(h1):+.4f}")
    print(f"min h2 (right)       : {np.min(h2):+.4f}")
    print(f"margin < 0           : {violation.sum()} samples ({100 * violation.mean():.1f}%)")
    print(f"CBF active           : {active.sum()} samples ({100 * active.mean():.1f}%)")
    if v2:
        unsolved = col("qp_status") != 1
        print(f"QP not solved        : {unsolved.sum()} samples (safest-input fallback used)")
        print(f"|v_safe - v_des|     : max {np.max(np.abs(col('v_safe') - col('v_desired'))):.3f}")
    else:
        infeasible = col("omega_lower") > col("omega_upper") + 1e-9
        print(f"infeasible QP        : {infeasible.sum()} samples")
    print(f"|omega_safe - des|   : max {np.max(np.abs(col('omega_safe') - col('omega_desired'))):.3f} rad/s")
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


def top_legend(ax, ncol):
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), fontsize=8,
              frameon=False, borderaxespad=0.2, ncol=ncol)


def plot(t, d, labels, title):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    col = make_col(d, labels)
    v2 = labels is LABELS_V2

    p = col("lane_position")
    p_ref = col("p_hat") if v2 else p          # what the margins are measured from
    h1, h2 = col("h1_left_margin"), col("h2_right_margin")
    om_des, om_safe = col("omega_desired"), col("omega_safe")
    om_lo, om_hi = col("omega_lower"), col("omega_upper")
    active = np.nan_to_num(col("cbf_active")) > 0.5
    violation = (np.nan_to_num(h1, nan=1) < 0) | (np.nan_to_num(h2, nan=1) < 0)

    # p_min / p_max are recoverable from the margins
    p_min = np.nanmedian(p_ref - h1)
    p_max = np.nanmedian(p_ref + h2)

    if v2:
        fig, axes = plt.subplots(5, 1, sharex=True, figsize=(12, 12),
                                 gridspec_kw={"height_ratios": [1.3, 1, 1.3, 0.8, 0.8]})
        ax_p, ax_h, ax_w, ax_v, ax_th = axes
    else:
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 10),
                                 gridspec_kw={"height_ratios": [1.3, 1, 1.3, 0.7]})
        ax_p, ax_h, ax_w, ax_v = axes

    for ax in axes:
        for a, b in true_spans(t, active):
            ax.axvspan(a, b, color=C_ACTIVE, alpha=0.18, linewidth=0)
        for a, b in true_spans(t, violation):
            ax.axvspan(a, b, color=C_VIOLATION, alpha=0.15, linewidth=0)

    # 1. Lane position
    ax_p.axhspan(p_min, p_max, color=C_SAFE_BAND, alpha=0.6, linewidth=0,
                 label=f"safe set [{p_min:.2f}, {p_max:.2f}]")
    ax_p.axhline(0.5, color=C_INK, linestyle="--", linewidth=1, label="lane center")
    if v2:
        sd = col("p_hat_std")
        ax_p.fill_between(t, p_ref - 2 * sd, p_ref + 2 * sd, color=C_BOUND, alpha=0.4,
                          linewidth=0, label="p_hat ± 2σ")
        ax_p.plot(t, p, color=C_MEAS, linewidth=1, label="p measured")
        ax_p.plot(t, p_ref, color=C_BLUE, linewidth=2, label="p_hat (EKF)")
        top_legend(ax_p, 5)
    else:
        ax_p.plot(t, p, color=C_BLUE, linewidth=2, label="p (lane position)")
        top_legend(ax_p, 3)
    ax_p.set_ylim(-0.02, 1.02)
    style_axis(ax_p, "p  (0=left, 1=right)")

    # 2. Barrier margins
    src = "p_hat" if v2 else "p"
    ax_h.axhline(0.0, color=C_VIOLATION, linewidth=1.2)
    ax_h.plot(t, h1, color=C_ORANGE, linewidth=2, label=f"h1 = {src} - p_min (left)")
    ax_h.plot(t, h2, color=C_AQUA, linewidth=2, label=f"h2 = p_max - {src} (right)")
    style_axis(ax_h, "margin h")
    top_legend(ax_h, 2)

    # 3. Omega and admissible band
    band_label = "admissible ω given v_safe" if v2 else "CBF admissible [ω_lower, ω_upper]"
    ax_w.fill_between(t, om_lo, om_hi, color=C_BOUND, alpha=0.35, linewidth=0, label=band_label)
    ax_w.plot(t, om_lo, color=C_BOUND, linewidth=1)
    ax_w.plot(t, om_hi, color=C_BOUND, linewidth=1)
    ax_w.plot(t, om_des, color=C_ORANGE, linewidth=1.5, linestyle="--", label="ω_desired")
    ax_w.plot(t, om_safe, color=C_BLUE, linewidth=2, label="ω_safe (sent)")
    ax_w.axhline(0.0, color=C_INK, linewidth=0.8)
    style_axis(ax_w, "ω  [rad/s]")
    top_legend(ax_w, 3)

    # 4. Speed
    if v2:
        ax_v.plot(t, col("v_desired"), color=C_ORANGE, linewidth=1.5, linestyle="--", label="v_desired")
        ax_v.plot(t, col("v_safe"), color=C_BLUE, linewidth=2, label="v_safe (sent)")
        ax_v.plot(t, col("v_hat"), color=C_AQUA, linewidth=1.5, label="v_hat (EKF)")
        top_legend(ax_v, 3)
    else:
        ax_v.plot(t, col("speed_cmd"), color=C_BLUE, linewidth=2)
    ax_v.set_ylim(bottom=0)
    style_axis(ax_v, "speed")

    # 5. Heading estimate
    if v2:
        ax_th.axhline(0.0, color=C_INK, linewidth=0.8)
        ax_th.plot(t, col("theta_hat"), color=C_BLUE, linewidth=2, label="θ_hat (EKF), + = left")
        style_axis(ax_th, "θ  [rad]")
        top_legend(ax_th, 1)

    axes[-1].set_xlabel("time  [s]", color=C_INK)

    fig.legend(
        handles=[Patch(color=C_ACTIVE, alpha=0.4, label="CBF modifying command"),
                 Patch(color=C_VIOLATION, alpha=0.3, label="margin < 0")],
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

    t_abs, data, labels = read_cbf_debug(args.bag, args.topic)
    t = t_abs - t_abs[0]

    mask = np.ones_like(t, dtype=bool)
    if args.t0 is not None:
        mask &= t >= args.t0
    if args.t1 is not None:
        mask &= t <= args.t1
    t, data = t[mask], data[mask]
    if len(t) == 0:
        sys.exit("No samples in the requested time window.")

    summarize(t, data, labels)

    if args.csv:
        np.savetxt(args.csv, np.column_stack([t, data]), delimiter=",",
                   header="t," + ",".join(labels), comments="", fmt="%.6f")
        print(f"Wrote {args.csv}")

    t_plot, d_plot = break_gaps(t, data)
    fig = plot(t_plot, d_plot, labels,
               title=f"Lane-keeping CBF — {os.path.basename(os.path.normpath(args.bag))}")

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Wrote {args.output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
