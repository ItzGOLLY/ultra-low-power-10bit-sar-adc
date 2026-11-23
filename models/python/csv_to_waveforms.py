#!/usr/bin/env python3
"""
csv_to_waveforms.py - turn the SAR ADC CSV outputs into waveforms
=================================================================

Reads the two CSV files written by task1_sar_adc_conventional.py (or task2_sar_adc_vcm_based.py):

    task1_bit_waveforms.csv    time_us + one column per signal (clocks, comparator, SAR register,
                               CDAC bottom plates, output register, code)
    task1_digital_output.csv   one row per conversion (input sample, code, b9..b0, reconstruction,
                               error, energy)

and writes, into --outdir:

    bit_waveforms_all.png               every signal over every conversion in the file
    bit_waveforms_conversion<N>.png     one conversion zoomed in: the SAR deciding b9 ... b0
    digital_output.png                  input vs reconstructed output, output bits, error, energy,
                                        plus an equivalent-time view when the input is a single tone
    <name>.vcsv                         same signals for Cadence ViVA: File > Open Results
    <name>_ascii.txt                    space-separated copy for ViVA's getAsciiWave() calculator function

How a CSV becomes a waveform
    * the time column is the x axis (time_us for bit waveforms, t_sample_s for the digital output)
    * every other numeric column becomes one trace against it
    * 0/1 columns are drawn as logic lanes, held flat between samples (zero-order hold)
    * multi-bit columns (the output code) are drawn as a bus lane with the value written in it
    * per-conversion values hold until the next conversion, so they are drawn as steps

Usage
    python csv_to_waveforms.py                                   # finds the CSVs next to it / in task1_outputs
    python csv_to_waveforms.py --bits task1_outputs/task1_bit_waveforms.csv \\
                               --digital task1_outputs/task1_digital_output.csv
    python csv_to_waveforms.py --frame 3 --first 48              # zoom on conversion 3, show 48 conversions
    python csv_to_waveforms.py --logic-high 1.8                  # logic 1 = 1.8 V in the ViVA files
    python csv_to_waveforms.py --to-vcsv task2_outputs/          # every CSV in a folder -> .vcsv only
    python csv_to_waveforms.py --to-vcsv a.csv b.csv --outdir vcsv

--to-vcsv reads the x axis from each file (time_us, t_sample_s, frequency_Hz, code, or Virtuoso's
own "/name X,/name Y" column pairs). Files without such a column (e.g. the summary table) are skipped.

Requires numpy; matplotlib for the PNG figures (the ViVA files are written without it).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import numpy as np

# ------------------------------------------------------------------ colours (validated palette)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # primary text
INK2 = "#52514e"         # secondary text, logic lanes
MUTED = "#898781"        # axis labels, annotations
GRID = "#e1e0d9"         # hairline grid
AXIS = "#c3c2b7"         # axis rules
BLUE = "#2a78d6"         # slot 1: the analog input
ORANGE = "#eb6834"       # slot 2: ADC / CDAC output
AQUA = "#1baf7a"         # slot 3: third identity (CDAC energy)

STEP_EPS = 1e-12         # pre-edge offset used to make sharp steps in the ViVA files [s]


# =============================================================================================
# reading
# =============================================================================================
def read_csv(path):
    """Return (header, dict name -> float array). Empty cells become NaN; text columns are skipped."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip() for h in rows[0]]
    body = [r for r in rows[1:] if r and any(c.strip() for c in r)]
    cols = {}
    for j, name in enumerate(header):
        if name.endswith("binary"):
            continue                                   # text column, e.g. code_binary
        try:
            cols[name] = np.array([float(r[j]) if j < len(r) and r[j].strip() else np.nan
                                   for r in body])
        except ValueError:
            continue
    return header, cols


def kind_of(name, v):
    """logic (0/1), level (few discrete voltages), bus (integer code), const, analog."""
    u = np.unique(v)
    if u.size == 1:
        return "const"
    if "code" in name.lower() and np.all(np.abs(v - np.round(v)) < 1e-9):
        return "bus"
    if set(u.tolist()) <= {0.0, 1.0}:
        return "logic"
    if u.size <= 3 and u.min() >= 0:
        return "level"                                 # e.g. bottom plates 0 / Vcm / Vref
    return "analog"


def pretty(name):
    m = re.fullmatch(r"BP_(\d+C)_(b\d)_V", name)
    if m:
        return f"{m.group(1)} plate ({m.group(2)})"
    if name == "BP_dummy_V":
        return "dummy plate"
    table = {"fs": "fs (convert)", "PHI1": "PHI1 (track)", "PHI1b": "PHI1b", "PHI2": "PHI2",
             "CLK_cmp": "CLK (comparator)", "Dcmp": "Dcmp", "Dcmpb": "Dcmpb", "EOC": "EOC",
             "Dout_code": "code", "code": "code", "Vin_V": "Vin", "Vx_V": "Vx (top plate)",
             "Vcm_V": "Vcm"}
    if name in table:
        return table[name]
    m = re.fullmatch(r"SAR_(b\d)", name)
    if m:
        return f"SAR {m.group(1)}"
    return name.replace("_", " ")


# =============================================================================================
# ViVA files
# =============================================================================================
def stepify(t, cols, step_names, analog_names):
    """Insert a point just before every sample where a step signal changes, so that a viewer
    that joins points with straight lines still shows sharp edges. Analog columns are
    linearly interpolated at the inserted points."""
    n = t.size
    change = np.zeros(n, bool)
    for name in step_names:
        v = cols[name]
        change[1:] |= v[1:] != v[:-1]
    idx = np.where(change)[0]
    t_new = np.empty(n + idx.size)
    out = {k: np.empty(n + idx.size) for k in list(step_names) + list(analog_names)}
    pos = 0
    prev = 0
    for i in list(idx) + [n]:
        seg = slice(prev, i)
        m = i - prev
        t_new[pos:pos + m] = t[seg]
        for k in out:
            out[k][pos:pos + m] = cols[k][seg]
        pos += m
        if i < n:
            te = t[i] - min(STEP_EPS, 0.01 * (t[i] - t[i - 1]))
            t_new[pos] = te
            for k in step_names:
                out[k][pos] = cols[k][i - 1]
            for k in analog_names:
                a = (te - t[i - 1]) / (t[i] - t[i - 1])
                out[k][pos] = cols[k][i - 1] + a * (cols[k][i] - cols[k][i - 1])
            pos += 1
        prev = i
    return t_new, out


def y_unit(name, kind, logic_high):
    """Unit string written in the VCSV header ('yunit' is what ViVA itself writes for no unit)."""
    if re.search(r"_V$|_V\(|^V", name) or kind == "level":
        return "V"
    if kind == "logic":
        return "V" if logic_high != 1.0 else "yunit"
    if "dB" in name:
        return "dB"
    return "yunit"


def safe_name(name, used):
    """ViVA wants plain alphanumeric signal names, unique within a file."""
    s = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "signal"
    base, k = s, 2
    while s in used:
        s, k = f"{base}_{k}", k + 1
    used.add(s)
    return s


def write_vcsv(path, x, signals, xname="time", xunit="s"):
    """signals: list of (name, y_array, y_unit), all sharing x.
    VCSV = 6 header lines starting with ';', then one (x, y) column pair per signal."""
    n = len(signals)
    with open(path, "w") as f:
        f.write(";Version, 1, 0\n")
        f.write(";" + ";".join(nm for nm, _, _ in signals) + "\n")
        f.write(";" + ";".join(["X, Y"] * n) + "\n")
        f.write(";" + ";".join(["Re, Re"] * n) + "\n")
        f.write(";" + ";".join([f"{xname}, Y"] * n) + "\n")
        f.write(";" + ";".join(f"{xunit}, {u}" for _, _, u in signals) + "\n")
        for i in range(x.size):
            xs = f"{x[i]:.12g}"
            f.write(",".join(f"{xs},{y[i]:.8g}" for _, y, _ in signals) + "\n")


def write_viva_files(base, t_s, cols, names, step_names, logic_names, logic_high, kinds=None,
                     ascii=True):
    analog_names = [k for k in names if k not in step_names]
    t2, data = stepify(t_s, cols, step_names, analog_names)
    for k in logic_names:
        data[k] = data[k] * logic_high
    kinds = kinds or {}
    used = set()
    write_vcsv(base + ".vcsv", t2,
               [(safe_name(k, used), data[k], y_unit(k, kinds.get(k, "analog"), logic_high)) for k in names])
    if not ascii:
        return base + ".vcsv", None, t2.size
    # plain columns for getAsciiWave(file xColumn yColumn); ';' lines are comments
    with open(base + "_ascii.txt", "w") as f:
        f.write("; column 1 = time_s\n")
        for c, k in enumerate(names, start=2):
            f.write(f"; column {c} = {k}\n")
        for i in range(t2.size):
            f.write(f"{t2[i]:.12g} " + " ".join(f"{data[k][i]:.8g}" for k in names) + "\n")
    return base + ".vcsv", base + "_ascii.txt", t2.size


# =============================================================================================
# plotting helpers
# =============================================================================================
def style_axes(ax, show_x=True):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8, width=0.6, length=3)
    ax.grid(True, axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    if not show_x:
        ax.tick_params(labelbottom=False)


def zoh(t, v, t_end):
    """Extend a per-sample series so the last value holds until t_end (for step drawing)."""
    return np.append(t, t_end), np.append(v, v[-1])


def draw_logic(ax, t, v, y0, h, t_end):
    tt, vv = zoh(t, v, t_end)
    ax.fill_between(tt, y0, y0 + h * vv, step="post", color=INK2, alpha=0.10, linewidth=0)
    ax.step(tt, y0 + h * vv, where="post", color=INK2, linewidth=0.9)


def draw_bus(ax, fig, t, v, y0, h, t_end, fmt=lambda x: f"{int(round(x))}"):
    """Bus lane: one chamfered box per constant run, value written inside when it fits."""
    renderer = fig.canvas.get_renderer()
    x0, x1 = ax.get_xlim()
    bbox = ax.get_window_extent(renderer)
    px_per_x = bbox.width / (x1 - x0)
    starts = np.r_[0, np.where(v[1:] != v[:-1])[0] + 1]
    ends = np.r_[starts[1:], v.size]
    mid = y0 + h / 2
    for a, b in zip(starts, ends):
        ta = max(t[a], x0)
        tb = min(t[b] if b < v.size else t_end, x1)
        if tb <= ta:
            continue
        d = min(3.0 / px_per_x, 0.3 * (tb - ta))
        xs = [ta, ta + d, tb - d, tb, tb - d, ta + d, ta]
        ys = [mid, y0 + h, y0 + h, mid, y0, y0, mid]
        ax.fill(xs, ys, color=INK2, alpha=0.10, linewidth=0)
        ax.plot(xs, ys, color=INK2, linewidth=0.9)
        label = fmt(v[a])
        txt = ax.text((ta + tb) / 2, mid, label, ha="center", va="center", fontsize=7, color=INK)
        w = txt.get_window_extent(renderer).width
        if w + 8 > (tb - ta) * px_per_x:           # does not fit: leave it out, never clip
            txt.remove()


def lane_layout(groups, pitch=1.0, gap=0.9):
    """Assign a baseline y to every lane (top to bottom). Returns list of (group, [(name, y0)])."""
    y = 0.0
    out = []
    for gname, members in groups:
        if not members:
            continue
        y -= gap
        lanes = []
        for name in members:
            y -= pitch
            lanes.append((name, y))
        out.append((gname, lanes))
    return out, y


def draw_lanes(ax, fig, layout, t, cols, kinds, t_end, h=0.68):
    for gname, lanes in layout:
        ax.text(-0.135, lanes[0][1] + 1.0, gname, transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=8, color=INK2, fontweight="bold")
        for name, y0 in lanes:
            v = cols[name]
            ax.text(-0.008, y0 + h / 2, pretty(name), transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=7.5, color=INK)
            if kinds[name] == "bus":
                draw_bus(ax, fig, t, v, y0, h, t_end)
            else:
                top = max(np.max(np.abs(v)), 1e-12)
                draw_logic(ax, t, v / top, y0, h, t_end)


# =============================================================================================
# bit waveforms
# =============================================================================================
BIT_GROUPS = [
    ("Clocks", [r"^fs$", r"^PHI1$", r"^PHI1b$", r"^PHI2$", r"^CLK"]),
    ("Comparator", [r"^outp$", r"^outn$", r"^Dcmp$", r"^Dcmpb$"]),
    ("SAR register", [r"^SAR_b\d$"]),
    ("CDAC bottom plates (0 / Vcm / Vref)", [r"^BP_"]),
    ("Output register", [r"^EOC$", r"^D\d$", r"code$"]),
]


def group_columns(names, kinds, patterns):
    used = set()
    groups = []
    for gname, pats in patterns:
        members = []
        for p in pats:
            for n in names:
                if n not in used and re.search(p, n) and kinds[n] in ("logic", "level", "bus", "const"):
                    members.append(n)
                    used.add(n)
        groups.append((gname, members))
    rest = [n for n in names if n not in used and kinds[n] in ("logic", "level", "bus")]
    if rest:
        groups.append(("Other", rest))
    return groups


def plot_bit_waveforms(path_png, t_us, cols, kinds, analog, groups, t_lo, t_hi, title,
                       frame_marks=None, cycle_marks=None):
    import matplotlib.pyplot as plt

    layout, y_bottom = lane_layout(groups)
    n_lanes = sum(len(l) for _, l in layout)
    lanes_h = 0.205 * (n_lanes + 0.9 * len(layout)) + 0.2
    strip_h = 1.9
    top_m, gap_m, bot_m = 0.75, 0.38, 0.55
    H = top_m + strip_h + gap_m + lanes_h + bot_m
    W = 13.0
    left, right = 1.75 / W, 1.0 / W
    fig = plt.figure(figsize=(W, H), dpi=160, facecolor=SURFACE)
    ax_a = fig.add_axes([left, (bot_m + lanes_h + gap_m) / H, 1 - left - right, strip_h / H])
    ax_l = fig.add_axes([left, bot_m / H, 1 - left - right, lanes_h / H], sharex=ax_a)
    style_axes(ax_a, show_x=False)
    style_axes(ax_l)
    ax_l.spines["left"].set_visible(False)
    ax_l.tick_params(axis="y", left=False, labelleft=False)

    # analog strip: input, top plate, threshold
    sel = (t_us >= t_lo) & (t_us <= t_hi)
    colors = [BLUE, ORANGE]
    k = 0
    for name in analog:
        v = cols[name]
        if kinds[name] == "const":
            what = "comparator threshold" if name.lower().startswith("vcm") else "constant"
            ax_a.axhline(v[0], color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)),
                         label=f"{pretty(name)} ({what})")
            ax_a.text(1.005, v[0], f"{pretty(name)} {v[0]:g} V", transform=ax_a.get_yaxis_transform(),
                      ha="left", va="center", fontsize=7.5, color=INK2, clip_on=False)
            k += 1
            continue
        ax_a.plot(t_us[sel], v[sel], color=colors[min(k, 1)], linewidth=1.3, label=pretty(name))
        k += 1
    ax_a.set_ylabel("V", color=INK2, fontsize=8, rotation=0, ha="right", va="center")
    ax_a.grid(True, axis="y", color=GRID, linewidth=0.5)
    if k >= 2:
        leg = ax_a.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=k, frameon=False,
                          fontsize=8, handlelength=1.6, borderaxespad=0.2)
        for txt in leg.get_texts():
            txt.set_color(INK)

    ax_l.set_xlim(t_lo, t_hi)
    ax_l.set_ylim(y_bottom - 0.2, 0.2)
    draw_lanes(ax_l, fig, layout, t_us, cols, kinds, t_end=t_us[-1] + (t_us[1] - t_us[0]))
    ax_l.set_xlabel("time (µs)", color=INK2, fontsize=8.5)

    for ax in (ax_a, ax_l):
        if frame_marks:
            for tm, _ in frame_marks:
                ax.axvline(tm, color=AXIS, linewidth=0.6)
        if cycle_marks:
            for tm, _ in cycle_marks:
                ax.axvline(tm, color=GRID, linewidth=0.6)
    lane_top = ax_l.get_ylim()[1]
    if frame_marks:
        for tm, lab in frame_marks:
            if t_lo <= tm < t_hi:
                ax_l.text(tm, lane_top, f" {lab}", ha="left", va="bottom", fontsize=7, color=MUTED)
    if cycle_marks:
        for i, (tm, lab) in enumerate(cycle_marks):
            t_next = cycle_marks[i + 1][0] if i + 1 < len(cycle_marks) else t_hi
            if t_lo <= tm < t_hi:
                ax_l.text((tm + min(t_next, t_hi)) / 2, lane_top, lab, ha="center", va="bottom",
                          fontsize=7, color=MUTED)

    fig.text(left, 1 - 0.28 / H, title, ha="left", va="center", fontsize=11, color=INK)
    fig.savefig(path_png, facecolor=SURFACE)
    plt.close(fig)


def cycle_boundaries(t_us, cols, frame_mask):
    """Track / bit / EOC cycles of one conversion, from the fs, CLK and EOC columns."""
    t = t_us[frame_mask]
    marks = [(t[0], "track")]
    if "fs" in cols:
        fs = cols["fs"][frame_mask]
        rise = np.where((fs[1:] > 0.5) & (fs[:-1] < 0.5))[0]
        if rise.size:
            t_conv = t[rise[0] + 1]
            eoc_t = None
            if "EOC" in cols:
                e = cols["EOC"][frame_mask]
                er = np.where((e[1:] > 0.5) & (e[:-1] < 0.5))[0]
                eoc_t = t[er[0] + 1] if er.size else None
            nbits = sum(1 for k in cols if re.fullmatch(r"SAR_b\d", k)) or 10
            t_end = eoc_t if eoc_t is not None else t[-1]
            tb = (t_end - t_conv) / nbits
            for j in range(nbits):
                marks.append((t_conv + j * tb, f"b{nbits - 1 - j}"))
            if eoc_t is not None:
                marks.append((eoc_t, "EOC"))
    return marks


# =============================================================================================
# digital output
# =============================================================================================
def plot_digital_output(path_png, cols, kinds, first, title):
    import matplotlib.pyplot as plt

    t_s = cols["t_sample_s"]
    n_all = t_s.size
    n = min(first, n_all)
    t_us = t_s[:n] * 1e6
    period = np.median(np.diff(t_s)) * 1e6 if n_all > 1 else 1.0
    t_end = t_us[-1] + period
    bits = [k for k in cols if re.fullmatch(r"b\d", k)]
    bits.sort(key=lambda k: -int(k[1:]))

    # equivalent-time view: only when the input is one coherent tone
    fold = None
    if "Vin_ideal_V" in cols and n_all >= 64:
        x = cols["Vin_ideal_V"] - cols["Vin_ideal_V"].mean()
        p = np.abs(np.fft.rfft(x)) ** 2
        k0 = 1 + int(np.argmax(p[1:]))
        if p[k0] / p[1:].sum() > 0.995:
            phase = (k0 * np.arange(n_all) / n_all) % 1.0
            order = np.argsort(phase)
            fin = k0 / n_all / (np.median(np.diff(t_s)))
            fold = (phase[order] / fin * 1e6, cols["Vin_ideal_V"][order],
                    cols.get("Vout_reconstructed_V", cols["Vin_ideal_V"])[order], fin, k0)

    rows = [("a", 2.0), ("b", 0.205 * (len(bits) + 1.6) + 0.25), ("c", 1.15), ("d", 1.35)]
    if fold:
        rows.append(("e", 2.0))
    top_m, gap, bot_m = 0.75, 0.42, 0.6
    H = top_m + sum(h for _, h in rows) + gap * (len(rows) - 1) + bot_m + (0.35 if fold else 0)
    W = 13.0
    left, right = 1.75 / W, 1.25 / W
    fig = plt.figure(figsize=(W, H), dpi=160, facecolor=SURFACE)
    axes = {}
    y = H - top_m
    for key, h in rows:
        if key == "e":
            y -= 0.35
        y -= h
        axes[key] = fig.add_axes([left, y / H, 1 - left - right, h / H])
        y -= gap
    for key in ("b", "c", "d"):
        axes[key].sharex(axes["a"])

    def panel_title(ax, text):
        ax.text(0.0, 1.03, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5,
                color=INK2)

    # (a) input samples vs reconstructed output
    ax = axes["a"]
    style_axes(ax, show_x=False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.5)
    if "Vout_reconstructed_V" in cols:
        tt, vv = zoh(t_us, cols["Vout_reconstructed_V"][:n], t_end)
        ax.step(tt, vv, where="post", color=ORANGE, linewidth=1.3, label="ADC output (code x LSB)")
    if "Vin_ideal_V" in cols:
        ax.plot(t_us, cols["Vin_ideal_V"][:n], linestyle="none", marker="o", markersize=4.2,
                markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=0.8,
                label="input Vin at each sampling instant")
    ax.set_ylabel("V", color=INK2, fontsize=8, rotation=0, ha="right", va="center")
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=8)
    for txt in leg.get_texts():
        txt.set_color(INK)
    panel_title(ax, "Input and reconstructed output")

    # (b) output bits and code
    ax = axes["b"]
    style_axes(ax, show_x=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlim(t_us[0], t_end)
    lanes = [("Output bits", bits + (["code"] if "code" in cols else []))]
    layout, y_bottom = lane_layout(lanes, gap=0.35)
    ax.set_ylim(y_bottom - 0.2, 0.0)
    sub = {k: cols[k][:n] for k in bits + (["code"] if "code" in cols else [])}
    kinds_sub = {k: ("bus" if k == "code" else "logic") for k in sub}
    for _, ln in layout:
        for name, y0 in ln:
            ax.text(-0.008, y0 + 0.34, name, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=7.5, color=INK)
            if kinds_sub[name] == "bus":
                draw_bus(ax, fig, t_us, sub[name], y0, 0.68, t_end)
            else:
                draw_logic(ax, t_us, sub[name], y0, 0.68, t_end)
    panel_title(ax, "Output bits b9 (MSB) ... b0 and the code they form")

    # (c) quantisation error
    ax = axes["c"]
    style_axes(ax, show_x=False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.5)
    if "error_LSB" in cols:
        tt, vv = zoh(t_us, cols["error_LSB"][:n], t_end)
        ax.axhline(0, color=AXIS, linewidth=0.6)
        ax.step(tt, vv, where="post", color=INK2, linewidth=1.1)
    ax.set_ylabel("LSB", color=INK2, fontsize=8, rotation=0, ha="right", va="center")
    panel_title(ax, "Error: reconstructed output minus input (includes the model's offset and gain error)")

    # (d) energy per conversion
    ax = axes["d"]
    style_axes(ax)
    ax.grid(True, axis="y", color=GRID, linewidth=0.5)
    e_series = [(k, c, lab) for k, c, lab in (("E_total_pJ", INK2, "total"),
                                              ("E_CDAC_pJ", AQUA, "CDAC (from Vref)"),
                                              ("E_CDAC_Vref_pJ", AQUA, "CDAC (from Vref)"),
                                              ("E_total_ideal_Vcm_pJ", INK2, "total"))
                if k in cols]
    for k, c, lab in e_series:
        tt, vv = zoh(t_us, cols[k][:n], t_end)
        ax.step(tt, vv, where="post", color=c, linewidth=1.3, label=lab)
        ax.text(t_end, vv[-1], f" {lab}", ha="left", va="center", fontsize=7.5, color=INK2,
                clip_on=False)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("pJ", color=INK2, fontsize=8, rotation=0, ha="right", va="center")
    if len(e_series) >= 2:
        leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False,
                        fontsize=8)
        for txt in leg.get_texts():
            txt.set_color(INK)
    panel_title(ax, "Energy per conversion")
    ax.set_xlabel("time (µs)", color=INK2, fontsize=8.5)
    axes["a"].set_xlim(t_us[0], t_end)

    # (e) equivalent-time view
    if fold:
        te, vin_f, vout_f, fin, k0 = fold
        ax = axes["e"]
        style_axes(ax)
        ax.grid(True, axis="y", color=GRID, linewidth=0.5)
        ax.plot(te, vout_f, color=ORANGE, linewidth=1.3, label="ADC output")
        ax.plot(te, vin_f, color=BLUE, linewidth=0.9, label="input Vin")
        ax.set_xlim(0, te.max())
        ax.set_ylabel("V", color=INK2, fontsize=8, rotation=0, ha="right", va="center")
        ax.set_xlabel("time within one input period (µs)", color=INK2, fontsize=8.5)
        leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False,
                        fontsize=8)
        for txt in leg.get_texts():
            txt.set_color(INK)
        panel_title(ax, f"All {n_all} conversions folded onto one period of the "
                        f"{fin / 1e3:.4g} kHz input (equivalent-time view)")

    fig.text(left, 1 - 0.28 / H, title, ha="left", va="center", fontsize=11, color=INK)
    fig.savefig(path_png, facecolor=SURFACE)
    plt.close(fig)



# =============================================================================================
# any CSV -> VCSV
# =============================================================================================
X_AXES = [  # (column regex, x name in the VCSV, x unit, scale to SI, per-sample hold)
    (r"^time_us$", "time", "s", 1e-6, False),
    (r"^(time_s|time)$", "time", "s", 1.0, False),
    (r"^t_sample_s$", "time", "s", 1.0, True),
    (r"^(frequency_Hz|freq_Hz|frequency|freq)$", "freq", "Hz", 1.0, False),
    (r"^code$", "code", "code", 1.0, False),
]
SKIP_Y = {"n", "frame", "metastable_decisions"}


def csv_to_vcsv(path, outdir, logic_high=1.0):
    """Convert one CSV to <outdir>/<name>.vcsv. Returns (vcsv path, report) or (None, reason)."""
    header, cols = read_csv(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(outdir, stem + ".vcsv")

    # Virtuoso's own CSV export: "/sig X", "/sig Y" column pairs
    pairs = [(h[:-2], h, h[:-2] + " Y") for h in header if h.endswith(" X") and h[:-2] + " Y" in cols]
    if pairs and all(h in cols for _, h, _ in pairs):
        x = cols[pairs[0][1]]
        if all(np.array_equal(cols[hx], x) for _, hx, _ in pairs):
            used = set()
            sig = [(safe_name(nm.lstrip("/"), used), cols[hy], "V") for nm, _, hy in pairs]
            write_vcsv(out, x, sig, "time", "s")
            return out, f"{len(sig)} signal(s) x {x.size} points (time)"
        return None, "column pairs have different time axes"

    xcol = None
    for pat, xname, xunit, scale, hold in X_AXES:
        xcol = next((h for h in header if h in cols and re.fullmatch(pat, h)), None)
        if xcol:
            break
    if not xcol:
        return None, "no time / frequency / code column - not a waveform (left as CSV)"
    names = [h for h in header if h in cols and h != xcol and h not in SKIP_Y]
    if not names:
        return None, "no numeric signal columns"
    good = np.isfinite(cols[xcol])
    for k in names:
        good &= np.isfinite(cols[k])
    dropped = int((~good).sum())
    x = cols[xcol][good] * scale
    sub = {k: cols[k][good] for k in names}
    kinds = {k: kind_of(k, sub[k]) for k in names}

    if xname == "time":
        steps = names if hold else [k for k in names if kinds[k] in ("logic", "level", "bus", "const")]
        logic = [k for k in names if kinds[k] == "logic"]
        write_viva_files(os.path.join(outdir, stem), x, sub, names, steps, logic, logic_high, kinds,
                         ascii=False)
        npts = sum(1 for _ in open(out)) - 6
    else:                                      # frequency or code axis: plain x-y curves
        used = set()
        write_vcsv(out, x, [(safe_name(k, used), sub[k], y_unit(k, kinds[k], 1.0)) for k in names],
                   xname, xunit)
        npts = x.size
    note = f"; {dropped} row(s) with empty cells left out" if dropped else ""
    return out, f"{len(names)} signals x {npts} points ({xname}){note}"


def convert_paths_to_vcsv(paths, outdir, logic_high):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(".csv"))
        else:
            files.append(p)
    os.makedirs(outdir, exist_ok=True)
    for f in files:
        out, msg = csv_to_vcsv(f, outdir, logic_high)
        print(f"  {os.path.basename(f):32s} -> {os.path.basename(out) if out else 'skipped'}: {msg}")


# =============================================================================================
# main
# =============================================================================================
def find_default(name):
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.getcwd(), os.path.join(os.getcwd(), "task1_outputs"), here,
              os.path.join(here, "task1_outputs")):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert SAR ADC CSV outputs into waveforms")
    ap.add_argument("--bits", help="bit-waveform CSV (default: task1_bit_waveforms.csv)")
    ap.add_argument("--digital", help="digital-output CSV (default: task1_digital_output.csv)")
    ap.add_argument("--outdir", default="waveforms", help="output folder (default: waveforms)")
    ap.add_argument("--frame", type=int, default=1, help="conversion to zoom on (default 1)")
    ap.add_argument("--first", type=int, default=32, help="conversions shown in digital_output.png")
    ap.add_argument("--logic-high", type=float, default=1.0,
                    help="voltage written for logic 1 in the ViVA files (default 1.0)")
    ap.add_argument("--to-vcsv", nargs="+", metavar="CSV_OR_FOLDER",
                    help="only convert these CSV files / every CSV in these folders to .vcsv")
    a = ap.parse_args(argv)

    if a.to_vcsv:
        print(f"VCSV files in {os.path.abspath(a.outdir)}:")
        convert_paths_to_vcsv(a.to_vcsv, a.outdir, a.logic_high)
        return 0

    bits_path = a.bits or find_default("task1_bit_waveforms.csv") or find_default("task2_bit_waveforms.csv")
    dig_path = a.digital or find_default("task1_digital_output.csv") or find_default("task2_digital_output.csv")
    if not bits_path and not dig_path:
        sys.exit("No CSV found - pass --bits and/or --digital")
    os.makedirs(a.outdir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        have_mpl = True
    except ImportError:
        have_mpl = False
        print("matplotlib not installed: writing the ViVA files only (pip install matplotlib for PNGs)")

    made = []
    # ------------------------------------------------------------------ bit waveforms
    if bits_path:
        header, cols = read_csv(bits_path)
        tcol = "time_us" if "time_us" in cols else header[0]
        t_us = cols[tcol]
        names = [h for h in header if h in cols and h not in (tcol, "frame")]
        kinds = {k: kind_of(k, cols[k]) for k in names}
        groups = group_columns(names, kinds, BIT_GROUPS)
        in_lanes = {m for _, members in groups for m in members}
        analog = [k for k in names if kinds[k] in ("analog", "const") and k not in in_lanes]
        stem = os.path.splitext(os.path.basename(bits_path))[0]
        step_names = [k for k in names if kinds[k] in ("logic", "level", "bus", "const")]
        logic_names = [k for k in names if kinds[k] == "logic"]
        v1, v2, npts = write_viva_files(os.path.join(a.outdir, stem), t_us * 1e-6, cols, names,
                                        step_names, logic_names, a.logic_high, kinds)
        made += [v1, v2]
        if have_mpl:
            frames = cols.get("frame")
            marks = None
            if frames is not None:
                starts = np.r_[0, np.where(frames[1:] != frames[:-1])[0] + 1]
                marks = [(t_us[i], f"conversion {int(frames[i])}") for i in starts]
            p = os.path.join(a.outdir, "bit_waveforms_all.png")
            plot_bit_waveforms(p, t_us, cols, kinds, analog, groups, t_us[0],
                               t_us[-1] + (t_us[1] - t_us[0]),
                               f"{os.path.basename(bits_path)}: all signals, "
                               f"{len(marks) if marks else 1} conversions", frame_marks=marks)
            made.append(p)
            if frames is not None and a.frame in set(frames.astype(int).tolist()):
                m = frames.astype(int) == a.frame
                t_lo = t_us[m][0]
                t_hi = t_us[m][-1] + (t_us[1] - t_us[0])
                p = os.path.join(a.outdir, f"bit_waveforms_conversion{a.frame}.png")
                plot_bit_waveforms(p, t_us, cols, kinds, analog, groups, t_lo, t_hi,
                                   f"{os.path.basename(bits_path)}: conversion {a.frame}, "
                                   f"bit by bit", cycle_marks=cycle_boundaries(t_us, cols, m))
                made.append(p)

    # ------------------------------------------------------------------ digital output
    if dig_path:
        header, cols = read_csv(dig_path)
        names = [h for h in header if h in cols and h not in ("n", "t_sample_s", "metastable_decisions")]
        kinds = {k: kind_of(k, cols[k]) for k in names}
        for k in names:
            if k == "code":
                kinds[k] = "bus"
        stem = os.path.splitext(os.path.basename(dig_path))[0]
        # every per-conversion value holds until the next conversion
        v1, v2, npts = write_viva_files(os.path.join(a.outdir, stem), cols["t_sample_s"], cols, names,
                                        names, [k for k in names if kinds[k] == "logic"], a.logic_high,
                                        kinds)
        made += [v1, v2]
        if have_mpl:
            p = os.path.join(a.outdir, "digital_output.png")
            n_all = cols["t_sample_s"].size
            plot_digital_output(p, cols, kinds, a.first,
                                f"{os.path.basename(dig_path)}: first {min(a.first, n_all)} of "
                                f"{n_all} conversions")
            made.append(p)

    print("written:")
    for p in made:
        print("  ", os.path.abspath(p), f"({os.path.getsize(p) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
