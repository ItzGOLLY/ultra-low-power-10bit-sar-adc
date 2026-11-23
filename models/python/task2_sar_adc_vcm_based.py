#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 2 - Analog waveform file  ->  digital signal with a Vcm-BASED SAR ADC
==========================================================================

Reads an analog waveform (Cadence/Virtuoso CSV export, OCEAN ocnPrint text, or any time/voltage
CSV), converts it sample by sample with a 10-bit SAR ADC that uses Vcm-based switching
(Zhu et al., JSSC 2010, "A 10-bit 100-MS/s reference-free SAR ADC"), and reports the digital
output, bit waveforms, power consumption and the important performance figures - all as CSV.

The front end is the same as the Task 1 design: bootstrapped switch (M1-M6, Cb), StrongARM latch +
NAND SR latch, NOR non-overlapping clock generator and synchronous SAR logic. Only the CDAC and
its switching change:

  Vcm-based CDAC: 256C, 128C, ..., 1C + 1C dummy  (512 Cu - half of the conventional 1024 Cu)
    * PHI1 (track): top plate Vx follows Vin, every bottom plate sits at Vcm = Vref/2.
    * Cycle 1: compare Vx = Vin with Vcm -> b9 with NO switching.
    * Cycle k: depending on the bit just decided, ONE capacitor moves Vcm -> GND (bit = 1) or
      Vcm -> Vref (bit = 0); it is never switched back. 9 capacitors give the 10 decisions.
    * Average CDAC energy 85.1 C*Vref^2 single-ended (= 170.2 differential, Zhu 2010):
      87.5 % below conventional. The script re-checks this every run.

The SAME input is also run through the conventional procedure (Task 1 scheme) on an ideal array
so the energy saving for the converted signal is reported directly.

Outputs (CSV, in --outdir):
    task2_digital_output.csv   one row per conversion: time, Vin, code, b9..b0, reconstruction,
                               error, energy (Vcm-based and conventional for comparison)
    task2_bit_waveforms.csv    fs, PHI1, PHI1b, PHI2, CLK, Vin, Vx, outp, outn, Dcmp, Dcmpb,
                               SAR register, tri-level bottom-plate drives, EOC, D9..D0, code
    task2_summary.csv          power breakdown, SNDR/SNR/THD/SFDR/ENOB, INL/DNL, FoM, timing,
                               Vcm-based vs conventional comparison
    task2_inl_dnl.csv          ramp code-density test
    task2_spectrum.csv         output spectrum (dBFS)
    task2_energy_vs_code.csv   ideal energy per code: Vcm-based vs conventional
    analog_input_demo.csv      (only when no --input is given) the demo analog file that was converted

Usage:
    python task2_sar_adc_vcm_based.py                          # makes + converts a demo analog file
    python task2_sar_adc_vcm_based.py --input vin.csv          # your file (time, voltage columns)
    python task2_sar_adc_vcm_based.py --input tran.csv --tcol 0 --vcol 3      # pick columns
    python task2_sar_adc_vcm_based.py --input sig.csv --autoscale             # fit signal to range
    python task2_sar_adc_vcm_based.py --input vin.csv --fs 1e3 --cu 2e-15     # change specs
    python task2_sar_adc_vcm_based.py --params my.json         # override any DEFAULT_PARAMS key

The input range is 0 .. Vref (centred on Vcm). A file centred on 0 V can be shifted with
--vin-offset 0.9 or mapped automatically with --autoscale. A single-column file is treated as
samples already taken at --fs. Requires Python 3.8+ and numpy.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import re
import sys

import numpy as np

KB = 1.380649e-23

# =============================================================================================
# 1. MODEL PARAMETERS  (180 nm-class typical values - edit, or override with --params file.json)
# =============================================================================================
DEFAULT_PARAMS = {
    # ---------------- converter ----------------
    "N_BITS": 10,
    "VDD": 1.8,              # supply [V]
    "VREF": 1.8,             # CDAC reference [V]; input range 0 .. VREF
    "VCM_ERROR": 0.0,        # error of the Vcm reference vs VREF/2 [V] (Vcm-based is sensitive to it)
    "FS": 100e3,             # sampling rate [S/s]
    "TRACK_CYCLES": 1,
    "EOC_CYCLES": 1,
    "T_LOGIC": 0.3e-9,
    "TEMP_K": 300.0,
    "SEED": 7,
    # ---------------- CDAC ----------------
    "CU": 10e-15,            # unit capacitor [F]
    "SIGMA_CU": 0.005,       # relative sigma of one unit cap (cap of m units: sigma/sqrt(m))
    "CP_TOP_FIXED": 20e-15,  # top-plate parasitic: fixed part (switch junction, routing) [F]
    "CP_TOP_FRAC": 0.012,    # top-plate parasitic: fraction of the array capacitance
    "TAU_DAC": 2e-9,         # bottom-plate settling time constant [s]
    "C_DRV_MIN": 2e-15,      # control-input capacitance of the smallest bottom-plate switch [F]
    "C_DRV_PER_UNIT": 4e-18, # extra control capacitance per unit capacitor [F]
    "CTRL_TOGGLES_PER_MOVE": 2.0,  # tri-level switch: two controls change per plate move
    # ---------------- MOS model ----------------
    "KPN": 300e-6, "VTN0": 0.45, "GAMMA": 0.45, "PHI2F": 0.8, "COX": 8.6e-3, "COV": 0.35e-9,
    # ---------------- bootstrapped switch ----------------
    "M1_W": 10e-6, "M1_L": 0.18e-6, "CB": 1e-12, "CG_PAR": 5e-15,
    "M2_GATE_TO_G": False,   # False = M2 gated by PHI (as drawn); True = gate tied to G
    "INJ_FRACTION": 0.5, "JITTER_RMS": 10e-12, "TRACK_STEPS": 120,
    # ---------------- StrongARM comparator + SR latch ----------------
    "MN12_W": 4e-6, "MN12_L": 0.36e-6, "AVT": 5e-9, "VOS": None, "CMP_NOISE_RMS": 0.3e-3,
    "CMP_T0": 150e-12, "CMP_TAU": 60e-12, "CMP_G0": 4.0,
    "C_OUT": 12e-15, "C_XY": 6e-15, "C_CLK_CMP": 8e-15, "K_DIP": 0.3, "C_CMP_IN": 10e-15,
    "C_SR_NODE": 4e-15, "T_INV": 40e-12, "T_NAND": 60e-12, "T_NOR": 80e-12,
    # ---------------- SAR logic, clocks, references ----------------
    "C_FF_CLK": 3e-15, "C_FF_Q": 5e-15, "C_CLKGEN_NODE": 5e-15, "C_PHI_LOAD": 25e-15,
    "P_VCM_GEN_STATIC": 0.0, # static power of the Vcm generator/buffer [W] (0 = ideal source)
    "P_LEAK": 2e-9,
}

IDEAL_OVERRIDES = {"SIGMA_CU": 0.0, "CP_TOP_FIXED": 0.0, "CP_TOP_FRAC": 0.0, "C_CMP_IN": 0.0, "VOS": 0.0, "CMP_NOISE_RMS": 0.0,
                   "JITTER_RMS": 0.0, "INJ_FRACTION": 0.0, "COV": 0.0, "VCM_ERROR": 0.0}


# =============================================================================================
# 2. CIRCUIT BLOCKS (identical to Task 1)
# =============================================================================================
class NonOverlapClockGen:
    """Event-driven model of the NOR non-overlapping clock generator (fs -> PHI1, PHI2, PHI1b)."""

    GATES = [
        ("fsb", "INV", ("fs",), "T_INV"),
        ("phi1", "NOR", ("fs", "phi2_d"), "T_NOR"),
        ("phi2", "NOR", ("fsb", "phi1_d"), "T_NOR"),
        ("phi1_m", "INV", ("phi1",), "T_INV"),
        ("phi1_d", "INV", ("phi1_m",), "T_INV"),
        ("phi2_m", "INV", ("phi2",), "T_INV"),
        ("phi2_d", "INV", ("phi2_m",), "T_INV"),
        ("phi1b", "INV", ("phi1",), "T_INV"),
    ]

    def __init__(self, p, t_track, t_frame):
        self.p, self.t_track, self.t_frame = p, t_track, t_frame
        self._simulate()

    @staticmethod
    def _eval(kind, vals):
        return 1 - vals[0] if kind == "INV" else (0 if any(vals) else 1)

    def _simulate(self):
        p = self.p
        v = {"fs": 0, "fsb": 1, "phi1": 1, "phi1_m": 0, "phi1_d": 1,
             "phi2": 0, "phi2_m": 1, "phi2_d": 0, "phi1b": 0}
        fanout = {}
        for g in self.GATES:
            for net in g[2]:
                fanout.setdefault(net, []).append(g)
        queue, seq = [], 0
        for t, val in ((self.t_track, 1), (self.t_frame, 0)):
            heapq.heappush(queue, (t, seq, "fs", val))
            seq += 1
        self.edges = {k: [] for k in v}
        self.toggles = {k: 0 for k in v}
        while queue:
            t, _, net, val = heapq.heappop(queue)
            if v[net] == val:
                continue
            v[net] = val
            self.toggles[net] += 1
            self.edges[net].append((t, val))
            for out, kind, ins, dkey in fanout.get(net, []):
                heapq.heappush(queue, (t + p[dkey], seq, out, self._eval(kind, [v[i] for i in ins])))
                seq += 1

        def edge(net, val, after):
            return min(t for t, x in self.edges[net] if x == val and t >= after)

        tf = self.t_frame
        self.t_phi1_fall = edge("phi1", 0, self.t_track)
        self.t_phi2_rise = edge("phi2", 1, self.t_track)
        self.t_phi1b_rise = edge("phi1b", 1, self.t_track)
        self.t_phi2_fall_rel = edge("phi2", 0, tf) - tf
        self.t_phi1_rise_rel = edge("phi1", 1, tf) - tf
        self.t_phi1b_fall_rel = edge("phi1b", 0, tf) - tf
        self.nonoverlap_12 = self.t_phi2_rise - self.t_phi1_fall
        self.nonoverlap_21 = self.t_phi1_rise_rel - self.t_phi2_fall_rel
        self.t_sample = self.t_phi1b_rise
        vdd = p["VDD"]
        cap = {k: p["C_CLKGEN_NODE"] for k in v}
        for k in ("phi1", "phi1b", "phi2"):
            cap[k] += p["C_PHI_LOAD"]
        self.energy_per_frame = sum(0.5 * cap[k] * vdd ** 2 * self.toggles[k] for k in v)


class BootstrappedSwitch:
    """Bootstrapped NMOS switch (M1-M6, Cb): V_G = alpha*(VDD + V_B); M2 gated by PHI limits V_B."""

    def __init__(self, p, c_sample):
        self.p = p
        w, l = p["M1_W"], p["M1_L"]
        self.beta = p["KPN"] * w / l
        self.c_gate = w * l * p["COX"] + 2 * w * p["COV"]
        self.c_g = self.c_gate + p["CG_PAR"]
        self.alpha = p["CB"] / (p["CB"] + self.c_g)
        self.c_ov = w * p["COV"]
        self.c_s = c_sample
        self.vb_max = self._solve_vb_max()

    def vth(self, vsb):
        p = self.p
        vsb = np.maximum(vsb, 0.0)
        return p["VTN0"] + p["GAMMA"] * (np.sqrt(p["PHI2F"] + vsb) - math.sqrt(p["PHI2F"]))

    def _solve_vb_max(self):
        if self.p["M2_GATE_TO_G"]:
            return float("inf")
        vb = self.p["VDD"] - self.p["VTN0"]
        for _ in range(60):
            vb = self.p["VDD"] - float(self.vth(vb))
        return vb

    def node_b(self, vin):
        return np.clip(vin, 0.0, self.vb_max)

    def gate_voltage(self, vin):
        return self.alpha * (self.p["VDD"] + self.node_b(vin))

    def vov(self, vin):
        return self.gate_voltage(vin) - vin - self.vth(vin)

    def ron(self, vin):
        return 1.0 / (self.beta * np.maximum(self.vov(vin), 0.02))

    def track(self, vin_func, t_start, t_end, vx0, nsteps):
        h = (t_end - t_start) / nsteps
        t = t_start.copy()
        vin_prev = vin_func(t)
        vx = vx0.copy()
        tau = self.ron(vin_prev) * self.c_s
        for _ in range(nsteps):
            t = t + h
            vin_next = vin_func(t)
            slope = (vin_next - vin_prev) / h
            tau = self.ron(0.5 * (vin_prev + vin_next)) * self.c_s
            decay = np.exp(-h / tau)
            vx = vin_next - slope * tau + (vx - vin_prev + slope * tau) * decay
            vin_prev = vin_next
        return vx, tau

    def hold(self, vx):
        p = self.p
        q_ch = p["M1_W"] * p["M1_L"] * p["COX"] * np.maximum(self.vov(vx), 0.0)
        dv_inj = -p["INJ_FRACTION"] * q_ch / self.c_s
        dv_ft = -self.c_ov * (vx + self.vth(vx)) / (self.c_s + self.c_ov)
        return vx + dv_inj + dv_ft

    def energy(self, vin):
        vdd = self.p["VDD"]
        return vdd * self.alpha * self.c_g * (vdd + self.node_b(vin))


class StrongArmComparator:
    """StrongARM latch + NAND SR latch: Dcmp = 1 when Vinp > Vinn (+offset, noise)."""

    def __init__(self, p, rng):
        self.p = p
        self.sigma_os = p["AVT"] / math.sqrt(p["MN12_W"] * p["MN12_L"])
        self.vos = p["VOS"] if p["VOS"] is not None else float(rng.normal(0.0, self.sigma_os))
        vdd = p["VDD"]
        self.e_decision = (vdd * (p["C_OUT"] * (1 + p["K_DIP"]) * vdd + 2 * p["C_XY"] * (vdd - p["VTN0"]))
                           + p["C_CLK_CMP"] * vdd ** 2)

    def decide(self, vinp, vinn, noise):
        p = self.p
        dv = vinp - vinn - self.vos + noise
        amp = np.maximum(np.abs(dv) * p["CMP_G0"], 1e-30)
        tdec = p["CMP_T0"] + p["CMP_TAU"] * np.log(np.maximum((p["VDD"] / 2) / amp, 1.0))
        return dv > 0, tdec, dv


# =============================================================================================
# 3. CAPACITIVE DACs  (shared charge-conservation engine)
# =============================================================================================
class _CDACBase:
    def initial_state(self, ns):
        return np.tile(self.Vb0, (ns, 1))

    def switch(self, vb_old, vb_new, vx_old):
        """Floating top plate: new Vx and the charge pushed into every bottom plate."""
        q = (self.C * (vx_old[:, None] - vb_old)).sum(1) + self.Cp * vx_old
        vx_new = (q + (self.C * vb_new).sum(1)) / (self.C.sum() + self.Cp)
        dq = self.C * ((vb_new - vx_new[:, None]) - (vb_old - vx_old[:, None]))
        return vx_new, dq

    def driven_charge(self, vb_old, vb_new, vx_old, vx_new):
        return self.C * ((vb_new - vx_new[:, None]) - (vb_old - vx_old[:, None]))

    def charge_from(self, dq, vb_new, level):
        """Charge delivered by the source at `level` (sum over plates connected to it)."""
        return (dq * np.isclose(vb_new, level)).sum(1)


class VcmCDAC(_CDACBase):
    """Vcm-based array: 256C ... 1C + 1C dummy. Index 0 = 256C (moves after b9), ...,
    index N-2 = 1C (moves after b1), index N-1 = dummy (stays at Vcm)."""

    name = "Vcm-based"

    def __init__(self, p, rng, ideal_caps=False):
        n = p["N_BITS"]
        self.n = n
        self.vref = p["VREF"]
        self.vcm = p["VREF"] / 2 + (0.0 if ideal_caps else p["VCM_ERROR"])
        self.units = np.array([2.0 ** (n - 2 - j) for j in range(n - 1)] + [1.0])
        sig = 0.0 if ideal_caps else p["SIGMA_CU"]
        eps = rng.normal(0.0, 1.0, self.units.size) * sig / np.sqrt(self.units) if sig > 0 else 0.0
        self.C = p["CU"] * self.units * (1.0 + eps)
        self.Cp = 0.0 if ideal_caps else p["CP_TOP_FIXED"] + p["CP_TOP_FRAC"] * self.C.sum() + p["C_CMP_IN"]
        self.Vb0 = np.full(self.units.size, self.vcm)
        self.c_drv = (p["C_DRV_MIN"] + p["C_DRV_PER_UNIT"] * self.units) * p["CTRL_TOGGLES_PER_MOVE"]
        self.c_drv[-1] = 0.0
        self.labels = [f"{int(self.units[j])}C_b{n - 1 - j}" for j in range(n - 1)] + ["dummy"]

    def next_state(self, vb, j, d):
        """After deciding bit j (j = 0 -> b9): move capacitor j once, Vcm -> GND (1) or Vref (0)."""
        vbn = vb.copy()
        vbn[:, j] = np.where(d, 0.0, self.vref)
        return vbn


class ConventionalCDAC(_CDACBase):
    """Task-1 conventional array (512C ... 1C + 1C dummy), used here for the energy comparison."""

    name = "conventional"

    def __init__(self, p, rng, ideal_caps=False):
        n = p["N_BITS"]
        self.n = n
        self.vref = p["VREF"]
        self.vcm = p["VREF"] / 2
        self.units = np.array([2.0 ** (n - 1 - j) for j in range(n)] + [1.0])
        sig = 0.0 if ideal_caps else p["SIGMA_CU"]
        eps = rng.normal(0.0, 1.0, self.units.size) * sig / np.sqrt(self.units) if sig > 0 else 0.0
        self.C = p["CU"] * self.units * (1.0 + eps)
        self.Cp = 0.0 if ideal_caps else p["CP_TOP_FIXED"] + p["CP_TOP_FRAC"] * self.C.sum() + p["C_CMP_IN"]
        self.Vb0 = np.zeros(self.units.size)
        self.Vb0[0] = self.vref
        self.c_drv = p["C_DRV_MIN"] + p["C_DRV_PER_UNIT"] * self.units
        self.c_drv[-1] = 0.0
        self.labels = [f"{int(self.units[j])}C_b{n - 1 - j}" for j in range(n)] + ["dummy"]

    def next_state(self, vb, j, d):
        vbn = vb.copy()
        vbn[:, j] = np.where(d, 0.0, self.vref)
        vbn[:, j + 1] = self.vref
        return vbn


def ideal_run(cdac, vin, steady_state_reset=True):
    """Ideal comparator + ideal array (build the CDAC with CU=1, VREF=1 to get C*Vref^2 units).
    Returns per-conversion energies: Vref (bit cycling + reset), Vcm net, Vcm sourced."""
    n = cdac.n
    ns = vin.size
    vb = cdac.initial_state(ns)
    vx = vin.copy()
    e_ref = np.zeros(ns)
    q_cm_net = np.zeros(ns)
    q_cm_src = np.zeros(ns)
    bits = np.zeros((ns, n), dtype=np.int8)
    for j in range(n):
        d = vx > cdac.vcm
        bits[:, j] = d
        if j < n - 1:
            vbn = cdac.next_state(vb, j, d)
            vxn, dq = cdac.switch(vb, vbn, vx)
            e_ref += cdac.vref * cdac.charge_from(dq, vbn, cdac.vref)
            qc = cdac.charge_from(dq, vbn, cdac.vcm)
            q_cm_net += qc
            q_cm_src += np.maximum(qc, 0.0)
            vb, vx = vbn, vxn
    vb_prev, vx_prev = (vb, vx) if steady_state_reset else (np.roll(vb, 1, 0), np.roll(vx, 1))
    vb0 = cdac.initial_state(ns)
    dq = cdac.driven_charge(vb_prev, vb0, vx_prev, vin)
    e_reset_ref = cdac.vref * cdac.charge_from(dq, vb0, cdac.vref)
    qc = cdac.charge_from(dq, vb0, cdac.vcm)
    code = bits.astype(np.int64) @ (2 ** np.arange(n - 1, -1, -1))
    return dict(code=code, e_conv_ref=e_ref, e_reset_ref=e_reset_ref, e_ref=e_ref + e_reset_ref,
                q_cm_net=q_cm_net + qc, q_cm_src=q_cm_src + np.maximum(qc, 0.0))


# =============================================================================================
# 4. THE CONVERTER (Vcm-based)
# =============================================================================================
class SARADC:
    def __init__(self, p):
        self.p = p
        self.n = p["N_BITS"]
        rng = np.random.default_rng(p["SEED"])
        self.cdac = VcmCDAC(p, rng)
        self.cmp = StrongArmComparator(p, rng)
        self.c_s = self.cdac.C.sum() + self.cdac.Cp
        self.sw = BootstrappedSwitch(p, self.c_s)
        self.vref = p["VREF"]
        self.vcm = self.cdac.vcm                         # comparator Vinn = the same Vcm node
        self.n_cycles = p["TRACK_CYCLES"] + self.n + p["EOC_CYCLES"]
        self.t_frame = 1.0 / p["FS"]
        self.t_b = self.t_frame / self.n_cycles
        self.t_track = p["TRACK_CYCLES"] * self.t_b
        self.clk = NonOverlapClockGen(p, self.t_track, self.t_frame)
        self.t_cmp_avail = self.t_b / 2
        self.t_settle = self.t_b / 2 - p["T_LOGIC"]
        self.ktc_sigma = math.sqrt(KB * p["TEMP_K"] / self.c_s)
        self.lsb = self.vref / 2 ** self.n
        self.n_ff = self.n_cycles + 2 * self.n

    def convert(self, vheld, noise, coin, keep=0):
        n, cd, vdd = self.n, self.cdac, self.p["VDD"]
        ns = vheld.size
        vb = cd.initial_state(ns)
        vx = vheld.copy()
        bits = np.zeros((ns, n), dtype=np.int8)
        vx_set = np.zeros((ns, n))
        tdec = np.zeros((ns, n))
        e_ref = np.zeros(ns)
        e_cm_net = np.zeros(ns)
        e_cm_buf = np.zeros(ns)
        drv_tog = np.zeros((ns, vb.shape[1]))
        hist = np.zeros((keep, n, vb.shape[1])) if keep else None
        serr = np.zeros(ns)
        meta = np.zeros(ns, dtype=int)
        settle_factor = math.exp(-self.t_settle / self.p["TAU_DAC"])
        for j in range(n):
            if keep:
                hist[:, j, :] = vb[:keep]
            vx_set[:, j] = vx
            d, td, _ = self.cmp.decide(vx - serr, self.vcm, noise[:, j])
            m = td > self.t_cmp_avail
            if m.any():
                d = np.where(m, coin[:, j], d)
                meta += m
            bits[:, j], tdec[:, j] = d, td
            if j < n - 1:
                vbn = cd.next_state(vb, j, d)
                vxn, dq = cd.switch(vb, vbn, vx)
                e_ref += self.vref * cd.charge_from(dq, vbn, self.vref)
                qc = cd.charge_from(dq, vbn, cd.vcm)
                e_cm_net += cd.vcm * qc
                e_cm_buf += vdd * np.maximum(qc, 0.0)    # a buffer sources from VDD, sinks to GND
                drv_tog += (vbn != vb)
                serr = (vxn - vx) * settle_factor
                vb, vx = vbn, vxn
        code = bits.astype(np.int64) @ (2 ** np.arange(n - 1, -1, -1))
        return dict(bits=bits, code=code, vx_set=vx_set, tdec=tdec, e_ref=e_ref, e_cm_net=e_cm_net,
                    e_cm_buf=e_cm_buf, drv_tog=drv_tog, vb_final=vb, vx_final=vx, hist=hist, meta=meta)

    def simulate(self, vin_func, n_conv, t0=0.0, seed_offset=1, keep=0):
        p, n = self.p, self.n
        rng = np.random.default_rng(p["SEED"] + seed_offset)
        idx = np.arange(n_conv)
        t_frame = t0 + idx * self.t_frame
        t_rise = t_frame + self.clk.t_phi1_rise_rel
        t_nom = t_frame + self.clk.t_sample
        t_s = t_nom + rng.normal(0.0, p["JITTER_RMS"], n_conv) if p["JITTER_RMS"] > 0 else t_nom
        noise = rng.normal(0.0, p["CMP_NOISE_RMS"], (n_conv, n)) if p["CMP_NOISE_RMS"] > 0 else np.zeros((n_conv, n))
        coin = rng.random((n_conv, n)) < 0.5
        ktc = rng.normal(0.0, self.ktc_sigma, n_conv)
        vin_s = vin_func(t_nom)
        vb0 = self.cdac.initial_state(n_conv)
        vx0 = np.full(n_conv, self.vcm)
        for _ in range(2):
            vtr, tau_tr = self.sw.track(vin_func, t_rise, t_s, vx0, p["TRACK_STEPS"])
            vheld = self.sw.hold(vtr) + ktc
            r = self.convert(vheld, noise, coin, keep)
            vx_after_reset, _ = self.cdac.switch(r["vb_final"], vb0, r["vx_final"])
            vx0 = np.roll(vx_after_reset, 1)
        r.update(t_frame=t_frame, t_s=t_s, vin_s=vin_s, vtrack=vtr, vheld=vheld, tau_track=tau_tr,
                 vx_after_reset=vx0)

        vb_prev = np.roll(r["vb_final"], 1, axis=0)
        vx_prev = np.roll(r["vx_final"], 1)
        dq = self.cdac.driven_charge(vb_prev, vb0, vx_prev, vheld)
        r["e_reset_ref"] = self.vref * self.cdac.charge_from(dq, vb0, self.vref)
        qc = self.cdac.charge_from(dq, vb0, self.cdac.vcm)
        r["e_cm_net"] = r["e_cm_net"] + self.cdac.vcm * qc
        r["e_cm_buf"] = r["e_cm_buf"] + p["VDD"] * np.maximum(qc, 0.0)

        vdd = p["VDD"]
        bits = r["bits"]
        prev_bits = np.roll(bits, 1, axis=0)
        q_tog = (bits[:, 0] != prev_bits[:, -1]).astype(int) + (bits[:, 1:] != bits[:, :-1]).sum(1)
        trial0 = np.zeros(n, dtype=np.int8)
        trial0[0] = 1
        tog_reg = (prev_bits != trial0).sum(1) + (1 - bits).sum(1) + (n - 1)
        tog_out = (bits != prev_bits).sum(1)
        tog_ring = 2 * self.n_cycles
        drv_tog = r["drv_tog"] + (vb_prev != vb0)
        r["energy"] = {
            "CDAC switching (Vref) - bit cycling": r["e_ref"],
            "CDAC switching (Vref) - reset/sampling": r["e_reset_ref"],
            "StrongARM comparator": np.full(n_conv, n * self.cmp.e_decision),
            "SR latch + output buffers": q_tog * 2 * p["C_SR_NODE"] * vdd ** 2,
            "SAR logic (sequencer + registers)": (self.n_cycles * self.n_ff * p["C_FF_CLK"] * vdd ** 2
                                                  + 0.5 * p["C_FF_Q"] * vdd ** 2 * (tog_reg + tog_out + tog_ring)),
            "CDAC tri-level switch drivers (gate drive)": 0.5 * vdd ** 2 * (drv_tog @ self.cdac.c_drv),
            "Bootstrapped switch": self.sw.energy(vtr),
            "Non-overlap clock generator": np.full(n_conv, self.clk.energy_per_frame),
        }
        r["e_total"] = sum(r["energy"].values())
        return r


# =============================================================================================
# 5. ANALYSIS HELPERS
# =============================================================================================
def bh4_window(n):
    x = 2 * np.pi * np.arange(n) / n
    return 0.35875 - 0.48829 * np.cos(x) + 0.14128 * np.cos(2 * x) - 0.01168 * np.cos(3 * x)


def dynamic_metrics(codes, fs, nbits, n_harm=9):
    x = codes.astype(float)
    n = x.size
    x = x - x.mean()
    pr = np.abs(np.fft.rfft(x)) ** 2
    if n < 16 or pr[1:].sum() <= 0:
        return None
    k_r = 1 + int(np.argmax(pr[1:]))
    coherent = pr[k_r] / pr[1:].sum() > 0.995
    w = np.ones(n) if coherent else bh4_window(n)
    span = 0 if coherent else 4
    pw = np.abs(np.fft.rfft(x * w)) ** 2
    nb = pw.size
    k0 = span + 1 + int(np.argmax(pw[span + 1:]))

    def lobe(k):
        return set(range(max(k - span, 0), min(k + span, nb - 1) + 1))

    sig, dc = lobe(k0), set(range(0, span + 1))
    harm_bins, harm = set(), []
    for h in range(2, n_harm + 1):
        kh = (h * k0) % n
        kh = n - kh if kh > n // 2 else kh
        b = lobe(kh) - sig - dc
        harm.append((h, kh, float(pw[list(b)].sum()) if b else 0.0))
        harm_bins |= b
    p_sig = pw[list(sig)].sum()
    p_harm = pw[list(harm_bins)].sum() if harm_bins else 0.0
    noise_mask = np.ones(nb, bool)
    noise_mask[list(sig | dc | harm_bins)] = False
    p_noise = pw[noise_mask].sum()
    spur_mask = np.ones(nb, bool)
    spur_mask[list(sig | dc)] = False
    a_fs = 2 ** (nbits - 1)
    amp = math.sqrt(4 * p_sig / (n * np.sum(w ** 2)))
    sig_dbfs = 20 * math.log10(amp / a_fs)
    sndr = 10 * math.log10(p_sig / (p_noise + p_harm))
    p_ref = (a_fs * np.sum(w) / 2) ** 2
    return dict(
        window="rectangular (coherent)" if coherent else "Blackman-Harris 4-term",
        fin=k0 * fs / n, n=n, tone_fraction=p_sig / pw[span + 1:].sum(), signal_dbfs=sig_dbfs, sndr=sndr,
        snr=10 * math.log10(p_sig / p_noise),
        thd=10 * math.log10(p_harm / p_sig) if p_harm > 0 else -np.inf,
        sfdr=10 * math.log10(pw[k0] / pw[spur_mask].max()),
        enob=(sndr - 1.76) / 6.02, enob_fs=(sndr - sig_dbfs - 1.76) / 6.02,
        noise_floor_dbfs=10 * math.log10(p_noise / max(noise_mask.sum(), 1) / p_ref),
        harmonics=[(h, kh * fs / n, 10 * math.log10(ph / p_sig) if ph > 0 else -np.inf) for h, kh, ph in harm],
        freq=np.arange(nb) * fs / n, spec_dbfs=10 * np.log10(pw / p_ref + 1e-30))


def static_test(adc, hits_per_code):
    m = 2 ** adc.n
    vref = adc.vref
    lo, hi = -0.03 * vref, 1.03 * vref
    n_conv = int(math.ceil(hits_per_code * m * (hi - lo) / vref))
    t_total = n_conv * adc.t_frame
    r = adc.simulate(lambda t: lo + (hi - lo) * t / t_total, n_conv, seed_offset=1000)
    codes, vs = r["code"], r["vin_s"]
    hits = np.bincount(codes, minlength=m)
    trans = np.interp(np.cumsum(hits)[:-1] - 0.5, np.arange(n_conv), vs)
    lsb_ideal = vref / m
    lsb_avg = (trans[-1] - trans[0]) / (m - 2)
    dnl = np.diff(trans) / lsb_avg - 1.0
    inl = (trans - trans[0]) / lsb_avg - np.arange(m - 1)
    gain_lsb = ((trans[-1] - trans[0]) - (m - 2) * lsb_ideal) / lsb_ideal
    return dict(hits=hits, trans=trans, dnl=dnl, inl=inl, missing=int(np.sum(hits[1:-1] == 0)),
                offset_lsb=(trans[0] - lsb_ideal) / lsb_ideal,
                offset_mid_lsb=(trans[m // 2 - 1] - (m // 2) * lsb_ideal) / lsb_ideal,
                gain_lsb=gain_lsb, gain_pct=100 * gain_lsb / (m - 2), n_conv=n_conv)


def closed_form_diff(n):
    """Literature closed forms for DIFFERENTIAL arrays [C*Vref^2]."""
    conv = sum(2.0 ** (n + 1 - 2 * i) * (2 ** i - 1) for i in range(1, n + 1))
    vcm = sum(2.0 ** (n - 2 - 2 * i) * (2 ** i - 1) for i in range(1, n))
    return conv, vcm


# =============================================================================================
# 6. INPUT HANDLING
# =============================================================================================
_SI = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3,
       "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9}
_NUM = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([fpnuµmkKMG]?)\s*(?:s|S|V|v|A|Hz)?\s*$")


def _parse_num(tok):
    m = _NUM.match(tok.strip().strip('"').strip("'"))
    if not m:
        return None
    val = float(m.group(1))
    return val * _SI[m.group(2)] if m.group(2) else val


def load_analog_file(path, tcol=0, vcol=1):
    """Time/voltage waveform from a Virtuoso CSV export, ocnPrint text, or any CSV/TSV/space file."""
    rows, header = [], None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("#", ";", "//")):
                continue
            if "," in line:
                toks = line.split(",")
            elif "\t" in line:
                toks = line.split("\t")
            elif ";" in line:
                toks = line.split(";")
            else:
                toks = line.split()
            vals = [_parse_num(t) for t in toks]
            if all(v is None for v in vals):
                header = [t.strip().strip('"') for t in toks]
                continue
            rows.append(vals)
    if not rows:
        raise ValueError(f"No numeric data found in {path}")
    ncol = min(len(r) for r in rows)
    if ncol == 1 or vcol >= ncol:
        return None, np.array([r[0] for r in rows if r[0] is not None], float), header
    data = [(r[tcol], r[vcol]) for r in rows if r[tcol] is not None and r[vcol] is not None]
    arr = np.array(data, float)
    order = np.argsort(arr[:, 0], kind="stable")
    t, v = arr[order, 0], arr[order, 1]
    keep = np.concatenate(([True], np.diff(t) > 0))
    return t[keep], v[keep], header


def largest_prime_le(x):
    x = int(x)
    while x > 2:
        if x % 2 and all(x % d for d in range(3, int(x ** 0.5) + 1, 2)):
            return x
        x -= 1
    return 1


def write_demo_analog_file(path, fs, n_conv, vcm, amp, fin, pts_per_frame=10):
    """A Virtuoso-style CSV ("/Vin X","/Vin Y") holding a sine, as if exported from a transient."""
    t = np.arange(int((n_conv + 1) * pts_per_frame) + 1) / (fs * pts_per_frame)
    v = vcm + amp * np.sin(2 * np.pi * fin * t + 0.3)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["/Vin X", "/Vin Y"])
        for ti, vi in zip(t, v):
            w.writerow([f"{ti:.9e}", f"{vi:.7f}"])


# =============================================================================================
# 7. OUTPUT WRITERS
# =============================================================================================
def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def fmt(x, nd=6):
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ""
    return f"{x:.{nd}g}"


def build_waveforms(adc, r, vin_func, n_frames, pts_per_cycle):
    p, n = adc.p, adc.n
    tb, tf, ttr = adc.t_b, adc.t_frame, adc.t_track
    ck = adc.clk
    t0 = r["t_frame"][0]
    dt = tb / pts_per_cycle
    npts = int(round(n_frames * tf / dt))
    t = t0 + np.arange(npts) * dt
    f = np.minimum(((t - t0) // tf).astype(int), n_frames - 1)
    tau = (t - t0) - f * tf
    fs_sig = (tau >= ttr).astype(int)
    phi1 = ((tau >= ck.t_phi1_rise_rel) & (tau < ck.t_phi1_fall)).astype(int)
    phi1b = 1 - ((tau >= ck.t_phi1b_fall_rel) & (tau < ck.t_phi1b_rise)).astype(int)
    phi2 = ((tau >= ck.t_phi2_rise) | (tau < ck.t_phi2_fall_rel)).astype(int)
    kc = np.floor((tau - ttr) / tb).astype(int)
    in_bit = (kc >= 0) & (kc < n)
    loc = (tau - ttr) - kc * tb
    clk = (in_bit & (loc >= tb / 2)).astype(int)
    kcc = np.clip(kc, 0, n - 1)
    d = r["bits"][:n_frames]
    tdec = r["tdec"][:n_frames]
    resolved = clk.astype(bool) & (loc - tb / 2 >= tdec[f, kcc])
    dd = d[f, kcc]
    outp = np.where(resolved, 1 - dd, 1)
    outn = np.where(resolved, dd, 1)
    t_dec_abs = (r["t_frame"][:n_frames, None] + ttr + np.arange(n)[None, :] * tb + tb / 2
                 + tdec + p["T_NAND"] + 2 * p["T_INV"])
    order = np.argsort(t_dec_abs.ravel())
    tda, dflat = t_dec_abs.ravel()[order], d.ravel()[order]
    k = np.searchsorted(tda, t, side="right") - 1
    q = np.where(k >= 0, dflat[np.maximum(k, 0)], 0)
    ndec = np.where(tau < ttr, 0, np.where(in_bit, kc, n))
    sar = np.zeros((npts, n), dtype=int)
    for j in range(n):
        sar[:, j] = np.where(j < ndec, d[f, j], (j == ndec).astype(int))
    state = np.where(tau < ttr, 0, kcc)
    bp = r["hist"][f, state, :]
    bp = np.where((tau < ttr)[:, None], adc.cdac.Vb0[None, :], bp)
    vin_t = vin_func(t)
    vx_set = r["vx_set"][:n_frames]
    tau_tr = r["tau_track"][f]
    vx_reset = r["vx_after_reset"][:n_frames]
    vin_rise = vin_func(r["t_frame"][:n_frames] + ck.t_phi1_rise_rel)
    vx_track = vin_t + (vx_reset[f] - vin_rise[f]) * np.exp(-np.maximum(tau - ck.t_phi1_rise_rel, 0) / tau_tr)
    prev, cur = vx_set[f, np.maximum(kcc - 1, 0)], vx_set[f, kcc]
    vx_bit = np.where(kcc == 0, cur, cur + (prev - cur) * np.exp(-np.maximum(loc, 0) / p["TAU_DAC"]))
    vx = np.where(tau < ck.t_phi1_rise_rel, vx_reset[f],
         np.where(tau < ck.t_sample, vx_track,
         np.where(tau < ttr, r["vheld"][:n_frames][f],
         np.where(kc >= n, vx_set[f, n - 1], vx_bit))))
    codes = r["code"][:n_frames]
    eoc = (kc >= n).astype(int)
    fo = np.where(eoc == 1, f, f - 1)
    dout = np.where(fo >= 0, codes[np.maximum(fo, 0)], 0)
    dbits = (dout[:, None] >> np.arange(n - 1, -1, -1)[None, :]) & 1
    header = (["time_us", "frame", "fs", "PHI1", "PHI1b", "PHI2", "CLK_cmp", "Vin_V", "Vx_V", "Vcm_V",
               "outp", "outn", "Dcmp", "Dcmpb"]
              + [f"SAR_b{n - 1 - j}" for j in range(n)]
              + [f"BP_{lab}_V" for lab in adc.cdac.labels]
              + ["EOC"] + [f"D{n - 1 - j}" for j in range(n)] + ["Dout_code"])
    rows = []
    for i in range(npts):
        rows.append([f"{(t[i] - t0) * 1e6:.6f}", int(f[i]), int(fs_sig[i]), int(phi1[i]), int(phi1b[i]),
                     int(phi2[i]), int(clk[i]), f"{vin_t[i]:.6f}", f"{vx[i]:.6f}", f"{adc.vcm:.6f}",
                     int(outp[i]), int(outn[i]), int(q[i]), int(1 - q[i])]
                    + [int(x) for x in sar[i]] + [f"{x:.4f}" for x in bp[i]]
                    + [int(eoc[i])] + [int(x) for x in dbits[i]] + [int(dout[i])])
    return header, rows


# =============================================================================================
# 8. MAIN
# =============================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="Task 2: analog file -> digital with a Vcm-based SAR ADC")
    ap.add_argument("--input", help="analog waveform file (CSV/TXT: time, voltage). Omit to use a demo file")
    ap.add_argument("--tcol", type=int, default=0, help="time column index (default 0)")
    ap.add_argument("--vcol", type=int, default=1, help="voltage column index (default 1)")
    ap.add_argument("--vin-gain", type=float, default=1.0, help="multiply file voltages by this")
    ap.add_argument("--vin-offset", type=float, default=0.0, help="then add this offset [V]")
    ap.add_argument("--autoscale", action="store_true", help="map the file's min/max to 5%%..95%% of the range")
    ap.add_argument("--max-conversions", type=int, default=None, help="limit the number of conversions")
    ap.add_argument("--fs", type=float, help="sampling rate [S/s]")
    ap.add_argument("--vdd", type=float, help="supply [V]")
    ap.add_argument("--vref", type=float, help="reference [V]")
    ap.add_argument("--cu", type=float, help="unit capacitor [F]")
    ap.add_argument("--demo-conversions", type=int, default=4096, help="length of the demo file in conversions")
    ap.add_argument("--demo-fin", type=float, default=None, help="demo sine frequency [Hz] (default ~0.11*fs)")
    ap.add_argument("--demo-amp-dbfs", type=float, default=-1.0, help="demo sine amplitude [dBFS]")
    ap.add_argument("--wave-frames", type=int, default=8, help="conversions written to the waveform CSV")
    ap.add_argument("--pts-per-cycle", type=int, default=40, help="waveform points per bit-clock cycle")
    ap.add_argument("--hits-per-code", type=int, default=32, help="ramp-test samples per code")
    ap.add_argument("--no-static", action="store_true", help="skip the ramp INL/DNL test")
    ap.add_argument("--ideal", action="store_true", help="switch off all non-idealities")
    ap.add_argument("--params", help="JSON file overriding DEFAULT_PARAMS keys")
    ap.add_argument("--seed", type=int, help="random seed")
    ap.add_argument("--outdir", default="task2_outputs", help="output folder")
    a = ap.parse_args(argv)

    p = dict(DEFAULT_PARAMS)
    if a.params:
        with open(a.params) as fh:
            user = json.load(fh)
        unknown = [k for k in user if k not in p]
        if unknown:
            sys.exit(f"Unknown parameter(s) in {a.params}: {unknown}")
        p.update(user)
    for key, val in (("FS", a.fs), ("VDD", a.vdd), ("VREF", a.vref), ("CU", a.cu), ("SEED", a.seed)):
        if val is not None:
            p[key] = val
    if a.ideal:
        p.update(IDEAL_OVERRIDES)
    os.makedirs(a.outdir, exist_ok=True)

    adc = SARADC(p)
    n, vref, fs, vdd = adc.n, adc.vref, p["FS"], p["VDD"]
    m = 2 ** n

    # ------------------------------------------------------------------ the analog file
    if a.input:
        path = a.input
        made_demo = False
    else:
        path = os.path.join(a.outdir, "analog_input_demo.csv")
        k_bin = largest_prime_le((a.demo_fin if a.demo_fin else 0.11 * fs) / fs * a.demo_conversions)
        demo_fin = k_bin * fs / a.demo_conversions
        demo_amp = (vref / 2) * 10 ** (a.demo_amp_dbfs / 20)
        write_demo_analog_file(path, fs, a.demo_conversions, vref / 2, demo_amp, demo_fin)
        made_demo = True
    t_file, v_file, file_header = load_analog_file(path, a.tcol, a.vcol)
    if t_file is None:
        t_file = np.arange(v_file.size) / fs
    v_file = v_file * a.vin_gain + a.vin_offset
    if a.autoscale:
        lo_f, hi_f = v_file.min(), v_file.max()
        v_file = adc.vcm + (v_file - 0.5 * (lo_f + hi_f)) * (0.9 * vref / max(hi_f - lo_f, 1e-12))
    n_conv = int(math.floor((t_file[-1] - t_file[0] - adc.clk.t_sample) / adc.t_frame)) + 1
    if a.max_conversions:
        n_conv = min(n_conv, a.max_conversions)
    if made_demo:
        n_conv = min(n_conv, a.demo_conversions)
    if n_conv < 2:
        sys.exit("The analog file is shorter than two conversion periods at this --fs.")
    vin_func = lambda t: np.interp(t, t_file, v_file)
    src = (f"{'demo file (sine ' + format(demo_fin, '.6g') + ' Hz, coherent)' if made_demo else 'file'}: "
           f"{os.path.basename(path)}, {t_file.size} points, {t_file[-1] - t_file[0]:.6g} s, "
           f"{v_file.min():.4g}..{v_file.max():.4g} V")

    # ------------------------------------------------------------------ convert
    keep = min(a.wave_frames, n_conv)
    r = adc.simulate(vin_func, n_conv, t0=t_file[0], keep=keep)
    codes, vin_s = r["code"], r["vin_s"]

    energy_mean = {k: float(np.mean(v)) for k, v in r["energy"].items()}
    e_dyn = sum(energy_mean.values())
    e_cm_net = float(np.mean(r["e_cm_net"]))          # ideal (well-decoupled) Vcm source
    e_cm_wc = float(np.mean(r["e_cm_buf"]))           # buffer sourcing every event from VDD, no recycling
    p_total = (e_dyn + max(e_cm_net, 0.0)) * fs + p["P_LEAK"] + p["P_VCM_GEN_STATIC"]
    p_total_wc = (e_dyn + e_cm_wc) * fs + p["P_LEAK"] + p["P_VCM_GEN_STATIC"]

    dyn = dynamic_metrics(codes, fs, n)
    tone = dyn is not None and dyn["tone_fraction"] > 0.9
    vrec = (codes + 0.5) * adc.lsb
    err = vrec - vin_s
    a_fit = np.vstack([vin_s, np.ones_like(vin_s)]).T
    coef, *_ = np.linalg.lstsq(a_fit, codes.astype(float), rcond=None)
    resid = codes - a_fit @ coef
    enob_fit = math.log2(m / (np.std(resid) * math.sqrt(12))) if np.std(resid) > 0 else float(n)
    out_of_range = int(np.sum((vin_s < 0) | (vin_s > vref)))
    rails = int(np.sum((codes == 0) | (codes == m - 1)))

    # ------------------------------------------------------------------ same signal, both schemes (ideal arrays)
    q1 = dict(p, CU=1.0, VREF=1.0)
    rng0 = np.random.default_rng(0)
    cd_v, cd_c = VcmCDAC(q1, rng0, ideal_caps=True), ConventionalCDAC(q1, rng0, ideal_caps=True)
    u = (np.clip(codes, 0, m - 1) + 0.5) / m                          # this signal's code sequence
    sv, sc = ideal_run(cd_v, u, steady_state_reset=False), ideal_run(cd_c, u, steady_state_reset=False)
    cv2 = p["CU"] * vref ** 2
    e_sig_v, e_sig_c = sv["e_ref"].mean(), sc["e_ref"].mean()

    # ------------------------------------------------------------------ energy vs code (uniform codes)
    allc = (np.arange(m) + 0.5) / m
    ev, ec = ideal_run(cd_v, allc), ideal_run(cd_c, allc)
    assert np.array_equal(ev["code"], np.arange(m)) and np.array_equal(ec["code"], np.arange(m))
    lit_conv, lit_vcm = closed_form_diff(n)

    dyn_alt = None
    if tone:   # design hint: same chip with the other M2 gate connection
        adc_alt = SARADC(dict(p, M2_GATE_TO_G=not p["M2_GATE_TO_G"]))
        dyn_alt = dynamic_metrics(adc_alt.simulate(vin_func, n_conv, t0=t_file[0])["code"], fs, n)

    st = None if a.no_static else static_test(adc, a.hits_per_code)

    # ================================================================== CSV: digital output
    rows = []
    for i in range(n_conv):
        b = r["bits"][i]
        rows.append([i, f"{r['t_s'][i]:.9e}", f"{vin_s[i]:.6f}", f"{r['vheld'][i]:.6f}", int(codes[i]),
                     "".join(str(int(x)) for x in b)] + [int(x) for x in b]
                    + [f"{vrec[i]:.6f}", f"{err[i] / adc.lsb:.4f}",
                       f"{(r['e_ref'][i] + r['e_reset_ref'][i]) * 1e12:.4f}", f"{r['e_cm_buf'][i] * 1e12:.4f}",
                       f"{r['e_total'][i] * 1e12:.4f}", f"{sc['e_ref'][i] * cv2 * 1e12:.4f}", int(r["meta"][i])])
    write_csv(os.path.join(a.outdir, "task2_digital_output.csv"),
              ["n", "t_sample_s", "Vin_ideal_V", "V_held_V", "code", "code_binary"]
              + [f"b{n - 1 - j}" for j in range(n)]
              + ["Vout_reconstructed_V", "error_LSB", "E_CDAC_Vref_pJ", "E_Vcm_buffer_worstcase_pJ",
                 "E_total_ideal_Vcm_pJ", "E_CDAC_if_conventional_pJ", "metastable_decisions"], rows)

    hdr, wrows = build_waveforms(adc, r, vin_func, keep, a.pts_per_cycle)
    write_csv(os.path.join(a.outdir, "task2_bit_waveforms.csv"), hdr, wrows)

    if st:
        rows = []
        for c in range(m):
            rows.append([c, int(st["hits"][c]), fmt(st["trans"][c - 1] if c >= 1 else None, 7),
                         fmt(st["dnl"][c - 1] if 1 <= c <= m - 2 else None, 5),
                         fmt(st["inl"][c - 1] if c >= 1 else None, 5)])
        write_csv(os.path.join(a.outdir, "task2_inl_dnl.csv"),
                  ["code", "hits", "transition_V(code-1->code)", "DNL_LSB", "INL_LSB"], rows)

    if dyn:
        write_csv(os.path.join(a.outdir, "task2_spectrum.csv"), ["frequency_Hz", "magnitude_dBFS"],
                  [[f"{fq:.6g}", f"{s:.3f}"] for fq, s in zip(dyn["freq"], dyn["spec_dbfs"])])

    rows = []
    for c in range(m):
        sav = 100 * (1 - ev["e_ref"][c] / ec["e_ref"][c]) if ec["e_ref"][c] > 0 else float("nan")
        rows.append([c, format(c, f"0{n}b"), f"{ev['e_ref'][c]:.4f}", f"{ev['q_cm_src'][c] * vdd / vref:.4f}",
                     f"{ec['e_ref'][c]:.4f}", f"{sav:.2f}", f"{ev['e_ref'][c] * cv2 * 1e15:.3f}",
                     f"{ec['e_ref'][c] * cv2 * 1e15:.3f}"])
    write_csv(os.path.join(a.outdir, "task2_energy_vs_code.csv"),
              ["code", "binary", "E_Vcm_based_Vref_CVref2", "E_Vcm_buffer_from_VDD_CVref2",
               "E_conventional_CVref2", "saving_percent_Vref", "E_Vcm_based_Vref_fJ", "E_conventional_fJ"], rows)

    # ================================================================== CSV: summary
    S = []

    def add(sec, name, val, unit="", note=""):
        S.append([sec, name, val if isinstance(val, str) else fmt(val), unit, note])

    sec = "Configuration"
    add(sec, "Architecture", "10-bit SAR, single-ended, top-plate sampled, Vcm-based switching (Zhu 2010)")
    add(sec, "Resolution", n, "bit")
    add(sec, "VDD", vdd, "V")
    add(sec, "Vref (input range 0..Vref)", vref, "V")
    add(sec, "Vcm (bottom-plate reset and comparator Vinn)", adc.vcm, "V", f"error vs Vref/2 = {p['VCM_ERROR'] * 1e3:g} mV")
    add(sec, "Sampling rate fs", fs, "S/s")
    add(sec, "Bit-clock frequency", 1 / adc.t_b, "Hz", f"{adc.n_cycles} cycles per conversion")
    add(sec, "LSB", adc.lsb * 1e3, "mV")
    add(sec, "Unit capacitor Cu", p["CU"] * 1e15, "fF")
    add(sec, "Unit caps in array", int(adc.cdac.units.sum()), "Cu", "256+128+...+1 + 1 dummy (conventional needs 1024)")
    add(sec, "Total CDAC capacitance (with mismatch)", adc.cdac.C.sum() * 1e12, "pF")
    add(sec, "Top-plate parasitic", adc.cdac.Cp * 1e15, "fF")
    att = adc.cdac.C.sum() / adc.c_s
    add(sec, "Effective input range", f"{adc.vcm - att * vref / 2:.4f} .. {adc.vcm + att * vref / 2:.4f}", "V",
        f"Vcm +/- {att:.4f} x Vref/2 (top-plate parasitic attenuation); beyond this the codes clip")
    add(sec, "Analog input", src)
    if file_header:
        add(sec, "File header", " | ".join(file_header), "", f"time column {a.tcol}, voltage column {a.vcol}")
    add(sec, "Conversions", n_conv)
    add(sec, "Ideal mode", "yes" if a.ideal else "no")

    sec = "Power"
    for name, val in energy_mean.items():
        add(sec, f"{name} - power", val * fs * 1e6, "uW",
            f"{val * 1e12:.4g} pJ/conversion; {100 * val / e_dyn:.1f}% of dynamic")
    add(sec, "Vcm reference - ideal decoupled source (net charge)", e_cm_net * fs * 1e6, "uW",
        "charge into and out of Vcm cancels on average; literature (Zhu 2010) counts Vref only")
    add(sec, "Vcm generator static power", p["P_VCM_GEN_STATIC"] * 1e6, "uW", "set P_VCM_GEN_STATIC for a real buffer")
    add(sec, "Static leakage", p["P_LEAK"] * 1e6, "uW")
    add(sec, "TOTAL POWER", p_total * 1e6, "uW", "ideal Vcm source")
    add(sec, "Energy per conversion", p_total / fs * 1e12, "pJ")
    add(sec, "Vcm buffer worst case (sources every event from VDD)", e_cm_wc * fs * 1e6, "uW",
        "bottom plates on Vcm carry the top-plate charge while tracking Vin; no recycling assumed")
    add(sec, "TOTAL POWER incl. worst-case Vcm buffer", p_total_wc * 1e6, "uW")

    sec = "Dynamic performance"
    if tone:
        add(sec, "FFT window", dyn["window"])
        add(sec, "Detected input frequency", dyn["fin"], "Hz")
        add(sec, "Signal level", dyn["signal_dbfs"], "dBFS")
        add(sec, "SNDR (SINAD)", dyn["sndr"], "dB")
        add(sec, "SNR", dyn["snr"], "dB")
        add(sec, "THD", dyn["thd"], "dBc")
        add(sec, "SFDR", dyn["sfdr"], "dBc")
        add(sec, "ENOB", dyn["enob"], "bit", "at the applied amplitude")
        add(sec, "ENOB (normalised to full scale)", dyn["enob_fs"], "bit")
        add(sec, "Noise floor (per bin)", dyn["noise_floor_dbfs"], "dBFS")
        for h, fh, lvl in dyn["harmonics"][:4]:
            add(sec, f"HD{h} (at {fh:.6g} Hz)", lvl, "dBc")
    else:
        add(sec, "FFT metrics", "n/a", "", "input is not a single tone - use the error-based metrics below")
    add(sec, "RMS error vs ideal input", float(np.sqrt(np.mean(err ** 2))) / adc.lsb, "LSB", "includes offset & gain error")
    add(sec, "Effective resolution after gain/offset fit", enob_fit, "bit", "valid for any input waveform")
    add(sec, "Out-of-range input samples", out_of_range, "", "below 0 V or above Vref (clipped)")
    add(sec, "Metastable comparator decisions", int(r["meta"].sum()))
    add(sec, "Samples at code 0 or full scale", rails, "", "non-zero means the input reaches the clipping range")

    sec = "Static performance (ramp code-density test)"
    if st:
        add(sec, "Ramp conversions", st["n_conv"])
        add(sec, "DNL max / min", f"{st['dnl'].max():+.3f} / {st['dnl'].min():+.3f}", "LSB")
        add(sec, "INL max / min (end-point fit)", f"{st['inl'].max():+.3f} / {st['inl'].min():+.3f}", "LSB")
        add(sec, "Missing codes", st["missing"])
        add(sec, "Offset error at mid-scale (511->512)", st["offset_mid_lsb"], "LSB",
            "comparator offset + switch charge injection/feed-through")
        add(sec, "Offset error at first transition (0->1)", st["offset_lsb"], "LSB", "includes the gain error (pivots about Vcm)")
        add(sec, "Gain error", st["gain_lsb"], "LSB", f"{st['gain_pct']:+.3f}% (top-plate parasitic attenuation)")
    else:
        add(sec, "Static test", "skipped")

    sec = "Figures of merit"
    if tone:
        add(sec, "Walden FoM = P/(2^ENOB*fs)", p_total / (2 ** dyn["enob_fs"] * fs) * 1e15, "fJ/conv-step", "full-scale ENOB")
        add(sec, "Walden FoM incl. worst-case Vcm buffer", p_total_wc / (2 ** dyn["enob_fs"] * fs) * 1e15, "fJ/conv-step")
        add(sec, "Schreier FoM = SNDR + 10log(fs/2/P)", dyn["sndr"] - dyn["signal_dbfs"] + 10 * math.log10(fs / 2 / p_total), "dB")
    else:
        add(sec, "Walden FoM (from fitted resolution)", p_total / (2 ** enob_fit * fs) * 1e15, "fJ/conv-step")
    add(sec, "Energy per conversion", p_total / fs * 1e12, "pJ")

    sec = "Vcm-based vs conventional (same input signal)"
    add(sec, "CDAC energy/conversion - Vcm-based (ideal array)", e_sig_v * cv2 * 1e12, "pJ", f"{e_sig_v:.2f} C*Vref^2")
    add(sec, "CDAC energy/conversion - conventional (ideal array)", e_sig_c * cv2 * 1e12, "pJ", f"{e_sig_c:.2f} C*Vref^2")
    add(sec, "CDAC switching-energy saving", 100 * (1 - e_sig_v / e_sig_c), "%")
    add(sec, "CDAC power at fs - Vcm-based / conventional",
        f"{e_sig_v * cv2 * fs * 1e6:.4g} / {e_sig_c * cv2 * fs * 1e6:.4g}", "uW")
    add(sec, "Total capacitance - Vcm-based / conventional", f"{int(cd_v.units.sum())} / {int(cd_c.units.sum())}", "Cu",
        "50% less capacitor area, 2x lighter input load")
    add(sec, "MSB decision", "free in both (top-plate sampling)")
    add(sec, "Plate moves per conversion", "9 one-way moves (Vcm-based) vs up to 18 moves incl. reversals (conventional)")
    add(sec, "Costs of Vcm-based", "third reference Vcm (buffer must source and sink), tri-level switches, Vcm accuracy sets DNL")

    sec = "CDAC switching-energy study (ideal array, all codes equally likely)"
    add(sec, "Vcm-based average (Vref energy)", ev["e_ref"].mean(), "C*Vref^2", "single-ended")
    add(sec, "Vcm-based differential equivalent (x2)", 2 * ev["e_ref"].mean(), "C*Vref^2",
        f"closed form (Zhu 2010) = {lit_vcm:.1f}")
    add(sec, "Conventional average", ec["e_ref"].mean(), "C*Vref^2", "single-ended")
    add(sec, "Conventional differential equivalent (x2)", 2 * ec["e_ref"].mean(), "C*Vref^2",
        f"closed form (Ginsburg/Liu) = {lit_conv:.1f}")
    add(sec, "Average saving", 100 * (1 - ev["e_ref"].mean() / ec["e_ref"].mean()), "%", "literature: 87.5%")
    add(sec, "Vcm-based max / min over codes", f"{ev['e_ref'].max():.2f} / {ev['e_ref'].min():.2f}", "C*Vref^2")

    sec = "Timing"
    add(sec, "Conversion period", adc.t_frame * 1e6, "us")
    add(sec, "Bit-clock period", adc.t_b * 1e9, "ns")
    add(sec, "Track (sampling) time", adc.t_track * 1e9, "ns")
    add(sec, "Latency: sampling instant -> data valid", (adc.t_track + n * adc.t_b - adc.clk.t_sample) * 1e9, "ns")
    add(sec, "PHI1 -> PHI2 / PHI2 -> PHI1 non-overlap",
        f"{adc.clk.nonoverlap_12 * 1e12:.0f} / {adc.clk.nonoverlap_21 * 1e12:.0f}", "ps")
    add(sec, "Comparator decision time (median / max)",
        f"{np.median(r['tdec']) * 1e12:.0f} / {r['tdec'].max() * 1e12:.0f}", "ps", f"available {adc.t_cmp_avail * 1e9:.4g} ns")
    add(sec, "DAC settling to 0.5 LSB (largest top-plate step Vref/4)", p["TAU_DAC"] * math.log(2 ** (n - 1)) * 1e9,
        "ns", f"available {adc.t_settle * 1e9:.4g} ns")
    add(sec, "Track-mode time constant (median)", float(np.median(r["tau_track"])) * 1e9, "ns")

    sec = "Block characteristics"
    vgrid = np.array([0.0, vref / 2, vref])
    ron = adc.sw.ron(vgrid)
    add(sec, "Bootstrap ratio Cb/(Cb+C_G)", adc.sw.alpha)
    vg3 = np.array([0.0, vref / 2, vref])
    inj = p["INJ_FRACTION"] * p["M1_W"] * p["M1_L"] * p["COX"] * np.maximum(adc.sw.vov(vg3), 0) / adc.c_s
    add(sec, "M1 charge-injection step at Vin = 0 / Vref/2 / Vref", " / ".join(f"{x / adc.lsb:.2f}" for x in inj),
        "LSB", "the variation across the range is what distorts; scale M1_W with the sampling capacitance")
    add(sec, "M1 Ron at Vin = 0 / Vref/2 / Vref", f"{ron[0]:.1f} / {ron[1]:.1f} / {ron[2]:.1f}", "ohm")
    add(sec, "kT/C noise", adc.ktc_sigma * 1e6, "uV rms", "larger than conventional: half the sampling capacitance")
    add(sec, "Comparator offset (this instance)", adc.cmp.vos * 1e3, "mV", f"Pelgrom sigma = {adc.cmp.sigma_os * 1e3:.3g} mV")
    add(sec, "Comparator energy/decision", adc.cmp.e_decision * 1e15, "fJ")
    rel = (adc.cdac.C / (p["CU"] * adc.cdac.units) - 1) * 100
    add(sec, "CDAC cap errors 256C..1C (%)", " ".join(f"{x:+.3f}" for x in rel[:-1]), "%")
    add(sec, "Clock-generator energy/frame", adc.clk.energy_per_frame * 1e15, "fJ")

    sec = "Design hints"
    if dyn_alt:
        now, alt = ("gate tied to G", "gate driven by PHI") if p["M2_GATE_TO_G"] else ("gate driven by PHI (as drawn)", "gate tied to G")
        hd2 = lambda d: d["harmonics"][0][2]
        add(sec, f"Bootstrapped switch M2 {now}", f"SFDR {dyn['sfdr']:.1f} dBc, THD {dyn['thd']:.1f} dBc, HD2 {hd2(dyn):.1f} dBc")
        add(sec, f"Same chip with M2 {alt}", f"SFDR {dyn_alt['sfdr']:.1f} dBc, THD {dyn_alt['thd']:.1f} dBc, HD2 {hd2(dyn_alt):.1f} dBc",
            "", "with PHI on M2's gate node B cannot pass Vin above VDD - Vth, so the boost and Ron degrade near full scale")
    add(sec, "Comparator reference", f"Vinn tied to the same Vcm node ({adc.vcm:.4g} V) that resets the bottom plates")

    sec = "Notes"
    add(sec, "Model", "behavioural/circuit-aware; replace DEFAULT_PARAMS with your Spectre numbers")
    add(sec, "Vcm accuracy", "try --params with VCM_ERROR (e.g. 0.005) to see its effect on DNL/INL")

    write_csv(os.path.join(a.outdir, "task2_summary.csv"), ["section", "parameter", "value", "unit", "note"], S)

    # ------------------------------------------------------------------ console
    print(f"\nTask 2 - Vcm-based SAR ADC ({n}-bit, fs = {fs:g} S/s, VDD = {vdd} V, Cu = {p['CU'] * 1e15:g} fF)")
    print(f"  analog input    : {src}")
    print(f"  converted       : {n_conv} samples -> task2_digital_output.csv")
    if tone:
        print(f"  SNDR / SFDR     : {dyn['sndr']:.2f} dB / {dyn['sfdr']:.2f} dBc  ->  ENOB {dyn['enob']:.2f} bit "
              f"({dyn['enob_fs']:.2f} at FS)")
    else:
        print(f"  resolution (fit): {enob_fit:.2f} bit  (input is not a single tone)")
    if st:
        print(f"  DNL / INL       : {st['dnl'].min():+.2f}..{st['dnl'].max():+.2f} / "
              f"{st['inl'].min():+.2f}..{st['inl'].max():+.2f} LSB, missing codes {st['missing']}")
    print(f"  total power     : {p_total * 1e6:.4g} uW   ({p_total / fs * 1e12:.4g} pJ/conversion, ideal Vcm source)")
    for name, val in sorted(energy_mean.items(), key=lambda kv: -kv[1]):
        print(f"     {name:48s} {val * fs * 1e6:9.4f} uW")
    print(f"  worst case      : {p_total_wc * 1e6:.4g} uW with a Vcm buffer that cannot recycle charge "
          f"(+{e_cm_wc * fs * 1e6:.4g} uW)")
    if tone:
        print(f"  Walden FoM      : {p_total / (2 ** dyn['enob_fs'] * fs) * 1e15:.2f} fJ/conv-step")
    print(f"  CDAC energy     : Vcm-based {e_sig_v * cv2 * 1e12:.3f} pJ vs conventional {e_sig_c * cv2 * 1e12:.3f} pJ "
          f"per conversion for this signal -> {100 * (1 - e_sig_v / e_sig_c):.1f}% saving")
    print(f"  literature check: Vcm-based {2 * ev['e_ref'].mean():.1f} (170.2), conventional "
          f"{2 * ec['e_ref'].mean():.1f} (1363.3) C*Vref^2 differential")
    if rails > 0.005 * n_conv:
        print(f"  WARNING         : {100 * rails / n_conv:.1f}% of samples sit at code 0 or {m - 1} - the input exceeds "
              f"the effective range (see task summary); reduce the amplitude or use --autoscale")
    print(f"  CSV files in    : {os.path.abspath(a.outdir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
