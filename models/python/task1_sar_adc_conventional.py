#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 1 - Working 10-bit SAR ADC model built from the project's circuit blocks
=====================================================================

    Vin --> Bootstrapped switch (M1-M6, Cb) --> 10-bit binary CDAC (512C ... 1C + 1C dummy,
            inverter bottom-plate drivers Vref/GND) --> StrongARM latch (MN1-MN5, MP1-MP4)
            --> NAND SR latch + 2-inverter buffer (Dcmp/Dcmpb) --> synchronous SAR logic --> D[9:0]
    Clocks: NOR-based non-overlapping generator  fs --> PHI1 (track, = PHI of the switch) / PHI2 (convert)

Switching scheme modelled: CONVENTIONAL (binary search, set-and-reset) with top-plate sampling
    * While PHI1 is high the bootstrapped switch drives the top plate Vx to Vin. The 512C bottom
      plate sits at Vref and every other plate (incl. the dummy) at GND, so the bottom-plate
      mean is Vref/2 = Vcm. The comparator's Vinn input is tied to Vcm.
    * Bit cycle 1 (b9): compare Vx = Vin against Vcm - no capacitor has to move.
    * Bit cycle k: if the bit just decided is 1, its capacitor is switched back Vref -> GND,
      and the next capacitor is raised GND -> Vref (the new trial). Dcmp = 1 <=> Vx > Vcm.
    * Code thresholds land on k*Vref/1024 and the average CDAC energy per cycle is exactly the
      textbook conventional value: 681.7 C*Vref^2 single-ended (= 1363.3 C*Vref^2 differential,
      Ginsburg & Chandrakasan / Liu et al. JSSC 2010). The script re-checks this every run.

What the model includes (all parameters are editable, see DEFAULT_PARAMS):
    bootstrapped switch - Ron(Vin) with body effect, bootstrap ratio Cb/(Cb+C_G), the PHI-gated M2
                          limit on node B, charge injection, clock feed-through, kT/C noise, jitter
    StrongARM + SR      - Pelgrom offset, input-referred noise, regeneration-time / metastability,
                          energy per decision from node capacitances
    CDAC                - unit-cap mismatch (sigma/sqrt(m)), top-plate parasitic (gain error),
                          settling, exact charge-conservation energy accounting from Vref
    clock generator     - event-driven gate-level simulation (non-overlap times, switching energy)
    SAR logic           - toggle-counted dynamic energy of sequencer + code/output registers

Outputs (CSV, written to --outdir):
    task1_digital_output.csv  one row per conversion: Vin, held value, code, b9..b0, reconstruction,
                              error, energy per conversion
    task1_bit_waveforms.csv   time-domain waveforms: fs, PHI1, PHI1b, PHI2, CLK, Vin, Vx, outp, outn,
                              Dcmp, Dcmpb, SAR register b9..b0, bottom-plate drives, EOC, D9..D0, code
    task1_summary.csv         power breakdown, SNDR/SNR/THD/SFDR/ENOB, INL/DNL, FoM, timing, block data
    task1_inl_dnl.csv         ramp code-density test: hits, transition voltage, DNL, INL per code
    task1_spectrum.csv        output spectrum (dBFS)
    task1_energy_vs_code.csv  ideal CDAC energy for every output code (C*Vref^2 and fJ)

Usage:
    python task1_sar_adc_conventional.py                         # built-in coherent sine test
    python task1_sar_adc_conventional.py --input vin.csv         # convert your own analog waveform
    python task1_sar_adc_conventional.py --fs 1e3 --cu 2e-15     # change specs
    python task1_sar_adc_conventional.py --params my.json        # override any DEFAULT_PARAMS key
    python task1_sar_adc_conventional.py --ideal                 # all non-idealities off (sanity check)

Analog input file (--input): CSV or text with time and voltage columns - e.g. a Virtuoso (ViVA)
CSV export ("/Vin X","/Vin Y") or OCEAN ocnPrint text (SI suffixes such as 10n, 1.2m are understood).
A single-column file is treated as samples already taken at --fs.

Requires Python 3.8+ and numpy.  All numbers are model estimates - replace the default
device/capacitance values with figures from your own Spectre runs for sign-off quality.
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

KB = 1.380649e-23  # Boltzmann constant [J/K]

# =============================================================================================
# 1. MODEL PARAMETERS  (180 nm-class typical values - edit, or override with --params file.json)
# =============================================================================================
DEFAULT_PARAMS = {
    # ---------------- converter ----------------
    "N_BITS": 10,
    "VDD": 1.8,              # supply [V]
    "VREF": 1.8,             # CDAC reference [V]; nominal input range 0 .. VREF
    "VCM": None,             # comparator reference (Vinn) [V]; None -> VREF/2
    "FS": 100e3,             # sampling rate [S/s]
    "TRACK_CYCLES": 1,       # bit-clock cycles spent tracking (PHI1 high)
    "EOC_CYCLES": 1,         # bit-clock cycles for output latch / DAC reset
    "T_LOGIC": 0.3e-9,       # SAR logic + driver delay after the CLK edge [s]
    "TEMP_K": 300.0,         # temperature [K] (kT/C noise)
    "SEED": 7,               # random seed (mismatch, offset, noise) - change for a new "chip"
    # ---------------- CDAC ----------------
    "CU": 10e-15,            # unit capacitor [F]
    "SIGMA_CU": 0.005,       # relative sigma of ONE unit cap (a cap of m units: sigma/sqrt(m))
    "CP_TOP_FIXED": 20e-15,  # top-plate parasitic: fixed part (switch junction, routing) [F]
    "CP_TOP_FRAC": 0.012,    # top-plate parasitic: fraction of the array capacitance
    "TAU_DAC": 2e-9,         # bottom-plate settling time constant (driver R x cap) [s]
    "C_DRV_MIN": 2e-15,      # input capacitance of the smallest bit driver [F]
    "C_DRV_PER_UNIT": 4e-18, # extra driver input capacitance per unit capacitor driven [F]
    # ---------------- MOS model (180 nm-class) ----------------
    "KPN": 300e-6,           # un*Cox [A/V^2]
    "VTN0": 0.45,            # NMOS threshold at Vsb = 0 [V]
    "GAMMA": 0.45,           # body-effect coefficient [V^0.5]
    "PHI2F": 0.8,            # 2*phi_F [V]
    "COX": 8.6e-3,           # gate-oxide capacitance [F/m^2]  (8.6 fF/um^2)
    "COV": 0.35e-9,          # gate overlap capacitance per width [F/m] (0.35 fF/um)
    # ---------------- bootstrapped switch ----------------
    "M1_W": 10e-6, "M1_L": 0.18e-6,   # main sampling switch M1
    "CB": 1e-12,             # bootstrap capacitor Cb [F]
    "CG_PAR": 5e-15,         # extra parasitic on gate node G (M4, M5, M6 junctions) [F]
    "M2_GATE_TO_G": False,   # False = M2 gated by PHI (as drawn); True = gate tied to G (Abo-Gray)
    "INJ_FRACTION": 0.5,     # fraction of M1 channel charge dumped onto Vx at turn-off
    "JITTER_RMS": 10e-12,    # sampling-clock jitter [s rms]
    "TRACK_STEPS": 120,      # integration steps for the track-mode RC model
    # ---------------- StrongARM comparator + SR latch ----------------
    "MN12_W": 4e-6, "MN12_L": 0.36e-6,  # input pair MN1/MN2
    "AVT": 5e-9,             # Pelgrom A_VT [V*m]  (5 mV*um)
    "VOS": None,             # input offset [V]; None -> one random draw from the Pelgrom sigma
    "CMP_NOISE_RMS": 0.3e-3, # input-referred noise [V rms]
    "CMP_T0": 150e-12,       # integration (pre-regeneration) delay [s]
    "CMP_TAU": 60e-12,       # regeneration time constant C/gm [s]
    "CMP_G0": 4.0,           # small-signal gain during integration
    "C_OUT": 12e-15,         # outp/outn node capacitance (incl. SR-latch NAND input) [F]
    "C_XY": 6e-15,           # X/Y node capacitance [F]
    "C_CLK_CMP": 8e-15,      # CLK load of MP3, MP4, MN5 [F]
    "K_DIP": 0.3,            # partial discharge of the winning output, fraction of VDD
    "C_CMP_IN": 10e-15,      # comparator input capacitance (adds to top-plate parasitic) [F]
    "C_SR_NODE": 4e-15,      # SR latch / output-buffer node capacitance [F]
    "T_INV": 40e-12, "T_NAND": 60e-12, "T_NOR": 80e-12,  # gate delays [s]
    # ---------------- SAR logic & clock generator ----------------
    "C_FF_CLK": 3e-15,       # effective clock capacitance per D flip-flop [F]
    "C_FF_Q": 5e-15,         # capacitance switched per flip-flop output toggle [F]
    "C_CLKGEN_NODE": 5e-15,  # internal node capacitance in the clock generator [F]
    "C_PHI_LOAD": 25e-15,    # load on each PHI1 / PHI1b / PHI2 line [F]
    "P_LEAK": 2e-9,          # total static leakage [W]
}

IDEAL_OVERRIDES = {  # used by --ideal
    "SIGMA_CU": 0.0, "CP_TOP_FIXED": 0.0, "CP_TOP_FRAC": 0.0, "C_CMP_IN": 0.0, "VOS": 0.0, "CMP_NOISE_RMS": 0.0,
    "JITTER_RMS": 0.0, "INJ_FRACTION": 0.0, "COV": 0.0,
}


# =============================================================================================
# 2. CIRCUIT BLOCKS
# =============================================================================================
class NonOverlapClockGen:
    """Event-driven gate-level model of the NOR-based non-overlapping clock generator.

        fsb   = INV(fs)
        PHI1  = NOR(fs,  PHI2_d)      PHI1_d = INV(INV(PHI1))   ("Delay (2 inv)")
        PHI2  = NOR(fsb, PHI1_d)      PHI2_d = INV(INV(PHI2))
        PHI1b = INV(PHI1)             (the PHIb needed by the bootstrapped switch)

    fs is LOW while the ADC tracks (PHI1 = sampling phase = PHI of the switch) and HIGH while it
    converts (PHI2). One frame is simulated; edge times are then reused for every frame.
    """

    GATES = [  # (output net, gate type, input nets, delay key)
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
        self.p = p
        self.t_track, self.t_frame = t_track, t_frame
        self._simulate()

    @staticmethod
    def _eval(kind, vals):
        if kind == "INV":
            return 1 - vals[0]
        if kind == "NOR":
            return 0 if any(vals) else 1
        raise ValueError(kind)

    def _simulate(self):
        p = self.p
        v = {"fs": 0, "fsb": 1, "phi1": 1, "phi1_m": 0, "phi1_d": 1,
             "phi2": 0, "phi2_m": 1, "phi2_d": 0, "phi1b": 0}  # steady state while tracking
        fanout = {}
        for g in self.GATES:
            for net in g[2]:
                fanout.setdefault(net, []).append(g)
        queue, seq = [], 0
        for t, val in ((self.t_track, 1), (self.t_frame, 0)):  # fs: rise (end of track), fall (next frame)
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
        self.t_phi2_fall_rel = edge("phi2", 0, tf) - tf      # relative to the next frame start
        self.t_phi1_rise_rel = edge("phi1", 1, tf) - tf
        self.t_phi1b_fall_rel = edge("phi1b", 0, tf) - tf
        self.nonoverlap_12 = self.t_phi2_rise - self.t_phi1_fall      # PHI1 falls -> PHI2 rises
        self.nonoverlap_21 = self.t_phi1_rise_rel - self.t_phi2_fall_rel  # PHI2 falls -> PHI1 rises
        self.t_sample = self.t_phi1b_rise    # PHIb rising (M4 grounds G) = sampling instant
        vdd = p["VDD"]
        cap = {k: p["C_CLKGEN_NODE"] for k in v}
        for k in ("phi1", "phi1b", "phi2"):
            cap[k] += p["C_PHI_LOAD"]
        cap["fs"] = p["C_CLKGEN_NODE"]
        self.energy_per_frame = sum(0.5 * cap[k] * vdd ** 2 * self.toggles[k] for k in v)


class BootstrappedSwitch:
    """Behavioural model of the bootstrapped sampling switch (M1-M6, Cb).

    Hold  (PHI=0, PHIb=1): M3 grounds B, M4 grounds G (M1 off), M6 charges A to VDD -> Cb holds VDD.
    Track (PHI=1, PHIb=0): M2 connects B to Vin, M5 connects A to G ->
            V_G = alpha*(VDD + V_B),  alpha = Cb/(Cb + C_G)   => Vgs(M1) ~ alpha*VDD - (1-alpha)*Vin.
    In the schematic M2's gate is PHI (= VDD), so node B only follows Vin up to VDD - Vth(M2);
    above that the boost degrades. That limit, the body effect of M1, channel-charge injection,
    clock feed-through, kT/C noise and aperture jitter are all included.
    """

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
        if self.p["M2_GATE_TO_G"]:      # boosted gate: node B follows Vin over the whole range
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
        """Integrate dVx/dt = (Vin - Vx)/(Ron*Cs) from PHI1 rising to the sampling instant.
        Exact first-order-hold solution per step (accurate for any step size)."""
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
        """Turn-off error: channel-charge injection + clock feed-through after cut-off."""
        p = self.p
        q_ch = p["M1_W"] * p["M1_L"] * p["COX"] * np.maximum(self.vov(vx), 0.0)
        dv_inj = -p["INJ_FRACTION"] * q_ch / self.c_s
        dv_ft = -self.c_ov * (vx + self.vth(vx)) / (self.c_s + self.c_ov)
        return vx + dv_inj + dv_ft

    def energy(self, vin):
        """VDD energy per sample: M6 restores the charge Cb lost to node G -> alpha*C_G*(VDD+V_B)."""
        vdd = self.p["VDD"]
        return vdd * self.alpha * self.c_g * (vdd + self.node_b(vin))


class StrongArmComparator:
    """StrongARM latch (MN1-MN5, MP1-MP4) followed by the NAND SR latch and 2-inverter buffer.
    With Vinp > Vinn, MN1 conducts more, X falls first, outp falls -> SR latch sets Q = Dcmp = 1."""

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


class ConventionalCDAC:
    """10-bit binary-weighted CDAC: 512C, 256C, ..., 1C + 1C dummy (top-plate sampled).
    Index 0 = MSB (b9) ... index N-1 = LSB (b0), index N = dummy (always GND)."""

    name = "conventional"

    def __init__(self, p, rng, ideal_caps=False):
        n = p["N_BITS"]
        self.n = n
        self.vref = p["VREF"]
        self.units = np.array([2.0 ** (n - 1 - j) for j in range(n)] + [1.0])
        sig = 0.0 if ideal_caps else p["SIGMA_CU"]
        eps = rng.normal(0.0, 1.0, self.units.size) * sig / np.sqrt(self.units) if sig > 0 else 0.0
        self.C = p["CU"] * self.units * (1.0 + eps)
        self.Cp = 0.0 if ideal_caps else p["CP_TOP_FIXED"] + p["CP_TOP_FRAC"] * self.C.sum() + p["C_CMP_IN"]
        self.Vb0 = np.zeros(self.units.size)
        self.Vb0[0] = self.vref                       # MSB at Vref, rest at GND during sampling
        self.c_drv = p["C_DRV_MIN"] + p["C_DRV_PER_UNIT"] * self.units
        self.c_drv[-1] = 0.0                           # dummy has no driver (tied to GND)
        self.labels = [f"{int(self.units[j])}C_b{n - 1 - j}" for j in range(n)] + ["dummy"]

    def initial_state(self, ns):
        return np.tile(self.Vb0, (ns, 1))

    def next_state(self, vb, j, d):
        """After deciding bit j: switch it back to GND if it was a 1, raise the next trial cap."""
        vbn = vb.copy()
        vbn[:, j] = np.where(d, 0.0, self.vref)
        vbn[:, j + 1] = self.vref
        return vbn

    def switch(self, vb_old, vb_new, vx_old):
        """Floating top plate: charge conservation -> new Vx and charge pushed into each bottom plate."""
        q = (self.C * (vx_old[:, None] - vb_old)).sum(1) + self.Cp * vx_old
        vx_new = (q + (self.C * vb_new).sum(1)) / (self.C.sum() + self.Cp)
        dq = self.C * ((vb_new - vx_new[:, None]) - (vb_old - vx_old[:, None]))
        return vx_new, dq

    def driven_charge(self, vb_old, vb_new, vx_old, vx_new):
        """Top plate driven (sampling): charge pushed into each bottom plate."""
        return self.C * ((vb_new - vx_new[:, None]) - (vb_old - vx_old[:, None]))

    def ref_charge(self, dq, vb_new):
        return (dq * np.isclose(vb_new, self.vref)).sum(1)


# =============================================================================================
# 3. THE CONVERTER
# =============================================================================================
class SARADC:
    def __init__(self, p):
        self.p = p
        self.n = p["N_BITS"]
        rng = np.random.default_rng(p["SEED"])
        self.cdac = ConventionalCDAC(p, rng)
        self.cmp = StrongArmComparator(p, rng)
        self.c_s = self.cdac.C.sum() + self.cdac.Cp
        self.sw = BootstrappedSwitch(p, self.c_s)
        self.vref = p["VREF"]
        self.vcm = p["VCM"] if p["VCM"] is not None else self.vref / 2
        self.n_cycles = p["TRACK_CYCLES"] + self.n + p["EOC_CYCLES"]
        self.t_frame = 1.0 / p["FS"]
        self.t_b = self.t_frame / self.n_cycles
        self.t_track = p["TRACK_CYCLES"] * self.t_b
        self.clk = NonOverlapClockGen(p, self.t_track, self.t_frame)
        self.t_cmp_avail = self.t_b / 2                  # CLK high = second half of each bit cycle
        self.t_settle = self.t_b / 2 - p["T_LOGIC"]       # DAC settles during the first half
        self.ktc_sigma = math.sqrt(KB * p["TEMP_K"] / self.c_s)
        self.lsb = self.vref / 2 ** self.n
        self.n_ff = self.n_cycles + 2 * self.n           # one-hot sequencer + SAR register + output register

    # -------------------------------------------------------------------- one SAR conversion
    def convert(self, vheld, noise, coin, keep=0):
        n = self.n
        ns = vheld.size
        vb = self.cdac.initial_state(ns)
        vx = vheld.copy()
        bits = np.zeros((ns, n), dtype=np.int8)
        vx_set = np.zeros((ns, n))
        vx_cmp = np.zeros((ns, n))
        tdec = np.zeros((ns, n))
        dvin = np.zeros((ns, n))
        e_ref = np.zeros(ns)
        drv_tog = np.zeros((ns, vb.shape[1]))
        hist = np.zeros((keep, n, vb.shape[1])) if keep else None
        serr = np.zeros(ns)
        meta = np.zeros(ns, dtype=int)
        settle_factor = math.exp(-self.t_settle / self.p["TAU_DAC"])
        for j in range(n):
            if keep:
                hist[:, j, :] = vb[:keep]
            vx_set[:, j] = vx
            vc = vx - serr                                   # incomplete DAC settling
            d, td, dv = self.cmp.decide(vc, self.vcm, noise[:, j])
            m = td > self.t_cmp_avail                        # metastable -> SAR latches a coin flip
            if m.any():
                d = np.where(m, coin[:, j], d)
                meta += m
            bits[:, j], vx_cmp[:, j], tdec[:, j], dvin[:, j] = d, vc, td, dv
            if j < n - 1:
                vbn = self.cdac.next_state(vb, j, d)
                vxn, dq = self.cdac.switch(vb, vbn, vx)
                e_ref += self.vref * self.cdac.ref_charge(dq, vbn)
                drv_tog += (vbn != vb)
                serr = (vxn - vx) * settle_factor
                vb, vx = vbn, vxn
        code = bits.astype(np.int64) @ (2 ** np.arange(n - 1, -1, -1))
        return dict(bits=bits, code=code, vx_set=vx_set, vx_cmp=vx_cmp, tdec=tdec, dvin=dvin,
                    e_ref=e_ref, drv_tog=drv_tog, vb_final=vb, vx_final=vx, hist=hist, meta=meta)

    # -------------------------------------------------------------------- many conversions
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
        for _ in range(2):   # pass 2 starts tracking from the true previous residue (memory effect)
            vtr, tau_tr = self.sw.track(vin_func, t_rise, t_s, vx0, p["TRACK_STEPS"])
            vheld = self.sw.hold(vtr) + ktc
            r = self.convert(vheld, noise, coin, keep)
            vx_after_reset, _ = self.cdac.switch(r["vb_final"], vb0, r["vx_final"])
            vx0 = np.roll(vx_after_reset, 1)
        r.update(t_frame=t_frame, t_s=t_s, vin_s=vin_s, vtrack=vtr, vheld=vheld, tau_track=tau_tr,
                 vx_after_reset=vx0)

        # ---- reset + sampling energy: previous final state -> (Vb0, Vheld) ----
        vb_prev = np.roll(r["vb_final"], 1, axis=0)
        vx_prev = np.roll(r["vx_final"], 1)
        dq = self.cdac.driven_charge(vb_prev, vb0, vx_prev, vheld)
        r["e_reset_ref"] = self.vref * self.cdac.ref_charge(dq, vb0)

        # ---- energy of every block, per conversion ----
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
        e = {
            "CDAC switching (Vref) - bit cycling": r["e_ref"],
            "CDAC switching (Vref) - reset/sampling": r["e_reset_ref"],
            "StrongARM comparator": np.full(n_conv, n * self.cmp.e_decision),
            "SR latch + output buffers": q_tog * 2 * p["C_SR_NODE"] * vdd ** 2,
            "SAR logic (sequencer + registers)": (self.n_cycles * self.n_ff * p["C_FF_CLK"] * vdd ** 2
                                                  + 0.5 * p["C_FF_Q"] * vdd ** 2 * (tog_reg + tog_out + tog_ring)),
            "CDAC bit drivers (gate drive)": 0.5 * vdd ** 2 * (drv_tog @ self.cdac.c_drv),
            "Bootstrapped switch": self.sw.energy(vtr),
            "Non-overlap clock generator": np.full(n_conv, self.clk.energy_per_frame),
        }
        r["energy"] = e
        r["e_total"] = sum(e.values())
        r["q_tog"] = q_tog
        return r


# =============================================================================================
# 4. ANALYSIS HELPERS
# =============================================================================================
def bh4_window(n):
    k = np.arange(n)
    x = 2 * np.pi * k / n
    return 0.35875 - 0.48829 * np.cos(x) + 0.14128 * np.cos(2 * x) - 0.01168 * np.cos(3 * x)


def dynamic_metrics(codes, fs, nbits, n_harm=9):
    """FFT metrics. Rectangular window when the record is coherent, Blackman-Harris otherwise."""
    x = codes.astype(float)
    n = x.size
    x = x - x.mean()
    pr = np.abs(np.fft.rfft(x)) ** 2
    if pr[1:].sum() <= 0:
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

    sig = lobe(k0)
    dc = set(range(0, span + 1))
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
    tone_fraction = p_sig / pw[span + 1:].sum()
    a_fs = 2 ** (nbits - 1)
    amp = math.sqrt(4 * p_sig / (n * np.sum(w ** 2)))
    sig_dbfs = 20 * math.log10(amp / a_fs)
    sndr = 10 * math.log10(p_sig / (p_noise + p_harm))
    p_ref = (a_fs * np.sum(w) / 2) ** 2
    out = dict(
        window="rectangular (coherent)" if coherent else "Blackman-Harris 4-term",
        fin=k0 * fs / n, bin=k0, n=n, tone_fraction=tone_fraction,
        signal_dbfs=sig_dbfs, sndr=sndr,
        snr=10 * math.log10(p_sig / p_noise),
        thd=10 * math.log10(p_harm / p_sig) if p_harm > 0 else -np.inf,
        sfdr=10 * math.log10(pw[k0] / pw[spur_mask].max()),
        enob=(sndr - 1.76) / 6.02,
        enob_fs=(sndr - sig_dbfs - 1.76) / 6.02,
        noise_floor_dbfs=10 * math.log10(p_noise / max(noise_mask.sum(), 1) / p_ref),
        harmonics=[(h, kh * fs / n, 10 * math.log10(ph / p_sig) if ph > 0 else -np.inf) for h, kh, ph in harm],
        freq=np.arange(nb) * fs / n,
        spec_dbfs=10 * np.log10(pw / p_ref + 1e-30),
    )
    return out


def static_test(adc, hits_per_code):
    """Ramp code-density test -> transition levels, DNL, INL (end-point fit), offset & gain error."""
    m = 2 ** adc.n
    vref = adc.vref
    lo, hi = -0.03 * vref, 1.03 * vref
    n_conv = int(math.ceil(hits_per_code * m * (hi - lo) / vref))
    t_total = n_conv * adc.t_frame
    r = adc.simulate(lambda t: lo + (hi - lo) * t / t_total, n_conv, seed_offset=1000)
    codes, vs = r["code"], r["vin_s"]
    hits = np.bincount(codes, minlength=m)
    cum = np.cumsum(hits)
    trans = np.interp(cum[:-1] - 0.5, np.arange(n_conv), vs)     # T[k], k = 1..m-1
    lsb_ideal = vref / m
    lsb_avg = (trans[-1] - trans[0]) / (m - 2)
    dnl = np.diff(trans) / lsb_avg - 1.0                          # codes 1..m-2
    inl = (trans - trans[0]) / lsb_avg - np.arange(m - 1)         # codes 1..m-1
    missing = int(np.sum(hits[1:-1] == 0))
    offset_lsb = (trans[0] - 1 * lsb_ideal) / lsb_ideal                       # at the first transition
    offset_mid_lsb = (trans[m // 2 - 1] - (m // 2) * lsb_ideal) / lsb_ideal     # at mid-scale (511 -> 512)
    gain_lsb = ((trans[-1] - trans[0]) - (m - 2) * lsb_ideal) / lsb_ideal
    return dict(hits=hits, trans=trans, dnl=dnl, inl=inl, missing=missing, offset_lsb=offset_lsb,
                offset_mid_lsb=offset_mid_lsb, gain_lsb=gain_lsb, gain_pct=100 * gain_lsb / (m - 2),
                n_conv=n_conv, lsb_avg=lsb_avg)


def ideal_energy_vs_code(p):
    """Ideal CDAC (unit caps, no parasitics), Cu = 1 and Vref = 1 -> energies in C*Vref^2.
    'cycle' = bit cycling + reset when the same code repeats (DC input).
    The textbook (bottom-plate-sampled) conventional procedure is evaluated alongside as a check."""
    n = p["N_BITS"]
    m = 2 ** n
    q = dict(p, CU=1.0, VREF=1.0)
    cd = ConventionalCDAC(q, np.random.default_rng(0), ideal_caps=True)
    codes = np.arange(m)
    vin = (codes + 0.5) / m
    vb = cd.initial_state(m)
    vx = vin.copy()
    e_conv = np.zeros(m)
    bits = np.zeros((m, n), dtype=np.int8)
    for j in range(n):
        d = vx > 0.5
        bits[:, j] = d
        if j < n - 1:
            vbn = cd.next_state(vb, j, d)
            vxn, dq = cd.switch(vb, vbn, vx)
            e_conv += cd.ref_charge(dq, vbn)
            vb, vx = vbn, vxn
    dq = cd.driven_charge(vb, cd.initial_state(m), vx, vin)
    e_reset = cd.ref_charge(dq, cd.initial_state(m))
    # textbook conventional: bottom-plate sampling, Vx = -Vin, trial caps raised one by one
    vb = np.zeros((m, n + 1))
    vx = -vin.copy()
    e_txt = np.zeros(m)
    tb = np.zeros((m, n), dtype=np.int8)
    for j in range(n):
        vbn = vb.copy()
        if j > 0:
            vbn[:, j - 1] = np.where(tb[:, j - 1] == 1, 1.0, 0.0)
        vbn[:, j] = 1.0
        vxn, dq = cd.switch(vb, vbn, vx)
        e_txt += cd.ref_charge(dq, vbn)
        vb, vx = vbn, vxn
        tb[:, j] = vx < 0
    code_chk = bits.astype(np.int64) @ (2 ** np.arange(n - 1, -1, -1))
    assert np.array_equal(code_chk, codes), "ideal conventional conversion failed"
    return dict(codes=codes, e_conv=e_conv, e_reset=e_reset, e_cycle=e_conv + e_reset, e_textbook=e_txt)


def closed_form_conventional_diff(n):
    """Ginsburg/Liu closed form for a differential conventional array [C*Vref^2]."""
    return sum(2.0 ** (n + 1 - 2 * i) * (2 ** i - 1) for i in range(1, n + 1))


# =============================================================================================
# 5. INPUT HANDLING
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
    """Read a time/voltage waveform (Virtuoso CSV export, ocnPrint text, any CSV/TSV/space file)."""
    rows = []
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
                continue                      # header line
            rows.append(vals)
    if not rows:
        raise ValueError(f"No numeric data found in {path}")
    ncol = min(len(r) for r in rows)
    if ncol == 1 or vcol >= ncol:
        v = np.array([r[0] for r in rows if r[0] is not None], float)
        return None, v
    data = [(r[tcol], r[vcol]) for r in rows if r[tcol] is not None and r[vcol] is not None]
    arr = np.array(data, float)
    order = np.argsort(arr[:, 0], kind="stable")
    t, v = arr[order, 0], arr[order, 1]
    keep = np.concatenate(([True], np.diff(t) > 0))
    return t[keep], v[keep]


def largest_prime_le(x):
    x = int(x)
    while x > 2:
        if x % 2 and all(x % d for d in range(3, int(x ** 0.5) + 1, 2)):
            return x
        x -= 1
    return 1


# =============================================================================================
# 6. OUTPUT WRITERS
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
    """Uniformly sampled logic/analog waveforms for the first n_frames conversions."""
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
    phi1b = 1 - (((tau >= ck.t_phi1b_fall_rel) & (tau < ck.t_phi1b_rise))).astype(int)
    phi2 = ((tau >= ck.t_phi2_rise) | (tau < ck.t_phi2_fall_rel)).astype(int)
    kc = np.floor((tau - ttr) / tb).astype(int)            # bit-cycle index (<0 during tracking)
    in_bit = (kc >= 0) & (kc < n)
    loc = (tau - ttr) - kc * tb                              # time inside the bit cycle
    clk = (in_bit & (loc >= tb / 2)).astype(int)
    kcc = np.clip(kc, 0, n - 1)
    d = r["bits"][:n_frames]
    tdec = r["tdec"][:n_frames]
    resolved = clk.astype(bool) & (loc - tb / 2 >= tdec[f, kcc])
    dd = d[f, kcc]
    outp = np.where(resolved, 1 - dd, 1)
    outn = np.where(resolved, dd, 1)
    # SR latch: Q follows the last resolved decision (chronological search)
    t_dec_abs = (r["t_frame"][:n_frames, None] + ttr + np.arange(n)[None, :] * tb + tb / 2
                 + tdec + p["T_NAND"] + 2 * p["T_INV"])
    order = np.argsort(t_dec_abs.ravel())
    tda, dflat = t_dec_abs.ravel()[order], d.ravel()[order]
    k = np.searchsorted(tda, t, side="right") - 1
    q = np.where(k >= 0, dflat[np.maximum(k, 0)], 0)
    # SAR register b9..b0
    ndec = np.where(tau < ttr, 0, np.clip(kc + 1, 0, n))     # decisions registered at end of each cycle
    ndec = np.where(in_bit, kc, ndec)
    ndec = np.where(kc >= n, n, ndec)
    sar = np.zeros((npts, n), dtype=int)
    for j in range(n):
        decided = j < ndec
        sar[:, j] = np.where(decided, d[f, j], (j == ndec).astype(int))
    # bottom-plate drives
    hist = r["hist"]
    state = np.where(tau < ttr, 0, np.clip(kc, 0, n - 1))
    bp = hist[f, state, :]
    bp = np.where((tau < ttr)[:, None], adc.cdac.Vb0[None, :], bp)
    # top plate Vx
    vin_t = vin_func(t)
    vx_set = r["vx_set"][:n_frames]
    tau_tr = np.atleast_1d(r["tau_track"])[f] if np.ndim(r["tau_track"]) else r["tau_track"]
    t_rise_rel, t_s_rel = ck.t_phi1_rise_rel, ck.t_sample
    vx_reset = r["vx_after_reset"][:n_frames]
    vin_rise = vin_func(r["t_frame"][:n_frames] + t_rise_rel)
    vx_track = vin_t + (vx_reset[f] - vin_rise[f]) * np.exp(-np.maximum(tau - t_rise_rel, 0) / tau_tr)
    prev = vx_set[f, np.maximum(kcc - 1, 0)]
    cur = vx_set[f, kcc]
    vx_bit = cur + (prev - cur) * np.exp(-np.maximum(loc, 0) / p["TAU_DAC"])
    vx_bit = np.where(kcc == 0, cur, vx_bit)
    vx = np.where(tau < t_rise_rel, vx_reset[f],
         np.where(tau < t_s_rel, vx_track,
         np.where(tau < ttr, r["vheld"][:n_frames][f],
         np.where(kc >= n, vx_set[f, n - 1], vx_bit))))
    # output register (latched at the start of the EOC cycle)
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
# 7. MAIN
# =============================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="Task 1: 10-bit conventional SAR ADC model of the given circuits")
    ap.add_argument("--input", help="analog waveform file (CSV/TXT: time, voltage)")
    ap.add_argument("--tcol", type=int, default=0, help="time column index in --input (default 0)")
    ap.add_argument("--vcol", type=int, default=1, help="voltage column index in --input (default 1)")
    ap.add_argument("--vin-gain", type=float, default=1.0, help="multiply file voltages by this")
    ap.add_argument("--vin-offset", type=float, default=0.0, help="then add this offset [V]")
    ap.add_argument("--autoscale", action="store_true", help="map the file's min/max to 5%%..95%% of the range")
    ap.add_argument("--max-conversions", type=int, default=None, help="limit conversions taken from --input")
    ap.add_argument("--fs", type=float, help="sampling rate [S/s]")
    ap.add_argument("--vdd", type=float, help="supply [V]")
    ap.add_argument("--vref", type=float, help="reference [V]")
    ap.add_argument("--cu", type=float, help="unit capacitor [F]")
    ap.add_argument("--nfft", type=int, default=4096, help="conversions for the built-in sine test")
    ap.add_argument("--fin", type=float, default=None, help="target sine frequency [Hz] (default ~0.49*fs)")
    ap.add_argument("--amp-dbfs", type=float, default=-1.0, help="built-in sine amplitude [dBFS]")
    ap.add_argument("--wave-frames", type=int, default=8, help="conversions written to the waveform CSV")
    ap.add_argument("--pts-per-cycle", type=int, default=40, help="waveform points per bit-clock cycle")
    ap.add_argument("--hits-per-code", type=int, default=32, help="ramp-test samples per code (INL/DNL)")
    ap.add_argument("--no-static", action="store_true", help="skip the ramp INL/DNL test")
    ap.add_argument("--ideal", action="store_true", help="switch off all non-idealities")
    ap.add_argument("--params", help="JSON file overriding DEFAULT_PARAMS keys")
    ap.add_argument("--seed", type=int, help="random seed")
    ap.add_argument("--outdir", default="task1_outputs", help="output folder")
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
    n, vref, fs = adc.n, adc.vref, p["FS"]
    m = 2 ** n

    # ------------------------------------------------------------------ input signal
    if a.input:
        t_file, v_file = load_analog_file(a.input, a.tcol, a.vcol)
        if t_file is None:                     # already-sampled data
            t_file = np.arange(v_file.size) / fs
        v_file = v_file * a.vin_gain + a.vin_offset
        if a.autoscale:
            lo_f, hi_f = v_file.min(), v_file.max()
            span = max(hi_f - lo_f, 1e-12)
            v_file = adc.vcm + (v_file - 0.5 * (lo_f + hi_f)) * (0.9 * vref / span)
        n_conv = int(math.floor((t_file[-1] - t_file[0] - adc.clk.t_sample) / adc.t_frame)) + 1
        if a.max_conversions:
            n_conv = min(n_conv, a.max_conversions)
        if n_conv < 2:
            sys.exit("Input file is shorter than two conversion periods at this --fs.")
        vin_func = lambda t: np.interp(t, t_file, v_file)
        t_start = t_file[0]
        src = f"file: {os.path.basename(a.input)} ({t_file.size} points, {t_file[-1] - t_file[0]:.6g} s)"
        fin, amp = None, None
    else:
        n_conv = a.nfft
        target = a.fin if a.fin else 0.49 * fs
        k_bin = largest_prime_le(target / fs * n_conv)
        fin = k_bin * fs / n_conv
        amp = (vref / 2) * 10 ** (a.amp_dbfs / 20)
        vin_func = lambda t: adc.vcm + amp * np.sin(2 * np.pi * fin * t + 0.3)
        t_start = 0.0
        src = f"built-in coherent sine: fin = {fin:.6g} Hz (bin {k_bin} of {n_conv}), {a.amp_dbfs:g} dBFS"

    keep = min(a.wave_frames, n_conv)
    r = adc.simulate(vin_func, n_conv, t0=t_start, keep=keep)
    codes = r["code"]
    vin_s = r["vin_s"]

    # ------------------------------------------------------------------ power
    energy_mean = {k: float(np.mean(v)) for k, v in r["energy"].items()}
    e_conv_total = sum(energy_mean.values())
    p_total = e_conv_total * fs + p["P_LEAK"]

    # ------------------------------------------------------------------ dynamic / error metrics
    dyn = dynamic_metrics(codes, fs, n)
    vrec = (codes + 0.5) * adc.lsb
    err = vrec - vin_s
    a_fit = np.vstack([vin_s, np.ones_like(vin_s)]).T
    coef, *_ = np.linalg.lstsq(a_fit, codes.astype(float), rcond=None)
    resid_lsb = codes - a_fit @ coef
    enob_fit = math.log2(m / (np.std(resid_lsb) * math.sqrt(12))) if np.std(resid_lsb) > 0 else float(n)
    out_of_range = int(np.sum((vin_s < 0) | (vin_s > vref)))
    rails = int(np.sum((codes == 0) | (codes == m - 1)))

    # ------------------------------------------------------------------ design hint: M2 gate
    dyn_alt = None
    if dyn and dyn["tone_fraction"] > 0.9:
        adc_alt = SARADC(dict(p, M2_GATE_TO_G=not p["M2_GATE_TO_G"]))   # same seed -> same "chip"
        dyn_alt = dynamic_metrics(adc_alt.simulate(vin_func, n_conv, t0=t_start)["code"], fs, n)

    # ------------------------------------------------------------------ static test
    st = None if a.no_static else static_test(adc, a.hits_per_code)

    # ------------------------------------------------------------------ ideal energy vs code
    ev = ideal_energy_vs_code(p)
    cu, cv2 = p["CU"], p["CU"] * vref ** 2

    # ================================================================== CSV 1: digital output
    rows = []
    for i in range(n_conv):
        b = r["bits"][i]
        rows.append([i, f"{r['t_s'][i]:.9e}", f"{vin_s[i]:.6f}", f"{r['vheld'][i]:.6f}", int(codes[i]),
                     "".join(str(int(x)) for x in b)] + [int(x) for x in b]
                    + [f"{vrec[i]:.6f}", f"{err[i] / adc.lsb:.4f}",
                       f"{(r['e_ref'][i] + r['e_reset_ref'][i]) * 1e12:.4f}", f"{r['e_total'][i] * 1e12:.4f}",
                       int(r["meta"][i])])
    write_csv(os.path.join(a.outdir, "task1_digital_output.csv"),
              ["n", "t_sample_s", "Vin_ideal_V", "V_held_V", "code", "code_binary"]
              + [f"b{n - 1 - j}" for j in range(n)]
              + ["Vout_reconstructed_V", "error_LSB", "E_CDAC_pJ", "E_total_pJ", "metastable_decisions"], rows)

    # ================================================================== CSV 2: bit waveforms
    hdr, wrows = build_waveforms(adc, r, vin_func, keep, a.pts_per_cycle)
    write_csv(os.path.join(a.outdir, "task1_bit_waveforms.csv"), hdr, wrows)

    # ================================================================== CSV 3: INL/DNL
    if st:
        rows = []
        for c in range(m):
            tr = st["trans"][c - 1] if c >= 1 else None
            dn = st["dnl"][c - 1] if 1 <= c <= m - 2 else None
            il = st["inl"][c - 1] if c >= 1 else None
            rows.append([c, int(st["hits"][c]), fmt(tr, 7), fmt(dn, 5), fmt(il, 5)])
        write_csv(os.path.join(a.outdir, "task1_inl_dnl.csv"),
                  ["code", "hits", "transition_V(code-1->code)", "DNL_LSB", "INL_LSB"], rows)

    # ================================================================== CSV 4: spectrum
    if dyn:
        write_csv(os.path.join(a.outdir, "task1_spectrum.csv"), ["frequency_Hz", "magnitude_dBFS"],
                  [[f"{fq:.6g}", f"{s:.3f}"] for fq, s in zip(dyn["freq"], dyn["spec_dbfs"])])

    # ================================================================== CSV 5: energy vs code
    rows = []
    for c in range(m):
        rows.append([c, format(c, f"0{n}b"), f"{ev['e_conv'][c]:.4f}", f"{ev['e_reset'][c]:.4f}",
                     f"{ev['e_cycle'][c]:.4f}", f"{ev['e_cycle'][c] * cv2 * 1e15:.3f}", f"{ev['e_textbook'][c]:.4f}"])
    write_csv(os.path.join(a.outdir, "task1_energy_vs_code.csv"),
              ["code", "binary", "E_bitcycling_CVref2", "E_reset_CVref2", "E_per_cycle_CVref2",
               "E_per_cycle_fJ", "E_textbook_conventional_CVref2"], rows)

    # ================================================================== CSV 6: summary
    S = []

    def add(sec, name, val, unit="", note=""):
        S.append([sec, name, val if isinstance(val, str) else fmt(val), unit, note])

    sec = "Configuration"
    add(sec, "Architecture", "10-bit SAR, single-ended, top-plate sampled, conventional switching")
    add(sec, "Resolution", n, "bit")
    add(sec, "VDD", p["VDD"], "V")
    add(sec, "Vref (input range 0..Vref)", vref, "V")
    add(sec, "Vcm (comparator Vinn)", adc.vcm, "V", "tie Vinn of the StrongARM to this")
    add(sec, "Sampling rate fs", fs, "S/s")
    add(sec, "Bit-clock frequency", 1 / adc.t_b, "Hz", f"{adc.n_cycles} cycles per conversion")
    add(sec, "LSB", adc.lsb * 1e3, "mV")
    add(sec, "Unit capacitor Cu", cu * 1e15, "fF")
    add(sec, "Unit caps in array", int(adc.cdac.units.sum()), "Cu", "512+256+...+1 + 1 dummy")
    add(sec, "Total CDAC capacitance (with mismatch)", adc.cdac.C.sum() * 1e12, "pF")
    add(sec, "Top-plate parasitic", adc.cdac.Cp * 1e15, "fF")
    att = adc.cdac.C.sum() / adc.c_s
    add(sec, "Effective input range", f"{adc.vcm - att * vref / 2:.4f} .. {adc.vcm + att * vref / 2:.4f}", "V",
        f"Vcm +/- {att:.4f} x Vref/2 (top-plate parasitic attenuation); beyond this the codes clip")
    add(sec, "Input source", src)
    add(sec, "Conversions simulated", n_conv)
    add(sec, "Ideal mode", "yes" if a.ideal else "no")

    sec = "Power"
    for name, val in energy_mean.items():
        add(sec, f"{name} - power", val * fs * 1e6, "uW", f"{val * 1e12:.4g} pJ/conversion; {100 * val / e_conv_total:.1f}% of dynamic")
    add(sec, "Static leakage", p["P_LEAK"] * 1e6, "uW")
    add(sec, "TOTAL POWER", p_total * 1e6, "uW")
    add(sec, "Energy per conversion", (p_total / fs) * 1e12, "pJ")
    e_ref_tot = energy_mean["CDAC switching (Vref) - bit cycling"] + energy_mean["CDAC switching (Vref) - reset/sampling"]
    add(sec, "Drawn from Vref (CDAC)", e_ref_tot * fs * 1e6, "uW")
    add(sec, "Drawn from VDD (everything else + leakage)", (e_conv_total - e_ref_tot) * fs * 1e6 + p["P_LEAK"] * 1e6, "uW")

    sec = "Dynamic performance"
    if dyn and dyn["tone_fraction"] > 0.9:
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
        add(sec, "FFT metrics", "n/a", "", "input is not a single tone - see error-based metrics")
    add(sec, "RMS error vs ideal input", np.sqrt(np.mean(err ** 2)) / adc.lsb, "LSB", "includes offset & gain error")
    add(sec, "Effective resolution after gain/offset fit", enob_fit, "bit", "valid for any input waveform")
    add(sec, "Out-of-range input samples", out_of_range, "", "below 0 V or above Vref")
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
        add(sec, "Offset error at first transition (0->1)", st["offset_lsb"], "LSB",
            "includes the gain error, which pivots about Vcm")
        add(sec, "Gain error", st["gain_lsb"], "LSB", f"{st['gain_pct']:+.3f}% (top-plate parasitic attenuation)")
    else:
        add(sec, "Static test", "skipped")

    sec = "Figures of merit"
    if dyn and dyn["tone_fraction"] > 0.9:
        add(sec, "Walden FoM = P/(2^ENOB*fs)", p_total / (2 ** dyn["enob_fs"] * fs) * 1e15, "fJ/conv-step", "uses full-scale ENOB")
        add(sec, "Schreier FoM = SNDR + 10log(fs/2/P)", dyn["sndr"] - dyn["signal_dbfs"] + 10 * math.log10(fs / 2 / p_total), "dB", "full-scale SNDR")
    add(sec, "Energy per conversion", p_total / fs * 1e12, "pJ")

    sec = "Timing"
    add(sec, "Conversion period", adc.t_frame * 1e6, "us")
    add(sec, "Bit-clock period", adc.t_b * 1e9, "ns")
    add(sec, "Track (sampling) time", adc.t_track * 1e9, "ns")
    add(sec, "Latency: sampling instant -> data valid", (adc.t_track + n * adc.t_b - adc.clk.t_sample) * 1e9, "ns")
    add(sec, "PHI1 -> PHI2 non-overlap", adc.clk.nonoverlap_12 * 1e12, "ps")
    add(sec, "PHI2 -> PHI1 non-overlap", adc.clk.nonoverlap_21 * 1e12, "ps")
    add(sec, "Comparator decision time (median / max)",
        f"{np.median(r['tdec']) * 1e12:.0f} / {r['tdec'].max() * 1e12:.0f}", "ps", f"time available {adc.t_cmp_avail * 1e9:.4g} ns")
    t_half = p["TAU_DAC"] * math.log(2 ** (n - 1))
    add(sec, "DAC settling to 0.5 LSB (largest top-plate step Vref/4)", t_half * 1e9, "ns",
        f"available {adc.t_settle * 1e9:.4g} ns")
    add(sec, "Track-mode time constant (median)", float(np.median(r["tau_track"])) * 1e9, "ns")

    sec = "Block characteristics"
    vgrid = np.array([0.0, vref / 4, vref / 2, 3 * vref / 4, vref])
    ron = adc.sw.ron(vgrid)
    add(sec, "Bootstrap ratio Cb/(Cb+C_G)", adc.sw.alpha)
    vg3 = np.array([0.0, vref / 2, vref])
    inj = p["INJ_FRACTION"] * p["M1_W"] * p["M1_L"] * p["COX"] * np.maximum(adc.sw.vov(vg3), 0) / adc.c_s
    add(sec, "M1 charge-injection step at Vin = 0 / Vref/2 / Vref", " / ".join(f"{x / adc.lsb:.2f}" for x in inj),
        "LSB", "the variation across the range is what distorts; scale M1_W with the sampling capacitance")
    add(sec, "M1 gate cap", adc.sw.c_gate * 1e15, "fF")
    if math.isinf(adc.sw.vb_max):
        add(sec, "Node B ceiling", "none - M2 gate tied to G, V_B follows Vin")
    else:
        add(sec, "Node B ceiling (M2 gated by PHI)", adc.sw.vb_max, "V", "above this Vin the boost degrades")
    add(sec, "M1 Ron at Vin = 0 / Vref/2 / Vref", f"{ron[0]:.1f} / {ron[2]:.1f} / {ron[4]:.1f}", "ohm")
    add(sec, "Track bandwidth at Vin = Vref/2", 1 / (2 * math.pi * ron[2] * adc.c_s) / 1e6, "MHz")
    add(sec, "kT/C noise", adc.ktc_sigma * 1e6, "uV rms")
    add(sec, "Bootstrapped switch energy/sample", float(np.mean(r["energy"]["Bootstrapped switch"])) * 1e15, "fJ")
    add(sec, "Comparator offset (this instance)", adc.cmp.vos * 1e3, "mV", f"Pelgrom sigma = {adc.cmp.sigma_os * 1e3:.3g} mV")
    add(sec, "Comparator noise", p["CMP_NOISE_RMS"] * 1e3, "mV rms")
    add(sec, "Comparator energy/decision", adc.cmp.e_decision * 1e15, "fJ")
    rel = (adc.cdac.C / (p["CU"] * adc.cdac.units) - 1) * 100
    add(sec, "CDAC cap errors b9..b0 (%)", " ".join(f"{x:+.3f}" for x in rel[:-1]), "%")
    add(sec, "Clock-generator energy/frame", adc.clk.energy_per_frame * 1e15, "fJ")

    sec = "CDAC switching-energy study (ideal array)"
    add(sec, "Average energy per cycle (this scheme)", ev["e_cycle"].mean(), "C*Vref^2", "bit cycling + reset, single-ended")
    add(sec, "  of which bit cycling", ev["e_conv"].mean(), "C*Vref^2")
    add(sec, "  of which reset/sampling", ev["e_reset"].mean(), "C*Vref^2")
    add(sec, "Textbook conventional (bottom-plate sampled)", ev["e_textbook"].mean(), "C*Vref^2", "single-ended")
    add(sec, "Differential equivalent (x2)", 2 * ev["e_cycle"].mean(), "C*Vref^2",
        f"closed form (Ginsburg/Liu) = {closed_form_conventional_diff(n):.1f}")
    add(sec, "Max / min energy over all codes", f"{ev['e_cycle'].max():.1f} / {ev['e_cycle'].min():.1f}", "C*Vref^2")
    add(sec, "Average in joules at this Cu, Vref", ev["e_cycle"].mean() * cv2 * 1e12, "pJ")

    sec = "Design hints"
    if dyn_alt:
        now, alt = ("gate tied to G", "gate driven by PHI") if p["M2_GATE_TO_G"] else ("gate driven by PHI (as drawn)", "gate tied to G")
        hd2 = lambda d: d["harmonics"][0][2]
        add(sec, f"Bootstrapped switch M2 {now}", f"SFDR {dyn['sfdr']:.1f} dBc, THD {dyn['thd']:.1f} dBc, HD2 {hd2(dyn):.1f} dBc")
        add(sec, f"Same chip with M2 {alt}", f"SFDR {dyn_alt['sfdr']:.1f} dBc, THD {dyn_alt['thd']:.1f} dBc, HD2 {hd2(dyn_alt):.1f} dBc",
            "", "with PHI on M2's gate node B cannot pass Vin above VDD - Vth, so the boost and Ron degrade near full scale")
    add(sec, "Comparator reference", f"tie Vinn to Vcm = {adc.vcm:.3g} V; the MSB plate sits at Vref during PHI1")
    add(sec, "Driver polarity", "bit drivers are inverters: drive input = NOT(bottom-plate state)")

    sec = "Notes"
    add(sec, "Model", "behavioural/circuit-aware; replace DEFAULT_PARAMS with your Spectre numbers")

    write_csv(os.path.join(a.outdir, "task1_summary.csv"), ["section", "parameter", "value", "unit", "note"], S)

    # ------------------------------------------------------------------ console
    print(f"\nTask 1 - conventional SAR ADC ({n}-bit, fs = {fs:g} S/s, VDD = {p['VDD']} V, Cu = {cu * 1e15:g} fF)")
    print(f"  input           : {src}")
    if dyn and dyn["tone_fraction"] > 0.9:
        print(f"  SNDR / SFDR     : {dyn['sndr']:.2f} dB / {dyn['sfdr']:.2f} dBc  ->  ENOB {dyn['enob']:.2f} bit "
              f"({dyn['enob_fs']:.2f} at FS)")
    if st:
        print(f"  DNL / INL       : {st['dnl'].min():+.2f}..{st['dnl'].max():+.2f} / "
              f"{st['inl'].min():+.2f}..{st['inl'].max():+.2f} LSB, missing codes {st['missing']}")
    print(f"  total power     : {p_total * 1e6:.4g} uW   ({p_total / fs * 1e12:.4g} pJ/conversion)")
    for name, val in sorted(energy_mean.items(), key=lambda kv: -kv[1]):
        print(f"     {name:42s} {val * fs * 1e6:9.4f} uW")
    if dyn and dyn["tone_fraction"] > 0.9:
        print(f"  Walden FoM      : {p_total / (2 ** dyn['enob_fs'] * fs) * 1e15:.2f} fJ/conv-step")
    print(f"  CDAC check      : {2 * ev['e_cycle'].mean():.1f} C*Vref^2 differential-equivalent "
          f"(literature 1363.3)")
    if rails > 0.005 * n_conv:
        print(f"  WARNING         : {100 * rails / n_conv:.1f}% of samples sit at code 0 or {m - 1} - the input exceeds "
              f"the effective range (see task summary); reduce the amplitude or use --autoscale")
    print(f"  CSV files in    : {os.path.abspath(a.outdir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
