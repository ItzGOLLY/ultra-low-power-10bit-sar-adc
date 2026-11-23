# Models

| File | What it is |
|---|---|
| `python/task1_sar_adc_conventional.py` | Circuit-aware model of the 10-bit SAR ADC with **conventional** switching (512C … 1C + dummy, two-level drivers) |
| `python/task2_sar_adc_vcm_based.py` | Same front end with **V<sub>cm</sub>-based** switching (256C … 1C + dummy, three-level drivers). It also runs the input through the conventional procedure so the energy saving for that signal is reported |
| `python/csv_to_waveforms.py` | Turns the CSV outputs into waveform figures (PNG), Cadence **VCSV** files for ViVA and ASCII files for `getAsciiWave()` |
| `veriloga/sar_adc_10b.va` | Verilog-A model of the complete converter for Virtuoso (ADE + Spectre), with selectable scheme. See [../cadence/README.md](../cadence/README.md) |

## How the Python models work

Each conversion is simulated cycle by cycle with the timing of the real circuit (1 track cycle, 10 bit cycles,
1 EOC cycle at a 1.2 MHz bit clock):

```
track:   Vx <- Vin(t_n) + charge injection + kT/C noise ; plates <- sampling positions (scheme dependent)
for bit k = 9 .. 0:
    decide  b_k <- (Vx - Vcm - Vos + noise > 0)                     # StrongARM + SR latch
    plates' <- next positions (conventional or Vcm-based rule)
    Vx'     <- Vx + sum_i C_i (VB_i' - VB_i) / C_T                   # charge conservation, incl. mismatch + parasitic
    E_ref   <- E_ref + Vref * sum_{i at Vref} C_i [(VB_i' - Vx') - (VB_i - Vx)]
end of conversion: code_n; add comparator, logic, clock and switch energies; reset plates
after all conversions: coherent FFT (SNDR, SFDR, THD, ENOB), ramp code-density test (DNL, INL), power, FoM
```

| Block | Modelled effects | Default parameters |
|---|---|---|
| Bootstrapped switch | signal-dependent R<sub>on</sub> with body effect, bootstrap ratio, M2-gate ceiling, track settling | M1 10/0.18 µm, C<sub>b</sub> = 1 pF, C<sub>G</sub> = 5 fF |
| Sampling | charge injection, clock jitter, kT/C noise | 50 % of channel charge, 10 ps rms, 300 K |
| CDAC | unit-capacitor mismatch, top-plate parasitic, bottom-plate settling | σ<sub>u</sub> = 0.5 %, C<sub>p</sub> = 20 fF + 1.2 % of array (+10 fF comparator input), τ = 2 ns |
| StrongARM comparator | Pelgrom offset, input noise, regeneration time, metastability | 4/0.36 µm pair, A<sub>VT</sub> = 5 mV·µm, 0.3 mV rms, τ = 60 ps |
| SR latch, SAR logic, clock generator | gate delays, switched capacitance per toggle, non-overlap timing | INV 40 ps, NAND 60 ps, NOR 80 ps, 3–5 fF per flip-flop node |
| Supplies | charge drawn from V<sub>ref</sub>, V<sub>DD</sub> and V<sub>cm</sub>, leakage | 2 nW leakage |

Every parameter is in `DEFAULT_PARAMS` at the top of each script and can be overridden with a JSON file, e.g.
`{"SIGMA_CU": 0.01, "M2_GATE_TO_G": true, "SEED": 3}` passed with `--params my.json`.

## Command-line options (both models)

| Option | Meaning |
|---|---|
| `--input FILE` | analog waveform to convert (CSV/TXT time, voltage; Virtuoso CSV export or OCEAN `ocnPrint` text, SI suffixes understood) |
| `--tcol / --vcol` | column indices of time and voltage in `--input` |
| `--vin-gain / --vin-offset / --autoscale` | scale or shift the input, or map its min/max to 5–95 % of the range |
| `--fs / --vdd / --vref / --cu` | change the specification |
| `--nfft --fin --amp-dbfs` (Task 1) | built-in coherent sine test |
| `--demo-conversions --demo-fin --demo-amp-dbfs` (Task 2) | demo waveform file generated when no `--input` is given |
| `--hits-per-code`, `--no-static` | ramp code-density test length, or skip it |
| `--ideal` | switch off every non-ideality (sanity check: ideal 10-bit codes) |
| `--params FILE.json`, `--seed N` | override model parameters / random seed (a new "chip") |
| `--outdir DIR` | output folder |

## Outputs

| File | One row is |
|---|---|
| `taskN_summary.csv` | one result: configuration, power breakdown, dynamic and static performance, FoM, timing, block data |
| `taskN_digital_output.csv` | one conversion: sample time, ideal input, held voltage, code, b9–b0, reconstruction, error (LSB), energy |
| `taskN_bit_waveforms.csv` | one time point (40 per bit cycle): clocks, V<sub>x</sub>, comparator outputs, SAR register, bottom plates, output bits |
| `taskN_inl_dnl.csv` | one output code: hits, transition level, DNL, INL |
| `taskN_spectrum.csv` | one FFT bin (dBFS) |
| `taskN_energy_vs_code.csv` | one output code: ideal CDAC energy (Task 2: both schemes and the saving) |
| `analog_input_demo.csv` | (Task 2) the 10.96 kHz demo input waveform |

Note: on macOS with NumPy ≥ 2.0 linked to Apple Accelerate, `RuntimeWarning: ... encountered in matmul`
messages can appear. They come from the BLAS library, and the results are unaffected.
