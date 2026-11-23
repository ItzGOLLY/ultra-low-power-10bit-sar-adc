# Ultra-Low-Power 10-bit SAR ADC: Conventional vs V<sub>cm</sub>-Based Capacitor Switching

[![models](https://github.com/ItzGOLLY/ultra-low-power-10bit-sar-adc/actions/workflows/models.yml/badge.svg)](https://github.com/ItzGOLLY/ultra-low-power-10bit-sar-adc/actions/workflows/models.yml)
![Cadence Virtuoso](https://img.shields.io/badge/Cadence-Virtuoso%20%7C%20ADE%20%7C%20ViVA-red)
![PDK](https://img.shields.io/badge/PDK-gpdk090%20(90%20nm)-blue)
![Verilog-A](https://img.shields.io/badge/HDL-Verilog--A-purple)
![Python](https://img.shields.io/badge/Python-NumPy-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

A 10-bit, 100 kS/s successive-approximation ADC for battery-powered wireless sensor nodes, designed in
**Cadence Virtuoso (gpdk090, 90 nm)** and used to compare two capacitor-DAC switching schemes on the same
front end: **conventional charge redistribution** and the **V<sub>cm</sub>-based switching** of Zhu et al.
(IEEE JSSC 2010).

The circuit blocks (bootstrapped sampling switch, StrongARM comparator, SR latch, non-overlapping clock
generator, capacitive DACs and the V<sub>cm</sub>-based switch logic) were drawn as schematics. Both converters
were evaluated with a circuit-aware behavioural model and a **Verilog-A model of the complete ADC**
simulated in ADE/Spectre. The waveforms were exported to Cadence's VCSV format and inspected in ViVA.

<p align="center">
  <img src="docs/images/block_diagram.png" width="780" alt="Block diagram of the 10-bit SAR ADC">
</p>

## Key results

| Metric @ 100 kS/s, 1.8 V | Task 1: Conventional | Task 2: V<sub>cm</sub>-based | Change |
|---|---|---|---|
| CDAC array | 1024 C<sub>u</sub> = 10.24 pF | 512 C<sub>u</sub> = 5.12 pF | **−50 % area** |
| Average CDAC switching energy | 681.7 C<sub>u</sub>V<sub>ref</sub>² | 85.1 C<sub>u</sub>V<sub>ref</sub>² | **−87.5 %** |
| Total power | 2.77 µW | 0.86 µW | **−69 %** |
| Energy per conversion | 27.7 pJ | 8.6 pJ | −69 % |
| Walden FoM | 31.8 fJ/conv-step | **10.0 fJ/conv-step** | 3.2× better |
| Schreier FoM | 163.1 dB | 168.0 dB | +4.9 dB |
| SNDR / ENOB (full scale) | 59.69 dB / 9.77 bit | 59.55 dB / 9.74 bit | same resolution |
| SFDR / THD | 79.4 dBc / −77.5 dBc | 72.7 dBc / −72.3 dBc | see [design insights](#design-insights) |
| DNL / INL (code density) | ±0.27 / +0.31…−0.16 LSB | ±0.27 / +0.54…−0.10 LSB | no missing codes |

Task 1 was tested with a 48.90 kHz (near-Nyquist) coherent sine and Task 2 with a 10.96 kHz sine read from a
waveform file, both with 4096 conversions. Static linearity comes from a 34 735-conversion ramp
(code-density test). All numbers come from the behavioural/Verilog-A models
([limitations](#limitations-and-future-work)) and are reproduced by the scripts in this repository.

## Contents

- [Architecture](#architecture)
- [Switching schemes](#switching-schemes)
- [Cadence Virtuoso schematics](#cadence-virtuoso-schematics-gpdk090)
- [Verilog-A model and Virtuoso testbench](#verilog-a-model-and-virtuoso-testbench)
- [Results](#results)
- [Design insights](#design-insights)
- [Reproduce the results](#reproduce-the-results)
- [Repository structure](#repository-structure)
- [Limitations and future work](#limitations-and-future-work)
- [Project report](#project-report)

## Architecture

| Parameter | Value |
|---|---|
| Resolution / architecture | 10 bit, single-ended charge-redistribution SAR, top-plate sampling |
| Supply / reference | V<sub>DD</sub> = V<sub>ref</sub> = 1.8 V (input range 0–1.8 V), V<sub>cm</sub> = 0.9 V |
| Sampling rate / bit clock | 100 kS/s; 1.2 MHz, 12 cycles per conversion (1 track + 10 bits + EOC) |
| LSB / unit capacitor | 1.758 mV / C<sub>u</sub> = 10 fF |
| Comparator | StrongARM latch + NAND SR latch |
| Sampling switch | Bootstrapped NMOS switch (C<sub>b</sub> = 1 pF) |
| Schematic technology | Cadence gpdk090, `nmos1v` / `pmos1v` |

The input is sampled onto the CDAC top plate (V<sub>x</sub>) by a bootstrapped switch. The StrongARM comparator
compares V<sub>x</sub> with V<sub>cm</sub> once per bit cycle, and the SR latch holds each decision while the
comparator resets. The SAR logic then sets the next trial value through the bottom-plate drivers. After the
last bit, the code is copied to D<sub>9</sub>–D<sub>0</sub> and EOC is raised. A NOR-based non-overlapping
clock generator produces the track (PHI1) and convert (PHI2) phases with a gap of about 160 ps.

<p align="center"><img src="docs/images/conversion_timing.png" width="700" alt="Timing of one conversion"></p>

## Switching schemes

Both designs share every block except the CDAC and its switch control:

| | Conventional (Task 1) | V<sub>cm</sub>-based (Task 2) |
|---|---|---|
| Array | 512C … 1C + 1C dummy | 256C … 1C + 1C dummy |
| Bottom plates while sampling | 512C at V<sub>ref</sub>, rest at GND | all at V<sub>cm</sub> |
| Bottom-plate levels | V<sub>ref</sub>, GND | V<sub>ref</sub>, V<sub>cm</sub>, GND |
| MSB decision | no switching | no switching |
| Plate moves per conversion | up to 18 (trial + undo) | 9, never undone |
| Extra requirement | – | a V<sub>cm</sub> reference that sources and sinks charge |

The energy drawn from V<sub>ref</sub> is found by charge conservation at every bit cycle. For an N-bit
differential array with equally likely codes, the averages are:

$$E_{conv}=\sum_{i=1}^{N}2^{N+1-2i}\left(2^{i}-1\right)C_uV_{ref}^2 \qquad E_{V_{cm}}=\sum_{i=1}^{N-1}2^{N-2-2i}\left(2^{i}-1\right)C_uV_{ref}^2$$

For N = 10 these give 1363.3 and 170.2 C<sub>u</sub>V<sub>ref</sub>² (681.7 and 85.1 for this single-ended
array). The models reproduce both values exactly over all 1024 codes.

<p align="center"><img src="docs/images/three_bit_switching_example.png" width="640" alt="Three-bit example of both switching schemes"></p>

## Cadence Virtuoso schematics (gpdk090)

All cells live in one Virtuoso library. The logic gates are reused inside the larger blocks.
[cadence/README.md](cadence/README.md) has the full cell list and the testbench settings.

| Bootstrapped sampling switch | StrongARM dynamic comparator |
|---|---|
| <img src="cadence/schematics/04_2BootStraped_bootstrapped_switch.jpg" width="420"> | <img src="cadence/schematics/06_2DynamicComparator_StrongARM.jpg" width="420"> |
| **Non-overlapping clock generator** | **NAND SR latch + output buffer** |
| <img src="cadence/schematics/05_2PhaseGenerator_nonoverlap_clock.jpg" width="420"> | <img src="cadence/schematics/08_2SRLatchOTPbuffer.jpg" width="420"> |
| **Capacitive DAC with inverter drivers** | **V<sub>cm</sub>-based split capacitor array** |
| <img src="cadence/schematics/09_2CapDAC_full.jpg" width="420"> | <img src="cadence/schematics/11_capacitive_ADC_vcm_split_array.jpg" width="420"> |
| **V<sub>cm</sub>-based switching logic** | **Preamp + latch comparator (base-paper type)** |
| <img src="cadence/schematics/14_Switch_Logic.jpg" width="420"> | <img src="cadence/schematics/07_project_preamp_latch_comparator.jpg" width="420"> |

The supporting cells (inverter, NOR2, NAND3, P/N switch-control slices, 10-bit output buffer, resistor-ladder
reconstruction DAC and op-amp) are in [`cadence/schematics/`](cadence/schematics).

The V<sub>cm</sub>-based switch control uses one P-slice and one N-slice per capacitor:

| AMP | L | D | P side | N side | Phase |
|---|---|---|---|---|---|
| 1 | × | × | V<sub>in</sub> | V<sub>in</sub> | sampling |
| 0 | 0 | × | V<sub>cm</sub> | V<sub>cm</sub> | waiting for its bit |
| 0 | 1 | 1 | GND | V<sub>DD</sub> | decided, comparator = 1 |
| 0 | 1 | 0 | V<sub>DD</sub> | GND | decided, comparator = 0 |

## Verilog-A model and Virtuoso testbench

[`models/veriloga/sar_adc_10b.va`](models/veriloga/sar_adc_10b.va) is an event-driven model of the whole
converter with the pins `Vin, CLK, VDD, GND, Vref, Vb, Dout[9:0], Done`. It works as follows:

- **Scheme:** `scheme = 0` selects conventional switching and `scheme = 1` selects V<sub>cm</sub>-based switching.
- **Charge conservation:** the model solves the top-plate voltage by charge conservation at every bit cycle.
- **Optional non-idealities:** unit-capacitor mismatch, top-plate parasitic, comparator offset and noise, and kT/C noise.
- **Supply currents:** the charge drawn at each edge becomes a short current pulse into `Vref`, `Vb` or `VDD`, so ADE can measure supply power the same way as a transistor-level run.
- **Run summary:** at the end of every transient run, the Spectre log gets the energy per conversion, power, ENOB, gain and offset errors, and the Walden FoM.

Before it was used in Virtuoso, the model was checked in gnucap and against an independent re-implementation of
its event logic. All 130 test conversions gave ideal codes, and the supply charge matched the analytical value
within 0.03 %.

<p align="center"><img src="cadence/schematics/18_Test_SAR_10_bit_testbench.jpg" width="640" alt="Verilog-A testbench Test_SAR_10_bit"></p>

## Results

### Waveforms in the Virtuoso waveform viewer (ViVA)

| Task 1: conventional, bottom plates show "set and undo" | Task 2: V<sub>cm</sub>-based, each plate moves once from V<sub>cm</sub> |
|---|---|
| <img src="results/figures/viva_task1_bottom_plates.jpg" width="420"> | <img src="results/figures/viva_task2_bottom_plates.jpg" width="420"> |

### One conversion, signal by signal

| Task 1, code 293 | Task 2, code 911 |
|---|---|
| <img src="results/figures/task1_one_conversion_code293.png" width="420"> | <img src="results/figures/task2_one_conversion_code911.png" width="420"> |

### Dynamic performance (4096-point coherent FFT)

| Task 1: 48.90 kHz input | Task 2: 10.96 kHz input |
|---|---|
| <img src="results/figures/spectrum_task1_conventional.png" width="420"> | <img src="results/figures/spectrum_task2_vcm.png" width="420"> |

### Static linearity (code-density test)

| Task 1 | Task 2 |
|---|---|
| <img src="results/figures/dnl_inl_task1_conventional.png" width="420"> | <img src="results/figures/dnl_inl_task2_vcm.png" width="420"> |

### Power and switching energy

| Power breakdown at 100 kS/s | CDAC energy for every output code |
|---|---|
| <img src="results/figures/power_breakdown.png" width="420"> | <img src="results/figures/energy_vs_code.png" width="420"> |

| Block (µW) | Conventional | V<sub>cm</sub>-based |
|---|---|---|
| CDAC switching from V<sub>ref</sub>: bit cycling | 1.350 | 0.263 |
| CDAC switching from V<sub>ref</sub>: reset and sampling | 0.830 | 0.000 |
| SAR logic | 0.413 | 0.412 |
| StrongARM comparator | 0.106 | 0.106 |
| Clock generator, SR latch, switch, drivers, leakage | 0.073 | 0.079 |
| **Total** | **2.772** | **0.859** |

### Comparison with the base paper

| | Zhu et al., JSSC 2010 [1] | This work, Task 1 | This work, Task 2 |
|---|---|---|---|
| Result type | measured silicon | simulated (behavioural) | simulated (behavioural) |
| Technology / supply | 90 nm CMOS / 1.2 V | gpdk090 schematics / 1.8 V model | gpdk090 schematics / 1.8 V model |
| Signal path / rate | differential / 100 MS/s | single-ended / 100 kS/s | single-ended / 100 kS/s |
| Switching scheme | V<sub>cm</sub>-based | conventional | V<sub>cm</sub>-based |
| Total CDAC capacitance | 2.75 pF | 10.24 pF | 5.12 pF |
| ENOB | 9.1 bit (1.8 MHz input) | 9.77 bit (full scale) | 9.74 bit (full scale) |
| Power / energy per conversion | 3 mW / 30 pJ | 2.77 µW / 27.7 pJ | 0.86 µW / 8.6 pJ |
| Walden FoM | 77 fJ/conv-step | 31.8 fJ/conv-step | 10.0 fJ/conv-step |

## Design insights

- **The energy bottleneck moves from the array to the logic.** In Task 1 the CDAC draws 79 % of the power.
  After V<sub>cm</sub>-based switching, the SAR logic (0.41 µW) is the largest consumer. The next step is
  dynamic or asynchronous logic.
- **The V<sub>cm</sub> reference costs very little if it can recycle charge.** An ideal decoupled V<sub>cm</sub>
  carries almost no net charge (about 0.6 nW). A buffer that can only source charge adds up to 0.30 µW, giving
  1.16 µW total. That is still 58 % below Task 1, at a Walden FoM of 13.5 fJ/conv-step.
- **Distortion comes from the sampling switch, not the switching scheme.** The schematic drives the gate of
  the bootstrapped switch's input-side transistor (M2) from the clock phase. Above about 1.13 V it can no
  longer pass the input, so the boost weakens near full scale. Tying M2's gate to the boosted node raises
  SFDR to about 84 dBc in both designs (Task 2: HD2 −72.7 → −96.6 dBc).
- **The smaller array has a cost.** Halving it barely raises kT/C noise (20 → 28 µV rms, under 2 % of an
  LSB). But the fixed top-plate parasitic becomes a larger fraction of the array, so the gain error grows
  (−1.48 % → −1.75 %, matching C<sub>p</sub>/(C<sub>DAC</sub>+C<sub>p</sub>)). The input source also sees half
  the load.
- **There is plenty of timing margin.** The slowest comparator decision takes under 1 ns of the 417 ns
  available, and the largest DAC step settles to ½ LSB in 12.5 ns, so the same blocks could run much faster.

## Reproduce the results

Requirements: Python ≥ 3.8 with NumPy (and Matplotlib for the waveform figures).

```bash
pip install -r requirements.txt
make all        # runs both models, draws the waveforms, writes VCSV files and checks the key metrics
```

Or step by step:

```bash
python models/python/task1_sar_adc_conventional.py --outdir results/task1        # conventional, 48.9 kHz coherent sine
python models/python/task2_sar_adc_vcm_based.py   --outdir results/task2        # Vcm-based, demo 10.96 kHz waveform file
python models/python/task2_sar_adc_vcm_based.py   --input my_tran.csv --autoscale   # convert your own Virtuoso CSV export
python models/python/csv_to_waveforms.py --bits results/task1/task1_bit_waveforms.csv \
       --digital results/task1/task1_digital_output.csv --outdir results/task1/waveforms   # PNG + .vcsv for ViVA
python tests/check_results.py results                                            # compare against the reference numbers
```

Every model parameter can be changed through `--params my.json`, and `--ideal` switches off all
non-idealities. [models/README.md](models/README.md) lists all options, and
[cadence/README.md](cadence/README.md) explains how to run the Verilog-A model in ADE and open the VCSV files
in ViVA. The full VCSV waveform sets are also attached to the
[GitHub release](https://github.com/ItzGOLLY/ultra-low-power-10bit-sar-adc/releases).

## Repository structure

```
├── models/
│   ├── python/
│   │   ├── task1_sar_adc_conventional.py   # circuit-aware model, conventional switching
│   │   ├── task2_sar_adc_vcm_based.py      # circuit-aware model, Vcm-based switching (+ conventional energy on the same input)
│   │   └── csv_to_waveforms.py             # CSV -> waveform figures and Cadence VCSV/ASCII files
│   └── veriloga/sar_adc_10b.va             # Verilog-A model of the full ADC for Virtuoso (ADE + Spectre)
├── cadence/
│   ├── README.md                           # cell list, testbench sources, ADE settings, ViVA import
│   └── schematics/                         # gpdk090 schematic screenshots of every cell
├── results/
│   ├── task1/, task2/                      # CSV outputs (summary, digital output, bit waveforms, INL/DNL, spectrum, energy vs code)
│   └── figures/                            # ViVA screenshots, spectra, DNL/INL, power and energy plots
├── tests/check_results.py                  # regression check of the key metrics (used by CI)
├── docs/
│   ├── images/                             # block diagram, timing, design flow, switching example
│   └── report/SAR_ADC_Project_Report.pdf   # full project report
├── Makefile
└── requirements.txt
```

## Limitations and future work

- **Behavioural results.** The performance numbers come from the circuit-aware behavioural model and the
  Verilog-A model, not from a transistor-level transient run of the complete converter. The comparator, logic
  and clock energies are estimates from node capacitances.
- **Schematics not yet sized.** The schematics use the gpdk090 default device size (W = 120 nm, L = 100 nm)
  and placeholder 1 pF capacitors. The sizes and capacitances used in the models are the targets for the next
  iteration. The model assumes a 1.8 V process, while the schematics use the kit's low-voltage core devices.
- **Simple parasitic and signal-path model.** Layout parasitics are modelled as one lumped top-plate
  capacitance, and the converter is single-ended.

Next steps: size each gpdk090 cell and characterise it in Spectre, then run the full converter at transistor
level using the Verilog-A model as the golden reference. After that, verify the M2-gate fix, design the
V<sub>cm</sub> buffer, and move to dynamic or asynchronous SAR logic. The last steps are layout, extraction
and Monte Carlo mismatch analysis.

## Project report

The complete report: [`docs/report/SAR_ADC_Project_Report.pdf`](docs/report/SAR_ADC_Project_Report.pdf)
(99 pages). It covers the literature survey, methodology, every schematic, results, discussion, the Verilog-A
source and the simulation settings.

**Author:** Aarush Jagannathan, B.Tech Electronics and Computer Engineering, VIT Chennai. The project was
carried out with teammate **Abhinav Chavva** under the guidance of **Dr. Manikandan P** (School of
Electronics Engineering, VIT Chennai).

## Key references

1. Y. Zhu et al., "A 10-bit 100-MS/s reference-free SAR ADC in 90 nm CMOS," *IEEE JSSC*, vol. 45, no. 6, pp. 1111–1121, 2010 (base paper).
2. C.-C. Liu et al., "A 10-bit 50-MS/s SAR ADC with a monotonic capacitor switching procedure," *IEEE JSSC*, vol. 45, no. 4, 2010.
3. B. P. Ginsburg and A. P. Chandrakasan, "An energy-efficient charge recycling approach for a SAR converter with capacitive DAC," *ISCAS*, 2005.
4. M. Saberi et al., "Analysis of power consumption and linearity in capacitive DACs used in SAR ADCs," *IEEE TCAS-I*, vol. 58, no. 8, 2011.
5. B. Razavi, "The StrongARM latch" and "The bootstrapped switch," *IEEE Solid-State Circuits Magazine*, 2015.
6. IEEE Std 1241-2010, Terminology and Test Methods for Analog-to-Digital Converters.

The full bibliography (31 references) is in the report.

## License

Code (Python models, Verilog-A model, scripts) is released under the [MIT License](LICENSE).
