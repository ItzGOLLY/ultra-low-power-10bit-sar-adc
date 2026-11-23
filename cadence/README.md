# Cadence Virtuoso design (gpdk090)

All cells were drawn in one Virtuoso library with the Cadence generic 90 nm kit (`gpdk090`, devices `nmos1v` /
`pmos1v`, analogLib capacitors, resistors, sources and ideal switches). The logic gates use the global nets
`vdd!` and `gnd!`, so they can be placed in any block without wiring the supply.

## Cells

| Cell | Function | Main components | Schematic |
|---|---|---|---|
| `NOT_Gate` | CMOS inverter | PM0, NM0 | [01](schematics/01_NOT_Gate.jpg) |
| `NOR2_Gate` | 2-input NOR | PM0, PM1 in series; NM0, NM1 in parallel | [02](schematics/02_NOR2_Gate.jpg) |
| `NAND3_Gate` | 3-input NAND | PM0–PM2 in parallel; NM0–NM2 in series | [03](schematics/03_NAND3_Gate.jpg) |
| `2BootStraped` | bootstrapped sampling switch | NM0–NM3, PM0, PM1, C0 = 1 pF | [04](schematics/04_2BootStraped_bootstrapped_switch.jpg) |
| `2PhaseGenerator` | two-phase non-overlapping clock generator | 2 × NOR2, 5 × inverter | [05](schematics/05_2PhaseGenerator_nonoverlap_clock.jpg) |
| `2DynamicComparator` | StrongARM dynamic comparator | NM0–NM4, PM0–PM3 | [06](schematics/06_2DynamicComparator_StrongARM.jpg) |
| `project` | 4-input preamplifier + latch comparator (base-paper type) | NM0–NM10, PM0–PM5, R0/R1 = 1 kΩ, switch S0 | [07](schematics/07_project_preamp_latch_comparator.jpg) |
| `2SRLatchOTPbuffer` | NAND SR latch with output buffer | 2 × NAND2, 2 × inverter | [08](schematics/08_2SRLatchOTPbuffer.jpg) |
| `2CapDAC` | binary CDAC with inverter bottom-plate drivers | C0–C10, one PMOS/NMOS driver per bit | [09](schematics/09_2CapDAC_full.jpg), [10](schematics/10_2CapDAC_detail.jpg) |
| `capacitive_ADC` | switch-level V<sub>cm</sub>-based split array | C0–C6, switches S0–S6 | [11](schematics/11_capacitive_ADC_vcm_split_array.jpg) |
| `P_slice`, `N_slice` | switch control for one capacitor (P and N side) | inverter, 2 × AND3, NOR2 | [12](schematics/12_P_slice.jpg), [13](schematics/13_N_slice.jpg) |
| `Switch_Logic` | switch control of the V<sub>cm</sub>-based array | P_slice and N_slice instances, inverters | [14](schematics/14_Switch_Logic.jpg) |
| `2Buffer10bit` | 10-bit output buffer | 20 × inverter | [15](schematics/15_2Buffer10bit.jpg) |
| `2DAC10bit` | resistor-ladder reconstruction DAC | R0–R19 = 1 kΩ | [16](schematics/16_2DAC10bit_reconstruction.jpg) |
| `2OpAmp` | operational amplifier (candidate V<sub>cm</sub> buffer) | NM0–NM6, PM0, PM2–PM4, I0 | [17](schematics/17_2OpAmp.jpg) |
| `Test_SAR_10_bit` | testbench of the Verilog-A converter | instance I0 of `sar_adc_10b` | [18](schematics/18_Test_SAR_10_bit_testbench.jpg) |

The schematics use the kit's default device size (W = 120 nm, L = 100 nm) and placeholder 1 pF capacitors. The
binary weights (512 C<sub>u</sub> … 1 C<sub>u</sub>, C<sub>u</sub> = 10 fF) and the device sizes used by the models
are the targets for the next, transistor-level iteration.

## Running the Verilog-A model in Virtuoso

1. In the library, create cellview `sar_adc_10b` with view type **veriloga** and paste
   [`../models/veriloga/sar_adc_10b.va`](../models/veriloga/sar_adc_10b.va). Check and save. This compiles the
   model in Spectre (ahdl), and Virtuoso offers to create a symbol.
2. Place the symbol in a testbench (`Test_SAR_10_bit`) and drive the pins from analogLib:

   | Pin | analogLib cell | Setting |
   |---|---|---|
   | VDD, Vref | vdc | 1.8 V |
   | Vb | vdc | 0.9 V (V<sub>ref</sub>/2) |
   | GND | gnd | 0 V |
   | CLK | vpulse | 0 → 1.8 V, period 833.333 ns, width 416.667 ns, rise/fall 1 ns |
   | Vin (sine test) | vsin | offset 0.9 V, amplitude 0.8 V, 10.961914 kHz (449 cycles in 4096 conversions) |
   | Vin (file test) | vpwlf | piecewise-linear file of time–voltage pairs |

3. ADE: transient analysis, stop time 41 ms for 4096 conversions (1 ms for a quick check). Set the instance
   parameters `scheme = 0` (conventional) or `1` (V<sub>cm</sub>-based) and `cu = 10f`, plus optional `cpar`,
   `sigma_cu`, `vos`, `vnoise` and `ktc_noise`. Save Vin, Dout<9:0>, Done and the VDD, Vref and Vb source currents.
4. Power is the average supply current × 1.8 V over a whole number of conversions (waveform calculator). At the
   end of the run, the model prints a summary in the Spectre log: conversions, measured sampling rate, energy
   per conversion by supply, average power, ENOB estimate, gain and offset errors and the Walden FoM.

| Verilog-A parameter | Default | Meaning |
|---|---|---|
| `scheme` | 0 | 0 = conventional (512C … 1C + dummy), 1 = V<sub>cm</sub>-based (256C … 1C + dummy) |
| `vb_is_vcm` | 1 | 1 = the Vb pin is V<sub>cm</sub>; 0 = V<sub>cm</sub> taken internally as V<sub>ref</sub>/2 |
| `cu`, `cpar`, `sigma_cu` | 10 fF, 0, 0 | unit capacitor, top-plate parasitic, unit-cap mismatch |
| `clk_th` | 0.45 | clock threshold as a fraction of V<sub>DD</sub> |
| `vos`, `vnoise`, `ktc_noise` | 0, 0, 0 | comparator offset, rms noise, kT/C noise on/off |
| `td`, `tr_out` | 1 ns, 100 ps | clock-to-output delay, output transition time |
| `e_cmp`, `e_dig`, `i_leak` | 118 fJ, 390 fJ, 1 nA | comparator energy per decision, logic + clock energy per cycle, static current (estimates) |
| `summary`, `verbose`, `log_en` | 1, 0, 0 | print summary, print every conversion, write a CSV log |

## Viewing the waveforms in ViVA

`csv_to_waveforms.py` converts the model outputs to VCSV (six `;`-prefixed header lines, then x, y pairs per
signal). In ViVA use **File → Open Results**, select the `.vcsv` file and plot any signal; markers and the
waveform calculator work as for a simulation result. Signals include `fs, PHI1, PHI1b, PHI2, CLK_cmp, Vin_V,
Vx_V, Vcm_V, outp, outn, Dcmp, Dcmpb, SAR_b9…SAR_b0, BP_<weight>C_<bit>_V, EOC, D9…D0, Dout_code` and, in the
digital-output file, `E_CDAC_pJ, E_total_pJ, V_held_V, Vin_ideal_V, Vout_reconstructed_V, code, error_LSB`.
