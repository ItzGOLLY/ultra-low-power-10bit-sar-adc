# Reproduce the SAR ADC results:  make all   (override the interpreter with  make PY=python3.11)
PY  ?= python3
OUT ?= results
M    = models/python

.PHONY: all task1 task2 waveforms check clean-waveforms

all: task1 task2 waveforms check

task1:
	$(PY) $(M)/task1_sar_adc_conventional.py --outdir $(OUT)/task1

task2:
	$(PY) $(M)/task2_sar_adc_vcm_based.py --outdir $(OUT)/task2

waveforms:
	$(PY) $(M)/csv_to_waveforms.py --bits $(OUT)/task1/task1_bit_waveforms.csv \
	    --digital $(OUT)/task1/task1_digital_output.csv --outdir $(OUT)/task1/waveforms
	$(PY) $(M)/csv_to_waveforms.py --bits $(OUT)/task2/task2_bit_waveforms.csv \
	    --digital $(OUT)/task2/task2_digital_output.csv --outdir $(OUT)/task2/waveforms

check:
	$(PY) tests/check_results.py $(OUT)

clean-waveforms:
	rm -f $(OUT)/task*/waveforms/*.vcsv $(OUT)/task*/waveforms/*_ascii.txt
