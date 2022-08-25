# Bash script to make all figures using output of analysis scripts
set -e
python fig1_script.py
python fig2_script.py
python fig3_script.py
python fig4_script.py
python fig_samples_script.py
python fig_ranges_script.py
