# bash script to perform all analysis steps
set -e
#python All_leaf.py
#python Delta_beta_leaf_make_results.py
#python Delta_beta_leaf_plot.py
#python New_figs_leaf.py
#python New_figs_basl.py
#python compare_histograms.py
python Fig1_large_panels.py
python tensor_figure.py
