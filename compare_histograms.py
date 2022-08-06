
import numpy as np
import sys
from scipy.stats import chi2_contingency, ks_2samp, ttest_ind
from pathlib import Path

def read_data(fn):
    d = []
    with open(fn, 'r') as f:
        try:
            while True:
                d.append(float(next(f)))
        except StopIteration:
            pass
    return d

d_t0 = read_data('output/plot_out/aggregate-t0--alpha.txt')
d_t5 = read_data('output/plot_out/stretch-t5--alpha.txt')
d_35S = read_data('output/plot_out/35S-basl--alpha.txt')
d_basl = read_data('output/plot_out/native-basl--alpha.txt')


r = (80, 100)

def get_counts(a, r):
    s_a =  np.sum((r[0]<=a) & (a < r[1]))
    return [s_a, len(a) - s_a]


def compare_hists(a, b, name_a, name_b, of=sys.stdout):

    counts_a = get_counts(a, r)
    counts_b = get_counts(b, r)


    print(f'contigency table {name_a} vs {name_b} angle range {r}\n', file=of)
    print('{} {} {}\n'.format(name_a, *counts_a), file=of)
    print('{} {} {}\n'.format(name_b, *counts_b), file=of)

    chi2 = chi2_contingency([ counts_a, counts_b ])

    print('chi2 (chi2, p, dof, table) ' +repr(chi2)+'\n', file=of)
    print('chi2 (chi2, p, dof, table) ' +str(chi2)+'\n', file=of)


    ks = ks_2samp(a, b)
    print(f'KS test {name_a} vs {name_b}', ks, file=of)


compare_output = 'output/compare_output/'
Path(compare_output).mkdir(exist_ok=True, parents=True)

of = open(compare_output + 'compare_fig2.txt', 'w')
compare_hists(np.abs(d_t0), np.abs(d_t5), 't0', 't5', of=of)

compare_hists(np.abs(d_t0), np.abs(d_basl), 't0', 'basl', of=of)

compare_hists(np.abs(d_t5), np.abs(d_basl), 't5', 'basl', of=of)

of.close()

