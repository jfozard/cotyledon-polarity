
## Scripts to analyse cotyledon polarity

### Installation

Analyses were performed on a Ryzen 3600 desktop with 64GB of ram and a 3090FE GPU

Ubuntu 20.04.01 LTS 5.15.0-46-generic #49~20.04.1-Ubuntu SMP

NVIDIA Driver Version: 470.141.03   CUDA Version: 11.4

Needs inkscape (1.0.1), latex (texlive-base/focal,focal,now 2019.20200218-1) and R (r-base-core/focal,now 3.6.3-2 amd64, r-recommended/focal,focal,now 3.6.3-2)

To install:
1. Install anaconda/conda from https://www.anaconda.com/products/distribution
2. Create a conda environment using `conda env create -n cotyledon -f env_simple.yml`
3. Activate this conda environment using `conda activate cotyledon`
4. Edit data_path.py to point to downloaded and extracted dataset (from https://osf.io/ufzj5 )
5. Run initial analysis scripts with `bash all_analysis.sh`
6. Run figure plotting scripts with `bash all_figs.sh`

Output figures generated in output/figures