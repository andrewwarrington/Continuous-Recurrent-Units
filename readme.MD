# Fork of: Modeling Irregular Time Series with Continuous Recurrent Units (CRUs)
This repository contains a modified version of the code released as part of 
[_Modeling Irregular Time Series with Continuous Recurrent Units_](https://proceedings.mlr.press/v162/schirmer22a/schirmer22a.pdf) 
by [Mona Schirmer](https://monasch.github.io/), [Mazin Eltayeb](https://www.linkedin.com/in/mazin-eltayeb-199a6a18), 
[Stefan Lessmann](https://www.wiwi.hu-berlin.de/en/Professorships/bwl/wi/personen-en/hl/standardseite-en) 
and [Maja Rudolph](http://maja-rita-rudolph.com/) published at ICML 2022.

This code was modified for use in _Simplified State Space Layers for Sequence Modeling_, 
for which the preprint is available [here](https://arxiv.org/abs/2208.04933).  We 
modify and distribute this version of the code permitted under the terms of the 
original GNU Affero General Public License (Version 3), for which copyright remains 
with the original authors and all original terms apply equally here.  The source 
contains code fragments from further repositories, as listed in `3rd-party-licenses.txt`. 

We modify the original code in three main ways:
- We remove nearly all of the code for applications that we did not consider.
- Added a validation dataset.
- Added basic logging and sanity check outputs for the _Pendulum_ example.

We intend for this version of the code to be used for generating the data for the _CRU 
(ours)_ baseline and the S5 application (in Section 6.3 of the S5 
[paper](https://arxiv.org/abs/2208.04933)).  It also runs and logs the CRU pendulum 
experiment to Weights and Biases (wandb) in a way that is compatible with the S5 
experiments.  This code should not be used for reproduction of the original CRU 
experiments.  

To prepare this repository:
```
git clone https://github.com/lindermanlab/S5.git
cd S5
mkdir -p cache_dir
git checkout pendulum
git clone https://github.com/andrewwarrington/Continuous-Recurrent-Units.git
cd Continuous-Recurrent-Units
pip install -r requirements.txt
wandb login  % If using wandb, follow the on-screen instructions.
```

To then actually run the Pendulum regression example, the use the lightly modified callstring:
```
python run_experiment.py --dataset pendulum --task regression -lsd 30 --sample-rate 0.5 --USE-WANDB 1 --wandb-project $WANDB_PROJ_NAME --wandb-entity $WANDB_USER_NAME --dir_name ./../cache_dir
```
where `$WANDB_PROJ_NAME, $WANDB_USER_NAME` are your wandb project name and username respectively.  Scripts for running the exact experiments in the context of S5 are contained in the S5 repository.

Running this will generate a file `./../cache_dir/pendulum/pend_regression.npz` which contains the train/val/test data used both by this CRU implementation and the S5 implementation.  On a reasonable GPU, the CRU code takes about 90 minutes to execute.

If you have any questions, feel free to reach out! 

-- The S5 Authors 




--- 

# Original Readme: Modeling Irregular Time Series with Continuous Recurrent Units (CRUs)

This repository contains the PyTorch implementation for the paper [Modeling Irregular Time Series with Continuous Recurrent Units](https://arxiv.org/pdf/2111.11344.pdf) by [Mona Schirmer](https://monasch.github.io/), [Mazin Eltayeb](https://www.linkedin.com/in/mazin-eltayeb-199a6a18), [Stefan Lessmann](https://www.wiwi.hu-berlin.de/en/Professorships/bwl/wi/personen-en/hl/standardseite-en) and [Maja Rudolph](http://maja-rita-rudolph.com/) published at ICML 2022.
>Schirmer, M., Eltayeb, M., Lessmann, S., & Rudolph, M. (2022, June). Modeling irregular time series with continuous recurrent units. In International Conference on Machine Learning (pp. 19388-19405). PMLR.
<p align="center">
  <img width="600" src="imgs/overview.PNG">
</p>

## Prerequisites
The code uses Python3 and PyTorch as auto-differentiation package. To set up an environment with the required packages, run
```
conda create -n cru python=3.9.7
conda activate cru
pip install -r requirements.txt
```


## Training and Evaluation
By default, datasets are generated or downloaded and processed when run the first time. 

 - Pendulum interpolation 
```
python run_experiment.py --dataset pendulum --task interpolation -lsd 30 --sample-rate 0.5 --impute-rate 0.5
```

 - Pendulum regression 
```
python run_experiment.py --dataset pendulum --task regression -lsd 30 --sample-rate 0.5
```

 - USHCN interpolation 
```
python run_experiment.py --dataset ushcn --task interpolation -lsd 10 --ts 0.3 --sample-rate 0.5 --unobserved-rate 0.2 --enc-var-activation square --dec-var-activation exp --trans-var-activation relu --grad-clip
```

 - USHCN extrapolation
```
python run_experiment.py --dataset ushcn --task extrapolation -lsd 10 --ts 0.3 --sample-rate 0.5 --unobserved-rate 0.2 --enc-var-activation square --dec-var-activation exp --trans-var-activation relu --grad-clip
```

 - Physionet interpolation
```
python run_experiment.py --dataset physionet --task interpolation -lsd 20 --ts 0.2 --enc-var-activation square --dec-var-activation exp --trans-var-activation relu --grad-clip --num-basis 20 --bandwidth 10 
```

 - Physionet extrapolation 
```
python run_experiment.py --dataset physionet --task extrapolation -lsd 20 --ts 0.2 --enc-var-activation square --dec-var-activation exp --trans-var-activation relu --grad-clip --num-basis 20 --bandwidth 10 --cut-time 24
```

## Running different models

 - CRU
```
python run_experiment.py --dataset pendulum --task regression -lsd 30 --sample-rate 0.5
```

 - f-CRU
```
python run_experiment.py --dataset pendulum --task regression -lsd 30 --sample-rate 0.5 --f-cru --lr 5e-3
```

 - RKN
```
python run_experiment.py --dataset pendulum --task regression -lsd 30 --sample-rate 0.5 --rkn 
```

 - RKN-Delta-t
```
python run_experiment.py --dataset pendulum --task regression -lsd 30 --sample-rate 0.5 --rkn --t-sensitive-trans-net
```


## Acknowledgements and References
The CRU builts on the repository of RKN by Becker et al. (2019):

[Becker, P., Pandya, H., Gebhardt, G., Zhao, C., Taylor, C. J., and Neumann, G. (2019). Recurrent kalman networks: Factorized inference in high-dimensional deep feature spaces. In International Conference on Machine Learning, pages 544-552. PMLR.](https://arxiv.org/pdf/1905.07357.pdf)

Preprocessing of USHCN has been adapted from Brouwer et al. (2019):

[Brouwer, E. D., Simm, J., Arany, A., and Moreau, Y. (2019). Gru-ode-bayes: continuous modeling of sporadically-observed time series. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pages 7379-7390.](https://github.com/edebrouwer/gru_ode_bayes)

Preprocessing of Physionet has been adapted from Rubanova et al. (2019):

[Rubanova, Y., Chen, R. T., and Duvenaud, D. Latent odes for irregularly-sampled time series. In Advances in Neural Information Processing Systems, pp. 5320–5330, 2019](https://github.com/YuliaRubanova/latent_ode)

## License

Modeling Irregular Time Series with Continuous Recurrent Units (CRUs) is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in Modeling Irregular Time Series with Continuous Recurrent Units (CRUs), see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).