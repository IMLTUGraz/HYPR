## Repository for Hybrid Propagation (HYPR)  
**From the paper:**  
*A Scalable Hybrid Training Approach for Recurrent Spiking Neural Networks*

---

### Running an Experiment

To run an experiment (e.g., the BRF model on SHD trained with HYPR), use:

```python main.py experiment=brf_shd_hypr data_dir=/path/to/datasets```

Replace `/path/to/datasets/` with the directory where the datasets are located (see below).

You can substitute `brf_shd_hypr` with any other experiment from the folder `config/experiment/`

Run `pip install -U -r requirements.txt` to install required libraries.

Tested with Python 3.12.9

---

### Dataset Setup

Make sure to create a folder for each dataset in your `/path/to/datasets/`:

- **ECG**  
  Create a folder named `ECG` and insert the following files (from Yin et al., 2019 - *Nature Communications*):
  - `QTDB_test.mat`
  - `QTDB_train.mat`

- **SHD**  
  Place the files 
  - `shd_train.h5` 
  - `shd_test.h5` 
  
  in a folder `SHD` in your data directory

- **PATHFINDER-E**  
  Download and unzip it from: https://storage.googleapis.com/long-range-arena/lra_release.gz
  
  Execute the below script after setting the variabless `root_dir` and `target_dir` in it:

  ```python data/process_pathfinder.py```


- **sMNIST**  
  This dataset will be downloaded automatically into a folder `MNIST` in your data directory.


- **sCIFAR**  
  This dataset will be downloaded automatically into a folder `CIFAR` in your data directory.

---

### Config files

We use Hydra (https://hydra.cc) for configuration management. You find all configuration in the config folder. There, the main.yaml is the main file, in which some global parameters are set. You can overwrite any parameter by adding it to the execution command, for example you can overwrite the learning rate:


```python main.py experiment=brf_shd_hypr data_dir=/path/to/datasets training.learning_rate=0.001```

or the random seed

```python main.py experiment=brf_shd_hypr data_dir=/path/to/datasets seed=1```

Experiments from the config/experiments folder only overwrite some of the parameters, most of the parameters are defined in the config/model folder or in config/dataset.

Hydra creates a new directory for each run, which will be located in `./results`.

The training.hypr flag enables or disables training with HYPR. If you look through the experiments (in the config/experiments folder) you find that for each experiment with the _hypr suffix we enable hypr by setting `training.hypr=True`. The experiments with SHD and ECG are relatively fast (mostly 5 minutes to 15 minutes on a recent GPU), sMNIST is slower (about 2 hours) and the slowest are sCIFAR and Pathfinder.

---

### Code Description

The main logic of HYPR is implemented in ```hypr_trainable_base.py``` and ```hypr_helpers.py``` in the folder 'hypr'.

In ```hypr_trainable_base.py```, a wrapper class is implemented to wrap a model such that it can be trained via hypr.

In the ```_generic_hypr_forward``` method, the subsequence is processed sequentially first, then the eligibility matrix at the last time step lambda of the subsequence is computed as in Eq. (11). However, no per-timestep losses are available here yet.

In the ```_generic_hypr_backward``` method, the loss is available since it is back-propagated (through layers, NOT through time) from the subsequent layers. Hence we have the loss signals dL<sup>t</sup>/ds<sup>t</sup> as described in Apendix I. We compute the q-values via the associative scan (see Appendix G) and combine them with the eligibility matrix e<sub>0</sub> as in Eq. (13).

One important thing to note is the ```hypr_args.num_chunks``` argument (see ```config/main.yaml```). It defines in how many subsequences the data is split: if ```hypr_args.num_chunks=1``` the entire time series is treated as one single subsequence (lambda is equal to the full sequence length). Note that the total sequence length must be divisible by ```hypr_args.num_chunks```. For example in SHD the sequence length is 250, so a valid value of ```hypr_args.num_chunks``` would be 25, which results in subsequence length lambda=10, because 250 / 25 = 10. Note that lambda (and hence num_chunks) does NOT influence the result of the training, but it influences the memory and runtime, as shown in Fig. 2.


