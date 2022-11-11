# quartLT22

Supplementary code for the paper:
> Sahel Mohammad Iqbal and Subhankar Mishra. 2023. Neural Networks at a Fraction with Pruned Quaternions. In 6th Joint International Conference on Data Science Management of Data (10th ACM IKDD CODS and 28th COMAD) (CODS-COMAD 2023), January 4–7, 2023, Mumbai, India. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3570991.3570997

This code is to run pruning experiments on quaternion neural networks. The methods to implement quaternion neural networks are borrowed from [hTorch](https://github.com/ispamm/hTorch), and the various pruning experiments are inspired from [open\_lth](https://github.com/facebookresearch/open\_lth).

## Setting up the environment.
Clone the repo and move into the folder. Create a python virtual environment and install PyTorch version specific to the system. Then install packages from `requirements.txt`, followed by the local package in the repo directory.
Example setup:
```
$ git clone https://github.com/Sahel13/pruning_quaternions.git
$ cd pruning_quaternions/
$ python3 -m venv env
$ source env/bin/activate
$ pip install torch torchvision
$ pip install -r requirements.txt
$ pip install -e .
```

## Running experiments.
Before running experiments, you need to specify where to save/load datasets from, and where to store the results of experiments. The default locations are `~/Documents/datasets` and `~/Documents/results` respectively. To change this, modify the methods `dataset_dir()` and `results_dir()` in `utils/misc.py`.

Now to run pruning experiments:
```
$ python prune.py -m model_name -o output_directory -g gpu -n 5
```

The experiments use a GPU by default (defaults to the first GPU). If you would like to run the experiments on CPU, set the `use_gpu` variable to `False` in `prune.py`.

## Visualizing the results.
The results of pruning experiments can be visualized with the methods provided in `utils/plot.py`. Example usage is illustrated in `plot.ipynb`. Create a folder named `images` in the project directory before running the file.

## Extending

### Add a new model.
To add a new model, create a new file in `models/`, and follow the template of `models/lenet_300_100.py`. Both real and quaternion implementations of the model must exist as different classes with names `Real` and `Quat` respectively. Hyper-parameters for training and pruning should be specified in the method `std_hparams()`. The model should then be imported inside `prune.py`.
