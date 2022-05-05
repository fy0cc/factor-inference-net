# Factor Iteration Graph Neural Network

codebase for Factor Iteration Graph Neural Network(figNN)

## Enviroments

- `Python 3.8`

- `PyTorch 1.10.0`: please follow instructions on [https://pytorch.org/](https://pytorch.org/) to install PyTorch based on your OS and hardware

- `PyTorch Geometric 2.0.4`: please follow instructions on [https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install Pytorch Geometric based on your OS and hardware

Other packages can be installed via:

```bash
pip install -r requirements.txt
```

## Generate datasets

### Gaussian 

Commands to generate train and test datasets of Gaussian tree graphs:

```bash
# Training set
python scripts/data/gaussian.py --outdir=data/gaussian/tree/train/n10 --structure=tree --ndata=30000 --nnode=10
# Testing sets
for i in `seq 10 10 50`; do
    python scripts/data/gaussian.py --outdir=data/gaussian/tree/test/n$i --structure=tree --ndata=2000 --nnode=$i
done
```

Commands to generate train and test datasets of general Gaussian graphical models (GGM) with various graph structures:

```bash
# Training set
python scripts/data/gaussian.py --outdir=data/gaussian/all/train/n10 --structure=all --ndata=10000 --nnode=10
# Testing sets
for i in `seq 10 10 50`; do
    python scripts/data/gaussian.py --outdir=data/gaussian/all/test/n$i --structure=all --ndata=2000 --nnode=$i
done
```

### Discrete spin-glass systems with third-order interactions

Commands to generate train and test datasets

```bash
julia --project -p 8 scripts/data/3spin.jl --outdir=data/third/discrete/train/n10 --gamma=0.5 --ndata=10000 --n=10
for i in `seq 5 1 15`; do
    julia --project -p 8 scripts/data/3spin.jl --outdir=data/third/discrete/test/n$i --gamma=0.5 --ndata=1000 --n=$i
done
```

### Continuous third-order graphical models

The continous third-order graphical model dataset could be downloaded from [here](https://drive.google.com/file/d/1DLwBNWXWn7-LtIdzL4_0sRSLqeWXQ69q/view?usp=sharing). After downloading and uncompressing, put the `continuous` folder in `data/third`.

## Train GNNs

### Gaussian datasets

* Train a figNN on the Tree GGM dataset. 

```bash
python train_lightning.py with jobs/gaussian_tree.yaml 
```

* Train a figNN on the loopy GMM dataset

```bash
python train_lightning.py with jobs/gaussian.yaml
```

### Discrete spin-glass dataset

```bash
python train_lightning.py with jobs/3discrete.yaml
```

### Continuous third-order dataset

```bash
python train_lightning.py with jobs/3continuous.yaml
```

### LDPC dataset

```bash
python train_lightning.py with jobs/LDPC.yaml
```