# Synesthetic Variational Autoencoders

Translating Visual Works of Art into Music

[TOC]

The Synesthetic Variational Autoencoder (SynVAE) attempts to translate images into the music domain in an unsupervised manner. This repository contains the code used in the associated experiments. For more details, please visit the project's website at [https://personads.me/x/synvae](https://personads.me/x/synvae).



## Installation

This project is written in Python 3 and requires a functioning set-up of magenta's [MusicVAE](https://magenta.tensorflow.org/music-vae). Additional Python packages can be installed using the `requirements.txt` file. The installation in a virtual environment is highly recommended.

```bash
$ pip install -r requirements.txt
```

## Datasets

Helper classes for different datasets are included in the `data/` directory. This implementation supports MNIST, CIFAR-10, BAM and has an additional class for loading latent vectors during the MINE evaluation. The base class `data/dataset.py` includes common functions such as making TensorFlow iterators or splitting training and validation data. All classes should inherit from this class.

### MNIST

The MNIST dataset is loaded from the TensorFlow-internal `keras.datasets` library and therefore does not require any external data sources. This can be overridden using the `data_path` argument if, for example, reconstructed images are to be loaded. This data should be stored in a pickled tuple `(images, labels)` with shapes identical to the keras implementation.

```python
from data import Mnist
dataset = Mnist(split='test')
dataset = Mnist(split='train', data_path='../data/mnist_recons/train.pkl') # load reconstructions
```

### CIFAR-10

CIFAR-10 data is loaded from the pickled format provided on the [dataset's website](https://www.cs.toronto.edu/~kriz/cifar.html). Place the data into appropriate training and testing directory. The class will load all pickles contained in the respective directory.

```python
from data import Cifar
dataset = Cifar('../data/cifar/train/')
```

### BAM

The [Behance Artistic Media dataset](https://bam-dataset.org/) (BAM) is a collection of annotated contemporary artworks. Images for these experiments must be in JPEG-format and sized 64 pixels on their shorter axis. The data directory should contain an `img/` directory with the scaled images named `MID.jpg` and a `labels.npy` file containing the appropriate labels sorted by MID.

```python
from data import Bam
dataset = Bam('../data/bam/train/')
```

### Latent Vectors

This helper class loads auditive and visual latent vectors from SynVAE in order for their mutual information to be estimated using MINE. The vectors themselves can be exported during the quantitative evaluation and should be available as numpy files.

```python
from data import Latents
dataset = Latents(vis_path='../exp/eval/vis_latents.npy', aud_path= '../exp/eval/aud_latents.npy')
```

## Models

All model classes can be found in the `models/` directory. The `BaseModel` class in `models/base.py` provides the superclass for all modality-specific models and implements training, validation and testing procedures.

### VisVAE

Visual VAE can be found in `models/visual.py` and are defined by the `VisualVae` superclass. It implements loss functions and overall build procedures. The dataset-specific VisVAEs provide a quick way to change hyperparameters while pre-defining constant properties such as image dimensions. Each VisVAE must implement its own `build_encoder(images)` and `build_decoder(latents)` functions.

```python
from models.visual import *
vis_vae = VisualVae(img_height=64, img_width=64, img_depth=3, latent_dim=512, beta=1.0, batch_size=128, learning_rate=1e-3)
vis_vae = MnistVae(latent_dim=50, beta=1.0, batch_size=256)
vis_vae = CifarVae(latent_dim=512, beta=1.0, batch_size=128)
vis_vae = BamVae(latent_dim=512, beta=1.0, batch_size=64)
vis_vae.build()
```

### MusicVAE

The `models/auditive.py` library contains a wrapper for MusicVAE. It can be used to rebuild pre-trained architectures and allows sampling music from the auditive latent space. The configuration must be defined in `magenta/models/music_vae/configs.py`. While this model cannot be trained on its own, its weights can be updated once placed in a SynVAE.

```python
from models.auditive import MusicVAE
music_vae = MusicVae(config_name='cat-mel_2bar_big', batch_size=128)
music_vae.build()
```

### SynVAE

These models use two single-modality models in order to build a single synesthetic architecture and can be found in `models/synesthetic.py`. To build its computation graph, simply initialize two single-modality VAEs and pass them to this model.

```python
from models.synesthetic import SynestheticVae
# initialize single-modality models without building their graphs
vis_vae = BamVae(latent_dim=512, beta=1.0, batch_size=64)
music_vae = MusicVae(config_name='cat-mel_2bar_big', batch_size=128)
# pass to SynVAE and build model
model = SynestheticVae(visual_model=vis_vae, auditive_model=music_vae, learning_rate=1e-3)
model.build()
```

### Classifiers

Visual classifiers are used in order to evaluate reconstruction classification accuracy. They are implemented in `models/classifiers.py` and follow CNN architectures with cross-entropy loss. Dataset-specific classifiers are built upon the `VisualCNN` superclass and generally follow the encoder architecture of their VisVAE counterpart.

```python
from models.classifiers import *
model = VisualCnn(img_height=64, img_width=64, img_depth=3, num_labels=10, batch_size=64, learning_rate=1e-3)
model = MnistCnn(batch_size=256)
model = CifarCnn(batch_size=128)
model = BamCnn(batch_size=64)
model.build()
```

### MINE

[Mutual Information Neural Estimation](https://arxiv.org/abs/1801.04062) (MINE) is used to estimate a lower-bound of the mutual information between auditive and visual latent vectors in SynVAE. The estimator is implemented in `models/mine.py`.

```python
from models.mine import Mine
model = Mine(latent_dim=512, batch_size=256, layer_size=128, learning_rate=1e-3)
model.build()
```

## Model Training

Training scripts for the aforementioned datasets are available in `run/`. Each experiment will produce an output directory which contains latest and best model checkpoints under `exp_dir/checkpoints/`, intermedia output from the validation set in `exp_dir/output/` (if enabled), TensorBoard summaries in `exp_dir/tensorboard/` (if enabled) and a log-file `exp_dir/experiment.log`.

### Training VisVAE

Training a VisVAE requires the task specification, output directory and data directory (except for original MNIST). Additional parameters control the beta hyperparameter, batch size and number of epochs.

```bash
# Training MNIST VisVAE
$ python run/vis_training.py mnist ../exp/mnist_vis/ '' --beta 1.0 --batch_size 256
# Training CIFAR-10 VisVAE
$ python run/vis_training.py cifar ../exp/cifar_vis/ ../data/cifar/train/ --beta 1.0 --batch_size 128
# Training BAM VisVAE
$ python run/vis_training.py bam ../exp/bam_vis/ ../data/bam/train/ --beta 1.0 --batch_size 64
```

To resume training a model, simply specify the initial epoch and the latest model will be loaded from the experiment's output directory.

```bash
$ python run/vis_training.py bam ../exp/bam_vis/ ../data/bam/train/ --beta 1.0 --batch_size 128 --init_epoch 20
```

### Training SynVAE

Training a SynVAE requires several additional arguments. In addition to task specification, output directory and data directory (except for original MNIST), a pre-trained VisVAE checkpoint can be used to initialize the visual components and a MusicVAE configuration and checkpoint are required for initializing the auditive components. Additional parameters control the beta hyperparameter, batch size and number of epochs.

```bash
# Training MNIST SynVAE (no initial VisVAE, no data path)
$ python run/syn_training.py mnist ../exp/mnist_syn/ '' '' cat-mel_2bar_big ../models/cat-mel_2bar_big.ckpt --beta 1.0 --batch_size 256
# Training CIFAR-10 SynVAE (no initial VisVAE)
$ python run/syn_training.py cifar ../exp/cifar_syn/ ../data/cifar/train/ '' cat-mel_2bar_big ../models/cat-mel_2bar_big.ckpt --beta 1.0 --batch_size 128
# Training BAM SynVAE
$ python run/syn_training.py bam ../exp/bam_syn/ ../data/bam/train/ ../exp/bam_vis/checkpoints/best_model.ckpt cat-mel_2bar_big ../models/cat-mel_2bar_big.ckpt --beta 1.0 --batch_size 64
```

To resume training a model, simply specify the initial epoch and the latest model will be loaded from the experiment's output directory. Note that this does not require the re-specification of VisVAE and MusicVAE checkpoints.

```bash
$ python run/syn_training.py bam ../exp/bam_syn/ ../data/bam/train/ '' cat-mel_2bar_big '' --beta 1.0 --batch_size 64 --init_epoch 20
```

## Quantitative Evaluation

### Nearest Neighbour Precision

The `run/vis_analysis.py` and `run/syn_analysis.py` scripts measure the precision at rank *n* for each data point embedded in latent space (both visual and auditive latent space for SynVAE). Result logs and potential output are stored in the specified evaluation directory.

```bash
$ python run/vis_analysis.py bam ../exp/bam_vis/checkpoints/best_model.ckpt ../data/bam/test/ test ../exp/bam_vis_eval/ --beta 1.0 --ranks '1,5,10' --export_latents --export_data
$ python run/syn_analysis.py bam cat-mel_2bar_big ../exp/bam_syn/checkpoints/best_model.ckpt ../data/bam/test/ test ../exp/bam_syn_eval/ --batch_size 64 --beta 1.0 --ranks '1,5,10' --export_latents --export_data
```

Exporting reconstructed images, audio translations and latent vectors is especially useful when employing further evaluation methods.

### Reconstruction Classification

This metric is measured by training a simple classification CNN on either the original image data or reconstructed image data and then testing it on the original test set or the reconstructed test set. The training script works similarly to the ones described above.

```bash
$ python run/cls_training.py bam ../exp/bam_cls/ ../data/bam_recons/train/
$ python run/cls_analysis.py bam ../exp/bam_cls/checkpoints/best_model.ckpt ../data/bam_recons/test/ ../exp/bam_cls_eval/
```

Results which include classification accuracy, precision and recall per class are logged to the specified output directory.

### MINE

[Mutual Information Neural Estimation](https://arxiv.org/abs/1801.04062) (MINE) is used to estimate a lower-bound of the mutual information between auditive and visual latent vectors in SynVAE. Use the latent vectors exported during the Nearest Neighbour Analysis. Since no separate testing procedure is needed, the best model's estimation can be seen as the result.	

```bash
$ python run/mine_training.py ../exp/bam_syn/vis_latents.npy ../exp/bam_syn/aud_latents.npy ../exp/bam_mine/
```

## Qualitative Evaluation

An additional qualitative evaluation can be performed using the accompanying [Syneval](https://personads.me/x/syneval-code) tool. It requires a JSON evaluation task configuration file and the corresponding audio-visual data. The `run/vis_evalgen.py` script generates these tasks based on exported latent vectors from the appropriate VisVAE model. Latent vectors, images and audios can all be exported using the `--export_latents` and `--export_data` flags of the quantitative evaluation scripts.

```bash
$ python run/vis_evalgen.py bam ../data/bam/test/ test ../exp/bam_vis_eval/latents.npy ../exp/bam_task/
```