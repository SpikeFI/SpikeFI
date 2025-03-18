<p align="center">
    <img src="https://github.com/SpikeFI/.github/blob/main/profile/spikefi_logo.png" width="400">
</p>

# SpikeFI
### *A Fault Injection Framework for Spiking Neural Networks*

This is the main repository of the *SpikeFI* framework.

## Abstract

Neuromorphic computing and spiking neural networks (SNNs) are gaining traction across various artificial intelligence (AI) tasks thanks to their potential for efficient energy usage and faster computation speed. This comparative advantage comes from mimicking the structure, function, and efficiency of the biological brain, which arguably is the most brilliant and green computing machine. As SNNs are eventually deployed on a hardware processor, the reliability of the application in light of hardware-level faults becomes a concern, especially for safety- and mission-critical applications. We propose *SpikeFI*, a fault injection framework for SNNs that can be used for automating the reliability analysis and test generation. *SpikeFI* is built upon the SLAYER PyTorch framework with fault injection experiments accelerated on a single or multiple GPUs. It has a comprehensive integrated neuron and synapse fault model library, in accordance to the literature in the domain, which is extendable by the user if needed. It supports: single and multiple faults; permanent and transient faults; specified, random layer-wise, and random network-wise fault locations; and pre-, during, and post-training fault injection. It also offers several optimization speedups and built-in functions for results visualization. *SpikeFI* is open-source.

## Publication

The article introducing *SpikeFI* has been submitted to IEEE for possible publication. A preprint version is available on arXiv [here](https://arxiv.org/abs/2412.06795).

### Citation

To cite our work, please use the following citation:

> T. Spyrou, S. Hamdioui and H.-G. Stratigopoulos, _"SpikeFI: A Fault Injection Framework for Spiking Neural Networks,"_ arXiv, 2024, https://arxiv.org/abs/2412.06795

```bibtex
@misc{spyrou2024spikefi,
      title={SpikeFI: A Fault Injection Framework for Spiking Neural Networks}, 
      author={Theofilos Spyrou and Said Hamdioui and Haralampos-G. Stratigopoulos},
      year={2024},
      eprint={2412.06795},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2412.06795}, 
}
```

## Acknowledgments
This work was funded by the ANR RE-TRUSTING project under Grant No ANR-21-CE24-0015-03 and by the European Network of Excellence dAIEDGE under Grant Agreement No 101120726. The work of T. Spyrou was supported by the Sorbonne Center for Artificial Intelligence (SCAI) through Fellowship.


## Requirements

SpikeFI requires Python 3 and it has been tested to work with version 3.11 or newer.

The following packages are required to be installed before proceeding with using SpikeFI:
- PyTorch (version 2.1.2 or newer*)
- matplotlib (version 3.7.5 or newer)
- numpy (version 1.26.4*)
- pyyaml (version 6.0.2 or newer)
- h5py (version 3.9.0 or newer)
- [SLAYER](https://github.com/bamsumit/slayerPytorch) (version 1.0)

_*numpy version 2.0 or newer is not yet supported. torch version needs to have support for numpy <2.0 (tested successfully up to torch 2.6.0)._

SLAYER requires a CUDA-enabled GPU for training SNN models.

*SpikeFI* has been tested with CUDA libraries version 12.1 up to 12.8 and GCC 11.5 to 13.2 on Almalinux 9.3, Ubuntu 24.04, and Red Hat Linux 9.5 using NVIDIA A100 80GB, NVIDIA Quadro RTX 4000, and NVIDIA GeForce RTX 2080 Ti GPUs.

## Installation

To properly install SpikeFI, please follow the instructions below at the correct order. A clean environment of Python is recommended (e.g., using conda).

#### 1. Clone the repository

Clone the SpikeFI and SLAYER repositories to the directory of your preference:

```
git clone https://github.com/SpikeFI/SpikeFI.git
git clone https://github.com/bamsumit/slayerPytorch.git
```

#### 2. Install PyTorch

Install any version of PyTorch from 2.1.2 onwards with the combination of the CUDA version supported by your system. For example, to install the latest version of PyTorch (i.e., 2.6.0 at the moment of writing) with CUDA 12.6 execute the following command:

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`

#### 3. Install the requirements

All the requirements are listed in the `requirements.txt` file. To install them you can use the following command (after switching to SpikeFI directory):

`pip3 install -r requirements.txt`

#### 4. Install SLAYER

From __within the SLAYER directory__, run the following command:

`python3 setup.py install`

For more information about the SLAYER framework, feel free to visit its GitHub page: https://github.com/bamsumit/slayerPytorch

#### 5. Install SpikeFI

Run the following command from SpikeFI directory to install the framework:

`pip3 install .`

#### 6. Putting it altogether

``` bash
git clone https://github.com/SpikeFI/SpikeFI.git
git clone https://github.com/bamsumit/slayerPytorch.git
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install -r ./SpikeFI/requirements.txt
cd ./slayerSNN/
python3 setup.py install
cd ../SpikeFI/
pip3 install .
```


## Package Structure

The *SpikeFI* package contains the following modules:
- __core__: main functionality
- __fault__: classes composing a Fault
- __models__: integrated fault library
- __visual__: visualization functions


### More examples

The additional *demo* package contains some examples demonstrating how to use the features and optimizations of *SpikeFI*:
- __bitfilp.py__: FI campaign for bitflipped synapse fault model
- __optimizations.py__: comparison between the supported optimizations
- __parametric.py__: FI campaign for neuron parametric faults
- __train_golden.py__: golden training of a SNN with SLAYER
- __train.py__: FI before/during training of a SNN

### Trained SNNs

The *nets* subpackage of the *demo* package contains the network classes (defined in SLAYER) for the N-MNIST and IBM's DVS128 Gesture SNNs, along with the classes to load their datasets. The paths to the dataset directories need to be indicated by the user in the .yaml configuration file of the network.

## License & Copyright

Theofilos Spyrou is the Author of the Software <br>
Copyright 2024 Sorbonne Université, Centre Nationale de la Recherche Scientifique
 
*SpikeFI* is free software: you can redistribute it and/or modify it under the terms of GNU General Public License version 3 as published by the Free Software Foundation.
 
*SpikeFI* is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 
You will found in the LICENSE file a copy of the GNU General Public License version 3.
