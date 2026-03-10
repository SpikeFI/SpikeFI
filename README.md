<p align="center">
    <img src="https://github.com/SpikeFI/.github/blob/main/profile/spikefi_logo.png" width="400">
</p>

# SpikeFI
### *A Fault Injection Framework for Spiking Neural Networks*

This is the main repository of the *SpikeFI* framework.


## Abstract

Neuromorphic computing and spiking neural networks (SNNs) are gaining traction across various artificial intelligence (AI) tasks thanks to their potential for efficient energy usage and faster computation speed. This comparative advantage comes from mimicking the structure, function, and efficiency of the biological brain, which arguably is the most brilliant and green computing machine. As SNNs are eventually deployed on a hardware processor, the reliability of the application in light of hardware-level faults becomes a concern, especially for safety- and mission-critical applications. We propose *SpikeFI*, a fault injection framework for SNNs that can be used for automating the reliability analysis and test generation. *SpikeFI* is built upon the SLAYER PyTorch framework with fault injection experiments accelerated on a single or multiple GPUs. It has a comprehensive integrated neuron and synapse fault model library, in accordance to the literature in the domain, which is extendable by the user if needed. It supports: single and multiple faults; permanent and transient faults; specified, random layer-wise, and random network-wise fault locations; and pre-, during, and post-training fault injection. It also offers several optimization speedups and built-in functions for results visualization. *SpikeFI* is open-source.


## Publication

The article introducing *SpikeFI* has been submitted for possible publication. A preprint version is available on arXiv [here](https://arxiv.org/abs/2412.06795).

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


## Dependencies

SpikeFI requires and has been tested to work with Python v3.11 or newer.

The following packages are required for using *SpikeFI*:
- `pytorch` ≥ v2.1
- `matplotlib` ≥ v3.7
- `numpy` ~= v1.26
- `pyyaml` ≥ v6.0
- `tqdm` ≥ v4.67
- `cycler` ≥ v0.12
- [`slayer`](https://github.com/bamsumit/slayerPytorch) ≥ v0.1

_*`numpy` version 2 is not yet supported. `torch` needs to have support for `numpy` < v2.0 (tested successfully up to `torch` v2.6)._

`slayer` requires a CUDA-enabled GPU for training SNN models.

*SpikeFI* has been tested with CUDA libraries version 12.1 up to 12.8 and GCC 11.5 to 13.2 on Almalinux 9.3, Ubuntu 24.04, and Red Hat Linux 9.5 using NVIDIA A100 80GB, NVIDIA Quadro RTX 4000, and NVIDIA GeForce RTX 2080 Ti GPUs.


## Installation

To properly install *SpikeFI*, please follow the instructions below at the correct order. A clean environment of Python is recommended.

### 1. Clone the repository

Clone the *SpikeFI* and SLAYER repositories to the directory of your preference:

```
git clone https://github.com/SpikeFI/SpikeFI.git
git clone https://github.com/bamsumit/slayerPytorch.git
```

### 2. Install PyTorch

Install any version of `pytorch` from v2.1 onwards with the combination of the CUDA version supported by your system. For example, to install the latest version of `pytorch` execute the following command:

`pip3 install torch torchvision torchaudio`

For more information, please consult the get-started instructions on the PyTorch website [here](https://pytorch.org/get-started/locally).

### 3. Install SLAYER

From __within the SLAYER directory__, run the following command:

`python3 setup.py install`

For more information about the SLAYER framework, feel free to visit its GitHub page [here](https://github.com/bamsumit/slayerPytorch).

### 4. Install SpikeFI

Run the following command from SpikeFI directory to install the framework:

`pip3 install .`

### Putting it altogether

``` bash
git clone https://github.com/SpikeFI/SpikeFI.git
git clone https://github.com/bamsumit/slayerPytorch.git
pip3 install torch torchvision torchaudio
cd ./slayerPytorch/
python3 setup.py install
cd ../SpikeFI/
pip3 install .
```


## Package Structure

The *SpikeFI* package contains the following modules:
- __core__: *main functionality*
- __fault__: *classes composing a Fault*
- __models__: *integrated fault library*
- __visual__: *visualization functions*
- __utils__: *utility functions*
    - __io__: *io and file helper*
    - __layer__: *snn layer information helper*
    - __progress__: *extended progress bar and info*
    - __quantization__: *weight quantization helper*


## Demonstration

*SpikeFI* is equipped with a `demo` package and a set of examples to easily get started and familiarize with its functionality.

### Pre-trained SNN models

*SpikeFI* is delivered with 3 pre-trained SNN models, namely, (i) `nmnist_mlp`, (ii) `nmnist_cnn`, and (iii) `gesture` networks. The network architectures and dataset classes in PyTorch can be found at `examples/demo/architectures` and the trained instances at `examples/demo/models`. The network configuration parameters, necessary for the SLAYER framework, can be found at `examples/demo/config`.

### Examples

The *examples* directory contains indicative cases of using *SpikeFI* in various scenarios, demonstrating how to use the framework's features and capabilities.

The available examples are the following:
- __Basic Example__: *Basic usage of SpikeFI's features*
- __Bit-flip Example__: *Reliability analysis of SNNs to bit-flip faults in the synaptic weights*
- __Optimizations Example__: *SpikeFI's optimization techniques and fault simulation acceleration*
- __Parametric Example__: *Reliability analysis of SNNs to parametric neuron faults*
- __Train Golden Example__: *Training a fault-free SNN with SpikeFI*
- __Train Faulty Example__: *Fault injection experiments before training*

Each example is available both as a python script file `.py` and as a Jupyter notebook `.ipynb`.

To execute an example, it is possible by opening and running the corresponding notebook, or by by executing the following command from SpikeFI main directory:

`python3 examples/<example_name>.py`

After the execution of an example is over, its results (output files) can be found in the `SpikeFI/examples/out` directory.


## License & Copyright

Theofilos Spyrou is the Author of the Software <br>
Copyright 2024 Sorbonne Université, Centre Nationale de la Recherche Scientifique
 
*SpikeFI* is free software: you can redistribute it and/or modify it under the terms of GNU General Public License version 3 as published by the Free Software Foundation.
 
*SpikeFI* is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 
You will found in the LICENSE file a copy of the GNU General Public License version 3.
