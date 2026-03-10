# Changelog

All notable changes to this project will be documented in this file.

## [1.3.0] – 2026-03-10

### Added
- Integrate `tonic` for automated dataset installation and loading, replacing previous manual dataset setup scripts ([5376c85](https://github.com/SpikeFI/SpikeFI/commit/5376c85), [85f88ec](https://github.com/SpikeFI/SpikeFI/commit/85f88ec))
- Add progress mode literals and allow passing the device as argument in `Campaign` ([b87b283](https://github.com/SpikeFI/SpikeFI/commit/b87b283))
- Add cached dataset support to accelerate dataset loading in repeated experiments ([52f3a58](https://github.com/SpikeFI/SpikeFI/commit/52f3a58))

### Changed
- Update network architectures and training setup; remove legacy dataset setup scripts and rely on dynamic dataset loading ([5376c85](https://github.com/SpikeFI/SpikeFI/commit/5376c85))
- Update gesture SNN architecture and add support for excluding the `"other"` class from the DVSGesture dataset ([2fc07f0](https://github.com/SpikeFI/SpikeFI/commit/2fc07f0))
- Improve training and testing loops and update statistics handling ([39bc5df](https://github.com/SpikeFI/SpikeFI/commit/39bc5df))
- Update training functionality and training pipeline for improved workflow and flexibility ([a62a47d](https://github.com/SpikeFI/SpikeFI/commit/a62a47d))
- Update examples to reflect the new dataset handling and cached loading mechanisms ([8623c52](https://github.com/SpikeFI/SpikeFI/commit/8623c52))

### Updated
- Update trained models and checkpoints for the gesture SNN architecture ([954c809](https://github.com/SpikeFI/SpikeFI/commit/954c809), [3f70662](https://github.com/SpikeFI/SpikeFI/commit/3f70662), [a08c093](https://github.com/SpikeFI/SpikeFI/commit/a08c093))

### Fixed
- Fix saving of the best-performing network during training and remove unused `get_loader` function from demo ([ca4f89c](https://github.com/SpikeFI/SpikeFI/commit/ca4f89c))
- Fix network loading and storing issues and update `train_golden_slayer` example ([7802418](https://github.com/SpikeFI/SpikeFI/commit/7802418))
- Fix DVSGesture dataset download issue when using `tonic` ([5c527e4](https://github.com/SpikeFI/SpikeFI/commit/5c527e4))

## [1.2.0] – 2025-12-02

### Added
- Implementation of distinct fault models for parametric neuron faults: `IntegrationFaultNeuron`, `RefractoryFaultNeuron`, `ThresholdFaultNeuron` ([cc5bc1d](https://github.com/SpikeFI/SpikeFI/commit/cc5bc1d))
- Implement new `SaturatedSynapse` and `StuckSynapse` fault models ([c93a752](https://github.com/SpikeFI/SpikeFI/commit/c93a752))
- Create `RandomFaultModel` classes to randomly select a fault model type: `RandomNeuron`, `RandomParametricNeuron`, `RandomSynapse` ([d0b6003](https://github.com/SpikeFI/SpikeFI/commit/d0b6003))
- Create `StuckNeuron` fault model ([17d888b](https://github.com/SpikeFI/SpikeFI/commit/17d888b))
- Allow default initialization for `ParametricNeuron`, `SaturatedSynapse`, and `PerturbedSynapse` fault models ([7a3e59a](https://github.com/SpikeFI/SpikeFI/commit/7a3e59a))
- Add support for custom unpickler in load functions to enhance compatibility with older versions of `Campaign` and `CampaignData` classes ([a5fc48c](https://github.com/SpikeFI/SpikeFI/commit/a5fc48c))
- Include `pyyaml` package as a requirement ([2af2e89](https://github.com/SpikeFI/SpikeFI/commit/2af2e89))

### Changed
- Create `NeuronHook` and `SynapseHook` classes to replace hook wrappers with nested functions in `Campaign` class ([bc5f660](https://github.com/SpikeFI/SpikeFI/commit/bc5f660))
- Create `RoundIndex` class to handle round index via reference in `Campaign` and access it more easily in the new hook classes ([bc5f660](https://github.com/SpikeFI/SpikeFI/commit/bc5f660))
- Initialize faults with empty site by default ([bc5f660](https://github.com/SpikeFI/SpikeFI/commit/bc5f660))
- Update `CampaignData` storing and handling ([9763033](https://github.com/SpikeFI/SpikeFI/commit/9763033))
- Revisit multiple random fault generation and injection functions to apply fault effects more efficiently and in parallel ([f516281](https://github.com/SpikeFI/SpikeFI/commit/f516281))
- Improve type hints in functions ([41ec8d3](https://github.com/SpikeFI/SpikeFI/commit/41ec8d3))
- Update module aliases ([bb3b05a](https://github.com/SpikeFI/SpikeFI/commit/bb3b05a))
- Unify `perturb_net` function for FI during and after training ([3dffb4d](https://github.com/SpikeFI/SpikeFI/commit/3dffb4d))
- Simplify `FaultSite` and `ParametricFault` initialization ([794176e](https://github.com/SpikeFI/SpikeFI/commit/794176e))
- Accept either an iterable or a single object in functions ([4b2f1f9](https://github.com/SpikeFI/SpikeFI/commit/4b2f1f9))
- Merge multiple faults of the same model during injection and guarantee the uniqueness of all fault sites in random multiple faults ([e8c88ae](https://github.com/SpikeFI/SpikeFI/commit/e8c88ae))
- Allow multiple bit positions in `BitflippedSynapse` and simplify quantization functions ([e863ac0](https://github.com/SpikeFI/SpikeFI/commit/e863ac0))
- Update quantization methods (use native PyTorch) ([f375711](https://github.com/SpikeFI/SpikeFI/commit/f375711))
- Improve quantization naming and readability ([c352f52](https://github.com/SpikeFI/SpikeFI/commit/c352f52))
- Improve `BitflippedSynapse` to receive quantization argument directly ([7a34eb1](https://github.com/SpikeFI/SpikeFI/commit/7a34eb1))
- Make save-to-file optional for visual functions ([c2fd323](https://github.com/SpikeFI/SpikeFI/commit/c2fd323))
- Improve bar plot label contrast ([20dede0](https://github.com/SpikeFI/SpikeFI/commit/20dede0))

### Fixed
- Fix PEP8 long-line violations ([2eb3406](https://github.com/SpikeFI/SpikeFI/commit/2eb3406))
- Fix bugs ([a090676](https://github.com/SpikeFI/SpikeFI/commit/a090676))
- Fix load issue with campaign data ([ddc9541](https://github.com/SpikeFI/SpikeFI/commit/ddc9541))
- Fix visualization of learning curves in faulty training ([c2fd323](https://github.com/SpikeFI/SpikeFI/commit/c2fd323))
- Fix multiple round evaluation issues
