# Speech-DPOAE Stimulus Generation and Experiment Evaluation

This repository contains the code used in our study on **distortion product otoacoustic emissions (DPOAEs)** evoked by human running speech (speech-DPOAEs).

If you use this code, **please cite**: LINK  

The repository includes:
- **Stimulus generation** code to reproduce (or modify) the stimuli used in our experiment. This part works independently of our experimental setup and can be directly applied (“plug-and-play”).
- **Experiment evaluation** code tailored to our dataset (available on [Zenodo](https://zenodo.org/records/16837673)), but adaptable to other datasets with minor modifications.

The accompanying publication is available as a preprint on arXiv: LINK  
Details on the theory behind our stimulus generation method can be found in [1].


## Background
Stimuli were designed to follow harmonic overtones *n* and *m* (*n < m*) of the speech signal’s fundamental frequency (*f₀*), with the expected lower-sideband DPOAE arising at harmonic *2n – m*.

The experiment consisted of:
1. **Single-speaker measurements** – either male or female voice presented in isolation (two trials per participant)  
2. **Competing-speaker measurements** – both voices presented simultaneously, with participants instructed to attend to one voice while ignoring the other (18 trials per participant)


## Repository Structure

### 1. Stimulus Generation
The stimulus generation scripts create `.wav` files representing waveforms of the desired harmonic overtones.

- **Main scripts:**
  - `stimulus_generation.py` — defines `stimulus_generator()` and `multiband_generator()` classes
  - `plot_stimulus_correlation.py` — helper for visualizing stimulus correlation
  - `run_stimulus_generation.py` — entry point to create stimuli; specify harmonics as `[n1, n2]` (integers or lists)

This module can be used independently of the experimental setup.


### 2. Experiment Evaluation
This section analyzes experimental data as provided on [Zenodo](https://zenodo.org/records/16837673).

- **Main scripts:**
  - `DPOAE_analysis.py` — defines `ExperimentEvaluator()` class to process raw recordings (stimuli + DPOAEs) via cross-correlation
  - `statistical_analysis.py` — defines `CrossCorrelationProcessor()` to compute grand averages, extract peak magnitudes/delays, and save results
  - `plot_statistics.py` — defines `StatsPlotter()` to reproduce figures from our publication

- **Entry points:**
  - `run_DPOAE_analysis.py` — runs the evaluation  
  - `run_statistical_analysis.py` — computes statistics and saves results (e.g., in `Data_Plots/`)  
  - `run_plot_statistics.py` — generates plots from saved statistics  

Note: The evaluation code assumes a data structure matching our PsychoPy-based experiment [2]. For your own data, you may need to adjust file paths and folder structures.



### 3. Helper Functions
- `butterworth_filter.py`
- `cross_correlation.py`
- `running_mean.py`  
These are utility scripts used by multiple modules.



## Data Availability
The dataset used in our publication, including recorded speech-DPOAE data, speech signals, and speech-DPOAE stimuli, is openly available on Zenodo:  
[https://zenodo.org/records/16837673](https://zenodo.org/records/16837673)



## Project Status
**Stable** — This repository is provided *as-is* and is no longer actively maintained, but remains available for reproducibility and reuse.



## References
[1] M. Saiz-Alía, P. Miller, T. Reichenbach, “Otoacoustic Emissions Evoked by the Time-Varying Harmonic Structure of Speech,” *eNeuro*, 8(2), (2021). doi:[10.1523/ENEURO.0428-20.2021](https://doi.org/10.1523/ENEURO.0428-20.2021)  
[2] J. Peirce *et al.*, “PsychoPy2: Experiments in behavior made easy,” *Behav Res*, 51(1), 195–203 (2019). doi:[10.3758/s13428-018-01193-y](https://doi.org/10.3758/s13428-018-01193-y)  


---

## Citation
If you use this repository in your research, please cite our paper:

> Your Name *et al.*, "Your Paper Title", *Nature Communications*, 2025. DOI: [link]  

A `CITATION.cff` file is provided for automated citation tools.
