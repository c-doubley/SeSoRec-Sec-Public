# SeSoRec-Sec

## Overview
SeSoRec-Sec is an implementation of the research presented in our upcoming paper (not yet published), which identifies vulnerabilities in the social recommendation framework **SeSoRec** from the paper *Secure Social Recommendation based on Secret Sharing*. We propose an attack method to exploit these vulnerabilities and introduce a secure social recommendation framework, **SeSoRec-Sec**, which incorporates a privacy-preserving matrix multiplication protocol (PPMM) to mitigate information leakage. This repository contains the code to reproduce the original SeSoRec framework, demonstrate the proposed attack, implement the secure SeSoRec-Sec framework, and compare recommendation quality with a classic social recommendation method (Soreg).

The project includes experiments to visualize information leakage using datasets like MNIST and to evaluate the privacy protection of PPMM on datasets such as FilmTrust, Epinions, and Douban.

## Features
- **Reproduction of SeSoRec**: Implements the original *Secure Social Recommendation based on Secret Sharing* framework with and without the Secret Sharing Matrix Multiplication (SSMM) protocol.
- **Attack Implementation**: Demonstrates vulnerabilities in SeSoRec by extracting and visualizing private user data (e.g., social relationships, image data).
- **Secure Framework (SeSoRec-Sec)**: Implements the proposed PPMM protocol to prevent information leakage.
- **Classic Soreg Implementation**: Includes a non-privacy-preserving social recommendation baseline (Soreg) for recommendation quality comparison.
- **Visualization**: Visualizes leakage using MNIST and CIFAR-10 datasets, generating images to illustrate attack results.
- **Evaluation**: Provides scripts to evaluate privacy leakage and recommendation quality on FilmTrust, Epinions, and Douban datasets.

## Repository Structure
```
SeSoRec-Sec/
├── data/                       # Datasets used in experiments
│   ├── CIFAR-10/              # CIFAR-10 dataset for visualization (not used in paper)
│   │   ├── class_0.png
│   │   ├── class_1.png
│   │   ├── class_2.png
│   │   ├── class_3.png
│   │   ├── class_4.png
│   │   ├── class_5.png
│   │   ├── class_6.png
│   │   ├── class_7.png
│   │   ├── class_8.png
│   │   └── class_9.png
│   ├── minist/                # Subset of MNIST dataset (10 images per digit) for leakage visualization
│   │   ├── 0_155.png
│   │   ├── ... (other MNIST images)
│   │   └── 9_3032.png
│   ├── epinions.txt           # Epinions dataset
│   ├── epinions_user_rating.txt
│   ├── ft_ratings.txt         # FilmTrust dataset
│   ├── ft_trust.txt
│   ├── out.douban             # Douban dataset
├── attack_experiments/         # Attack experiments to demonstrate vulnerabilities
│   ├── attack_Cifar_SSMM.py   # Visualizes leakage on CIFAR-10 dataset (not in paper)
│   ├── attack_MNIST_PPMM.py   # Visualizes PPMM protection on MNIST dataset (Figure 5)
│   ├── attack_MNIST_SSMM.py   # Visualizes leakage on MNIST dataset (Figure 3)
│   ├── attack_SeSoRec-Sec.py  # Evaluates PPMM protection on recommendation datasets
│   ├── attack_SeSoRec.py      # Extracts social relationship leakage (Table 1)
├── picture/                    # Visualized leakage images
│   ├── attack_Cifar_SSMM.png
│   ├── attack_MNIST_PPMM.png
│   ├── attack_MNIST_SSMM.png
│   ├── attack_MNIST_SSMM_one.png
│   ├── mnist_leak_image.png
│   ├── mnist_leak_images.png
│   ├── mnist_leak_left.png
│   ├── mnist_leak_right.png
│   ├── mnist_no_leak_images.png
│   ├── mnist_original_image.png
├── reader/                     # Scripts for reading dataset files
│   ├── cifar.py
│   ├── douban.py
│   ├── epinions.py
│   ├── mnist.py
│   ├── trust.py
├── recommended_quality_experiments/  # Implementation of SeSoRec, SeSoRec-Sec, and Soreg
│   ├── SeSoRec-Sec_ep_ILS_R.py    # SeSoRec-Sec on Epinions with ILS (Recommender)
│   ├── SeSoRec-Sec_ep_ILS_S.py    # SeSoRec-Sec on Epinions with ILS (Social)
│   ├── SeSoRec-Sec_ep_R.py        # SeSoRec-Sec on Epinions (Recommender)
│   ├── SeSoRec-Sec_ep_S.py        # SeSoRec-Sec on Epinions (Social)
│   ├── SeSoRec-Sec_tf_ILS_R.py    # SeSoRec-Sec on FilmTrust with ILS (Recommender)
│   ├── SeSoRec-Sec_tf_ILS_S.py    # SeSoRec-Sec on FilmTrust with ILS (Social)
│   ├── SeSoRec-Sec_tf_R.py        # SeSoRec-Sec on FilmTrust (Recommender)
│   ├── SeSoRec-Sec_tf_S.py        # SeSoRec-Sec on FilmTrust (Social)
│   ├── SeSoRec_ep_ILS_R.py        # SeSoRec on Epinions with ILS (Recommender)
│   ├── SeSoRec_ep_ILS_S.py        # SeSoRec on Epinions with ILS (Social)
│   ├── SeSoRec_ep_R.py            # SeSoRec on Epinions (Recommender)
│   ├── SeSoRec_ep_S.py            # SeSoRec on Epinions (Social)
│   ├── SeSoRec_tf_ILS_R.py        # SeSoRec on FilmTrust with ILS (Recommender)
│   ├── SeSoRec_tf_ILS_S.py        # SeSoRec on FilmTrust with ILS (Social)
│   ├── SeSoRec_tf_R.py            # SeSoRec on FilmTrust (Recommender)
│   ├── SeSoRec_tf_S.py            # SeSoRec on FilmTrust (Social)
│   ├── soreg_ep_ILS_R.py          # Soreg on Epinions with ILS (Recommender)
│   ├── soreg_ep_ILS_S.py          # Soreg on Epinions with ILS (Social)
│   ├── soreg_tf_ILS_R.py          # Soreg on FilmTrust with ILS (Recommender)
│   ├── soreg_tf_ILS_S.py          # Soreg on FilmTrust with ILS (Social)
│   ├── soreg_tf_R.py              # Soreg on FilmTrust (Recommender)
│   ├── soreg_tf_S.py              # Soreg on FilmTrust (Social)
│   ├── configx.py                 # Configuration parameters
│   ├── cross_validation.py        # Five-fold cross-validation
│   ├── data_wash.py               # Data preprocessing
│   ├── data/                      # Datasets for recommendation quality experiments
│   │   ├── Epinions/
│   │   │   ├── ratings.txt
│   │   │   └── user_rating.txt
│   │   ├── TrustFilm/
│   │   │   ├── ft_ratings.txt
│   │   │   └── ft_trust.txt
│   │   └── cv/
│   │       ├── Epinions/
│   │       │   ├── Epinions-0-train.txt
│   │       │   ├── Epinions-0.txt
│   │       │   ├── ... (other cross-validation files)
│   │       │   └── user_map.npy
│   │       └── TrustFilm/
│   │           ├── ft-0-train.txt
│   │           ├── ft-0.txt
│   │           ├── ... (other cross-validation files)
│   │           └── ft-4.txt
│   ├── run_folds.sh               # Script to run cross-validation folds
│   ├── result.txt                 # Recommendation quality results
├── results/                        # Experiment result images and tables
│   ├── Figure3.png
│   ├── Figure5.png
│   ├── Table1.png
│   ├── Table4.png
│   ├── attack_Cifar_SSMM.png
├── util/                           # Utility scripts
│   ├── painting.py                 # Visualization tools
│   ├── transfer.py                 # Matrix transformation utilities
├── README.md                       # This file
├── README1.md                      # Deprecated, to be removed
```

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/SeSoRec-Sec.git
   cd SeSoRec-Sec
   ```

2. **Set Up Python Environment**:
   - Ensure Python 3.8 is installed.
   - Install dependencies using `conda` or `pip`:
     ```bash
     conda env create -f environment.yml
     conda activate SeSoRec-Sec
     ```
     Alternatively:
     ```bash
     pip install numpy pandas scipy scikit-learn matplotlib pillow
     ```

3. **Verify Datasets**:
   - The `data/` directory contains necessary datasets (Epinions, FilmTrust, Douban, MNIST subset).
   - Ensure all files are present as listed in the repository structure.

## Usage
### Running Attack Experiments
- **Single-file Experiments**:
  Experiments in the `attack_experiments/` directory can be run directly. For example, to reproduce Figure 3 (MNIST leakage visualization):
  ```bash
  python attack_experiments/attack_MNIST_SSMM.py
  ```

### Running Recommendation Quality Experiments
- The `recommended_quality_experiments/` directory contains paired scripts (e.g., `SeSoRec_ep_ILS_R.py` and `SeSoRec_ep_ILS_S.py`). These simulate a two-party protocol:
  1. Start the recommender script first:
     ```bash
     python recommended_quality_experiments/SeSoRec_ep_ILS_R.py
     ```
  2. In a separate terminal, run the corresponding social script:
     ```bash
     python recommended_quality_experiments/SeSoRec_ep_ILS_S.py
     ```
  **Note**: Ensure the script names match (e.g., `SeSoRec_ep_ILS_R.py` with `SeSoRec_ep_ILS_S.py`). The suffix `_R` indicates the recommender party (with user-item data), and `_S` indicates the social party (with user-user trust data). `ILS` indicates evaluation with the Intra-List Similarity metric.

### Output
- **Attack Experiment Outputs**:
  - `attack_MNIST_SSMM.py` and `attack_MNIST_PPMM.py` generate images in the `picture/` directory.
  - `attack_SeSoRec.py` generates tabular results (Table 1 in the paper).
  - `attack_SeSoRec-Sec.py` outputs privacy evaluation metrics.
- **Recommendation Quality Outputs**:
  - Results are saved in `recommended_quality_experiments/result.txt` (corresponding to Table 4 in the paper).

## Obtaining Paper Results
This section explains how to reproduce the results presented in the paper.

### Table 1: Social Relationship Leakage
- **Description**: Extracts user social relationship information from FilmTrust, Epinions, and Douban datasets using the proposed attack on SeSoRec.
- **Output**: See `results/Table1.png`.
- **How to Run**:
  ```bash
  python attack_experiments/attack_SeSoRec.py
  ```

### Figure 3: MNIST Leakage Visualization
- **Description**: Visualizes information leakage from the SeSoRec framework using the MNIST dataset.
- **Output**: See `results/Figure3.png`.
- **How to Run**:
  ```bash
  python attack_experiments/attack_MNIST_SSMM.py
  ```
- **Note**: The output image name is configured in `util/painting.py` using either `mnist_stack_images` (for multiple stacked images) or `mnist_painting` (for a single image).

### Figure 5: PPMM Protection Visualization
- **Description**: Demonstrates that the proposed PPMM protocol prevents leakage on the MNIST dataset under the same attack.
- **Output**: See `results/Figure5.png`.
- **How to Run**:
  ```bash
  python attack_experiments/attack_MNIST_PPMM.py
  ```

### Table 4: Recommendation Quality Evaluation
- **Description**: Evaluates the recommendation quality of SeSoRec, SeSoRec-Sec, and Soreg on Epinions and FilmTrust datasets, with and without Intra-List Similarity (ILS) metrics.
- **Output**: See `results/Table4.png`.
- **How to Run**:
  Run the paired scripts in `recommended_quality_experiments/`. For example, to evaluate SeSoRec on Epinions with ILS:
  ```bash
  cd recommended_quality_experiments
  python SeSoRec_ep_ILS_R.py
  python SeSoRec_ep_ILS_S.py
  ```
  Each script includes running instructions. File naming conventions:
  - `SeSoRec` or `SeSoRec-Sec`: Framework type.
  - `ep` or `tf`: Dataset (Epinions or FilmTrust).
  - `ILS`: Includes Intra-List Similarity metric (omitted if not used).
  - `R` or `S`: Recommender or Social party.

### Additional Experiment: CIFAR-10 Leakage Visualization
- **Description**: Visualizes leakage on the CIFAR-10 dataset (not included in the paper due to less significant results).
- **Output**: See `results/attack_Cifar_SSMM.png`.
- **How to Run**:
  ```bash
  python attack_experiments/attack_Cifar_SSMM.py
  ```

## Datasets
- **Epinions**: User-item ratings and user-user trust data.
- **FilmTrust**: User-item ratings and social trust data.
- **Douban**: Social recommendation dataset.
- **MNIST (subset)**: 10 images per digit (0-9) for leakage visualization.
- **CIFAR-10**: Used for additional visualization (not included in the paper).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Notes
- The `README1.md` file is deprecated and will be removed in future updates.
- Contributions are not currently accepted, as this is a research implementation.
- For code review, an anonymous gist can be created by uploading this repository to [GitHub Gist](https://gist.github.com).