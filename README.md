# SeSoRec-Sec

## Overview
SeSoRec-Sec is an implementation of the research presented in our upcoming paper (not yet published), which identifies vulnerabilities in the social recommendation framework **SeSoRec** from the paper *Secure Social Recommendation based on Secret Sharing*. We propose an attack method to exploit these vulnerabilities and introduce a secure social recommendation framework, **SeSoRec-Sec**, which incorporates a privacy-preserving matrix multiplication protocol (PPMM) to mitigate information leakage. This repository contains the code to reproduce the original SeSoRec framework, demonstrate the proposed attack, and implement the secure SeSoRec-Sec framework.

The project includes experiments to visualize information leakage using datasets like MNIST and to evaluate the privacy protection of PPMM on datasets such as FilmTrust, Epinions, and Douban.

## Features
- **Reproduction of SeSoRec**: Implements the original *Secure Social Recommendation based on Secret Sharing* framework with and without the Secret Sharing Matrix Multiplication (SSMM) protocol.
- **Attack Implementation**: Demonstrates vulnerabilities in SeSoRec by extracting and visualizing private user data (e.g., social relationships, image data).
- **Secure Framework (SeSoRec-Sec)**: Implements the proposed PPMM protocol to prevent information leakage.
- **Visualization**: Visualizes leakage using MNIST and CIFAR-10 datasets, generating images to illustrate attack results.
- **Evaluation**: Provides scripts to evaluate privacy leakage on FilmTrust, Epinions, and Douban datasets.

## Repository Structure
```
SeSoRec-Sec/
├── data/                       # Datasets used in experiments
│   ├── CIFAR-10/              # CIFAR-10 dataset for visualization (not used in paper)
│   ├── minist/                # Subset of MNIST dataset (10 images per digit) for leakage visualization
│   ├── epinions.txt           # Epinions dataset
│   ├── epinions_user_rating.txt
│   ├── ft_ratings.txt         # FilmTrust dataset
│   ├── ft_trust.txt
│   ├── out.douban             # Douban dataset
├── experiments/                # Attack and evaluation experiments
│   ├── experiment1.py         # (Deprecated, planned for removal)
│   ├── experiment2.py         # Visualizes leakage on MNIST dataset (generates Figure 3 in paper)
│   ├── experiment3.py         # Extracts social relationship leakage from FilmTrust, Epinions, and Douban (generates Table 1 in paper)
│   ├── experiment4.py         # Visualizes leakage on CIFAR-10 dataset (not used in paper)
│   ├── experiment5.py         # Evaluates PPMM protocol on FilmTrust, Epinions, and Douban (results in Section 5.3)
│   ├── experiment6.py         # Visualizes PPMM protocol's protection on MNIST dataset (generates Figure 5 in paper)
├── picture/                    # Visualized leakage images from MNIST experiments
│   ├── comparison.png
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
├── sRecommender/               # Implementation of SeSoRec and SeSoRec-Sec frameworks
│   ├── Recommender1.py        # Baseline recommender (paired with Social1.py)
│   ├── Social1.py             # Baseline social component
│   ├── Recommender2.py        # SeSoRec without SSMM (paired with Social2.py)
│   ├── Social2.py
│   ├── Recommender3.py        # SeSoRec with SSMM on FilmTrust (paired with Social3.py)
│   ├── Social3.py
│   ├── Recommender4.py        # SeSoRec-Sec on FilmTrust (paired with Social4.py)
│   ├── Social4.py
│   ├── Recommender5.py        # SeSoRec with SSMM on Epinions (paired with Social5.py)
│   ├── Social5.py
│   ├── Recommender6.py        # SeSoRec-Sec on Epinions (paired with Social6.py)
│   ├── Social6.py
│   ├── configx.py             # Configuration parameters for recommendation algorithms
│   ├── cross_validation.py    # Five-fold cross-validation for dataset splitting
│   ├── data_wash.py           # Data preprocessing script
│   ├── data/                  # Datasets specific to sRecommender experiments
│   ├── run_folds.sh           # Script to run cross-validation folds
│   ├── result.txt             # Experiment results
├── util/                       # Utility scripts
│   ├── painting.py            # Visualization tools
│   ├── transfer.py            # Matrix transformation utilities
├── README.md                   # This file
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
### Running Experiments
- **Single-file Experiments**:
  Experiments in the `experiments/` directory can be run directly. For example, to reproduce Figure 3 from the paper (MNIST leakage visualization):
  ```bash
  python experiments/experiment2.py
  ```

- **Recommender Experiments**:
  The `sRecommender/` directory contains paired scripts (`RecommenderX.py` and `SocialX.py`, where `X` ranges from 1 to 6). These simulate a two-party protocol:
  1. Start the recommender script first:
     ```bash
     python sRecommender/RecommenderX.py
     ```
  2. In a separate terminal, run the corresponding social script:
     ```bash
     python sRecommender/SocialX.py
     ```
  **Note**: Ensure `X` matches for both scripts (e.g., `Recommender1.py` with `Social1.py`). Examples:
  - For SeSoRec without SSMM (FilmTrust):
    ```bash
    python sRecommender/Recommender2.py
    python sRecommender/Social2.py
    ```
  - For SeSoRec-Sec (Epinions):
    ```bash
    python sRecommender/Recommender6.py
    python sRecommender/Social6.py
    ```

### Output
- **Experiment Outputs**:
  - `experiment2.py` and `experiment6.py` generate images in the `picture/` directory.
  - `experiment3.py` generates tabular results (Table 1 in the paper).
  - `experiment5.py` outputs privacy evaluation metrics (Section 5.3 in the paper).
- **Recommender Outputs**:
  - Results are saved in `sRecommender/result.txt` (corresponding to Table 4 in the paper).

## Datasets
- **Epinions**: User-item ratings and user-user trust data.
- **FilmTrust**: User-item ratings and social trust data.
- **Douban**: Social recommendation dataset.
- **MNIST (subset)**: 10 images per digit (0-9) for leakage visualization.
- **CIFAR-10**: Used for additional visualization (not included in the paper).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Notes
- The `test/` directory and files like `Social_tmp.py` and `TTP.py` are deprecated and will be removed in future updates.
- Contributions are not currently accepted, as this is a research implementation.
- For code review, an anonymous gist can be created by uploading this repository to [GitHub Gist](https://gist.github.com).s