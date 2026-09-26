# Phishing URL Classification using Self-Organizing Maps (SOM)

Unsupervised detection of phishing URLs with a Self-Organizing Map, built in MATLAB. The project was completed for the *Data Driven Models for System Engineering* course of the MSc in Computer Engineering, Cybersecurity and Artificial Intelligence at the University of Cagliari.

## Overview

Phishing URLs are built to imitate legitimate sites, and signature- and rule-based filters struggle to keep up as attacks change. Supervised models can work, but they need large labeled datasets and may miss unseen patterns. This project asks a different question: can a SOM, which learns the structure of the data without labels, separate phishing URLs from legitimate ones using only numeric URL and page features?

Labels are used only after training, to name each neuron by majority vote, so the map itself is learned without supervision.

## Dataset

[PhiUSIIL Phishing URL dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) (UCI Machine Learning Repository): 235,795 URLs (100,945 phishing and 134,850 legitimate), described by numeric features such as URL length, character continuation rate, TLD probability, URL similarity index and subdomain count.

## Method

1. **Preprocessing:** drop the non-numeric columns (`FILENAME`, `URL`, `Domain`, `TLD`, `Title`) and remove `DomainTitleMatchScore`, which is highly correlated with `URLMatchScore`. A correlation heatmap supports the choice.
2. **Split and scaling:** 80/20 hold-out split. Z-score normalization is fitted on the training set only and applied to the test set, which avoids data leakage.
3. **SOM training:** a 3x3 grid (9 neurons) trained with MATLAB's `selforgmap` for 200 epochs.
4. **Neuron labeling:** each neuron takes the majority label of the training samples mapped to it.
5. **Testing:** each test URL is assigned to its best matching unit and inherits that neuron's label.
6. **Evaluation:** accuracy, confusion matrix, and per-class and macro-averaged precision, recall, F1 and specificity, plus SOM hit and weight-plane plots.

## Results

On the 47,159 held-out URLs the classifier reaches **99.11% accuracy**.

| Class | Precision | Recall | F1 | Specificity | Support |
|---|---|---|---|---|---|
| Phishing | 99.60% | 98.31% | 98.95% | 99.71% | 20,158 |
| Legitimate | 98.75% | 99.71% | 99.23% | 98.31% | 27,001 |
| **Macro average** | **99.18%** | **99.01%** | **99.09%** | **99.01%** | |

Of the 47,159 test URLs, 340 phishing URLs were classified as legitimate and 79 legitimate URLs as phishing.

## Report

The full write-up covers the motivation, dataset, methodology, results and future directions (feature engineering, PCA, combining the SOM with other classifiers).

**[Read the report (PDF)](report.pdf)**

## Repository contents

| File | Description |
|---|---|
| `source_code.m` | MATLAB implementation of the full pipeline |
| `report.pdf` | Project report |

## How to run

1. Download `PhiUSIIL_Phishing_URL_Dataset.csv` from the dataset page above and place it next to `source_code.m`.
2. Open `source_code.m` in MATLAB with the Deep Learning Toolbox (for `selforgmap`) and the Statistics and Machine Learning Toolbox.
3. Run the script. It prints the metrics and opens the correlation heatmap, confusion matrix and SOM plots.

## Author

Mohamed Njah, [LinkedIn](https://www.linkedin.com/in/mednjah), [GitHub](https://github.com/mohamednjah)
