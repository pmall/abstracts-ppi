# Fine-Tuning a Language Model for Text Classification Using Huggingface

## Introduction

### Context

This project explores the process of fine-tuning an existing language model using a real-world dataset. Specifically, a classification model was trained using [Huggingface](https://huggingface.co/) tools. Rather than aiming for a perfect model, the focus is on understanding each step of the fine-tuning process and identifying potential solutions to challenges that arise along the way.

### Dataset

The dataset originates from my previous research team, which studied human-viral protein-protein interactions (PPIs). It was developed as part of this research and served as the foundation for a [scientific publication](https://pubmed.ncbi.nlm.nih.gov/38252831/).

The dataset consists of approximately **80,000 scientific abstracts**, categorized into two classes:

- **Positive:** The abstract describes a publication detailing PPIs.
- **Negative:** The abstract does not describe a publication detailing PPIs.

The dataset was constructed over several years through the following steps:

1. **Querying:** A broad search query was run monthly on pubmed to retrieve all publications that could potentially describe a PPI, yielding a few thousand abstracts each time.
2. **Precuration:** Biologist curators manually reviewed each new batch of abstracts, a process that took a few days for every query result.
3. **Curation:** Curators analyzed the full text of selected publications, extracting all described PPIs and flagging any false positives that had passed the precuration step.

This process resulted in a **highly imbalanced dataset**:

- **5,919 positive abstracts**
- **78,274 negative abstracts**

Additionally, abstracts describing **human-human PPIs** were excluded, as they did not undergo the same manual curation process. Most of these abstracts were positive examples, and including them would have further skewed the dataset.

### Goal

The objective is to fine-tune an existing language model to serve as a **binary classifier**, replacing the precuration step. Since precuration is both time-consuming and mentally demanding, automating this step would be very valuable.

Given that the PPI curation process aims to be **exhaustive**, the cost of a **false negative** (missing a relevant publication) is much higher than that of a **false positive** (which will be filtered out later during the curation step). Therefore, the classifier must achieve both **high recall** and **strong precision**, ensuring a high **F1 score** for reliable performance.

## Methods

### Dataset Preparation

To clean the dataset, **encoded Unicode characters** (such as Greek letters in abstracts) were decoded, and **residual HTML tags** were removed. Abstracts with missing titles or fewer than **30 words** were discarded, resulting in:

- **5,893 positive examples**
- **77,464 negative examples**

The **titles and abstracts** were concatenated to form the input text.

Splitting strategy:

- **10%** of the dataset was set aside for both **evaluation** and **test** splits, preserving the real-world class imbalance.
- The remaining **80%** was used for training. However, to address the class imbalance, **positive examples were oversampled 20 times** to match the number of negative examples.

This resulted in the following dataset sizes:

- **Evaluation/Test Splits:**
  - **589** positive examples
  - **7,746** negative examples
- **Training Split (after oversampling):**
  - **61,275** positive examples (**4,715 unique**)
  - **61,972** negative examples

This is a **naïve oversampling approach**, and more advanced strategies are discussed in the [Conclusion and Perspectives](#conclusion-and-perspectives) section.

### Base Model Selection

The selected language model needed to:

- Understand **English**
- Fit within a **single 12GB VRAM GPU**
- Have a **context window** suitable for abstracts (~230 words on average)

A common choice for text classification is **[BERT Base Uncased](https://huggingface.co/google-bert/bert-base-uncased)**, which supports a **512-token** context window. However, a more specialized alternative exists: **[BiomedBERT Base Uncased Abstract](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract)**, which was trained specifically on **pubmed abstracts**. Given the nature of the dataset, **BiomedBERT** was chosen as the initial model.

### Training

Since the training set included **repeated positive examples** and all the negative examples, training was conducted for **only one epoch** to prevent excessive overfitting. The model was trained with a batch size of **16** for approximately **7,700 steps**, with training and evaluation loss recorded every **1,000 steps**. This setup ensured a balanced exposure to positive and negative examples while keeping training time reasonable.

## Results

### Training and Evaluation Loss

| Step | Training Loss | Validation Loss |
| :--: | :-----------: | :-------------: |
| 1000 |    0.3268     |     0.5548      |
| 2000 |    0.2588     |     0.2738      |
| 3000 |    0.2180     |     0.2580      |
| 4000 |    0.1990     |     0.3933      |
| 5000 |    0.1519     |     0.3477      |
| 6000 |    0.1311     |     0.3246      |
| 7000 |    0.1175     |     0.3275      |

The training loss consistently decreased, while the evaluation loss initially dropped, then spiked before decreasing again. This fluctuation suggests that the model **began to overfit** the training data at a certain point.

### Test Split Metrics

| Metric    | Score  |
| --------- | :----: |
| Accuracy  | 0.9394 |
| Recall    | 0.6978 |
| Precision | 0.5569 |
| F1 Score  | 0.6194 |

While the **accuracy** is high, the **recall** is insufficient for the intended application—meaning **30% of PPI-related publications would be missed**. Additionally, nearly **50% of selected abstracts would be false positives**.

## Conclusion and Perspectives

The model's performance is **not yet sufficient** for its intended use. The primary issue is **overfitting due to the imbalanced training data**. Several strategies could improve performance:

### 1. Using a More Powerful Base Model

A larger model could improve results, but it would require significantly more computational resources.

### 2. Addressing Overfitting to Specific Terms

The model may be learning **virus and protein names** instead of actual sentence structures. A possible solution is to use a **named entity recognition (NER) model**, such as **[pubmedBERT NER Gene](https://huggingface.co/pruas/BENT-pubmedBERT-NER-Gene)**, to identify and **mask** these terms during preprocessing. This would improve generalization and potentially allow the model to work for **any pathogen**.

### 3. Improving Data Augmentation

Simply repeating positive examples is not enough. More advanced augmentation techniques could include:

- **Randomly masking tokens** in positive examples to prevent memorization.
- **Using the base model for masked token filling**, ensuring that repeated examples are slightly different.
- **Generating synthetic positive examples** using a fine-tuned language model, followed by manual precuration to validate them.
