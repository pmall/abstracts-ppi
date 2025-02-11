# Labelling scientific abstracts about protein-protein interactions

## Source of the dataset

High quality dataset of almost 100k (93560) manually curated abstracts.

Abstracts are labelled into two categories: the publication is describing a protein-protein interaction (ppi) or not.

Publications describing ppi falls into two categories: human-human ppi (hh ppi) or human-viral ppi (vh ppi).

Steps creating the dataset were:

1. A detailed query is run on pubmed every month to retrieve all publications containing keywords related to ppi
2. The request yields multiple hundreds, thousands of publication every month
3. A small team of biologists manually review all the abstracts to discard those obviously not related to ppi (pre-curation)
4. All publications passing the pre-curation step is reviewed by biologists to extract ppi information - remaining non ppi publication are discarded

## Goal of the model

The goal is to replace step 3 with a model to pre-curate the abstracts.

It would be achieved by fine-tuning an existing pre-trained model familiar with english to create a classification model.

The final model would take an abstract as input and predict probabilities this is the abstract of a publication containing information about about ppi or not.

The classification would then be over two categories: ppi positive or ppi negative.

The original dataset also contains information about the publication being about hh ppi or vh ppi. The model could then even have 3 labels, hh ppi positive, vh ppi positive, ppi negative.

Due to the pre-curation/curation process, the model should aim to minimize as much as possible the false negatives (abstracts labelled as ppi negative while they are actually ppi positive). False positive are not such a concern because they will be wiped out during the curation step anyway.

## Dataset unbalance

The source dataset contains ~5k vh ppi, ~10k hh ppi and ~77k non ppi abstracts. So guessing ppi negative every time would result in 77% of good predictions.

As a first approach, vh ppi and hh ppi can be mixed to achieve a ~15k ppi positive set. The non ppi examples could be randomly under-sampled to a ~15k ppi negative set.

The resulting dataset would be a 30k balanced dataset which seems reasonable for this kind of task. In order to explore a large portion of the examples, many balanced datasets could be created by randomly under-sampling the ppi negative set. Many models could be trained, or a single model could be trained on those multiple datasets.

## Pre-trained model

An usual pre-trained model for this kind of task is the BERT model (https://huggingface.co/google-bert/bert-base-uncased)

This model is trained on an english language corpus and is particularly suited for fine tuning. As described:

> Note that this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked) to make decisions, such as sequence classification, token classification or question answering.

BERT model has a context window of 512 tokens (~words). It seems reasonable for most publication title + abstract pairs. A first approach to deal with longer title + abstract pairs would be to truncate it to 512 tokens.

A more advanced approach would be to randomly sample windows of 512 tokens within the dataset of abstracts but this is a lot of preprocessing in comparison to the potential prediction improvement.

There exist two variations of the BERT model that could be suited for this task:

- sciBERT (https://huggingface.co/allenai/scibert_scivocab_uncased) trained on a corpus of scientific publications
- bioBERT (https://huggingface.co/dmis-lab/biobert-base-cased-v1.2) specifically trained on a corpus of biomedical publications

We will start with the original BERT model as it is widely used. We will then benchmark it against sciBERT and bioBERT models.

## Specific concern

Another concern I have with this modeling task is I fear the model learns more about protein names than about the english structure of the sentences in the abstracts. I hope the dataset is balanced enough so there are positive and negative examples for each protein names. The model could also fail at predicting when presented with an abstract containing unseen protein names. One improvement could be to mask the protein names in order to force the model to learn from the structure of the sentences. I don't know if this is an usual concern in ML/DL modeling.

This concern can be addressed by using an NER model specifically trained on biomedical data like https://huggingface.co/pruas/BENT-PubMedBERT-NER-Gene. Processing the whole dataset through a NER model could take time.

## First approach

- Randomly under-sample down to 15k the ppi negative set in order to have a 30k balanced dataset
- Randomly sample a validation set and a test set
- Ensure train, validation and test sets are balanced
- Assign 2 labels to each example: ppi positive and ppi negative
- Tokenize all title + abstracts pairs through the BERT tokenizer and truncate text longer than 512 tokens
- Fine tune the BERT model on those classification data
- Assess the performance of the model considering we aim for very low false negatives at the expense of false positives
