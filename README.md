# Identifying-Bias-in-AI
## *Bias in Toxicity Classification* 

Exploring Unintended Bias in NLP Models Using the Civil Comments Dataset

## Overview

This project explores unintended bias in machine learning models for text classification, using real-world data from the Civil Comments platform and the Jigsaw Unintended Bias in Toxicity Classification Kaggle competition.

The goal is not just to train a model that detects toxic language, but to critically examine how bias emerges, how it affects different identity groups, and why naïvely “high accuracy” models can still cause harm in practice.

This repository is based on a Kaggle learning exercise, but extended and reframed to emphasize ethical ML, bias awareness, and interpretability rather than leaderboard optimization.

## Background

In late 2017, the Civil Comments platform shut down and released an open archive of approximately 2 million public comments. Jigsaw sponsored the effort to comprehensively annotate the data, including toxicity labels and identity-related attributes.

In 2019, Kaggle hosted the Jigsaw Unintended Bias in Toxicity Classification competition to encourage the ML community to develop models that:

Detect toxic comments

Avoid disproportionately misclassifying comments referencing specific identity groups

This dataset has since become a canonical example for studying bias in NLP systems.
Overview

The analysis is implemented in Google Colab for accessibility and reproducibility, and documented here for transparency.

## Dataset

Each comment includes:

- Free-text content

- A continuous toxicity score

For this project, comments are converted into a binary label:

- Toxic if target > 0.7

- Not toxic otherwise

The dataset is downloaded programmatically via kagglehub, ensuring reproducibility outside the Kaggle environment.

## Methodology

Text is vectorized using a bag-of-words representation (CountVectorizer)

A logistic regression classifier is trained on the resulting word counts

The model is evaluated using standard accuracy on a held-out test set

This intentionally simple setup prioritizes interpretability over sophistication, allowing direct inspection of learned word-level associations.

## Model Exploration

The notebook walks through several stages of investigation:

Basic sanity checks
 Testing the model on simple, hand-written sentences to confirm expected behavior.

Coefficient inspection
 Examining which words receive the highest toxicity coefficients.

Counterfactual testing
 Comparing predictions for sentences that are identical except for identity-related terms (e.g., religion or race).

These steps reveal how the model’s predictions can change solely due to identity tokens, even when intent and structure remain neutral.

## Key Findings

The model achieves high overall accuracy, but this masks problematic behavior.

Certain identity-related words receive disproportionately large toxicity coefficients.

Structurally identical sentences can flip from not toxic to toxic based only on the identity term used.

These patterns reflect correlations in the training data, not an understanding of harmful intent.

## Key Lessons
### *1. High Accuracy Does Not Mean Fair Behavior* 

Strong aggregate performance can conceal systematic errors affecting specific groups.

### *2. Models Learn Social Patterns, Not Intent*

The classifier responds to statistical correlations in the data rather than the meaning or intent of a sentence, leading to biased outcomes for identity-related language.

### *3. Interpretability Reveals Bias — It Doesn’t Prevent It* 

Using a simple, transparent model makes bias visible, but does not resolve it. Mitigation requires deliberate data, evaluation, and deployment choices.

### *4. Bias Emerges Across the ML Lifecycle* 

This project surfaces multiple forms of bias, including:

- Historical and representation bias in the data

- Measurement bias (e.g., through translation)

- Evaluation bias when test data mirrors training data

- Deployment bias when models are used in new linguistic or cultural contexts

No single fix addresses all of these.

### *5. Ethical Risk Is a Deployment Concern* 

Many of the most harmful effects only appear after deployment. Responsible use of toxicity classifiers requires ongoing monitoring, subgroup evaluation, and explicit consideration of who bears the cost of errors.

## Why There Is No Kaggle Autograder Here

The original Kaggle notebook relies on learntools and q_X.check() cells for automated grading.
These components are Kaggle-specific and are intentionally removed in this version.

They are replaced with written analysis and reflection, which better reflects real-world ML practice where correctness is demonstrated through reasoning, not passing an autograder.

## How to Run

Open the linked Google Colab notebook

Run cells sequentially (no local setup required)

The data-loading step takes ~30 seconds

## Outputs include:

1. Confirmation of data loading

2. Sample toxic and non-toxic comments

3. Model accuracy

4. Word-level coefficient analysis

5. Counterfactual bias probes

## Disclaimer

This project is educational.
The model is not suitable for real-world deployment without further fairness evaluation, mitigation strategies, and domain-specific testing.

## Acknowledgments

Civil Comments for releasing the dataset

Jigsaw for sponsoring the annotation effort
