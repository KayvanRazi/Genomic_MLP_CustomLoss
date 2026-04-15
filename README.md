## Deep Learning for Genomic Selection
This script implements a Multi-Layer Perceptron (MLP) in R/Keras specifically designed for animal breeding scenarios.

### Key Highlights:
* **Custom Hybrid Loss:** Optimizes both Mean Squared Error and Pearson Correlation to maximize selection accuracy ($r_{g\hat{g}}$).
* **Regularization Suite:** Incorporates L2 regularization, Batch Normalization, and Gaussian Noise to handle high-dimensional SNP data (110k markers).
* **Automated Training:** Uses early stopping and model checkpointing based on validation correlation to ensure the best weights are preserved.
