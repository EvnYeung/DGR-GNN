# DGR-GNN

This repository provides the implementation of the GNN-based fraud detector **DGR-GNN**, proposed in our paper "Dynamic Graph Rewiring for Fraud Detection Based on One-class Homophily".

## Dependencies

- pytorch==1.12.0
- torch\_geometric==2.3.1
- numpy==1.21.6
- pandas==1.3.5
- scikit\_learn==1.0.2
- scipy==1.7.3
- matplotlib==3.0.3

## Model Training

Train and test the model:

```python
# Modify the argument in "train.py" to specify the target dataset
python train.py
```
