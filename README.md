# Reconstructing Social Network Structure with Limited Information

## Introduction
This study explores the possibility of using limited user information to reconstruct social
network structures, thereby assessing the risk of privacy leakage.  
Here, subgraphs generated by GNNExplainer are used as limited yet representative user information. The aim is to see to what extent reconstruction can be achieved with a small amount of explanatory subgraph data.

## Overview
The experiment uses two datasets, namely Facebook and GitHub, and it can primarily be divided into three stages:
- Stage1: Train the base GNN model. (Here, I use GCN)
- Stage2: Use GNNExplainer to obtain subgraph explanations.
- Stage3: Attempt to reconstruct the original graph using the subgraphs from GNNExplainer. (Link Prediction)
![overview](/plot/overview.png)

## File Description
- `module.py`: Performing data preprocessing on the subgraph generated by GNN Explainer, which will be used in stage 2.
- `Facebook_stage1&2.ipynb`: Stage 1 and Stage 2 on Facebook dataset.
- `Facebook_stage3.ipynb`: Stage 3 on Facebook dataset.
- `Github_stage1&2.ipynb`: Stage 1 and Stage 2 on Github dataset.
- `Github_stage3.ipynb`: Stage 3 on Github dataset.
