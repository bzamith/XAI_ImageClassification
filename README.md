# XAI_ImageClassification

**Explainable AI (XAI) for Image Classification**  
This repository explores the application of XAI techniques to understand and improve image classification models.

## Overview

The project investigates the use of **local XAI methods** (e.g., LIME, SHAP, Grad-CAM) to explain predictions made by convolutional neural networks (CNNs) on various image datasets. The goal is to analyze model behavior, identify biases, and propose enhancements for better interpretability and performance.

## Features

- **CNN Architectures:** Simple and robust CNN models for image classification.
- **XAI Methods:**
  - **LIME:** Local Interpretable Model-Agnostic Explanations.
  - **SHAP:** SHapley Additive Explanations.
  - **Grad-CAM:** Gradient-weighted Class Activation Mapping.
- **Datasets:**  
  - Dogs vs. Wolves  
  - Multiclass insect classification (Beetles, Cockroach, Dragonflies)  
  - PneumoniaMNIST (Pneumonia vs. Normal)

## Results

- Highlighted differences in model focus (e.g., overemphasis on irrelevant features such as tags or backgrounds).  
- Demonstrated how XAI tools can reveal insights and suggest areas for improvement, such as removing noise or rebalancing data.  
- Showcased the robustness of Grad-CAM and SHAP in distinguishing features across classes.  

## Installation

1. Clone the repository:
```bash
   git clone https://github.com/bzamith/XAI_ImageClassification.git
   cd XAI_ImageClassification
```

2. Install dependencies:
```bash
	source env/bin/activate
	pip install -r requirements.txt
```

## Key Findings
- LIME: Intuitive visualizations but unstable across runs. Useful for highlighting "certain" and "uncertain" regions.
- SHAP: Excellent for comparing feature contributions across classes but less intuitive for quick insights.
- Grad-CAM: Strong visual feedback, especially in robust models, focusing on entire regions rather than scattered points.

## Future Work
Further reduce false negatives in pneumonia classification.
Address hyper-focus on specific patterns (e.g., leaves in insect classification).
Extend analysis to additional datasets and complex architectures.