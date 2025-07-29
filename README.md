# AI & Python Project

This repository contains a collection of my AI and machine learning projects developed while learning Python and various AI techniques.

## 🎓 Learning Journey

### AI + Python for Beginners Course
I completed the [AI Python for Beginners course](https://learn.deeplearning.ai/courses/ai-python-for-beginners/lesson/z57gn/introduction) from DeepLearning.AI to build foundational skills in AI and Python programming. As part of this course, I created a project analyzing restaurants in Sydney, which helped me understand data processing and AI applications in real-world scenarios.

### Diffusion Models Exploration
Drawing from my previous coursework, I revisited and expanded on diffusion model projects. These models demonstrate generative AI capabilities and showcase different approaches to creating synthetic data through probabilistic modeling techniques.

## 🔢 Digit Recognition Project

### Overview
The digit recognition project implements and compares multiple machine learning algorithms to classify handwritten digits from the sklearn digits dataset. This project demonstrates practical application of classification algorithms and model evaluation techniques.

### Features
- **Multiple Algorithm Comparison**: 
  - Logistic Regression
  - K-Nearest Neighbors (KNN) with different k values
  - Support Vector Machine (SVM)
  - Random Forest Classifier

- **Comprehensive Evaluation**:
  - Cross-validation scoring for robust performance estimates
  - Confusion matrices for detailed classification analysis
  - Classification reports with precision, recall, and F1-scores
  - Misclassification analysis with visual examples

- **Data Visualization**:
  - Sample digit displays from the dataset
  - Model performance comparison charts
  - Feature importance heatmaps
  - Confusion matrices for top-performing models

### What the Model Does
The digit recognition system:
1. Loads the sklearn handwritten digits dataset (8x8 pixel images)
2. Preprocesses the data with feature scaling where appropriate
3. Trains multiple classification models simultaneously
4. Evaluates each model using cross-validation and test accuracy
5. Identifies the best-performing model automatically
6. Provides detailed analysis including misclassified examples
7. Visualizes results through multiple chart types and confusion matrices

## 🛠️ Technical Setup

### Virtual Environment
This project uses a Python virtual environment to manage dependencies and ensure reproducibility:

```bash
# Create virtual environment
python3 -m venv venv    

# Activate virtual environment
# On macOS:
source venv/bin/activate  

# Install required packages
pip install scikit-learn matplotlib numpy

# Run your code
python digitRecognition.py
```

## 🎯 Learning Outcomes
Through these projects, I've gained experience in:
- Python programming for AI applications
- Machine learning algorithm implementation and comparison
- Data preprocessing and feature engineering
- Model evaluation and validation techniques
- Data visualization and results interpretation
- Virtual environment management for Python projects
- Computing with numpy and scikit-learn

*This repository represents my journey in learning AI and machine learning, from foundational concepts to practical implementations.*
