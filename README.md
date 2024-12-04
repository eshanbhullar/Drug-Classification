# Drug Classification Project

## Project Title

**Drug Classification Using Machine Learning**

## Description

This project aims to classify drug types based on various health metrics using machine learning models. The dataset includes information such as age, gender, blood pressure, cholesterol, and sodium-to-potassium ratio for patients. The goal is to predict the drug that a patient is likely to be prescribed based on these factors.

The project uses algorithms such as Neural Networks (MLPClassifier) and Decision Trees (DecisionTreeClassifier) to perform the classification task.

## Getting Started

These instructions will guide you in setting up the project on your local machine for development and testing purposes.

### Prerequisites

You need to have Python installed on your computer, along with the following libraries:

- **pandas**: For data manipulation and analysis
- **numpy**: For numerical computing
- **scikit-learn**: For machine learning algorithms and model evaluation
- **matplotlib**: For visualizations

You can install these dependencies by running:

```bash
pip install pandas numpy scikit-learn matplotlib
```
Installing
Clone the repository: First, clone this repository to your local machine using the following command:

```bash
git clone https://github.com/yourusername/Drug-Classification.git
```
Navigate to the project directory:

```bash
cd Drug-Classification
```
Running the Code
Open your terminal or command prompt and navigate to the project directory.

Run the Python script:

```bash
python drug_classification.py
```
The script will:
Load and process the data.
Train the machine learning models (Neural Network and Decision Tree).
Display the evaluation results, including confusion matrices and classification reports.

<h1>Running the Tests</h1>
The tests in this project include evaluating the machine learning models (Neural Network and Decision Tree) using the following metrics:

<h2>Confusion Matrix: To visualize the classification performance.</h2>
Classification Report: To evaluate precision, recall, F1-score, and accuracy.</br>
Breakdown of Tests</br>
Neural Network (MLPClassifier): A Multi-Layer Perceptron model to predict the drug classification.</br>
Decision Tree (DecisionTreeClassifier): A decision tree model to predict the same.</br>
Both models are evaluated using classification_report and confusion_matrix from sklearn.metrics.</br>

python
Copy code
from sklearn.metrics import classification_report, confusion_matrix
