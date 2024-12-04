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

<h2>Confusion Matrix:</h2> To visualize the classification performance.</br>

<h2>Classification Report:</h2> To evaluate precision, recall, F1-score, and accuracy.</br>

<h2>Breakdown of Tests</h2>
<h3>1. Neural Network (MLPClassifier):</h3> A Multi-Layer Perceptron model to predict the drug classification.</br>
<h3>2. Decision Tree (DecisionTreeClassifier):</h3> A decision tree model to predict the same.</br>

Both models are evaluated using classification_report and confusion_matrix from sklearn.metrics.</br>

```
from sklearn.metrics import classification_report, confusion_matrix
```
<h2>Deployment</h2>
To deploy the model in a live system, you can follow these steps:

<h4>1. Train the model using a larger dataset.</h4>

<h4>2. Serialize the model using joblib or pickle.</h4>

```
import joblib
joblib.dump(mlp, 'drug_classification_model.pkl')
```

<h4>3. Deploy the serialized model on a web application or API for predictions.</h4>

<h1>Author</h1>
<h3>Eshandeep Singh -  GitHub: eshanbhullar</h3>

<h1>License</h1>
<h3>This project is licensed under the MIT License - see the LICENSE.md file for details.</h3>

<h1>Acknowledgments</h1>
Thanks to the creators of scikit-learn for providing powerful machine learning tools.</br>
Had tip to anyone whose code was used as inspiration or reference in this project.</br>
Special thanks to the contributors to the dataset, which was essential for training the models.</br>


