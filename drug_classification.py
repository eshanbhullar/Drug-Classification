In [1]:
#Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
In [5]:
#Load Data
data = pd.read_csv(r"C:\Users\ssr98\Downloads\drugdataset (1).csv")
data.head()
Out[5]:
Age	Sex	BP	Cholesterol	Na_to_K	Drug
0	23	1	2	1	25.355	drugY
1	47	0	1	1	13.093	drugC
2	47	0	1	1	10.114	drugC
3	28	1	0	1	7.798	drugX
4	61	1	1	1	18.043	drugY
In [7]:
data.describe()
Out[7]:
Age	Sex	BP	Cholesterol	Na_to_K
count	200.000000	200.000000	200.000000	200.000000	200.000000
mean	44.315000	0.480000	1.090000	0.515000	16.084485
std	16.544315	0.500854	0.821752	0.501029	7.223956
min	15.000000	0.000000	0.000000	0.000000	6.269000
25%	31.000000	0.000000	0.000000	0.000000	10.445500
50%	45.000000	0.000000	1.000000	1.000000	13.936500
75%	58.000000	1.000000	2.000000	1.000000	19.380000
max	74.000000	1.000000	2.000000	1.000000	38.247000
In [9]:
data.Drug.unique()
Out[9]:
array(['drugY', 'drugC', 'drugX', 'drugA', 'drugB'], dtype=object)
In [11]:
#Create x and y variables
X = data.drop('Drug',axis=1).to_numpy()
y = data['Drug'].to_numpy()

#Create Train and Test datasets
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size = 0.20,random_state=100)

#Scale the data
from sklearn.preprocessing import StandardScaler  
sc = StandardScaler()  
x_train2 = sc.fit_transform(X_train)
x_test2 = sc.transform(X_test)
In [13]:
#Script for Neural Network
from sklearn.neural_network import MLPClassifier  
mlp = MLPClassifier(hidden_layer_sizes=(5, 4, 5),
                    activation='relu',solver='adam',
                    max_iter=10000,random_state=100)  
mlp.fit(x_train2, y_train) 
predictions = mlp.predict(x_test2) 

#Evaluation Report and Matrix
from sklearn.metrics import classification_report, confusion_matrix  
target_names=['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions,target_names=target_names))
[[ 5  0  0  0  0]
 [ 0  2  0  0  1]
 [ 0  0  3  0  0]
 [ 0  0  0 11  0]
 [ 0  0  0  1 17]]
              precision    recall  f1-score   support

       drugA       1.00      1.00      1.00         5
       drugB       1.00      0.67      0.80         3
       drugC       1.00      1.00      1.00         3
       drugX       0.92      1.00      0.96        11
       drugY       0.94      0.94      0.94        18

    accuracy                           0.95        40
   macro avg       0.97      0.92      0.94        40
weighted avg       0.95      0.95      0.95        40

In [15]:
#Script for Decision Tree
from sklearn.tree import DecisionTreeClassifier  

for name,method in [('DT', DecisionTreeClassifier(random_state=100))]: 
    method.fit(x_train2,y_train)
    predict = method.predict(x_test2)
    print('\nEstimator: {}'.format(name)) 
    print(confusion_matrix(y_test,predict))  
    print(classification_report(y_test,predict))
Estimator: DT
[[ 4  1  0  0  0]
 [ 0  3  0  0  0]
 [ 0  0  3  0  0]
 [ 0  0  0 10  1]
 [ 0  0  0  0 18]]
              precision    recall  f1-score   support

       drugA       1.00      0.80      0.89         5
       drugB       0.75      1.00      0.86         3
       drugC       1.00      1.00      1.00         3
       drugX       1.00      0.91      0.95        11
       drugY       0.95      1.00      0.97        18

    accuracy                           0.95        40
   macro avg       0.94      0.94      0.93        40
weighted avg       0.96      0.95      0.95        40

In [21]:
# Create Forecast Table comparing Actual vs Predicted
forecast_table = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

# Optionally, add a column to show if the prediction is correct or not (True/False)
forecast_table['Correct'] = forecast_table['Actual'] == forecast_table['Predicted']

# Print the Forecast Table
print('\nForecast Table:')
print(forecast_table.head())
Forecast Table:
  Actual Predicted  Correct
0  drugA     drugA     True
1  drugX     drugX     True
2  drugB     drugB     True
3  drugY     drugY     True
4  drugX     drugX     True
In [ ]:
  
