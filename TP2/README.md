# 2do TRABAJO PRÁCTICO: Fashion-MNIST

## Librerias necesarias para ejecutar el programa:

pandas
numpy 
matplotlib.pyplot
os
duckdb
sklearn
sklearn.model_selection (usamos train_test_split, GridSearchCV)
sklearn.neighbors (usamos KNeighborsClassifier)
sklearn.tree (usamos DecisionTreeClassifier)
sklearn.metrics (usamos accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, classification_report)
random

Para poder correr el programa, se debe utilizar el dataSet pedido por la consigna, el cual debe ser guardado como "Fashion-MNIST.csv" en el mismo directorio donde se encuentre el archivo .py. 

Aclaración: El gráfico n°5 no va a aparecer en el código ya que apartamos el caso de hiperparámetro mf: 'None' para que el resultado de mejor hiperparámetros fuera el elegido por nosotros. Aparece en su lugar un gráfico con los casos 50%, log2 y sqr. Para obtener el grafico n°5 basta con agregar nuevamente None como opción en el grid search y correr el código.
