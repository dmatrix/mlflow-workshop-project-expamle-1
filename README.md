# MLflow Project Keras Example 

![](images/nn_linear_regression.png)

This example demonstrates how you can package an [MLflow Project](https://mlflow.org/docs/latest/projects.html) into GitHub and share it with 
others to reproduce runs.

Problem: Build a simple Linear NN Model that predicts Celsius temperaturers from training data with Fahrenheit degree

![](images/temperature-conversion.png)


You can execute this project from your local host as:

* ```mlflow run git@github.com:dmatrix/mlflow-workshop-project-expamle-1.git -P batch_size=5 -P epochs=500```
* ```mlflow run https://github.com/dmatrix/mlflow-workshop-project-expamle-1 -P batch_size=10 -P epochs=1000```
