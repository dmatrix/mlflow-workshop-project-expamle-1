import sys
import warnings
import tensorflow as tf
import keras
import numpy as np
import mlflow.keras
import mlflow.pyfunc

# source: https://androidkt.com/linear-regression-model-in-keras/
# Modified and extended

#
# Generate Data
#

def f2c(f):
  return (f - 32) * 5.0/9.0

def gen_data():
    X_fahrenheit = np.arange(-212, 10512, 2, dtype=float)
    # Randomize the input
    np.random.shuffle(X_fahrenheit)
    y_celsius = np.array(np.array([f2c(f) for f in X_fahrenheit]))

    predict_data =[]
    [predict_data.append(t) for t in range (212, 170, -5)]

    return (X_fahrenheit, y_celsius, predict_data)

def predict_keras_model(uri, data):
  model = mlflow.keras.load_model(uri)
  return [(f"(F={f}, C={model.predict([f])[0]})") for f in data]
#
# Define the model
#
def baseline_model():
   model = keras.Sequential([
      keras.layers.Dense(64, activation='relu', input_shape=[1]),
      keras.layers.Dense(64, activation='relu'),
      keras.layers.Dense(1)
   ])

   optimizer = keras.optimizers.RMSprop(0.001)

   # Compile the model
   model.compile(loss='mean_squared_error',
                 optimizer=optimizer,
                 metrics=['mean_absolute_error', 'mean_squared_error'])
   return model

def mlflow_run(params, X, y, run_name="Keras Linear Regression"):
    # Start MLflow run and log everyting...
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_uuid
        exp_id = run.info.experiment_id

        model = baseline_model()
        # single line of MLflow Fluent API obviates the need to log
        # individual parameters, metrics, model, artifacts etc...
        # https://mlflow.org/docs/latest/python_api/mlflow.keras.html#mlflow.keras.autolog
        mlflow.keras.autolog()
        model.fit(X, y, batch_size=params['batch_size'], epochs=params['epochs'])

        return (exp_id, run_id)

# Use the model
if __name__ == '__main__':
   # suppress any deprecated warnings
   warnings.filterwarnings("ignore", category=DeprecationWarning)
   print(f"mlflow version={mlflow.__version__};keras version={keras.__version__};tensorlow version={tf.__version__}")
   batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
   epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
   params = {'batch_size': batch_size, 'epochs': epochs}
   (X, y, predict_data) = gen_data()
   (exp_id, run_id) = mlflow_run(params, X,y)

   print(f"Finished Experiment id={exp_id} and run id = {run_id}")

   # Load this Keras Model Flavor as a Keras native model flavor and make a prediction
   model_uri = f"runs:/{run_id}/model"
   print(f"Loading the Keras Model={model_uri} as Keras Model")
   predictions = predict_keras_model(model_uri, predict_data)
   print(predictions)

