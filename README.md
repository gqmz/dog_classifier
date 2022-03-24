# Dog Classifier Streamlit App

Dog Classifier is a Streamlit app that takes a user-uploaded .jpg image of a dog and predicts it's breed. 
The classifier is a pre-trained [EfficientNetB0 network](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/EfficientNetB0) achieving 81% accuracy on the [stanford_dogs](https://www.tensorflow.org/datasets/catalog/stanford_dogs) dataset.

## Repository structure
```

```



## Usage

To run this application, you'll need [Git](https://git-scm.com/) installed on your machine. From your command line:

```
# Clone this repository
$ git clone https://github.com/gqmz/dog_classifier.git

# Go in the repository
$ cd dog_classifier

# Install dependencies
$ pip install -r requirements.txt
```

### Launch Streamlit app
```
# Run the app on localhost
$ streamlit run main.py
```
The repository is setup to be deployed on [Heroku](https://www.heroku.com/)

## Exploring neural network architectures for classification
Refer to [notebook](/training_notebooks/mlflow_training.ipynb) for experiments trained on Colab.
Experiment tracking with [MLflow on localhost](https://www.mlflow.org/docs/latest/tracking.html#scenario-1-mlflow-on-localhost). All experimental runs stored in ./mlruns. 

### Launch MLflow dashboard
```
# Launch MLflow dashboard
$ mlflow ui
```

### Model progression
Transfer learning feature extraction models were built using the ResNet50V2 & EfficientNetB0 architectures. While both overfit the training set, the latter has ~10% higher accuracy on the test set.

## Future work
Stay tuned for updates!