# Dog Classifier Streamlit App

Dog Classifier is a Streamlit app that takes a user uploaded .jpg image of a dog and predicts it's breed. The classifier is a fine-tuned EfficientNetB0 neural network that achieves 81% accuracy on the stanford_dogs Tensorflow Dataset.


## Usage

To run this application, you'll need [Git][https://git-scm.com/] installed on your machine. From your command line:

'''bash
# Clone this repository
$ git clone https://github.com/gqmz/dog_classifier.git

# Go in the repository
$ cd dog_classifier

# Install dependencies
$ pip install -r requirements.txt

# Run the app on localhost
$ streamlit run main.py
'''

The repository is setup to be deployed on [Heroku][https://www.heroku.com/]