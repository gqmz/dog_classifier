#tensorflow
import tensorflow as tf

#sklearn
from sklearn.metrics import confusion_matrix

#plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#general
import numpy as np
from itertools import product
import random
import os
import pandas as pd
import datetime
import zipfile

#MODEL EVALUATION
def plot_confusion_matrix(y_true, y_pred, classes=None, figsize=(10,10), text_size=10, norm=False, savefig=False):
    """
    Make labelled confusion matrix with rows:ground truth, columns: predictions
    If classes is passed, confusion matrix will be labelled, 
    else integer class values will be used

    Args:
        y_true (np.array): ground truth classes (must be same shape as y_pred)
        y_pred (np.array): predicted classes (must be same shape as y_true)
        classes (array) : array of class labels (str). If 'None', int labels are used
        figsize (tuple): size of output figure
        text_size: size of output figure text
        norm (bool): normalize displayed values or not
        savefig (bool): save confusion matrix to file or not
    """
    assert y_pred.shape == y_true.shape, "ground truth and predictions must be same shape"

    #create confusion matrix
    cm = confusion_matrix(y_true, tf.round(y_pred))
    cm_norm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis] #normalize the confusion matrix
    n_classes = cm.shape[0]

    #matrix plot
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    #create classes
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    #plot labels
    ax.set(
        title='Confusion Matrix',
        xlabel='Predicted Label',
        ylabel='True Label',
        xticks=np.arange(n_classes), #create enough ticks to cover all classes
        yticks=np.arange(n_classes),
        xticklabels=labels, #label ticks
        yticklabels=labels
    )

    #set x-axis labels to bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    #adjust label size
    ax.xaxis.label.set_size(text_size)
    ax.yaxis.label.set_size(text_size)
    ax.title.set_size(text_size)

    ### Added: Rotate xticks for readability & increase font size (required due to such a large confusion matrix)
    plt.xticks(rotation=70, fontsize=text_size)
    plt.yticks(fontsize=text_size)

    #plot text each cell
    threshold = (cm.max() + cm.min())/2.
    for i,j in product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i,
                    f"{cm[i,j]} ({cm_norm[i,j]*100:.1f}%)",
                    horizontalalignment='center',
                    color="white" if cm[i,j] > threshold else "black",
                    size=text_size)
        else:
            plt.text(j, i,
                    f"{cm[i,j]}",
                    horizontalalignment='center',
                    color="white" if cm[i,j] > threshold else "black",
                    size=text_size)

    #save figure
    if savefig:
        fig.savefig("confusion_matrix.png")

def predict_and_plot(filename, model, class_names):
    """
    make prediction on filename using model and plot image + prediction
    """
    #image to sized
    img = jpeg_to_tensor(filename, img_shape=224)
    #make prediction
    pred = model.predict(tf.expand_dims(img, axis=0)) #add dimension for batch

    #assign class to prediction
    if len(pred[0]) > 1: #multiclass prediction
        prediction_label = class_names[pred.argmax()]
    else: #single class
        prediction_label = class_names[int(tf.round(pred[0][0]))]

    #plot image and prediction
    plt.figure(figsize=(10,7))
    plt.imshow(img)
    plt.title(f"Prediction: {prediction_label}")
    plt.grid(False)

def plot_loss_curves(history):
    """
    plot loss curves from history object
    assumes metric is accuracy
    """
    #loss curves
    df = pd.DataFrame(history.history)
    df['epoch'] = df.index + 1

    plt.figure(figsize=(10,5), tight_layout=True)
    plt.subplot(1,2,1)
    plt.plot(df['epoch'], df['loss'], label='Training loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation loss')
    plt.xlabel('epochs'); plt.ylabel('Loss'); plt.title('Loss')
    plt.grid('on'); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(df['epoch'], df['accuracy'], label='Training accuracy')
    plt.plot(df['epoch'], df['val_accuracy'], label='Validation accuracy')
    plt.xlabel('epochs'); plt.ylabel('Accuracy'); plt.title('Accuracy')
    plt.grid('on'); plt.legend()

def compare_historys(history1, history2):
    """
    Compare two model history objects. Assume history2 resumes after history1 ends
    Args:
        history1 (dict): history of model trained first
        history2 (dict): history of model trained next
    """
    #pre-processing histories
    history1_df = pd.DataFrame(history1.history)
    history1_df['epoch'] = history1_df.index

    history2_df = pd.DataFrame(history2.history)
    history2_df['epoch'] = history2_df.index + history1_df['epoch'].max() + 1

    #accumulatation
    df = pd.concat([history1_df, history2_df], axis=0)

    #plot    
    plt.figure(figsize=(10,5), tight_layout=True)
    plt.subplot(1,2,1)
    plt.plot(df['epoch'], df['loss'], label='Training loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation loss')
    plt.xlabel('epochs'); plt.ylabel('Loss'); plt.title('Loss')
    plt.grid('on'); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(df['epoch'], df['accuracy'], label='Training accuracy')
    plt.plot(df['epoch'], df['val_accuracy'], label='Validation accuracy')
    plt.xlabel('epochs'); plt.ylabel('Accuracy'); plt.title('Accuracy')
    plt.grid('on'); plt.legend()

#DATA LOADING
#view a random image
def view_random_image(target_dir, target_class):
    """
    Args:
        target_dir: train or test directory
        target_class: label of target class ()
    Returns:
        numpy array of image
    """
    
    #set up target folder
    data_dir = target_dir.joinpath(target_class)

    #pick random image
    image = random.choice(os.listdir(data_dir))

    #plot image
    img = mpimg.imread(data_dir.joinpath(image)) #np.array of image
    plt.imshow(img)
    plt.title(target_class)
    plt.axis('off')

    print(f"Image shape {img.shape}")
    return img

# Create function to unzip a zipfile into current working directory 
def unzip_data(filename):
  """
  Unzips filename into the current working directory.

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  with zipfile.ZipFile(filename, "r") as zip_ref:
    zip_ref.extractall()

def jpeg_to_tensor(filename, img_shape=224, scale=False):
    """
    convert jpeg image to resized & normalized tensor
    Args:
        filename (str path): path to image file
        img_shape (int): dimension of square image
        scale (bool): True - scale pixel values to [0,1]
    """
    #read file
    img = tf.io.read_file(filename)
    #convert into tensor
    img = tf.image.decode_image(img)
    #resize image to target size
    img = tf.image.resize(img, size=(img_shape, img_shape))
    #normalization
    if scale:
        img = img/255.
    return img

def walk_through_dir(directory):
    """
    Walk through directory and print out details
    Args:
        directory (str or Pathlib object): directory to walk through
    """
    for dirpath, dirnames, filenames in os.walk(directory):
        print(f"There are {len(dirnames)} directories and {len(filenames)} filenames in {dirpath}")

#CALLBACKS
#create tensorboard callback
def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a tensorboard callback to save logs to a directory
    
    Args:
        dir_name (Pathlib path)
        experiment name (str)
    Returns:
        Tensorboard callback
    """
    #define log directory
    # log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = dir_name.joinpath(experiment_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
    )
    print(f"Saving Tensorboard log files to: {log_dir}")
    return tensorboard_cb
