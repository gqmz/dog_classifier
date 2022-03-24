"""
code to create plots for keras multiclass classification models
"""

#imports
import matplotlib.pyplot as plt
import pandas as pd
#sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np

class KPlot():
    def __init__(self):
        return

    def plot_loss_curves(self, history):
        """
        Generate matplotlib plot of accuracy and loss metrics
        Args:
            history: history object returned by fit function
        Returns:
            figure instance
        """
        #convert to dataframe
        df = pd.DataFrame(history.history)
        df['epoch'] = df.index + 1

        #make figure
        fig, ax = plt.subplots(1, 2, figsize=(8,5), tight_layout=True)

        ax[0].plot(df['epoch'], df['loss'], label='Training loss', c='r')
        ax[0].plot(df['epoch'], df['val_loss'], label='Validation loss', c='b')
        ax[0].set_xlabel('epochs'); ax[0].set_ylabel('Loss'); ax[0].set_title('Loss')
        ax[0].grid('on'); ax[0].legend()

        ax[1].plot(df['epoch'], df['accuracy'], label='Training accuracy', c='r')
        ax[1].plot(df['epoch'], df['val_accuracy'], label='Validation accuracy', c='b')
        ax[1].set_xlabel('epochs'); ax[1].set_ylabel('Accuracy'); ax[1].set_title('Accuracy')
        ax[1].grid('on'); ax[1].legend()

        plt.close('all')
        return fig

    def plot_confusion_matrix(self, y_true, y_pred, classes=None, figsize=(10,10), text_size=10, norm=False):
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
        Returns:
            figure instance
        """
        assert y_pred.shape == y_true.shape, "ground truth and predictions must be same shape"
        from itertools import product

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
        plt.xticks(rotation=90, fontsize=text_size)
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
        plt.close('all')
        return fig

    def plot_classification_report(self, y_true, y_pred, classes, figsize=(12, 25)):
        """
        Plot histogram of F1-score for each class
        Args:
            y_true (np.array): ground truth classes (must be same shape as y_pred)
            y_pred (np.array): predicted classes (must be same shape as y_true)
            classes (array) : array of class labels (str). If 'None', int labels are used
        Returns:
            figure instance
        """
        report = classification_report(y_true, y_pred, output_dict=True)

        #transform dictionary
        class_f1_scores = {}
        for k,v in report.items():
            if k == 'accuracy':
                break
            else:
                class_f1_scores[classes[int(k)]] = v['f1-score']
        df = pd.DataFrame({'class_name': list(class_f1_scores.keys()),
                            'f1_score': list(class_f1_scores.values())})
        df = df.sort_values(by='f1_score', ascending=False)

        #plot histogram
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

        cax = ax.barh(range(len(df)), df['f1_score'])
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(list(df['class_name']))
        ax.set_title("F1-Scores for 10 different classes")
        ax.set_xlabel("F1-score")
        ax.invert_yaxis()

        def autolabel(rects): # Modified version of: https://matplotlib.org/examples/api/barchart_demo.html
            # Attach a text label above each bar displaying its height (it's value).
            for rect in rects:
                width = rect.get_width()
                ax.text(1.03*width, rect.get_y() + rect.get_height()/1.5,
                        f"{width:.2f}",
                        ha='center', va='bottom')

        autolabel(cax)
        plt.close('all')
        return fig