"""
code to train & evaluate keras multiclass classification model

"""
#imports
import tensorflow as tf
import tempfile
import numpy as np
import os
import mlflow

from graphs_keras import KPlot
from data_keras import KData

class KTrain():
    def __init__(self):
        return

    def compile_and_fit(self, model, train_set, test_set, 
                        epochs=5, 
                        loss=tf.keras.losses.SparseCategoricalCrossentropy, 
                        optimizer=tf.keras.optimizers.Adam, 
                        lr=0.001, metric='accuracy', verbose=1, output_dir='\tmp'):
        """
        compile model & fit on train_set, test_set (assumes data is tf.Dataset)
        Args:
            model (keras model)
            train_set
            test_set
        Returns:
            history: history object returned by model.fit
        """
        print(f"Compiling model...")
        #compile
        model.compile(loss=loss(), optimizer=optimizer(learning_rate=lr),
                    metrics=[metric,])
        #callbacks
        print(f"Writing logs to {output_dir}")
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=output_dir)
        history = model.fit(train_set, epochs=epochs,
                            steps_per_epoch=len(train_set),
                            validation_data=test_set,
                            validation_steps=int(0.15*len(test_set)),
                            verbose=verbose,
                            callbacks=[tensorboard_cb, ])

        return history

    def evaluate_model(self, model, test_set):
        """
        evaluate model on test data
        Args:
            model (keras model)
            test_set (tf dataset)
        Returns:
            evaluation scores
        """
        return model.evaluate(test_set)

    def predict_model(self, model, test_set):
        """
        get prediction probabilities on test data
        Args:
            model (keras model)
            test_set (tf dataset)
        Returns:
            Probabilities of all samples in test_set (m, len(class_names))
        """
        return model.predict(test_set)

    def get_train_acc(self, history):
        acc = history.history['accuracy']
        return acc[-1]
    
    def get_train_loss(self, history):
        loss = history.history['loss']
        return loss[-1]

    def get_validation_acc(self, history):
        validation_acc = history.history['val_accuracy']
        return validation_acc[-1]

    def get_validation_loss(self, history):
        validation_loss = history.history['val_loss']
        return validation_loss[-1]

    def print_metrics(self, history):
        print(f"Train loss {self.get_train_loss(history)}")
        print(f"Train accuracy {self.get_train_acc(history)}")
        print(f"Validation loss {self.get_validation_loss(history)}")
        print(f"Validation accuracy {self.get_validation_acc(history)}")

    def train_model(self, model, train_set, test_set, class_names, *, 
                    epochs=1, lr=0.001, expt_name=None, run_name=None):
        """
        Args:
            model (keras model)

        """
        # temporary directories
        logs_dir = tempfile.mkdtemp() #temp directory for tensorboard logs
        graphs_dir = tempfile.mkdtemp() #images
        model_dir = tempfile.mkdtemp() #saved model

        #train model
        ktrain_obj = KTrain()
        history = ktrain_obj.compile_and_fit(model, train_set, test_set, epochs=epochs, lr=lr, output_dir=logs_dir)

        #evaluate & predict
        test_loss, test_accuracy = ktrain_obj.evaluate_model(model, test_set)
        y_test_pred = np.argmax(ktrain_obj.predict_model(model, test_set), axis=1)

        #plots
        kgraphs_obj = KPlot()
        kdata_obj = KData()
        y_test_true = kdata_obj.get_data_labels(test_set)
        loss_fig = kgraphs_obj.plot_loss_curves(history)
        cm_fig = kgraphs_obj.plot_confusion_matrix(y_true=y_test_true,
                                                y_pred=y_test_pred,
                                                classes=class_names,
                                                figsize=(35,35),
                                                text_size=8)
        f1_fig = kgraphs_obj.plot_classification_report(y_true=y_test_true,
                                                y_pred=y_test_pred,
                                                classes=class_names)

        #save images
        loss_fig.savefig(os.path.join(graphs_dir, 'loss.png'))
        cm_fig.savefig(os.path.join(graphs_dir, 'confusion.png'))
        f1_fig.savefig(os.path.join(graphs_dir, 'f1_report.png'))

        #log in mlflow
        mlflow.set_experiment(expt_name)
        with mlflow.start_run(run_name=run_name) as run:
            run_uuid = run.info.run_uuid
            print(f"MLflow Experiment ID: {run.info.experiment_id}")
            print(f"MLflow Run ID: {run.info.run_id}")

            #log parameters
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("lr", lr)

            #log metrics
            mlflow.log_metric("train_acc", ktrain_obj.get_train_acc(history))
            mlflow.log_metric("train_loss", ktrain_obj.get_train_loss(history))
            mlflow.log_metric("test_acc", test_accuracy)
            mlflow.log_metric("test_loss", test_loss)

            #log model - 2 ways
            mlflow.keras.log_model(model, "saved_model")
            model.save(os.path.join(model_dir, 'model.h5')) #saved below
            #log graphs
            mlflow.log_artifacts(graphs_dir, artifact_path="images")
            mlflow.log_artifacts(logs_dir, artifact_path="events")

            #write model summary to file
            summary = []
            model.summary(print_fn=summary.append)
            summary = "\n".join(summary)
            summary_file = os.path.join(model_dir, 'model_summary.txt')
            with open(summary_file, "w") as f:
                f.write(summary)
            mlflow.log_artifacts(model_dir, artifact_path='model_h5') #writes .h5 model & summary to artifact_path