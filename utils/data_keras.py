import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#image tensor pre-preprocessing
def preprocess_image(image, label, img_shape=224):
    """
    resize image tensor to (img_shape, img_shape, color_channels) & convert to float32 
    Args:
        image (tensor): 
        label (tensor):
        img_shape (int): height & width of output tensor
    """
    image = tf.image.resize(image, (img_shape, img_shape))
    image = tf.cast(image, dtype=tf.float32)

    return image, label

class KData():
    def __init__(self):
        return

    def load_data(self, name='stanford_dogs'):
        """
        Load dataset
        Returns:
            train_set, test_set, class_names
        """
        (train_set, test_set), ds_info = tfds.load(name='stanford_dogs',
                                                    split=['train', 'test'],
                                                    as_supervised=True,
                                                    with_info=True,
                                                    shuffle_files=False)
        class_names = ds_info.features['label'].names
        return train_set, test_set, class_names

    def get_train_batches(self, train_set):
        """
        Batch training data
        Args:
            train_set (tf dataset)
        Returns:
            train_set: batched dataset
        """
        train_set = train_set.map(map_func=preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        train_set = train_set.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)
        return train_set

    def get_test_batches(self, test_set):
        """
        Batch test data
        Args:
            test_set (tf dataset)
        Returns:
            test_set: batched dataset
        """
        test_set = test_set.map(map_func=preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        test_set = test_set.batch(32).prefetch(tf.data.AUTOTUNE)
        return test_set

    def get_data_labels(self, data_set):
        """
        Extract class labels as 1D array
        data_set is an array of (image, label) tuples - return an array of labels
        after concatenating along axis=0
        Args:
            data_set (tf dataset): train or test
        """
        return np.concatenate([y for x,y in data_set], axis=0)

    def get_random_sample(self, data_set, class_names, *, n=1):
        """
        Take random sample from data_set
        data_set (tf dataset): train or test - must be unbatched!
        n (int): number of samples to take from data_set
        """
        sample = data_set.take(count=n)

        #sample details
        for image, label in sample:
            # print(f"""
            # Image shape: {image.shape}
            # Image dtype: {image.dtype}
            # Target class from stanford_dogs (tensor form): {label}
            # Target class (str form): {class_names[label.numpy()]}
            # """)
            #plot an image tensor
            
            plt.imshow(image)
            plt.title(class_names[label.numpy()])
            plt.axis(False)