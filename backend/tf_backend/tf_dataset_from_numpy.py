import tensorflow as tf
import numpy as np


def get_filename(batch_number, path_to_data):
    return path_to_data + "/features/features_batch_" + str(
        batch_number.numpy()
    ) + ".npy", path_to_data + "/labels/labels_batch_" + str(
        batch_number.numpy()
    ) + ".npz"


def get_data_from_filename(batch_number, path_to_data, nClasses):
    x_file, y_file = get_filename(batch_number, path_to_data=path_to_data)
    x = np.load(x_file)
    y_npz = np.load(y_file)
    id_indices = y_npz["id_indices"]
    label_indices = y_npz["label_indices"]
    y = np.zeros((x.shape[0], nClasses))
    y[id_indices, label_indices] = 1.0
    return x, y


def get_data_wrapper(batch_number, path_to_data, nClasses):
    features, labels = tf.py_function(
        lambda batch_number:
        get_data_from_filename(batch_number, path_to_data, nClasses),
        [batch_number],
        (tf.float32, tf.float32),
    )
    return tf.data.Dataset.from_tensor_slices((features, labels))


def get_dataset_from_numpy(nBatches, path_to_data, nClasses, nDim=2048):
    ds = tf.data.Dataset.from_tensor_slices(np.arange(nBatches))
    ds = ds.flat_map(
        lambda batch_number:
        get_data_wrapper(batch_number, path_to_data, nClasses)
    )

    def _local_set_shape(feature, label):
        feature.set_shape((nDim, ))
        label.set_shape((nClasses, ))
        return feature, label

    ds = ds.map(_local_set_shape)
    return ds


class Dataset_np_to_tf():
    def __init__(
        self,
        nBatches,
        path_to_data,
        nClasses,
        nDim,
        use_input_layers_only=False,
        batch_size=512,
        shuffle=True,
        shuffle_buffer_size=1000,
        prefetch_buffer_size=1000,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_buffer_size = prefetch_buffer_size
        
        self.nBatches = nBatches
        self.path_to_data = path_to_data
        self.nClasses = nClasses
        self.nDim = nDim
        self.use_input_layers_only = use_input_layers_only

        self.build_ds()

        if self.shuffle:
            self.dataset = self.dataset.shuffle(
                buffer_size=self.shuffle_buffer_size
            )
        self.dataset = self.dataset.prefetch(
            buffer_size=self.prefetch_buffer_size
        )
        self.dataset = self.dataset.batch(self.batch_size)

    def build_ds(self):
        self.dataset = tf.data.Dataset.from_tensor_slices(
            np.arange(self.nBatches)
        )
        self.dataset = self.dataset.flat_map(
            lambda batch_number:
            get_data_wrapper(batch_number, self.path_to_data, self.nClasses)
        )

        def _local_set_shape(feature, label):
            feature.set_shape((self.nDim, ))
            label.set_shape((self.nClasses, ))
            if self.use_input_layers_only:
                return {"input_feature":feature,"input_label":label},label
            else:
                return feature, label

        self.dataset = self.dataset.map(_local_set_shape)
        
    def take(self, i):
        return self.dataset.take(i)
    

# Test that the dataset pipeline works as expected
if __name__ == "__main__":
    path_to_scratch = "/scratch/maxcauch/"
    DATA = "train"
    path_to_data = path_to_scratch + "OpenImageProcessed/" + DATA
    nBatches = 1
    nClasses = 500

    ds = get_dataset_from_numpy(nBatches, path_to_data, nClasses)
    ds_bis = Dataset_np_to_tf(nBatches, path_to_data, nClasses, nDim=2048)

    for x, y in ds.take(10):
        print(x.numpy().shape, y.numpy().shape)

    for x, y in ds_bis.take(1):
        print(x.numpy().shape, y.numpy().shape)