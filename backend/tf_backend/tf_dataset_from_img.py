import tensorflow as tf
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE

import argparse


"""
Dataset with images utils
"""

def _decode_img(img, nDim, convert_to_float=True, resize=True):
    """
    Loads an image and processes it 
    Returns an image with size nDim
    and numbers between 0 and 1
    """

    img_height, img_width, img_channels = nDim
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=img_channels)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    if convert_to_float:
        img = tf.image.convert_image_dtype(img, tf.float32)

    if resize:
        # resize the image to the desired size.
        return tf.image.resize(img, [img_height, img_width])
    else:
        return img


def _process_path(file_path, nDim, convert_to_float=True, resize=True):
    """
    Returns the processed image from its file path
    """
    img = tf.io.read_file(file_path)
    img = _decode_img(img, nDim, convert_to_float, resize)
    return img


def _process_label(label_indices, nClasses):
    """
    Returns the label in a one-hot encoder form, 
    from a sparse vector from [label_1,...,label_k]
    """
    # Not Optimal
    #     return tf.math.reduce_sum(tf.one_hot(label_indices, nClasses), axis=0)

    #Not Optimal
    label_indices = label_indices[label_indices < nClasses]
    label_indices_stacked = tf.transpose(
        tf.stack([label_indices, tf.zeros_like(label_indices)])
    )
    sparse_label = tf.sparse.SparseTensor(
        label_indices_stacked, tf.ones_like(label_indices), [nClasses, 1]
    )
    return tf.reshape(tf.sparse.to_dense(sparse_label), [-1])


"""
TF Records Utils
"""

# The following functions can be used to convert a value to a type compatible
# with tf.Example.


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy(
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint list"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _serialize_example_path(file_path, label):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'label': _int64_list_feature(label),
        'file_path': _bytes_feature(file_path),
    }
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example_proto.SerializeToString()


def _tf_serialize_example_path(file_path, label):
    tf_string = tf.py_function(
        serialize_example,
        (file_path, label),  # pass these args to the above function.
        tf.string
    )  # the return type is `tf.string`.
    return tf.reshape(tf_string, ())  # The result is a scalar


def _serialize_example_img(file_path, label, nDim):
    img = _process_path(file_path, nDim)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    img_bytes = tf.image.encode_jpeg(img)
    feature = {
        'label': _int64_list_feature(label),
        'image_raw': _bytes_feature(img_bytes),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example_proto.SerializeToString()


class Dataset_img_to_tf:
    """
    Class Dataset which takes the image directly in input
    Saves the dataset in a tfRecords format with the labels and the img_path
    Returns X, y where X has size (batch_size, nDim) and y has size (batch_size, nClasses)
    """
    def __init__(
        self,
        path_to_data,
        path_to_tf_records,
        nClasses,
        nDim,
        batch_size=512,
        shuffle=True,
        shuffle_buffer_size=1000,
        prefetch_buffer_size="AUTOTUNE",
        cache=True,
        batch_and_fetch=True,
        process_data=True,
        use_input_layers_only=False,
        is_ds_saved_in_tf_record=False,
        is_img_saved_in_tf=False,

    ):
        # Save all parameters
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        if prefetch_buffer_size == "AUTOTUNE":
            self.prefetch_buffer_size = AUTOTUNE
        else:
            self.prefetch_buffer_size = prefetch_buffer_size
        self.cache = cache
        self.batch_and_fetch = batch_and_fetch
        self.process_data = process_data

        self.path_to_data = path_to_data
        self.path_to_tf_records = path_to_tf_records
        self.is_ds_saved_in_tf_record = is_ds_saved_in_tf_record

        self.nClasses = nClasses
        self.nDim = nDim
        self.use_input_layers_only = use_input_layers_only

        self.is_img_saved_in_tf = is_img_saved_in_tf

        # Write labels and id in tf_records if needed
        if not (self.is_ds_saved_in_tf_record):
            self.write_ds_into_tf_records()

        # Load dataset from a tf_record format
        # If shuffle=True, also shuffles the data
        self.load_ds_from_tf_records()

        # Batches and Prefetchs the dataset
        if self.batch_and_fetch:
            self.prepare_for_training()

    def load_ds_from_tf_records(self, path_to_tf_records=None):
        if path_to_tf_records is not None:
             self.path_to_tf_records = path_to_tf_records
        print("Loading from: ",
            self.path_to_tf_records + "/data.tfrecord")
        
#         file_img_dataset = tf.data.TFRecordDataset(
#             self.path_to_tf_records + "/data.tfrecord*")

        files = tf.data.Dataset.list_files(
            self.path_to_tf_records + "/data.tfrecord*"
        )
        file_img_dataset = files.interleave(
            tf.data.TFRecordDataset,
            cycle_length=AUTOTUNE,
            num_parallel_calls=AUTOTUNE
        )

        # Create a dictionary describing the features.
        if self.is_img_saved_in_tf:
            image_feature_description = {
                'label': tf.io.VarLenFeature(tf.int64),
                'image_raw': tf.io.FixedLenFeature([], tf.string),
            }
        else:
            image_feature_description = {
                'label': tf.io.VarLenFeature(tf.int64),
                'file_path': tf.io.FixedLenFeature([], tf.string),
            }

        def _parse_example(example_proto):
            # Parses the input tf.Example proto using the dictionary above.
            parsed_example = tf.io.parse_single_example(
                example_proto, image_feature_description
            )
            if self.is_img_saved_in_tf:
                return parsed_example['image_raw'], tf.sparse.to_dense(
                    parsed_example['label']
                )
            else:
                return parsed_example['file_path'], tf.sparse.to_dense(
                    parsed_example['label']
                )

        # Parses all the examples in our tf_records file
        parsed_img_dataset = file_img_dataset.map(
            _parse_example, num_parallel_calls=AUTOTUNE
        )

        # Shuffles the dataset prior to reading the images
        if self.shuffle:
            parsed_img_dataset = parsed_img_dataset.shuffle(
                buffer_size=self.shuffle_buffer_size
            )

        # Maps the file_path to the image and returns the label in a one-hot format
        def _process_label_and_path(file_path, label):
            if self.use_input_layers_only:
                label = _process_label(label, self.nClasses)
                return {
                    "input_1": _process_path(file_path, self.nDim),
                    "input_label": label
                }, label
            else:
                return _process_path(file_path, self.nDim), _process_label(
                    label, self.nClasses
                )

        def _process_label_and_img(img, label):
            label = _process_label(label, self.nClasses)
            img = _decode_img(img, self.nDim)
            if self.use_input_layers_only:   
                return {"input_1": img, "input_label": label}, label
            else:
                return img , label

        if self.process_data:
            if self.is_img_saved_in_tf:
                self.dataset = parsed_img_dataset.map(
                    _process_label_and_img, num_parallel_calls=AUTOTUNE
                )
            else:
                self.dataset = parsed_img_dataset.map(
                    _process_label_and_path, num_parallel_calls=AUTOTUNE
                )
        else:
            self.dataset = parsed_img_dataset

    def take(self, i):
        return self.dataset.take(i)

    def prepare_for_training(self):
        if self.cache:
            if isinstance(self.cache, str):
                self.dataset = self.dataset.cache(cache)
            else:
                self.dataset = self.dataset.cache()

        # Batches the dataset
        self.dataset = self.dataset.batch(self.batch_size)

        # Prefetches the dataset when training
        self.dataset = self.dataset.prefetch(
            buffer_size=self.prefetch_buffer_size
        )

    def write_ds_into_tf_records(self, id_to_label_numpy):
        self.dataset = tf.data.Dataset.list_files(
            self.path_to_data + "/*/*.jpg"
        )

        ### TO BE UPDATED
        def _get_label_with_file_path(file_path):
            return file_path, self.tf_get_labels_from_id(file_path)

        self.dataset = self.dataset.map(
            _get_label_with_file_path, num_parallel_calls=AUTOTUNE
        )
        self.dataset = self.dataset.map(
            _tf_serialize_example_path, num_parallel_calls=AUTOTUNE
        )

        if self.path_to_tf_records is None:
            self.path_to_tf_records = self.path_to_data

        self.dataset = self.dataset.apply(tf.data.experimental.ignore_errors())

        print("Starting writing dataset in tfRecords in: ")
        print(self.path_to_tf_records + "/data.tfrecord")

        writer = tf.data.experimental.TFRecordWriter(
            self.path_to_tf_records + "/data.tfrecord"
        )
        writer.write(self.dataset)
        self.is_ds_saved_in_tf_record = True
        print("Done writing Dataset in tfrecords")

    def transform_from_file_path_to_img(self, new_path, num_shards=100):
        tf.io.gfile.mkdir(new_path)
        if not (self.is_img_saved_in_tf):
            print("Starting writing dataset in tfRecords in: ")
            print(new_path + "/data.tfrecord")

            def reduce_func(key, dataset):
                filename = tf.strings.join(
                    [new_path + "/data.tfrecord",
                     tf.strings.as_string(key)]
                )
                writer = tf.data.experimental.TFRecordWriter(filename)
                writer.write(dataset.map(lambda _, x: x))
                return tf.data.Dataset.from_tensors(filename)

            def _serialize_example_img_with_known_dim(file_path, label):
                return _serialize_example_img(
                    file_path, label, nDim=self.nDim
                )

            def _tf_serialize_example_img(file_path, label):
                tf_string = tf.py_function(
                    _serialize_example_img_with_known_dim,
                    (file_path,
                     label),  # pass these args to the above function.
                    tf.string
                )  # the return type is `tf.string`.
                return tf.reshape(tf_string, ())

            serialized_dataset = self.dataset.map(
                _tf_serialize_example_img, num_parallel_calls=AUTOTUNE
            )

        return serialized_dataset
    
    def forward_and_save(self, model, path_to_write, layer_before_last=2, 
                         return_parsed_features=False, n_shards=5):
        tf.io.gfile.mkdir(path_to_write)
        
        model_for_inner_layer = tf.keras.models.Model(
            inputs=model.input, outputs=model.layers[-layer_before_last].output
        )

        
        def _forward_map(x, y):
            return model_for_inner_layer.predict(x), y.numpy()


        def _serialize_tf_records_forward(feature, label):
            feature = {
                'label': _int64_list_feature(tf.where(label)),
                'feature': _float_feature(feature),
            }
            example_proto = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            return example_proto.SerializeToString()

        def _tf_serialize_tf_records_forward(feature, label):
            tf_string = tf.py_function(
                _serialize_tf_records_forward,
                (feature, label),
                tf.string
            )
            if return_parsed_features:
                return tf.reshape(tf_string, ()), feature, label
            else:
                return tf.reshape(tf_string, ())
        
        writer_ds = []
        for i in range(n_shards):
            writer_ds.append(
                tf.io.TFRecordWriter(path_to_write + "/data.tfrecord" + str(i))
            )
    
        for i, (x,y) in enumerate(self.dataset):
            if i % 10 == 0:
                print(i)
            features, label = _forward_map(x, y)
            for k in range(features.shape[0]):
                example_k = _serialize_tf_records_forward(features[k,:], label[k,:])
                writer_ds[i % n_shards].write(example_k)
            
        for i in range(n_shards):
            writer_ds[i].close()
        
        
        

    

# Test that the dataset pipeline works as expected
if __name__ == "__main__":
    import pandas as pd
    
    data_folder = "OpenImagesData"
    tf_folder = "OpenImagesTfRecords"
    tf_folder_with_img = "OpenImagesTfRecordsWithImg"



    nDim = 2048
    img_width = 224
    img_height = 224
    img_channels = 3
    nClasses = 500

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="train", type=str)
    args = parser.parse_args()

    DATA = args.dataset
    
    path_to_scratch = "/britten1/maxcauch/"
    path_to_data = path_to_scratch + data_folder + "/" + DATA
    path_to_tf_records = path_to_scratch + tf_folder + "/" + DATA
    path_to_tf_records_with_img = path_to_scratch + tf_folder_with_img + "/" + DATA
    
    write_to_tf_records = True
    exportClasses = False

    ### Read labels in csv
    if write_to_tf_records:
        df_labels = pd.read_csv(
            path_to_data_labels + DATA + "/" + DATA +
            "-annotations-human-imagelabels-boxable.csv"
        )
        df_positive_labels = df_labels.loc[(df_labels.Confidence == 1), :]

        ### If needed export class names
        if exportClasses:
            CLASS_NAMES = np.array(
                list(
                    pd.value_counts(df_positive_labels["LabelName"])[:N_CLASSES].index
                )
            )

            np.save(path_to_scratch + "OpenImagesMetaData/classesId.npy", CLASS_NAMES)

        else:
            CLASS_NAMES = np.load(path_to_scratch + "OpenImagesMetaData/classesId.npy")

        ### Map every class name to an integer
        map_classes_to_integers = dict(zip(CLASS_NAMES, np.arange(len(CLASS_NAMES))))

        ### Only select labels in CLASS_NAMES and map them to their integers
        df_labels_selected = df_positive_labels.loc[
            df_positive_labels.LabelName.apply(lambda x: x in CLASS_NAMES), :]
        label_image_id = df_labels_selected.ImageID.values.astype("<U16")
        label_integers = df_labels_selected.LabelName.apply(
            lambda x: map_classes_to_integers[x]
        ).values

        del (df_labels)
        del (df_labels_selected)
        del (df_positive_labels)

        id_to_label_np = np.dstack((label_image_id, label_integers))[0, :, :]
        dt = np.dtype([('id', '<U16'), ('label', int)])
        id_to_label_np = np.zeros(len(label_image_id), dtype=dt)

        ### Build a dictionary between ids and labels
        id_to_label_np['id'] = label_image_id
        id_to_label_np['label'] = label_integers
        id_to_label_np = np.sort(id_to_label_np, order=['id', 'label'])
    else:
        id_to_label_np = None
    
    print("Finished Loading Labels - Building Dataset")
    ds = Dataset_img_to_tf(
        path_to_data = path_to_data,
        path_to_tf_records = path_to_tf_records,
        nClasses = 500, 
        nDim= (img_width, img_height, img_channels),
        use_input_layers_only=True,
        batch_size=128,
        shuffle=False,
        is_ds_saved_in_tf_record=not(write_to_tf_records),
        id_to_label_np=id_to_label_np
    )
    
    print("Testing")
    for x,y in ds.take(2):
        print(x.shape, y.shape)

    params_ds = {
        "path_to_data": path_to_data,
        "path_to_tf_records": path_to_tf_records,
        "nClasses": nClasses,
        "nDim": (img_width, img_height, img_channels),
        "use_input_layers_only": True,
        "batch_size": 1,
        "shuffle": False,
        "is_ds_saved_in_tf_record": True,
        "cache": False,
        "is_img_saved_in_tf": False,
        "process_data": False,
        "batch_and_fetch": False
    }
    
    ds = Dataset_img_to_tf(**params_ds)
    ds.dataset = ds.dataset.apply(tf.data.experimental.ignore_errors())
    ds.transform_from_file_path_to_img(
        path_to_tf_records_with_img)