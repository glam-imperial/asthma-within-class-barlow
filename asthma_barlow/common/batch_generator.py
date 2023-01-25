from pathlib import Path

import tensorflow as tf

from common.augmentation import tf_specaugment

try:
    FixedLenFeature = tf.FixedLenFeature
    parse_single_example = tf.parse_single_example
    decode_raw = tf.decode_raw
    random_normal = tf.random_normal
    Iterator = tf.data.Iterator
    TFRecordDataset = tf.data.TFRecordDataset
except AttributeError:
    FixedLenFeature = tf.compat.v1.FixedLenFeature
    parse_single_example = tf.compat.v1.parse_single_example
    decode_raw = tf.compat.v1.decode_raw
    random_normal = tf.compat.v1.random_normal
    Iterator = tf.compat.v1.data.Iterator
    TFRecordDataset = tf.compat.v1.data.TFRecordDataset


# TODO: Padded shape for unequal size sequences.


class BatchGenerator:
    def __init__(self,
                 tf_records_folder,
                 is_training,
                 partition,
                 are_test_labels_available,
                 name_to_metadata,
                 input_type_list,
                 output_type_list,
                 batch_size,
                 buffer_size,
                 path_list=None,
                 use_autopad=False,
                 augmentation_configuration=None):
        self.tf_records_folder = tf_records_folder
        self.is_training = is_training
        self.partition = partition
        self.are_test_labels_available = are_test_labels_available
        self.name_to_metadata = name_to_metadata
        self.input_type_list = input_type_list
        self.output_type_list = output_type_list
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.path_list = path_list
        self.use_autopad = use_autopad
        self.augmentation_configuration = augmentation_configuration

        if (self.path_list is None) or (len(self.path_list) == 0):
            if isinstance(self.tf_records_folder, str):
                root_path = Path(self.tf_records_folder)
                self.path_list = [str(x) for x in root_path.glob('*.tfrecords')]
            elif isinstance(self.tf_records_folder, list):
                self.path_list = list()
                for folder in tf_records_folder:
                    root_path = Path(folder)
                    self.path_list.extend([str(x) for x in root_path.glob('*.tfrecords')])
            else:
                raise ValueError

        print("Number of files:", len(self.path_list))

    def get_tf_dataset(self):
        dataset = TFRecordDataset(self.path_list,
                                  num_parallel_reads=8)

        features_dict = dict()
        for attribute_name, attribute_metadata in self.name_to_metadata.items():
            dtype = attribute_metadata["tfrecords_type"]
            variable_type = attribute_metadata["variable_type"]
            if not ((self.partition == "test") and (not self.are_test_labels_available) and (variable_type == "y")):
                features_dict[attribute_name] = FixedLenFeature([], dtype)

        dataset = dataset.map(lambda x: parse_single_example(x,
                                                             features=features_dict),
                              num_parallel_calls=8)

        def map_func(attribute):
            for attribute_name, attribute_value in attribute.items():
                attribute_metadata = self.name_to_metadata[attribute_name]
                variable_type = attribute_metadata["variable_type"]
                shape = self.name_to_metadata[attribute_name]["shape"]
                # shape = self.name_to_metadata[attribute_name]["numpy_shape"]
                dtype = self.name_to_metadata[attribute_name]["tf_dtype"]

                if variable_type == "id":
                    attribute[attribute_name] = tf.cast(tf.reshape(attribute[attribute_name],
                                                                   shape),
                                                        dtype)
                elif variable_type in ["x", "y", "support"]:
                    attribute[attribute_name] = tf.reshape(decode_raw(attribute[attribute_name],
                                                                         dtype),
                                                           shape)
                    if variable_type == "x":
                        if self.augmentation_configuration is not None:
                            if "amplitude_aug" in self.augmentation_configuration.keys():
                                if attribute_name in self.augmentation_configuration["amplitude_aug"][
                                    "inputs_to_augment"]:
                                    attribute[attribute_name] = attribute[attribute_name] * (1.0 + tf.random.uniform([],
                                                                                                                     minval=-
                                                                                                                     self.augmentation_configuration[
                                                                                                                         "amplitude_aug"][
                                                                                                                         "factor"],
                                                                                                                     maxval=
                                                                                                                     self.augmentation_configuration[
                                                                                                                         "amplitude_aug"][
                                                                                                                         "factor"],
                                                                                                                     dtype=tf.float32))

                            if "specaug" in self.augmentation_configuration.keys():
                                if attribute_name in self.augmentation_configuration["specaug"]["inputs_to_augment"]:
                                    attribute[attribute_name] = tf_specaugment.time_masking(
                                        mel_spectrogram=attribute[attribute_name],
                                        time_masking_para=self.augmentation_configuration["specaug"][
                                            "time_masking_para"],
                                        time_mask_num=self.augmentation_configuration["specaug"]["time_mask_num"])
                                    attribute[attribute_name] = tf_specaugment.frequency_masking(
                                        mel_spectrogram=attribute[attribute_name],
                                        frequency_masking_para=self.augmentation_configuration["specaug"][
                                            "frequency_masking_para"],
                                        frequency_mask_num=self.augmentation_configuration["specaug"][
                                            "frequency_mask_num"])

                            if "input_gaussian_noise" in self.augmentation_configuration.keys():
                                if attribute_name in self.augmentation_configuration["input_gaussian_noise"][
                                    "inputs_to_augment"]:
                                    attribute[attribute_name] = attribute[attribute_name] + random_normal(
                                        tf.shape(attribute[attribute_name]), mean=.0,
                                        stddev=self.augmentation_configuration["input_gaussian_noise"][
                                            "standard_deviation"])
                else:
                    raise ValueError

            input_list = list()
            output_list = list()

            for input_type in self.input_type_list:
                input_list.append(attribute[input_type])

            for output_type in self.output_type_list:
                output_list.append(attribute[output_type])

            return tuple(input_list), tuple(output_list)

        dataset = dataset.map(map_func,
                              num_parallel_calls=8)

        if self.is_training:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        input_padded_shapes = list()
        for input_type in self.input_type_list:
            if self.use_autopad:
                padded_shape = self.name_to_metadata[input_type]["padded_shape"]
            else:
                padded_shape = self.name_to_metadata[input_type]["numpy_shape"]
            input_padded_shapes.append(padded_shape)
        output_padded_shapes = list()
        if not ((self.partition == "test") and (not self.are_test_labels_available) and (variable_type == "y")):
            for output_type in self.output_type_list:
                if self.use_autopad:
                    padded_shape = self.name_to_metadata[output_type]["padded_shape"]
                else:
                    padded_shape = self.name_to_metadata[output_type]["numpy_shape"]
                output_padded_shapes.append(padded_shape)

        # print(dataset.output_types,
        #       dataset.output_shapes)

        dataset = dataset.padded_batch(self.batch_size,
                                       padded_shapes=(tuple(input_padded_shapes),
                                                      tuple(output_padded_shapes)))
        take_num = len(self.path_list)
        dataset = dataset.take(take_num)
        # if self.partition == "train":
        #     if take_num % self.batch_size != 0:
        #         take_num = take_num - (take_num % self.batch_size)
        #         take_num = take_num + (self.batch_size - (take_num % self.batch_size))
        # dataset = dataset.take(take_num)

        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        iterator = Iterator.from_structure(dataset.output_types,
                                                   dataset.output_shapes)

        # print(dataset.output_types,
        #       dataset.output_shapes)

        next_element = iterator.get_next()

        init_op = iterator.make_initializer(dataset)

        return dataset, iterator, next_element, init_op
