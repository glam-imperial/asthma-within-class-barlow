import os
import collections

import tensorflow as tf


def process_metadata(name_to_metadata):
    for variable_name in name_to_metadata.keys():
        numpy_shape = name_to_metadata[variable_name]["numpy_shape"]
        variable_type = name_to_metadata[variable_name]["variable_type"]

        if variable_type == "x":
            tfrecords_type = tf.string
            tf_dtype = tf.float32
            shape = tuple([-1, ] + [d for d in numpy_shape[1:]])
            padded_shape = tuple([None, ] + [d for d in numpy_shape[1:]])
            placeholder_shape = tuple([None, None] + [d for d in numpy_shape[1:]])
        elif variable_type == "y":
            tfrecords_type = tf.string
            tf_dtype = tf.float32
            if len(numpy_shape) > 1:
                shape = tuple([-1, ] + [d for d in numpy_shape[1:]])
                padded_shape = tuple([None, ] + [d for d in numpy_shape[1:]])
                placeholder_shape = tuple([None, None] + [d for d in numpy_shape[1:]])
            else:
                shape = numpy_shape
                padded_shape = numpy_shape
                placeholder_shape = tuple([None, ] + [d for d in numpy_shape[0:]])
        elif variable_type == "support":
            tfrecords_type = tf.string
            tf_dtype = tf.float32
            shape = tuple([-1, ] + [d for d in numpy_shape[1:]])
            padded_shape = tuple([None, ] + [d for d in numpy_shape[1:]])
            placeholder_shape = tuple([None, None] + [d for d in numpy_shape[1:]])
        elif variable_type == "id":
            tfrecords_type = tf.int64
            tf_dtype = tf.int32
            shape = numpy_shape
            padded_shape = numpy_shape
            if len(numpy_shape) > 1:
                placeholder_shape = tuple([None, ] + [d for d in numpy_shape[1:]])
            else:
                placeholder_shape = tuple([None, ] + [1,])
        else:
            raise ValueError("Invalid variable type.")

        name_to_metadata[variable_name]["tfrecords_type"] = tfrecords_type
        name_to_metadata[variable_name]["tf_dtype"] = tf_dtype
        name_to_metadata[variable_name]["shape"] = shape
        name_to_metadata[variable_name]["padded_shape"] = padded_shape
        name_to_metadata[variable_name]["placeholder_shape"] = placeholder_shape

    return name_to_metadata


def filter_names(all_path_list,
                 pos_variations,
                 neg_variations):
    path_dict = collections.defaultdict(list)

    for path in all_path_list:
        path_split = path[:-10].split("_")
        segment_id = int(path_split[-2])
        version_id = int(path_split[-1])
        name = "_".join(path_split[1:4])
        path_dict[name].append(path)

    all_path_list_new = list()
    for k, v in path_dict.items():
        number_of_variations = len(v)
        for i, vv in enumerate(v):
            if i < number_of_variations:
                all_path_list_new.append(v[i])
    return all_path_list_new


def get_dataset_info(tfrecords_folder):
    partitions = ["train",
                  "devel",
                  "test"]
    path_list_dict = dict()
    partition_size_dict = dict()
    for partition in partitions:
        partition_eff = partition

        all_path_list = os.listdir(tfrecords_folder + "/" + partition_eff)

        if partition_eff == "train":
            all_path_list = filter_names(all_path_list,
                                         pos_variations=None,  # Multiple versions per positive sample exist -- offline random time shift.
                                         neg_variations=None)  # Get all negatives.
        elif partition_eff in ["devel", "test"]:
            all_path_list = filter_names(all_path_list,
                                         pos_variations=None,   # Only 1 version per positive sample exists (and should exist).
                                         neg_variations=None)  # Get all negatives.
        else:
            raise ValueError

        all_path_list = [tfrecords_folder + "/" + partition_eff + "/" + name for name in all_path_list]

        path_list_dict[partition] = all_path_list

        partition_size_dict[partition] = len(all_path_list)
    return path_list_dict, partition_size_dict
