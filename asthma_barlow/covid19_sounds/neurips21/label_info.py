import collections

import numpy as np


def get_classification_label_info(sess,
                                  init_op_train_list,
                                  init_op_devel_list,
                                  init_op_test_list,
                                  train_steps_per_epoch_list,
                                  devel_steps_per_epoch_list,
                                  test_steps_per_epoch_list,
                                  next_element_train_list,
                                  next_element_devel_list,
                                  next_element_test_list,
                                  output_type_list):
    assert len(init_op_train_list) == len(train_steps_per_epoch_list) == len(next_element_train_list)
    assert len(init_op_devel_list) == len(devel_steps_per_epoch_list) == len(next_element_devel_list)
    assert len(init_op_test_list) == len(test_steps_per_epoch_list) == len(next_element_test_list)

    train_len = len(init_op_train_list)
    devel_len = len(init_op_devel_list)
    test_len = len(init_op_test_list)

    pos_weights = dict()
    true_train_dict = collections.defaultdict(list)
    for t in range(train_len):
        sess.run(init_op_train_list[t])
        for i in range(train_steps_per_epoch_list[t]):
            batch_data = sess.run(next_element_train_list[t])
            for o, output_type in enumerate(output_type_list):
                true_train_dict[output_type].append(batch_data[1][o])
    for o, output_type in enumerate(output_type_list):
        true_train_dict[output_type] = np.vstack(true_train_dict[output_type])
    for o, output_type in enumerate(output_type_list):
        pos_weights[output_type] = true_train_dict[output_type].sum(axis=0)
        pos_weights[output_type] = (true_train_dict[output_type].shape[0]) / pos_weights[output_type]

    # print(true_train_dict)

    print(pos_weights)

    # Get the true labels for devel and test.
    true_devel_dict = collections.defaultdict(list)
    for t in range(devel_len):
        sess.run(init_op_devel_list[t])
        for i in range(devel_steps_per_epoch_list[t]):
            batch_data = sess.run(next_element_devel_list[t])
            for o, output_type in enumerate(output_type_list):
                true_devel_dict[output_type].append(batch_data[1][o])
    for o, output_type in enumerate(output_type_list):
        true_devel_dict[output_type] = np.vstack(true_devel_dict[output_type])

    true_test_dict = collections.defaultdict(list)
    for t in range(test_len):
        sess.run(init_op_test_list[t])
        for i in range(test_steps_per_epoch_list[t]):
            batch_data = sess.run(next_element_test_list[t])
            for o, output_type in enumerate(output_type_list):
                true_test_dict[output_type].append(batch_data[1][o])
    for o, output_type in enumerate(output_type_list):
        true_test_dict[output_type] = np.vstack(true_test_dict[output_type])

    return pos_weights,\
           true_train_dict,\
           true_devel_dict,\
           true_test_dict
