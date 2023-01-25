import collections
import os
import math
import statistics

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.generic_utils import to_list

from common.common import safe_make_dir
from common.evaluation.monitor import CustomSaver
from covid19_sounds.neurips21.heterogeneous_batch_generator import HeterogeneousBatchGenerator
from covid19_sounds.neurips21.label_info import get_classification_label_info

try:
    Session = tf.Session
    variable_scope = tf.variable_scope
except AttributeError:
    Session = tf.compat.v1.Session
    variable_scope = tf.compat.v1.variable_scope


def experiment_run(config_dict):
    tfrecords_folder = config_dict["tfrecords_folder"]
    output_folder = config_dict["output_folder"]
    gpu = config_dict["gpu"]
    are_test_labels_available = config_dict["are_test_labels_available"]
    path_list_dict = config_dict["path_list_dict"]
    train_batch_size = config_dict["train_batch_size"]
    devel_batch_size = config_dict["devel_batch_size"]
    test_batch_size = config_dict["test_batch_size"]
    name_to_metadata = config_dict["model_configuration"]["name_to_metadata"]
    method_string = config_dict["method_string"]
    augmentation_configuration = config_dict["augmentation_configuration"]
    model_configuration = config_dict["model_configuration"]
    initial_learning_rate = config_dict["initial_learning_rate"]
    number_of_epochs = config_dict["number_of_epochs"]
    val_every_n_epoch = config_dict["val_every_n_epoch"]
    patience = config_dict["patience"]
    monitor_target_to_measures = config_dict["monitor_target_to_measures"]
    target_to_task_type = config_dict["target_to_task_type"]
    output_channel_targets = config_dict["output_channel_targets"]
    homogeneous_batches = config_dict["homogeneous_batches"]
    barlow_twins = config_dict["barlow_twins"]
    input_type_list = model_configuration["input_type_list"]
    output_type_list = model_configuration["output_type_list"]

    losses = config_dict["losses_module"]
    evaluation = config_dict["evaluation_module"]
    architecture = config_dict["architecture_module"]

    method_output_prefix = output_folder + "/" + method_string
    safe_make_dir(method_output_prefix)
    # best_model_chackpoint_path = method_output_prefix + "/best_model.h5"

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = repr(gpu)

    g = tf.Graph()
    with g.as_default():
        with Session() as sess:
            het_b_g_train_obj = HeterogeneousBatchGenerator(tf_records=tfrecords_folder,
                                                            is_training=True,
                                                            partition="train",
                                                            are_test_labels_available=are_test_labels_available,
                                                            name_to_metadata=name_to_metadata,
                                                            input_type_list=input_type_list,
                                                            output_type_list=output_type_list,
                                                            batch_size=train_batch_size,
                                                            buffer_size=15 * train_batch_size,
                                                            path_list_dict=path_list_dict["train"],
                                                            augmentation_configuration=augmentation_configuration)
            het_b_g_train = het_b_g_train_obj.get_tf_dataset()
            het_b_g_devel_obj = HeterogeneousBatchGenerator(tf_records=tfrecords_folder,
                                                            is_training=False,
                                                            partition="devel",
                                                            are_test_labels_available=are_test_labels_available,
                                                            name_to_metadata=name_to_metadata,
                                                            input_type_list=input_type_list,
                                                            output_type_list=output_type_list,
                                                            batch_size=devel_batch_size,
                                                            buffer_size=15 * devel_batch_size,
                                                            path_list_dict=path_list_dict["devel"],
                                                            augmentation_configuration=None)
            het_b_g_devel = het_b_g_devel_obj.get_tf_dataset()
            input_type_list_eff = ["voice_logmel_spectrogram",
                                   "voice_logmel_spectrogram_support",
                                   "breath_logmel_spectrogram",
                                   "breath_logmel_spectrogram_support",
                                   "cough_logmel_spectrogram",
                                   "cough_logmel_spectrogram_support",
                                   ]
            # input_type_list_eff = ["cough_wav2vec_embeddings",
            #                        "cough_wav2vec_embeddings_support",
            #                        "breath_wav2vec_embeddings",
            #                        "breath_wav2vec_embeddings_support",
            #                        "voice_wav2vec_embeddings",
            #                        "voice_wav2vec_embeddings_support"]
            het_b_g_test_all_three_obj = HeterogeneousBatchGenerator(tf_records=tfrecords_folder,
                                                                     is_training=False,
                                                                     partition="test",
                                                                     are_test_labels_available=are_test_labels_available,
                                                                     name_to_metadata=name_to_metadata,
                                                                     input_type_list=input_type_list_eff,
                                                                     output_type_list=output_type_list,
                                                                     batch_size=test_batch_size,
                                                                     buffer_size=15 * test_batch_size,
                                                                     path_list_dict=path_list_dict["test"],
                                                                     augmentation_configuration=None)
            het_b_g_test_all_three = het_b_g_test_all_three_obj.get_tf_dataset()
            input_type_list_eff = ["voice_logmel_spectrogram",
                                   "voice_logmel_spectrogram_support"]
            # input_type_list_eff = ["voice_wav2vec_embeddings",
            #                        "voice_wav2vec_embeddings_support"]
            het_b_g_test_all_voice_obj = HeterogeneousBatchGenerator(tf_records=tfrecords_folder,
                                                                     is_training=False,
                                                                     partition="test",
                                                                     are_test_labels_available=are_test_labels_available,
                                                                     name_to_metadata=name_to_metadata,
                                                                     input_type_list=input_type_list_eff,
                                                                     output_type_list=output_type_list,
                                                                     batch_size=test_batch_size,
                                                                     buffer_size=15 * test_batch_size,
                                                                     path_list_dict=path_list_dict["test"],
                                                                     augmentation_configuration=None)
            het_b_g_test_all_voice = het_b_g_test_all_voice_obj.get_tf_dataset()
            input_type_list_eff = ["breath_logmel_spectrogram",
                                   "breath_logmel_spectrogram_support"]
            # input_type_list_eff = ["breath_wav2vec_embeddings",
            #                        "breath_wav2vec_embeddings_support"]
            het_b_g_test_all_breath_obj = HeterogeneousBatchGenerator(tf_records=tfrecords_folder,
                                                                      is_training=False,
                                                                      partition="test",
                                                                      are_test_labels_available=are_test_labels_available,
                                                                      name_to_metadata=name_to_metadata,
                                                                      input_type_list=input_type_list_eff,
                                                                      output_type_list=output_type_list,
                                                                      batch_size=test_batch_size,
                                                                      buffer_size=15 * test_batch_size,
                                                                      path_list_dict=path_list_dict["test"],
                                                                      augmentation_configuration=None)
            het_b_g_test_all_breath = het_b_g_test_all_breath_obj.get_tf_dataset()
            input_type_list_eff = ["cough_logmel_spectrogram",
                                   "cough_logmel_spectrogram_support"]
            # input_type_list_eff = ["cough_wav2vec_embeddings",
            #                        "cough_wav2vec_embeddings_support"]
            het_b_g_test_all_cough_obj = HeterogeneousBatchGenerator(tf_records=tfrecords_folder,
                                                                     is_training=False,
                                                                     partition="test",
                                                                     are_test_labels_available=are_test_labels_available,
                                                                     name_to_metadata=name_to_metadata,
                                                                     input_type_list=input_type_list_eff,
                                                                     output_type_list=output_type_list,
                                                                     batch_size=test_batch_size,
                                                                     buffer_size=15 * test_batch_size,
                                                                     path_list_dict=path_list_dict["test"],
                                                                     augmentation_configuration=None)
            het_b_g_test_all_cough = het_b_g_test_all_cough_obj.get_tf_dataset()

            pos_weights, \
            true_train_dict, \
            true_devel_dict, \
            true_test_dict = get_classification_label_info(sess,
                                                           [het_b_g_train[k][3] for k in sorted(het_b_g_train.keys())],
                                                           [het_b_g_devel[k][3] for k in sorted(het_b_g_devel.keys())],
                                                           [het_b_g_test_all_cough[k][3] for k in sorted(het_b_g_test_all_cough.keys())],
                                                           [het_b_g_train_obj.steps_per_epoch[k] for k in sorted(het_b_g_train.keys())],
                                                           [het_b_g_devel_obj.steps_per_epoch[k] for k in sorted(het_b_g_devel.keys())],
                                                           [het_b_g_test_all_cough_obj.steps_per_epoch[k] for k in sorted(het_b_g_test_all_cough.keys())],
                                                           [het_b_g_train[k][2] for k in sorted(het_b_g_train.keys())],
                                                           [het_b_g_devel[k][2] for k in sorted(het_b_g_devel.keys())],
                                                           [het_b_g_test_all_cough[k][2] for k in sorted(het_b_g_test_all_cough.keys())],
                                                           output_type_list)

            with variable_scope("Model"):
                model_configuration_effective = {k: v for k, v in model_configuration.items()}

                pred_train,\
                pred_test, \
                keras_model_train,\
                keras_model_test, \
                other_outputs, \
                custom_objects = architecture.get_model(name_to_metadata=name_to_metadata,
                                                        model_configuration=model_configuration_effective)

            loss, \
            info_loss = losses.get_loss(pred_train=pred_train,
                                        model_configuration=model_configuration,
                                        output_type_list=output_type_list,
                                        other_outputs=other_outputs,
                                        pos_weights=pos_weights,
                                        barlow_twins=barlow_twins)

            optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

            # for modality_combination in ["single", "double", "triple"]:
            for modality_combination in ["single_voice",
                                         "single_breath",
                                         "single_cough",
                                         "double_voice_breath",
                                         "double_voice_cough",
                                         "double_breath_cough",
                                         "triple_voice_breath_cough"]:
                keras_model_train[modality_combination].compile(optimizer,
                                                                loss[modality_combination],
                                                                metrics=None)
                keras_model_test[modality_combination].compile(optimizer,
                                                               loss[modality_combination],
                                                               metrics=None)

            custom_saver = CustomSaver(output_folder=output_folder,
                                       method_string=method_string,
                                       monitor_target_to_measures=monitor_target_to_measures,
                                       keras_model_test=keras_model_test["triple_voice_breath_cough"])  # This generalises as parameters are the same.
            current_patience = 0
            performance_monitor = evaluation.PerformanceMonitor(output_folder=output_folder,
                                                                method_string=method_string,
                                                                custom_saver=custom_saver,
                                                                monitor_target_to_measures=monitor_target_to_measures,
                                                                target_to_task_type=target_to_task_type,
                                                                are_test_labels_available=are_test_labels_available,
                                                                output_type_list=output_type_list,
                                                                model_configuration=model_configuration)

            print("Start training base model.")
            print("Fresh base model.")
            for ee, epoch in enumerate(range(number_of_epochs)):
                print("EPOCH:", epoch + 1)
                loss_list = list()
                counter = 0
                for data in het_b_g_train_obj.heterogeneous_generation(sess, shuffle=True):
                    # print(data[0], data[1][0][0].shape, data[1][0][1].shape, data[1][1][0].shape)
                    # for m in range(data[1][0][0].shape[0]):
                    #     plt.imsave("/data/testims/" + repr(counter), data[1][0][0][m, :])
                    #     counter += 1

                    modality_combination = data[0]

                    data_eff = [d[:, :, :64] for d in data[1][0]]
                    # data_eff = data[1][0]

                    history = keras_model_train[modality_combination].fit(x=data_eff,
                                                                          # y=data[1][1][0],
                                                                          y=data[1][1],
                                                                          epochs=1,
                                                                          verbose=0,
                                                                          callbacks=None,
                                                                          batch_size=train_batch_size,
                                                                          validation_steps=None,
                                                                          validation_data=None,
                                                                          workers=1,
                                                                          use_multiprocessing=True)
                    loss_list.append(history.history["loss"][0])
                print(statistics.mean(loss_list))

                if (ee + 1) % val_every_n_epoch == 0:
                    pred_devel_np = collections.defaultdict(list)
                    true_devel_np = collections.defaultdict(list)
                    for data in het_b_g_devel_obj.heterogeneous_generation(sess, shuffle=False):
                        data_eff = [d[:, :, :64] for d in data[1][0]]
                        # data_eff = data[1][0]

                        keras_output = keras_model_test[data[0]].predict(x=data_eff,
                                                                         verbose=0,
                                                                         batch_size=devel_batch_size,
                                                                         callbacks=None,
                                                                         workers=1,
                                                                         use_multiprocessing=True)
                        for o_i, output_type in enumerate(output_type_list):
                            if len(output_type_list) > 1:
                                pred_devel_np[output_type].append(keras_output[0])
                            else:
                                pred_devel_np[output_type].append(keras_output)
                            true_devel_np[output_type].append(data[1][1][o_i])  # TODO: Check if correct.

                    for o_i, output_type in enumerate(output_type_list):
                        pred_devel_np[output_type] = np.vstack(pred_devel_np[output_type])
                        true_devel_np[output_type] = np.vstack(true_devel_np[output_type])
                    # print(pred_devel_np)
                    # print(pred_devel_np.shape)
                    # print(true_devel_np)
                    # print(true_devel_np.shape)

                    # bbb = true_devel_np - true_devel_dict["asthma"]
                    #
                    # print("Error:", bbb.sum())

                    devel_items = dict()
                    for t_i, target_name in enumerate(output_type_list):
                        devel_items[target_name] = dict()
                        devel_items[target_name]["pred"] = pred_devel_np[target_name]
                        devel_items[target_name]["true"] = true_devel_np[target_name]

                    performance_monitor.get_measures(items=devel_items,
                                                     partition="devel")
                    performance_monitor.report_measures(partition="devel",
                                                        output_channel_targets=output_channel_targets)

                    noticed_improvement = performance_monitor.monitor_improvement()

                    if noticed_improvement:
                        current_patience = 0
                    else:
                        current_patience += 1
                        if current_patience > patience:
                            break

            results_summary_dict = dict()
            items_summary_dict = dict()
            for target in monitor_target_to_measures.keys():
                ###########
                # All three
                ###########
                for measure in monitor_target_to_measures[target]:
                    keras_model_test["triple_voice_breath_cough"] = custom_saver.load_model(target=target,
                                                                                            measure=measure,
                                                                                            custom_objects=custom_objects)

                    # pred_test_np = list()
                    pred_test_np = collections.defaultdict(list)
                    true_test_np = collections.defaultdict(list)
                    for data in het_b_g_test_all_three_obj.heterogeneous_generation(sess, shuffle=False):

                        data_eff = [d[:, :, :64] for d in data[1][0]]
                        # data_eff = data[1][0]
                        print(keras_model_test.keys())
                        keras_output = keras_model_test[data[0]].predict(x=data_eff,
                                                                         verbose=0,
                                                                         batch_size=test_batch_size,
                                                                         callbacks=None,
                                                                         workers=1,
                                                                         use_multiprocessing=True)
                        for o_i, output_type in enumerate(output_type_list):
                            if len(output_type_list) > 1:
                                pred_test_np[output_type].append(keras_output[o_i])
                            else:
                                pred_test_np[output_type].append(keras_output)
                            true_test_np[output_type].append(data[1][1][o_i])
                    label_metadata_array = list()
                    # pred_test_np = np.vstack(pred_test_np)
                    for o_i, output_type in enumerate(output_type_list):
                        pred_test_np[output_type] = np.vstack(pred_test_np[output_type])
                        true_test_np[output_type] = np.vstack(true_test_np[output_type])
                    #     label_metadata_array.append(true_test_np[output_type])
                    # label_metadata_array = np.hstack(label_metadata_array)
                    # np.save("/data/Downloads/metadata_array.npy", label_metadata_array)
                    # print("metadata_saved")

                    test_items = dict()
                    for t_i, target_name in enumerate(output_type_list):
                        test_items[target_name] = dict()
                        test_items[target_name]["pred"] = pred_test_np[target_name]
                        test_items[target_name]["true"] = true_test_dict[target_name]

                    performance_monitor.get_test_measures(test_items=test_items,
                                                          target=target,
                                                          measure=measure)
                performance_monitor.report_best_performance_measures(output_channel_targets=output_channel_targets)

                results_summary, items_summary = performance_monitor.get_results_summary()
                results_summary_dict["all_three"] = results_summary
                items_summary_dict["all_three"] = items_summary

                ###########
                # Voice
                ###########
                for measure in monitor_target_to_measures[target]:
                    # keras_model_test["single_voice"] = custom_saver.load_model(target=target,
                    #                                                         measure=measure,
                    #                                                         custom_objects=custom_objects)

                    pred_test_np = collections.defaultdict(list)
                    true_test_np = collections.defaultdict(list)
                    for data in het_b_g_test_all_voice_obj.heterogeneous_generation(sess, shuffle=False):
                        data_eff = [d[:, :, :64] for d in data[1][0]]
                        # data_eff = data[1][0]
                        keras_output = keras_model_test[data[0]].predict(x=data_eff,
                                                                         verbose=0,
                                                                         batch_size=test_batch_size,
                                                                         callbacks=None,
                                                                         workers=1,
                                                                         use_multiprocessing=True)
                        for o_i, output_type in enumerate(output_type_list):
                            if len(output_type_list) > 1:
                                pred_test_np[output_type].append(keras_output[o_i])
                            else:
                                pred_test_np[output_type].append(keras_output)
                            true_test_np[output_type].append(data[1][1][o_i])
                    for o_i, output_type in enumerate(output_type_list):
                        pred_test_np[output_type] = np.vstack(pred_test_np[output_type])
                        true_test_np[output_type] = np.vstack(true_test_np[output_type])

                    test_items = dict()
                    for t_i, target_name in enumerate(output_type_list):
                        test_items[target_name] = dict()
                        test_items[target_name]["pred"] = pred_test_np[target_name]
                        test_items[target_name]["true"] = true_test_dict[target_name]

                    performance_monitor.get_test_measures(test_items=test_items,
                                                          target=target,
                                                          measure=measure)
                performance_monitor.report_best_performance_measures(output_channel_targets=output_channel_targets)

                results_summary, items_summary = performance_monitor.get_results_summary()
                results_summary_dict["voice"] = results_summary
                items_summary_dict["voice"] = items_summary

                ###########
                # Breath
                ###########

                for measure in monitor_target_to_measures[target]:
                    # keras_model_test["single_breath"] = custom_saver.load_model(target=target,
                    #                                                         measure=measure,
                    #                                                         custom_objects=custom_objects)

                    pred_test_np = collections.defaultdict(list)
                    true_test_np = collections.defaultdict(list)
                    for data in het_b_g_test_all_breath_obj.heterogeneous_generation(sess, shuffle=False):
                        data_eff = [d[:, :, :64] for d in data[1][0]]
                        # data_eff = data[1][0]
                        keras_output = keras_model_test[data[0]].predict(x=data_eff,
                                                                         verbose=0,
                                                                         batch_size=test_batch_size,
                                                                         callbacks=None,
                                                                         workers=1,
                                                                         use_multiprocessing=True)
                        for o_i, output_type in enumerate(output_type_list):
                            if len(output_type_list) > 1:
                                pred_test_np[output_type].append(keras_output[o_i])
                            else:
                                pred_test_np[output_type].append(keras_output)
                            true_test_np[output_type].append(data[1][1][o_i])
                    for o_i, output_type in enumerate(output_type_list):
                        pred_test_np[output_type] = np.vstack(pred_test_np[output_type])
                        true_test_np[output_type] = np.vstack(true_test_np[output_type])

                    test_items = dict()
                    for t_i, target_name in enumerate(output_type_list):
                        test_items[target_name] = dict()
                        test_items[target_name]["pred"] = pred_test_np[target_name]
                        test_items[target_name]["true"] = true_test_dict[target_name]

                    performance_monitor.get_test_measures(test_items=test_items,
                                                          target=target,
                                                          measure=measure)
                performance_monitor.report_best_performance_measures(output_channel_targets=output_channel_targets)

                results_summary, items_summary = performance_monitor.get_results_summary()
                results_summary_dict["breath"] = results_summary
                items_summary_dict["breath"] = items_summary

                ###########
                # Cough
                ###########
                for measure in monitor_target_to_measures[target]:
                    # keras_model_test["single_cough"] = custom_saver.load_model(target=target,
                    #                                            measure=measure,
                    #                                            custom_objects=custom_objects)

                    pred_test_np = collections.defaultdict(list)
                    true_test_np = collections.defaultdict(list)
                    for data in het_b_g_test_all_cough_obj.heterogeneous_generation(sess, shuffle=False):
                        data_eff = [d[:, :, :64] for d in data[1][0]]
                        # data_eff = data[1][0]
                        keras_output = keras_model_test[data[0]].predict(x=data_eff,
                                                                         verbose=0,
                                                                         batch_size=test_batch_size,
                                                                         callbacks=None,
                                                                         workers=1,
                                                                         use_multiprocessing=True)
                        for o_i, output_type in enumerate(output_type_list):
                            if len(output_type_list) > 1:
                                pred_test_np[output_type].append(keras_output[o_i])
                            else:
                                pred_test_np[output_type].append(keras_output)
                            true_test_np[output_type].append(data[1][1][o_i])
                    for o_i, output_type in enumerate(output_type_list):
                        pred_test_np[output_type] = np.vstack(pred_test_np[output_type])
                        true_test_np[output_type] = np.vstack(true_test_np[output_type])

                    test_items = dict()
                    for t_i, target_name in enumerate(output_type_list):
                        test_items[target_name] = dict()
                        test_items[target_name]["pred"] = pred_test_np[target_name]
                        test_items[target_name]["true"] = true_test_dict[target_name]

                    performance_monitor.get_test_measures(test_items=test_items,
                                                          target=target,
                                                          measure=measure)
                performance_monitor.report_best_performance_measures(output_channel_targets=output_channel_targets)

                results_summary, items_summary = performance_monitor.get_results_summary()
                results_summary_dict["cough"] = results_summary
                items_summary_dict["cough"] = items_summary

            return results_summary_dict, items_summary_dict
