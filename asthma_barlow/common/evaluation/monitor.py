import math

import numpy as np
import tensorflow as tf

from common.common import safe_make_dir
from variational.layers import VariationalLayer, DenseReparameterisation, Conv2dReparameterization, VariationalGRUCell

# TODO: Store other stuff, apart from the prediction.
# TODO: Update for multiple binary classification.
# "opt_pos_j",
#           "opt_neg_j",
#           "opt_macro_j",
#           "opt_pos_f1",
#           "opt_neg_f1",
#           "opt_macro_f1",
#           "opt_pos_p",
#           "opt_neg_p",
#           "opt_macro_p",
#           "opt_pos_r",
#           "opt_neg_r",
#           "opt_macro_r",

BEST_VALUE_INITIALISER = dict()
MONITOR_FUNCTION = dict()

for M in ["accuracy",  # binary
          "balanced_accuracy",
          "balanced_accuracy_adjusted",
          "au_roc",
          "au_pr",
          "pos_precision",
          "neg_precision",
          "macro_precision",
          "weighted_precision",
          "micro_precision",
          "pos_recall",
          "neg_recall",
          "macro_recall",
          "weighted_recall",
          "micro_recall",
          "pos_f1",
          "neg_f1",
          "macro_f1",
          "weighted_f1",
          "micro_f1",
          "cohen_kappa",
          "mcc",
          "mcc_alt",
          "opt_pos_f1_from_au_pr",
          "opt_neg_f1_from_au_pr",
          "opt_macro_f1_from_au_pr",
          "opt_pos_p_from_au_pr",
          "opt_neg_p_from_au_pr",
          "opt_macro_p_from_au_pr",
          "opt_pos_r_from_au_pr",
          "opt_neg_r_from_au_pr",
          "opt_macro_r_from_au_pr",
          "opt_pos_f1_from_au_roc",
          "opt_neg_f1_from_au_roc",
          "opt_macro_f1_from_au_roc",
          "opt_pos_p_from_au_roc",
          "opt_neg_p_from_au_roc",
          "opt_macro_p_from_au_roc",
          "opt_pos_r_from_au_roc",
          "opt_neg_r_from_au_roc",
          "opt_macro_r_from_au_roc",
          "au_roc_macro_ovr"  # multiclass
          "au_roc_macro_ovo"
          "au_roc_weighted_ovr"
          "au_roc_weighted_ovo"
          "macro_accuracy",  # multitask binary
          "weighted_accuracy",
          "macro_balanced_accuracy",
          "weighted_balanced_accuracy",
          "macro_balanced_accuracy_adjusted",
          "weighted_balanced_accuracy_adjusted",
          "macro_au_roc",
          "weighted_au_roc",
          "macro_au_pr",
          "weighted_au_pr",
          "macro_pos_precision",
          "macro_neg_precision",
          "weighted_pos_precision",
          "weighted_neg_precision",
          "macro_macro_precision",
          "weighted_macro_precision",
          "macro_weighted_precision",
          "weighted_weighted_precision",
          "macro_micro_precision",
          "weighted_micro_precision",
          "macro_pos_recall",
          "macro_neg_recall",
          "weighted_pos_recall",
          "weighted_neg_recall",
          "macro_macro_recall",
          "weighted_macro_recall",
          "macro_weighted_recall",
          "weighted_weighted_recall",
          "macro_micro_recall",
          "weighted_micro_recall",
          "macro_pos_f1",
          "macro_neg_f1",
          "weighted_pos_f1",
          "weighted_neg_f1",
          "macro_macro_f1",
          "weighted_macro_f1",
          "macro_weighted_f1",
          "weighted_weighted_f1",
          "macro_micro_f1",
          "weighted_micro_f1",
          "macro_cohen_kappa",
          "weighted_cohen_kappa"
          "macro_mcc",
          "weighted_mcc"
          "macro_mcc_alt",
          "weighted_mcc_alt"
          ]:

    BEST_VALUE_INITIALISER[M] = -1.0
    MONITOR_FUNCTION[M] = lambda best, new: best < new

for M in ["ece",
          "mce",
          "macro_ece",
          "weighted_ece",
          "macro_mce"
          "weighted_mce"]:

    BEST_VALUE_INITIALISER[M] = math.inf
    MONITOR_FUNCTION[M] = lambda best, new: best > new


def get_measures_per_task_type(target_to_task_type):
    measures_per_task_type = {"binary_classification":
                                  ["accuracy",
                                   "balanced_accuracy",
                                   "balanced_accuracy_adjusted",
                                   "au_roc",
                                   "au_pr",
                                   "pos_precision",
                                   "neg_precision",
                                   "macro_precision",
                                   "weighted_precision",
                                   "micro_precision",
                                   "pos_recall",
                                   "neg_recall",
                                   "macro_recall",
                                   "weighted_recall",
                                   "micro_recall",
                                   "pos_f1",
                                   "neg_f1",
                                   "macro_f1",
                                   "weighted_f1",
                                   "micro_f1",
                                   "cohen_kappa",
                                   "mcc",
                                   # "mcc_alt",
                                   "opt_pos_f1_from_au_pr",
                                   "opt_neg_f1_from_au_pr",
                                   "opt_macro_f1_from_au_pr",
                                   "opt_pos_p_from_au_pr",
                                   "opt_neg_p_from_au_pr",
                                   "opt_macro_p_from_au_pr",
                                   "opt_pos_r_from_au_pr",
                                   "opt_neg_r_from_au_pr",
                                   "opt_macro_r_from_au_pr",
                                   "opt_pos_f1_from_au_roc",
                                   "opt_neg_f1_from_au_roc",
                                   "opt_macro_f1_from_au_roc",
                                   "opt_pos_p_from_au_roc",
                                   "opt_neg_p_from_au_roc",
                                   "opt_macro_p_from_au_roc",
                                   "opt_pos_r_from_au_roc",
                                   "opt_neg_r_from_au_roc",
                                   "opt_macro_r_from_au_roc",
                                   "ece",
                                   "mce"],
                              "multiclass_classification":
                                  ["accuracy",
                                   "balanced_accuracy",
                                   "balanced_accuracy_adjusted",
                                   "au_roc_macro_ovr",
                                   "au_roc_macro_ovo",
                                   "au_roc_weighted_ovr",
                                   "au_roc_weighted_ovo",
                                   "macro_au_pr",
                                   "weighted_au_pr",
                                   "macro_precision",
                                   "weighted_precision",
                                   "micro_precision",
                                   "macro_recall",
                                   "weighted_recall",
                                   "micro_recall",
                                   "macro_f1",
                                   "weighted_f1",
                                   "micro_f1",
                                   "cohen_kappa",
                                   "mcc",
                                   "ece",
                                   "mce",
                                   "macro_ece",
                                   "macro_mce",
                                   "weighted_ece",
                                   "weighted_mce",
                               ],
                              "multiple_binary_classification":
                                  ["macro_accuracy",
                                   "weighted_accuracy",
                                   "macro_au_roc",
                                   "weighted_au_roc",
                                   "macro_au_pr",
                                   "weighted_au_pr",
                                   "macro_pos_precision",
                                   "macro_neg_precision",
                                   "weighted_pos_precision",
                                   "weighted_neg_precision",
                                   "macro_macro_precision",
                                   "weighted_macro_precision",
                                   "macro_micro_precision",
                                   "weighted_micro_precision",
                                   "macro_pos_recall",
                                   "macro_neg_recall",
                                   "weighted_pos_recall",
                                   "weighted_neg_recall",
                                   "macro_macro_recall",
                                   "weighted_macro_recall",
                                   "macro_micro_recall",
                                   "weighted_micro_recall",
                                   "macro_pos_f1",
                                   "macro_neg_f1",
                                   "weighted_pos_f1",
                                   "weighted_neg_f1",
                                   "macro_macro_f1",
                                   "weighted_macro_f1",
                                   "macro_micro_f1",
                                   "weighted_micro_f1",
                                   "macro_mcc",
                                   "weighted_mcc",
                                   "macro_ece",
                                   "weighted_ece",
                                   "macro_mce",
                                   "weighted_mce"]}

    report_target_to_measures = dict()
    for target, task_type in target_to_task_type.items():
        report_target_to_measures[target] = measures_per_task_type[task_type]
    return report_target_to_measures


class CustomSaverVirtual:
    def __init__(self,
                 output_folder,
                 method_string,
                 monitor_target_to_measures,
                 keras_model_test):
        self.output_folder = output_folder
        self.method_string = method_string
        self.monitor_target_to_measures = monitor_target_to_measures
        self.keras_model_test = keras_model_test
        self.saver_paths = dict()
        self.saver_dict = dict()

        self.method_output_prefix = output_folder + "/" + method_string

        safe_make_dir(self.method_output_prefix)

        for monitor_target in monitor_target_to_measures.keys():
            self.saver_paths[monitor_target] = dict()
            self.saver_dict[monitor_target] = dict()
            safe_make_dir(self.method_output_prefix + "/" + monitor_target)
            for monitor_measure in monitor_target_to_measures[monitor_target]:
                safe_make_dir(self.method_output_prefix + "/" + monitor_target + "/" + monitor_measure)
                self.saver_paths[monitor_target][monitor_measure] = self.method_output_prefix + "/" + monitor_target + "/" + monitor_measure + "/"
                self.saver_dict[monitor_target][monitor_measure] = self.keras_model_test
                # safe_make_dir(self.method_output_prefix + "/" + target + "/" + measure)

    def save_model(self,
                   target,
                   measure):
        if isinstance(self.saver_dict[target][measure], list):
            for t in range(len(self.saver_dict[target][measure])):
                self.saver_dict[target][measure][t].save_weights(self.saver_paths[target][measure] + "_model_" + repr(t))
        elif isinstance(self.saver_dict[target][measure], dict):
            for k, v in self.saver_dict[target][measure].items():
                self.saver_dict[target][measure][k].save_weights(self.saver_paths[target][measure] + "_model_" + k)
        else:
            # self.saver_dict[target][measure].save(self.saver_paths[target][measure] + "_model.tf")
            self.saver_dict[target][measure].save_weights(self.saver_paths[target][measure] + "_model")

    def load_model(self,
                   target,
                   measure,
                   custom_objects):
        # print(custom_objects)
        if isinstance(self.saver_dict[target][measure], list):
            for t in range(len(self.saver_dict[target][measure])):
                self.saver_dict[target][measure][t].load_weights(self.saver_paths[target][measure] + "_model_" + repr(t))
        elif isinstance(self.saver_dict[target][measure], dict):
            for k, v in self.saver_dict[target][measure].items():
                self.saver_dict[target][measure][k].load_weights(self.saver_paths[target][measure] + "_model_" + k)
        else:
            # self.saver_dict[target][measure] = tf.saved_model.load(self.saver_paths[target][measure] + "_model",
            #                                                               custom_objects=custom_objects)
            # self.saver_dict[target][measure] = tf.keras.models.load_model(self.saver_paths[target][measure] + "_model.tf")
            self.saver_dict[target][measure].load_weights(self.saver_paths[target][measure] + "_model")
        return self.saver_dict[target][measure]


class CustomSaver(CustomSaverVirtual):
    def __init__(self,
                 output_folder,
                 method_string,
                 monitor_target_to_measures,
                 keras_model_test):
        super().__init__(output_folder,
                         method_string,
                         monitor_target_to_measures,
                         keras_model_test)


class PerformanceMonitorVirtual:
    def __init__(self,
                 output_folder,
                 method_string,
                 custom_saver,
                 monitor_target_to_measures,
                 target_to_task_type,
                 are_test_labels_available,
                 output_type_list,
                 model_configuration):
        self.output_folder = output_folder
        self.method_string = method_string
        self.custom_saver = custom_saver
        self.monitor_target_to_measures = monitor_target_to_measures
        self.target_to_task_type = target_to_task_type
        self.are_test_labels_available = are_test_labels_available
        self.output_type_list = output_type_list
        self.model_configuration = model_configuration

        self.method_output_prefix = output_folder + "/" + method_string

        self.report_target_to_measures = get_measures_per_task_type(self.target_to_task_type)

        # Contains measure summary for last run.
        self.measures = dict()

        # I may want to monitor multiple performance measures per multiple tasks/targets separately.

        # Contains test items and summary, dependent on target and measure
        self.test_measures_dict = dict()
        self.test_items_dict = dict()
        self.best_performance_dict = dict()
        self.monitor_function_dict = dict()
        self.best_optimal_threshold = dict()
        self.current_optimal_threshold = dict()
        for monitor_target in self.monitor_target_to_measures.keys():
            self.best_performance_dict[monitor_target] = dict()
            self.monitor_function_dict[monitor_target] = dict()
            self.test_measures_dict[monitor_target] = dict()
            self.test_items_dict[monitor_target] = dict()
            self.best_optimal_threshold[monitor_target] = dict()
            self.current_optimal_threshold[monitor_target] = dict()
            for monitor_measure in self.monitor_target_to_measures[monitor_target]:
                self.best_performance_dict[monitor_target][monitor_measure] = dict()
                self.monitor_function_dict[monitor_target][monitor_measure] = dict()
                self.best_optimal_threshold[monitor_target][monitor_measure] = dict()
                self.current_optimal_threshold[monitor_target][monitor_measure] = dict()
                for report_target in self.report_target_to_measures.keys():
                    self.best_performance_dict[monitor_target][monitor_measure][report_target] = dict()
                    self.monitor_function_dict[monitor_target][monitor_measure][report_target] = dict()
                    for report_measure in self.report_target_to_measures[report_target]:
                        self.best_performance_dict[monitor_target][monitor_measure][report_target][report_measure] =\
                            BEST_VALUE_INITIALISER[monitor_measure]
                        self.monitor_function_dict[monitor_target][monitor_measure][report_target][report_measure] =\
                            MONITOR_FUNCTION[monitor_measure]

    def get_measures(self,
                     items,
                     partition):
        raise NotImplementedError

    def report_measures(self,
                        partition,
                        output_channel_targets=None):
        measures = self.measures[partition]

        for report_target in self.report_target_to_measures.keys():
            if output_channel_targets is not None:
                if report_target not in output_channel_targets:
                    continue
            print(partition + " measures on: " + report_target)
            for report_measure in self.report_target_to_measures[report_target]:
                print(report_measure + ":", measures[report_target][report_measure])

    def monitor_improvement(self):
        noticed_improvement = False

        for monitor_target in self.monitor_target_to_measures.keys():
            for monitor_measure in self.monitor_target_to_measures[monitor_target]:
                if self.monitor_function_dict[monitor_target][monitor_measure][monitor_target][monitor_measure](self.best_performance_dict[monitor_target][monitor_measure][monitor_target][monitor_measure],
                                                                                                                self.measures["devel"][monitor_target][monitor_measure]):
                    self.best_performance_dict[monitor_target][monitor_measure] = self.measures["devel"]
                    noticed_improvement = True
                    self.custom_saver.save_model(target=monitor_target,
                                                 measure=monitor_measure)

                    self.best_optimal_threshold[monitor_target][monitor_measure] = self.current_optimal_threshold[monitor_target][monitor_measure]
        return noticed_improvement

    def get_test_measures(self,
                          test_items,
                          target,
                          measure):
        if self.are_test_labels_available:
            self.get_measures(items=test_items,
                              partition="test")
            self.test_measures_dict[target][measure] = self.measures["test"]
        self.test_items_dict[target][measure] = test_items

    def report_best_performance_measures(self,
                                         output_channel_targets=None):
        for monitor_target in self.monitor_target_to_measures.keys():
            for monitor_measure in self.monitor_target_to_measures[monitor_target]:
                print("Model selected on " + monitor_measure + " of " + monitor_target)
                print("Best devel " + monitor_measure + ":", self.best_performance_dict[monitor_target][monitor_measure][monitor_target][monitor_measure])

                if self.are_test_labels_available:
                    for report_target in self.report_target_to_measures.keys():
                        if output_channel_targets is not None:
                            if report_target not in output_channel_targets:
                                continue
                        print("Test measures on: " + report_target)
                        for report_measure in self.report_target_to_measures[report_target]:
                            print(report_measure + ":",
                                  self.test_measures_dict[monitor_target][monitor_measure][report_target][report_measure])

    def get_results_summary(self):
        results = dict()
        items = dict()

        results["method_string"] = self.method_string
        items["method_string"] = self.method_string

        for monitor_target in self.monitor_target_to_measures.keys():
            results[monitor_target] = dict()
            items[monitor_target] = dict()
            for monitor_measure in self.monitor_target_to_measures[monitor_target]:
                results[monitor_target][monitor_measure] = dict()
                items[monitor_target][monitor_measure] = dict()
                # for report_target in self.report_target_to_measures.keys():
                for report_target in ["asthma",]:
                    results[monitor_target][monitor_measure][report_target] = dict()

                    for report_measure in self.report_target_to_measures[report_target]:
                        results[monitor_target][monitor_measure][report_target]["best_devel_" + report_measure] = self.best_performance_dict[monitor_target][monitor_measure][report_target][report_measure]

                for y_pred_name in self.output_type_list:
                    items[monitor_target][monitor_measure][y_pred_name] = dict()

                    items[monitor_target][monitor_measure][y_pred_name]["test_pred"] = \
                        self.test_items_dict[monitor_target][monitor_measure][y_pred_name]["pred"]
                    # print(self.test_items_dict[monitor_target][monitor_measure][y_pred_name]["pred"])
                    # print(self.test_items_dict[monitor_target][monitor_measure][y_pred_name]["pred"].shape)
                    np.save(
                        self.output_folder + "/" + self.method_string + "/" + monitor_target + "/" + monitor_measure + "/" + y_pred_name + "_" + "test_pred.npy",
                        self.test_items_dict[monitor_target][monitor_measure][y_pred_name]["pred"])

                # TODO: Store other stuff, apart from the prediction.

        if self.are_test_labels_available:
            for monitor_target in self.monitor_target_to_measures:
                for monitor_measure in self.monitor_target_to_measures[monitor_target]:
                    # if monitor_target in self.test_measures_dict[monitor_target][monitor_measure].keys():
                    #     for report_measure in self.test_measures_dict[monitor_target][monitor_measure][monitor_target].keys():
                    # for report_target in self.report_target_to_measures.keys():
                    for report_target in ["asthma",]:
                        for report_measure in self.report_target_to_measures[report_target]:
                            results[monitor_target][monitor_measure][report_target]["test_" + report_measure] = self.test_measures_dict[monitor_target][monitor_measure][report_target][report_measure]

                    for y_pred_name in self.output_type_list:
                        items[monitor_target][monitor_measure][y_pred_name]["test_true"] = \
                            self.test_items_dict[monitor_target][monitor_measure][y_pred_name]["true"]
                        np.save(
                            self.output_folder + "/" + self.method_string + "/" + monitor_target + "/" + monitor_measure + "/" + y_pred_name + "_" + "test_true.npy",
                            self.test_items_dict[monitor_target][monitor_measure][y_pred_name]["true"])

        return results, items
