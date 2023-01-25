from common.evaluation.monitor import PerformanceMonitorVirtual
from common.evaluation.measures import get_binary_classification_measures, get_multiclass_classification_measures
from variational.activations import sigmoid_moments_np


class PerformanceMonitor(PerformanceMonitorVirtual):
    def __init__(self,
                 output_folder,
                 method_string,
                 custom_saver,
                 monitor_target_to_measures,
                 target_to_task_type,
                 are_test_labels_available,
                 output_type_list,
                 model_configuration):
        super().__init__(output_folder,
                         method_string,
                         custom_saver,
                         monitor_target_to_measures,
                         target_to_task_type,
                         are_test_labels_available,
                         output_type_list,
                         model_configuration)

    def get_measures(self,
                     items,
                     partition):
        if "bayesian" in self.model_configuration.keys():
            is_bayesian = True
        else:
            is_bayesian = False

        measures = dict()

        if is_bayesian:
            pred = sigmoid_moments_np(items["asthma"]["pred"][:, :1],
                                      items["asthma"]["pred"][:, 1:])

            measures["asthma"] = get_binary_classification_measures(true=items["asthma"]["true"][:, 0],
                                                                    pred=pred[:, 0],
                                                                    are_logits=False)

            # TODO: Categories for Bayesian.

        else:
            print(items["asthma"]["pred"].shape)

            if partition == "test":
                optimal_threshold = self.best_optimal_threshold["asthma"]["au_pr"]
            else:
                optimal_threshold = None

            measures["asthma"],\
            optimal_threshold = get_binary_classification_measures(true=items["asthma"]["true"][:, 0],
                                                                   pred=items["asthma"]["pred"][:, 0],
                                                                   # pred=items["asthma"]["pred"][:, 1],
                                                                   are_logits=True,
                                                                   optimal_threshold=optimal_threshold)

        self.measures[partition] = measures

        if partition == "devel":
            self.current_optimal_threshold["asthma"]["au_pr"] = optimal_threshold
