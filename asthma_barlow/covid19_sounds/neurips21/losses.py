import collections

import numpy as np

from common.losses import _calculate_weighted_binary_crossentropy,\
    _calculate_weighted_binary_softmax_crossentropy,\
    _calculate_barlow_twins_loss,\
    _calculate_norm_loss


def get_loss(pred_train,
             model_configuration,
             output_type_list,
             other_outputs,
             pos_weights,
             ssl_regulariser,
             ssl_type):
    print(pos_weights)

    def loss_factory(modality_combination,
                     target_name,
                     task_type,
                     pos_weight,
                     weight):

        def loss_instance(y_true,
                          y_pred):
            loss_value = 0.0

            print(target_name)
            print(y_true.shape)
            print(pred_train[modality_combination][target_name].shape)

            if task_type == "binary_classification":
                loss_value = loss_value + \
                             _calculate_weighted_binary_crossentropy(target=y_true[:, 0],
                                                                     output=pred_train[modality_combination][
                                                                                target_name][:, 0],
                                                                     positive_weight=pos_weight) * weight
            elif task_type == "multiclass_classification":
                loss_value = loss_value + \
                             _calculate_weighted_binary_softmax_crossentropy(target=y_true,
                                                                             output=pred_train[modality_combination][
                                                                                 target_name],
                                                                             positive_weight=pos_weight) * weight
            else:
                raise ValueError

            if ssl_regulariser > 0.0:
                if ssl_type == "barlow_twins":
                    loss_value = loss_value + _calculate_barlow_twins_loss(embedding=pred_train[modality_combination]["embedding"]) * weight * ssl_regulariser
                elif ssl_type == "norm":
                    loss_value = loss_value + _calculate_norm_loss(embedding=pred_train[modality_combination]["embedding"]) * weight * ssl_regulariser
                else:
                    raise ValueError("Invalid SSL method name.")

            return loss_value
        return loss_instance

    loss = collections.defaultdict(dict)
    for m_i, modality_combination in enumerate(["single_voice",
                                                "single_breath",
                                                "single_cough",
                                                "double_voice_breath",
                                                "double_voice_cough",
                                                "double_breath_cough",
                                                "triple_voice_breath_cough"]):
        for t_i, target in enumerate(output_type_list):
            if target in ["asthma",
                          "copd",
                          "longterm",
                          "lung",
                          "pulmonary",
                          "drycough",
                          "wetcough",
                          "sorethroat",
                          "shortbreath"]:
                task_type = "binary_classification"
                pos_weight = pos_weights[target][0]
                pos_weight = np.nan_to_num(pos_weight, posinf=0.0)
                if target == "asthma":
                    weight = 1.0
                else:
                    weight = 0.2 / 14.0
            elif target in ["age",
                            "sex",
                            "smoking",
                            "language",
                            "covid_tested",
                            "recording_source"]:
                task_type = "multiclass_classification"
                pos_weight = pos_weights[target]
                pos_weight = np.nan_to_num(pos_weight, posinf=0.0)
                weight = 0.2 / 14.0
            else:
                raise ValueError

            # if "single" in modality_combination:
            #     m_n = 0
            # elif "double" in modality_combination:
            #     m_n = 0
            # elif "triple" in modality_combination:
            #     m_n = 0
            # else:
            #     raise ValueError

            # if t_i == 0:
            #     layer_name = "ff_" + repr(m_n)
            # else:
            #     layer_name = "ff_" + repr(m_n) + "_" + repr(t_i)

            if t_i == 0:
                layer_name = "ff_" + repr(0)
            else:
                layer_name = "ff_" + repr(0) + "_" + repr(t_i)

            loss[modality_combination][layer_name] =\
                loss_factory(modality_combination,
                             target,
                             task_type,
                             pos_weight,
                             weight)

    info_loss = None

    return loss, info_loss
