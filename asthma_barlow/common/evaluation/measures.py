import sklearn
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_curve, precision_recall_curve

# TODO: Each Corpus, might have different validation.
# TODO: Make multiclass.
# TODO: Clean up get_multiple_binary_classification_measures.

# Binary classification balanced accuracy.


def get_multiple_binary_classification_measures(true, pred, are_logits):
    multitask_measures = dict()
    per_task_measures = dict()
    y_measures_list = list()
    y_weight_list = list()

    number_of_classes = true.shape[1]

    for c in range(number_of_classes):
        true_c = true[:, c]
        pred_c = pred[:, c]
        y_measures_list.append(get_binary_classification_measures(true_c,
                                                                  pred_c,
                                                                  are_logits))
        y_weight_list.append(true_c.sum(axis=0))
    y_weight_list = np.array(y_weight_list, dtype=np.float32)
    y_weight_list = y_weight_list / np.sum(y_weight_list)
    y_weight_list = list(y_weight_list)

    for measure_name in ["accuracy",
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
                         "ece",
                         "mce"]:
        multitask_measures["macro_" + measure_name] = np.mean([m[measure_name] for m in y_measures_list])
        multitask_measures["weighted_" + measure_name] = np.sum(
            [m[measure_name] * w for m, w in zip(y_measures_list, y_weight_list)])

    for c in range(number_of_classes):
        per_task_measures["label_" + repr(c)] = y_measures_list[c]

    return multitask_measures, per_task_measures


def get_multiclass_classification_measures(true, pred, are_logits):
    number_of_classes = pred.shape[1]

    target_measures = dict()

    true_indicator = true

    if are_logits:
        pred_logits = pred
        pred_logits = np.nan_to_num(pred_logits)
        pred_prob = stable_softmax(pred_logits)
    else:
        pred_prob = pred
        pred_prob = np.nan_to_num(pred_prob)

    true_labels = np.argmax(true_indicator, axis=1)
    pred_labels_indicator = make_indicator_from_probabilities_multiclass(pred_prob)
    pred_labels = np.argmax(pred_labels_indicator, axis=1)

    # Accuracy.
    accuracy = sklearn.metrics.accuracy_score(true_labels, pred_labels, normalize=True, sample_weight=None)
    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(true_labels,
                                                        pred_labels,
                                                        sample_weight=None,
                                                        adjusted=False)
    balanced_accuracy_adjusted = sklearn.metrics.balanced_accuracy_score(true_labels,
                                                                 pred_labels,
                                                                 sample_weight=None,
                                                                 adjusted=True)
    target_measures["accuracy"] = accuracy
    target_measures["balanced_accuracy"] = balanced_accuracy
    target_measures["balanced_accuracy_adjusted"] = balanced_accuracy_adjusted

    # AU-ROC.
    au_roc_macro_ovr = sklearn.metrics.roc_auc_score(true_labels,
                                                     pred_prob,
                                                     average="macro",
                                                     multi_class="ovr")
    au_roc_macro_ovo = sklearn.metrics.roc_auc_score(true_labels,
                                                     pred_prob,
                                                     average="macro",
                                                     multi_class="ovo")
    au_roc_weighted_ovr = sklearn.metrics.roc_auc_score(true_labels,
                                                        pred_prob,
                                                        average="weighted",
                                                        multi_class="ovr")
    au_roc_weighted_ovo = sklearn.metrics.roc_auc_score(true_labels,
                                                        pred_prob,
                                                        average="weighted",
                                                        multi_class="ovo")
    target_measures["au_roc_macro_ovr"] = au_roc_macro_ovr
    target_measures["au_roc_macro_ovo"] = au_roc_macro_ovo
    target_measures["au_roc_weighted_ovr"] = au_roc_weighted_ovr
    target_measures["au_roc_weighted_ovo"] = au_roc_weighted_ovo

    # AU-PR
    au_prc_list = list()
    for c in range(number_of_classes):
        au_prc = sklearn.metrics.average_precision_score(true_indicator[:, c], pred_prob[:, c], average="macro")
        au_prc_list.append(au_prc)
    au_prc_list = np.array(au_prc_list, dtype=np.float64)
    au_prc_macro = np.mean(au_prc_list)

    for c in range(number_of_classes):
        au_prc_list[c] = au_prc_list[c] * true_indicator[:, c].sum()
    au_prc_list = au_prc_list / true_indicator.sum()
    au_prc_weighted = np.mean(au_prc_list)
    target_measures["macro_au_pr"] = au_prc_macro
    target_measures["weighted_au_pr"] = au_prc_weighted

    # Precision, Recall, F1
    precision_classes, recall_classes, f1_classes, _ = sklearn.metrics.precision_recall_fscore_support(
        true_labels,
        pred_labels,
        zero_division=0,
        average=None)
    precision_weighted, recall_weighted, f1_weighted, _ = sklearn.metrics.precision_recall_fscore_support(true_labels,
                                                                                                          pred_labels,
                                                                                                          zero_division=0,
                                                                                                          average="weighted")
    precision_micro, recall_micro, f1_micro, _ = sklearn.metrics.precision_recall_fscore_support(true_labels,
                                                                                                 pred_labels,
                                                                                                 zero_division=0,
                                                                                                 average="micro")
    # target_measures["pos_precision"] = precision_classes[1]
    # target_measures["neg_precision"] = precision_classes[0]
    target_measures["macro_precision"] = np.mean(precision_classes)
    target_measures["weighted_precision"] = precision_weighted
    target_measures["micro_precision"] = precision_micro
    # target_measures["pos_recall"] = recall_classes[1]
    # target_measures["neg_recall"] = recall_classes[0]
    target_measures["macro_recall"] = np.mean(recall_classes)
    target_measures["weighted_recall"] = recall_weighted
    target_measures["micro_recall"] = recall_micro
    # target_measures["pos_f1"] = f1_classes[1]
    # target_measures["neg_f1"] = f1_classes[0]
    target_measures["macro_f1"] = np.mean(f1_classes)
    target_measures["weighted_f1"] = f1_weighted
    target_measures["micro_f1"] = f1_micro

    # Cohen's kappa.
    cohen_kappa = sklearn.metrics.cohen_kappa_score(true_labels, pred_labels)
    target_measures["cohen_kappa"] = cohen_kappa

    # MCC
    mcc = sklearn.metrics.matthews_corrcoef(true_labels,
                                            pred_labels)
    target_measures["mcc"] = mcc

    # Calibration.
    try:
        cal = calibration(true_indicator, pred_prob, num_bins=10)
        ece = cal["ece"]
        mce = cal["mce"]
    except ZeroDivisionError:
        ece = 1.0
        mce = 1.0

    ece_list = list()
    mce_list = list()
    for c in range(number_of_classes):
        true_labels_c = np.reshape(true_indicator[:, c], (true_indicator.shape[0], 1))
        true_labels_c = np.hstack([1.0 - true_labels_c, true_labels_c])
        pred_prob_c = np.reshape(pred_prob[:, c], (pred_prob.shape[0], 1))
        pred_prob_c = np.hstack([1.0 - pred_prob_c, pred_prob_c])
        cal_c = calibration(true_labels_c, pred_prob_c, num_bins=10)
        ece_list.append(cal_c["ece"])
        mce_list.append(cal_c["mce"])

    ece_list = np.array(ece_list, dtype=np.float64)
    mce_list = np.array(mce_list, dtype=np.float64)
    ece_macro = np.mean(ece_list)
    mce_macro = np.mean(mce_list)

    for c in range(number_of_classes):
        ece_list[c] = ece_list[c] * true_indicator[:, c].sum()
        mce_list[c] = mce_list[c] * true_indicator[:, c].sum()
    ece_list = ece_list / true_indicator.sum()
    mce_list = mce_list / true_indicator.sum()
    ece_weighted = np.mean(ece_list)
    mce_weighted = np.mean(mce_list)

    target_measures["ece"] = ece
    target_measures["mce"] = mce
    target_measures["macro_ece"] = ece_macro
    target_measures["macro_mce"] = mce_macro
    target_measures["weighted_ece"] = ece_weighted
    target_measures["weighted_mce"] = mce_weighted

    return target_measures


def get_binary_classification_measures(true,  # 1-D array.
                                       pred,  # 1-D array.
                                       are_logits,
                                       optimal_threshold=None):

    target_measures = dict()

    true_indicator = None

    if are_logits:
        pred_logits = pred
        pred_logits = np.nan_to_num(pred_logits)
        pred_prob = sigmoid(pred_logits)
    else:
        pred_prob = pred
        pred_prob = np.nan_to_num(pred_prob)

    true_labels = true
    pred_labels = make_indicator_from_probabilities_binary(pred_prob,
                                                           0.5)

    # print(true_labels[true_labels == 1.0][0:10])
    # print(pred_prob[true_labels == 1.0][0:10])
    # print(true_labels[true_labels == 0.0][0:10])
    # print(pred_prob[true_labels == 0.0][0:10])
    # print(pred_prob[true_labels == 1.0].mean())
    # print(pred_prob[true_labels == 0.0].mean())

    # Accuracy.
    accuracy = sklearn.metrics.accuracy_score(true_labels,
                                              pred_labels,
                                              normalize=True,
                                              sample_weight=None)
    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(true_labels,
                                                        pred_labels,
                                                        sample_weight=None,
                                                        adjusted=False)
    balanced_accuracy_adjusted = sklearn.metrics.balanced_accuracy_score(true_labels,
                                                                 pred_labels,
                                                                 sample_weight=None,
                                                                 adjusted=True)
    target_measures["accuracy"] = accuracy
    target_measures["balanced_accuracy"] = balanced_accuracy
    target_measures["balanced_accuracy_adjusted"] = balanced_accuracy_adjusted

    # AU-ROC.
    au_roc_macro = sklearn.metrics.roc_auc_score(true_labels, pred_prob, average="macro")
    target_measures["au_roc"] = au_roc_macro

    # AU-PR
    au_prc_macro = sklearn.metrics.average_precision_score(true_labels, pred_prob, average="macro")
    target_measures["au_pr"] = au_prc_macro

    # Precision, Recall, F1
    precision_classes, recall_classes, f1_classes, _ = sklearn.metrics.precision_recall_fscore_support(
        true_labels,
        pred_labels,
        zero_division=0,
        average=None)
    precision_weighted, recall_weighted, f1_weighted, _ = sklearn.metrics.precision_recall_fscore_support(true_labels,
                                                                                                          pred_labels,
                                                                                                          zero_division=0,
                                                                                                          average="weighted")
    precision_micro, recall_micro, f1_micro, _ = sklearn.metrics.precision_recall_fscore_support(true_labels,
                                                                                                 pred_labels,
                                                                                                 zero_division=0,
                                                                                                 average="micro")

    # Cohen's kappa.
    cohen_kappa = sklearn.metrics.cohen_kappa_score(true_labels, pred_labels)

    # MCC.
    mcc = sklearn.metrics.matthews_corrcoef(true_labels,
                                            pred_labels)
    # mcc_alt = binary_matthews_correlation_coefficient(true_labels,
    #                                                   pred_labels)

    # Calibration.
    true_labels_2 = np.reshape(true_labels, (true_labels.size, 1))
    true_labels_2 = np.hstack([1.0 - true_labels_2, true_labels_2])
    pred_prob_2 = np.reshape(pred_prob, (pred_prob.size, 1))
    pred_prob_2 = np.hstack([1.0 - pred_prob_2, pred_prob_2])
    try:
        cal = calibration(true_labels_2, pred_prob_2, num_bins=10)
    except ZeroDivisionError:
        cal = {"ece": 1.0, "mce": 1.0}

    # Optimal threshold stuff.
    # true_labels_2 = np.hstack([1.0 - true_labels.reshape((-1, 1)), true_labels.reshape((-1, 1))])
    # pred_prob_2 = np.hstack([1.0 - pred_prob.reshape((-1, 1)), pred_prob.reshape((-1, 1))])
    measures_at_threshold_au_roc, \
    thresholds_au_roc, \
    number_of_classes_au_roc = get_rates_and_thresholds(true_labels_2,
                                                        pred_prob_2,
                                                        "ROC")

    measures_at_threshold_au_pr, \
    thresholds_au_pr, \
    number_of_classes_au_pr = get_rates_and_thresholds(true_labels_2,
                                                       pred_prob_2,
                                                       "PR")

    # print(measures_at_threshold_au_roc)
    # print(thresholds_au_roc)
    # print(measures_at_threshold_au_pr)
    # print(thresholds_au_pr)

    # measure_per_class_j, optimal_threshold_per_class_j = \
    #     get_optimal_threshold_per_class(measures_at_threshold_au_roc,
    #                                     thresholds_au_roc,
    #                                     number_of_classes_au_roc,
    #                                     "J")
    #
    # measure_per_class_f1, optimal_threshold_per_class_f1 = \
    #     get_optimal_threshold_per_class(measures_at_threshold_au_pr,
    #                                     thresholds_au_pr,
    #                                     number_of_classes_au_pr,
    #                                     "F1")

    optimal_threshold_r, optimal_threshold_f1 = get_optimal_threshold_r(measures_at_threshold_au_pr,
                                                                        thresholds_au_pr)

    if optimal_threshold is None:
        optimal_threshold = dict()
        optimal_threshold["j"] = optimal_threshold_r
        optimal_threshold["f1"] = optimal_threshold_f1
    else:
        if "f1" not in optimal_threshold.keys():
            optimal_threshold["f1"] = 0.5
        if "j" not in optimal_threshold.keys():
            optimal_threshold["j"] = 0.5

    # print(optimal_threshold)

    pred_labels = make_indicator_from_probabilities_binary(pred_prob,
                                                           optimal_threshold["f1"])

    precision_classes, recall_classes, f1_classes, _ = sklearn.metrics.precision_recall_fscore_support(
        true_labels,
        pred_labels,
        zero_division=0,
        average=None)

    opt_pos_f1_from_au_pr = f1_classes[1]
    opt_neg_f1_from_au_pr = f1_classes[0]
    opt_macro_f1_from_au_pr = np.mean(f1_classes)

    pred_labels = make_indicator_from_probabilities_binary(pred_prob,
                                                           optimal_threshold["f1"])
    precision_classes, recall_classes, f1_classes, _ = sklearn.metrics.precision_recall_fscore_support(
        true_labels,
        pred_labels,
        zero_division=0,
        average=None)

    opt_pos_p_from_au_pr = precision_classes[1]
    opt_neg_p_from_au_pr = precision_classes[0]
    opt_macro_p_from_au_pr = np.mean(precision_classes)

    pred_labels = make_indicator_from_probabilities_binary(pred_prob,
                                                           optimal_threshold["f1"])
    precision_classes, recall_classes, f1_classes, _ = sklearn.metrics.precision_recall_fscore_support(
        true_labels,
        pred_labels,
        zero_division=0,
        average=None)

    opt_pos_r_from_au_pr = recall_classes[1]
    opt_neg_r_from_au_pr = recall_classes[0]
    opt_macro_r_from_au_pr = np.mean(recall_classes)

    pred_labels = make_indicator_from_probabilities_binary(pred_prob,
                                                           optimal_threshold["j"])

    precision_classes, recall_classes, f1_classes, _ = sklearn.metrics.precision_recall_fscore_support(
        true_labels,
        pred_labels,
        zero_division=0,
        average=None)

    opt_pos_f1_from_au_roc = f1_classes[1]
    opt_neg_f1_from_au_roc = f1_classes[0]
    opt_macro_f1_from_au_roc = np.mean(f1_classes)

    pred_labels = make_indicator_from_probabilities_binary(pred_prob,
                                                           optimal_threshold["j"])
    precision_classes, recall_classes, f1_classes, _ = sklearn.metrics.precision_recall_fscore_support(
        true_labels,
        pred_labels,
        zero_division=0,
        average=None)

    opt_pos_p_from_au_roc = precision_classes[1]
    opt_neg_p_from_au_roc = precision_classes[0]
    opt_macro_p_from_au_roc = np.mean(precision_classes)

    pred_labels = make_indicator_from_probabilities_binary(pred_prob,
                                                           optimal_threshold["j"])
    precision_classes, recall_classes, f1_classes, _ = sklearn.metrics.precision_recall_fscore_support(
        true_labels,
        pred_labels,
        zero_division=0,
        average=None)

    opt_pos_r_from_au_roc = recall_classes[1]
    opt_neg_r_from_au_roc = recall_classes[0]
    opt_macro_r_from_au_roc = np.mean(recall_classes)

    target_measures["pos_precision"] = precision_classes[1]
    target_measures["neg_precision"] = precision_classes[0]
    target_measures["macro_precision"] = np.mean(precision_classes)
    target_measures["weighted_precision"] = precision_weighted
    target_measures["micro_precision"] = precision_micro
    target_measures["pos_recall"] = recall_classes[1]
    target_measures["neg_recall"] = recall_classes[0]
    target_measures["macro_recall"] = np.mean(recall_classes)
    target_measures["weighted_recall"] = recall_weighted
    target_measures["micro_recall"] = recall_micro
    target_measures["pos_f1"] = f1_classes[1]
    target_measures["neg_f1"] = f1_classes[0]
    target_measures["macro_f1"] = np.mean(f1_classes)
    target_measures["weighted_f1"] = f1_weighted
    target_measures["micro_f1"] = f1_micro
    target_measures["cohen_kappa"] = cohen_kappa
    target_measures["mcc"] = mcc
    # target_measures["mcc_alt"] = mcc_alt
    target_measures["opt_pos_f1_from_au_pr"] = opt_pos_f1_from_au_pr
    target_measures["opt_neg_f1_from_au_pr"] = opt_neg_f1_from_au_pr
    target_measures["opt_macro_f1_from_au_pr"] = opt_macro_f1_from_au_pr
    target_measures["opt_pos_p_from_au_pr"] = opt_pos_p_from_au_pr
    target_measures["opt_neg_p_from_au_pr"] = opt_neg_p_from_au_pr
    target_measures["opt_macro_p_from_au_pr"] = opt_macro_p_from_au_pr
    target_measures["opt_pos_r_from_au_pr"] = opt_pos_r_from_au_pr
    target_measures["opt_neg_r_from_au_pr"] = opt_neg_r_from_au_pr
    target_measures["opt_macro_r_from_au_pr"] = opt_macro_r_from_au_pr
    target_measures["opt_pos_f1_from_au_roc"] = opt_pos_f1_from_au_roc
    target_measures["opt_neg_f1_from_au_roc"] = opt_neg_f1_from_au_roc
    target_measures["opt_macro_f1_from_au_roc"] = opt_macro_f1_from_au_roc
    target_measures["opt_pos_p_from_au_roc"] = opt_pos_p_from_au_roc
    target_measures["opt_neg_p_from_au_roc"] = opt_neg_p_from_au_roc
    target_measures["opt_macro_p_from_au_roc"] = opt_macro_p_from_au_roc
    target_measures["opt_pos_r_from_au_roc"] = opt_pos_r_from_au_roc
    target_measures["opt_neg_r_from_au_roc"] = opt_neg_r_from_au_roc
    target_measures["opt_macro_r_from_au_roc"] = opt_macro_r_from_au_roc
    target_measures["ece"] = cal["ece"]
    target_measures["mce"] = cal["mce"]

    return target_measures, optimal_threshold


def get_optimal_threshold_r(measures_at_threshold_au_pr,
                            thresholds_au_pr):
    thresholds_au_pr[0] = 1.0 - thresholds_au_pr[0]
    thresholds_au_pr[0] = thresholds_au_pr[0][::-1]

    measures_at_threshold_au_pr[0][0] = measures_at_threshold_au_pr[0][0][::-1]
    measures_at_threshold_au_pr[1][0] = measures_at_threshold_au_pr[1][0][::-1]

    # print(thresholds_au_pr)
    # print(measures_at_threshold_au_pr)

    i = 0
    j = 0

    merged_thresholds = list()
    measure_f1_list = list()
    measure_r_list = list()

    p_pos = measures_at_threshold_au_pr[0][1][0]
    p_neg = measures_at_threshold_au_pr[0][0][0]
    r_pos = measures_at_threshold_au_pr[1][1][0]
    r_neg = measures_at_threshold_au_pr[1][0][0]

    while (i < thresholds_au_pr[0].size) and (j < thresholds_au_pr[1].size):
        if i == thresholds_au_pr[0].size:
            get_first = False
        elif j == thresholds_au_pr[1].size:
            get_first = True
        else:
            if thresholds_au_pr[0][i] < thresholds_au_pr[1][j]:
                get_first = True
            else:
                get_first = False
        if get_first:
            p_neg = measures_at_threshold_au_pr[0][0][i]
            r_neg = measures_at_threshold_au_pr[1][0][i]

            merged_thresholds.append(thresholds_au_pr[0][i])

            i += 1
        else:
            p_pos = measures_at_threshold_au_pr[0][1][j]
            r_pos = measures_at_threshold_au_pr[1][1][j]

            merged_thresholds.append(thresholds_au_pr[1][j])

            j += 1

        macro_r = (r_pos + r_neg) / 2.0
        macro_f1 = ((p_neg * r_neg) / (
                    p_neg + r_neg)) + (
                               (p_pos * r_pos) / (
                                   p_pos + r_pos))
        macro_f1 = np.nan_to_num(macro_f1)
        measure_r_list.append(macro_r)
        measure_f1_list.append(macro_f1)

    merged_thresholds = np.array(merged_thresholds)
    measure_f1_list = np.array(measure_f1_list)
    measure_r_list = np.array(measure_r_list)

    optimal_threshold_r = merged_thresholds[np.argmax(measure_r_list)]
    optimal_threshold_f1 = merged_thresholds[np.argmax(measure_f1_list)]

    # print(optimal_threshold_r, optimal_threshold_f1)
    # print(np.max(measure_r_list), np.max(measure_f1_list))

    return optimal_threshold_r, optimal_threshold_f1


def make_indicator_from_probabilities_binary(y_pred_prob,
                                             threshold):
    y_pred_indicator = np.zeros_like(y_pred_prob)
    y_pred_indicator[y_pred_prob >= threshold] = 1.0
    y_pred_indicator[y_pred_prob < threshold] = 0.0

    return y_pred_indicator


def make_indicator_from_probabilities_multiclass(y_pred_prob):
    y_pred_indicator = np.zeros_like(y_pred_prob)
    max_indices = np.argmax(y_pred_prob, axis=1).reshape((-1, 1))
    max_indices = np.hstack((np.arange(y_pred_prob.shape[0], dtype=np.int32).reshape((-1, 1)),
                             max_indices))
    y_pred_indicator[max_indices] = 1.0

    return y_pred_indicator


def binary_matthews_correlation_coefficient(y_true,
                                            y_pred):  # These are labels; not probabilities or logits.
    # y_pos_true = y_true[:, 1]
    y_pos_true = y_true
    # y_pos_pred_indicator = y_pred[:, 1]
    y_pos_pred_indicator = y_pred

    TP = np.count_nonzero(np.multiply(y_pos_pred_indicator,
                                      y_pos_true))
    TN = np.count_nonzero(np.multiply((y_pos_pred_indicator - 1.0),
                                      (y_pos_true - 1.0)))
    FP = np.count_nonzero(np.multiply(y_pos_pred_indicator,
                                      (y_pos_true - 1.0)))
    FN = np.count_nonzero(np.multiply((y_pos_pred_indicator - 1.0),
                                      y_pos_true))

    if np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) == 0.0:
        mcc = 0.0
    else:
        mcc = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    return mcc


def get_rates_and_thresholds(y_true,
                             y_pred,
                             curve_function_type):
    measures_at_threshold = (dict(), dict())
    thresholds = dict()

    if y_true.shape[-1] != y_pred.shape[-1]:
        raise ValueError("Y true and y pred do not cover the same number of classes.")

    number_of_classes = y_true.shape[-1]

    for i in range(number_of_classes):
        if curve_function_type == "ROC":
            measures_at_threshold[0][i],\
            measures_at_threshold[1][i],\
            thresholds[i] = roc_curve(y_true[:, i],
                                      y_pred[:, i],
                                      drop_intermediate=False)
        elif curve_function_type == "PR":
            measures_at_threshold[0][i],\
            measures_at_threshold[1][i], \
            thresholds[i] = precision_recall_curve(y_true[:, i],
                                                   y_pred[:, i])
        else:
            raise ValueError("Invalid curve function type.")

    return measures_at_threshold,\
           thresholds,\
           number_of_classes


def get_optimal_threshold_per_class(measures_at_threshold,
                                    thresholds,
                                    number_of_classes,
                                    function_name):
    measure_per_class_j = [None] * number_of_classes
    measure_per_class_f1 = [None] * number_of_classes
    optimal_threshold_per_class_j = [None] * number_of_classes
    optimal_threshold_per_class_f1 = [None] * number_of_classes

    for i in range(number_of_classes):
        if function_name == "J":
            measure_per_class_j[i] = youden_j_statistic(fpr=measures_at_threshold[0][i],
                                                        tpr=measures_at_threshold[1][i])
            optimal_threshold_per_class_j[i] = thresholds[i][np.argmax(measure_per_class_j[i])]
        elif function_name == "F1":
            measure_per_class_f1[i] = f1_measure(precision=measures_at_threshold[0][i],
                                                 recall=measures_at_threshold[1][i])

            optimal_threshold_per_class_f1[i] = thresholds[i][np.argmax(measure_per_class_f1[i])]
        else:
            raise ValueError

    if function_name == "J":
        return measure_per_class_j, optimal_threshold_per_class_j
    elif function_name == "F1":
        return measure_per_class_f1, optimal_threshold_per_class_f1


def youden_j_statistic(fpr, tpr):
    return tpr - fpr


def f1_measure(precision, recall):
    return (2 * precision * recall) / (precision + recall)


def stable_softmax(X):
    exps = np.exp(X - np.max(X, 1).reshape((X.shape[0], 1)))
    return exps / np.sum(exps, 1).reshape((X.shape[0], 1))


def sigmoid(x):
    x = np.nan_to_num(x)
    return expit(x)
    # return 1. / (1. + np.exp(-x))


def calibration(y, p_mean, num_bins=10):
    """Compute the calibration. -- https://github.com/google-research/google-research/blob/master/uncertainties/sources/postprocessing/metrics.py
    References:
    https://arxiv.org/abs/1706.04599
    https://arxiv.org/abs/1807.00263
    Args:
        y: one-hot encoding of the true classes, size (?, num_classes)
        p_mean: numpy array, size (?, num_classes)
                containing the mean output predicted probabilities
        num_bins: number of bins
    Returns:
        cal: a dictionary
             {reliability_diag: realibility diagram
        ece: Expected Calibration Error
        mce: Maximum Calibration Error
             }
    """
    # Compute for every test sample x, the predicted class.
    class_pred = np.argmax(p_mean, axis=1)
    # and the confidence (probability) associated with it.
    conf = np.max(p_mean, axis=1)
    # Convert y from one-hot encoding to the number of the class
    y = np.argmax(y, axis=1)
    # Storage
    acc_tab = np.zeros(num_bins)  # empirical (true) confidence
    mean_conf = np.zeros(num_bins)  # predicted confidence
    nb_items_bin = np.zeros(num_bins)  # number of items in the bins
    tau_tab = np.linspace(0, 1, num_bins+1)  # confidence bins
    for i in np.arange(num_bins):  # iterate over the bins
        # select the items where the predicted max probability falls in the bin
        # [tau_tab[i], tau_tab[i + 1)]
        sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
        nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
        # select the predicted classes, and the true classes
        class_pred_sec, y_sec = class_pred[sec], y[sec]
        # average of the predicted max probabilities
        mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
        # compute the empirical confidence
        acc_tab[i] = np.mean(
          class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

    # Cleaning
    mean_conf = mean_conf[nb_items_bin > 0]
    acc_tab = acc_tab[nb_items_bin > 0]
    nb_items_bin = nb_items_bin[nb_items_bin > 0]

    # Reliability diagram
    reliability_diag = (mean_conf, acc_tab)

    weights = nb_items_bin.astype(np.float) / np.sum(nb_items_bin)
    if np.sum(weights) == 0.0:
        weights = np.ones_like(nb_items_bin.astype(np.float)) / num_bins

    # Expected Calibration Error
    try:
        ece = np.average(
            np.absolute(mean_conf - acc_tab),
            weights=weights)
        # Maximum Calibration Error
        mce = np.max(np.absolute(mean_conf - acc_tab))
    except ZeroDivisionError as e:
        # ece = 0.0
        # mce = 0.0
        raise e
    # Saving
    cal = {'reliability_diag': reliability_diag,
           'ece': ece,
           'mce': mce}
    return cal
