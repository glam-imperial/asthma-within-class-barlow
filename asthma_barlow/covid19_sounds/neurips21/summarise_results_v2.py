import statistics
import os.path

from common.common import load_pickle

PROJECT_FOLDER = '/data/Downloads/COVIDSounds/NeurIPS2021-data'
OUTPUT_FOLDER = PROJECT_FOLDER + '/Results'


def trial_average(summary_list, name, return_list=False):
    value_list = list()
    for s in summary_list:
        if name in s.keys():
            value_list.append(s[name])
            print(s[name])
    if len(value_list) > 1:
        m_v = statistics.mean(value_list)
        std_v = statistics.stdev(value_list)
        max_v = max(value_list)
    elif len(value_list) == 0:
        m_v = 0.0
        std_v = 0.0
        max_v = 0.0
    else:
        m_v = value_list[0]
        std_v = 0.0
        max_v = value_list[0]

    if return_list:
        return (m_v, std_v, max_v), value_list
    else:
        return (m_v, std_v, max_v)

import numpy as np
from common.evaluation.measures import get_binary_classification_measures


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_results_from_items_summary(test_pred, test_true):
    pred = sigmoid(test_pred)
    results_summary_temp,\
    _ = get_binary_classification_measures(true=test_true[:, 0],
                                                              pred=pred[:, 0],
                                                              are_logits=False)
    results_summary = {"test_" + k: v for k, v in results_summary_temp.items()}
    # print(results_summary)

    return results_summary


results_dict = dict()
results_dict["all_three"] = dict()
results_dict["voice"] = dict()
results_dict["breath"] = dict()
results_dict["cough"] = dict()

results_dict_secondary = dict()
results_dict_secondary["all_three"] = dict()
results_dict_secondary["voice"] = dict()
results_dict_secondary["breath"] = dict()
results_dict_secondary["cough"] = dict()

for name in [
        # "core-all-three-ResNetish-avg",
        # "core-voice-ResNetish-avg",
        # "all-all-three-ResNetish-avg",
        # "all-voice-ResNetish-avg",
        # "all-voice-VGGish-avg",
        # "core-voice-VGGish-avg",
        # "core-voice-VGGish-avg-2",
        # "core-voice-VGGish-avg-3",
        # "core-voice-VGGish-avg-4",
        # "core-voice-VGGish-avg-4",
        "all-voice-ResNet-avg-opt_thresh-mt-hom",
        "all-voice-VGGish-avg-opt_thresh-mt-hom",
        # "all-voice-att-VGGish-avg-opt_thresh-hom",
        # "all-voice-Wav2Vec2-avg",
        # "all-voice-Wav2Vec2-avg-hom",
        # "all-voice-Wav2Vec2-avg-hom-cleaner",
        "all-all-VGGish-avg-opt_thresh-hom",
        # "all-all-VGGish-avg-opt_thresh-hom-barlow",
        # "all-voice-VGGish-avg-opt_thresh-hom-barlow-cdpl",
        "all-breath-VGGish-avg-opt_thresh",
        "all-cough-VGGish-avg-opt_thresh",
        "all-voice-VGGish-avg-opt_thresh",
        "all-voice-VGGish-avg-opt_thresh-mt",
        "all-voice-VGGish-avg-opt_thresh-gr",
        "all-voice-VGGish-avg-opt_thresh-hom",
        "all-voice-VGGish-avg-opt_thresh-hom-selfreg",
        "all-voice-VGGish-avg-opt_thresh-hom-barlow02",
        "all-voice-VGGish-avg-opt_thresh-hom-barlow",
        # "all-voice-VGGish-avg-opt_thresh-hom-cleaner",
        # "all-voice-ResNetish-avg-opt_thresh",
        # "all-voice-ResNetish-avg-opt_thresh-hom-barlow",
        # "all-voice-ResNet-avg-opt_thresh",
        # "all-voice-ResNetish-avg-opt_thresh-hom-barlow",
        # "all-voice-SEResNetish-avg-opt_thresh",
        # "all-voice-SEResNetish-avg-opt_thresh-hom-barlow",

]:

    print(name)
    trial_summaries = dict()
    for comb in ["all_three", "voice", "breath", "cough"]:
        trial_summaries[comb] = list()

    for t in range(10):
        if not os.path.exists(OUTPUT_FOLDER + "/" + name + "/results_summary_trial" + repr(t) + ".pkl"):
            continue
        print("Trial: ", t)
        filepath = OUTPUT_FOLDER + "/" + name + "/results_summary_trial" + repr(t) + ".pkl"
        try:
            results_summary = load_pickle(filepath)
        except FileNotFoundError:
            continue
        for comb in ["all_three", "voice", "breath", "cough"]:
            # print(t, results_summary[comb]["asthma"]["au_pr"]["asthma"]["test_au_pr"])
            results_dict[comb][name] = results_summary[comb]["asthma"]["au_pr"]["asthma"]
            # trial_summaries[comb].append(results_summary[comb]["asthma"]["au_pr"]["asthma"])

            filepath = OUTPUT_FOLDER + "/" + name + "/items_summary_trial" + repr(t) + ".pkl"
            try:
                items_summary = load_pickle(filepath)
            except FileNotFoundError:
                continue
            results_dict_secondary[comb][name] = get_results_from_items_summary(
                items_summary[comb]["asthma"]["au_pr"]["asthma"]["test_pred"],
                items_summary[comb]["asthma"]["au_pr"]["asthma"]["test_true"])
            # print(results_dict_secondary[comb].keys())
            # print(results_dict_secondary[comb][name].keys())
            # print(results_dict_secondary[comb][name])
            trial_summaries[comb].append(results_dict_secondary[comb][name])

    print("Trial averages.")
    # print("Best devel AU PR:", trial_average(trial_summaries[comb], "best_devel_au_pr"))

    if True:
        # for comb in ["all_three", "voice", "breath", "cough"]:
        print("overall")
        # for comb in ["voice", "all_three"]:
        # for comb in ["voice", "breath", "cough"]:
        for comb in ["voice", ]:
            print(comb)
            print("Test  AU PR:    ", trial_average(trial_summaries[comb], "test_au_pr"))
            print("Test  AU ROC:   ", trial_average(trial_summaries[comb], "test_au_roc"))
            print("Test  ECE:   ", trial_average(trial_summaries[comb], "test_ece"))
            print("Test  Macro F1:       ", trial_average(trial_summaries[comb], "test_macro_f1"))
            print("Test  Macro F1:       ", trial_average(trial_summaries[comb], "test_opt_macro_f1_from_au_pr"))
            print("Test  Macro Recall:   ", trial_average(trial_summaries[comb], "test_macro_recall"))
            print("Test  Macro Recall:   ", trial_average(trial_summaries[comb], "test_opt_macro_r_from_au_roc"))
