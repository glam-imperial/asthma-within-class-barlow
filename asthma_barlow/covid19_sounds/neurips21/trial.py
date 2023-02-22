import os.path

from common.common import store_pickle
from covid19_sounds.neurips21.experiment import experiment_run

# TODO: Best across trials (measures & model) -- option to keep all, or just the best?


def run_experiments(config_dict_list):
    all_experiments_results_summary_list = list()
    all_experiments_items_summary_list = list()
    for config_dict in config_dict_list:
        single_experiment_results_summary,\
        single_experiment_items_summary = run_trials(config_dict=config_dict)
        all_experiments_results_summary_list.append(single_experiment_results_summary)
        all_experiments_items_summary_list.append(single_experiment_items_summary)

    return all_experiments_results_summary_list, all_experiments_items_summary_list


def run_trials(config_dict):
    all_trials_results_summary_list = list()
    all_trials_items_summary_list = list()
    print("Method name:")
    print(config_dict["method_string"])
    for t in range(config_dict["number_of_trials"]):
        config_dict_effective = {k: v for k, v in config_dict.items()}
        config_dict_effective["current_trial"] = t
        single_trial_results_summary,\
        single_trial_items_summary = run_single_trial(config_dict=config_dict_effective)
        single_trial_results_summary["configuration_dict"] = config_dict_effective

        # Set to None because we cannot pickle modules.
        for module_name in ["model_module",
                            "architecture_module",
                            "losses_module",
                            "evaluation_module"]:
            if module_name in single_trial_results_summary["configuration_dict"].keys():
                single_trial_results_summary["configuration_dict"][module_name] = None

        # Make sure we are not overwriting stored result dictionaries.
        t_eff = t
        while os.path.exists(config_dict["results_summary_path"] + "_trial" + repr(t_eff) + ".pkl"):
            t_eff += 1

        store_pickle(config_dict["results_summary_path"] + "_trial" + repr(t_eff) + ".pkl",
                     single_trial_results_summary)

        # t_eff = t
        # while os.path.exists(config_dict["items_summary_path"] + "_trial" + repr(t_eff) + ".pkl"):
        #     t_eff += 1

        store_pickle(config_dict["items_summary_path"] + "_trial" + repr(t_eff) + ".pkl",
                     single_trial_items_summary)

        all_trials_results_summary_list.append(single_trial_results_summary)
        all_trials_items_summary_list.append(single_trial_items_summary)

    # TODO: Best across trials (measures & model) -- option to keep all, or just the best?

    return all_trials_results_summary_list, all_trials_items_summary_list


def run_single_trial(config_dict):
    single_trial_results_summary,\
    single_trial_items_summary = experiment_run(config_dict=config_dict)
    return single_trial_results_summary, single_trial_items_summary
