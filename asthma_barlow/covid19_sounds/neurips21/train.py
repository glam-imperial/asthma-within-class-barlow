from covid19_sounds.neurips21.trials import run_experiments
from covid19_sounds.neurips21.configuration import get_config_dict_from_yaml


def make_config_dict_list():
    config_dict_list = list()

    for name in [
        "all-voice-VGGish-avg",
        "all-breath-VGGish-avg",
        "all-cough-VGGish-avg",
        ]:  # These are the names of the YAML files in folder: experiment_configurations.
        config_dict = get_config_dict_from_yaml(name)
        config_dict_list.append(config_dict)

    return config_dict_list


if __name__ == '__main__':
    run_experiments(make_config_dict_list())
