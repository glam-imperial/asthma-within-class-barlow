import yaml
import collections
import os

from common.metadata_util import process_metadata, filter_names

#########################################################################################################
# Edit these.
#########################################################################################################
PROJECT_FOLDER = '/data/PycharmProjects/asthma-within-class-barlow'
DATA_FOLDER = "/data/Downloads/COVIDSounds/NeurIPS2021-data"

# Can leave the below as they are.
AUDIO_FOLDER = DATA_FOLDER + "/covid19_data_0426/covid19_data_0426"
TFRECORDS_FOLDER = DATA_FOLDER + "/tfrecords_fair"
OUTPUT_FOLDER = DATA_FOLDER + '/Results'

YAML_CONFIGURATION_FOLDER = PROJECT_FOLDER + "/Tool/covid19_sounds/neurips21/experiment_configurations"


FOLDER_MODALITIES = dict()
FOLDER_MODALITIES["none"] = ["neg_plus_only",
                             "pos_plus_only",
                             "neg_train_only",
                             "pos_train_only",
                             "neg_test_only",
                             "pos_test_only",
                             "neg_devel_only",
                             "pos_devel_only",
                             "neg_web_only",
                             "pos_web_only"]
FOLDER_MODALITIES["single"] = ["neg_plus_only_breath",
                               "pos_plus_only_breath",
                               "neg_train_only_breath",
                               "pos_train_only_breath",
                               "neg_test_only_breath",
                               "pos_test_only_breath",
                               "neg_devel_only_breath",
                               "pos_devel_only_breath",
                               "neg_web_only_breath",
                               "pos_web_only_breath",
                               "neg_plus_only_cough",
                               "pos_plus_only_cough",
                               "neg_train_only_cough",
                               "pos_train_only_cough",
                               "neg_test_only_cough",
                               "pos_test_only_cough",
                               "neg_devel_only_cough",
                               "pos_devel_only_cough",
                               "neg_web_only_cough",
                               "pos_web_only_cough",
                               "neg_plus_only_voice",
                               "pos_plus_only_voice",
                               "neg_train_only_voice",
                               "pos_train_only_voice",
                               "neg_test_only_voice",
                               "pos_test_only_voice",
                               "neg_devel_only_voice",
                               "pos_devel_only_voice",
                               "neg_web_only_voice",
                               "pos_web_only_voice"]
FOLDER_MODALITIES["double"] = ["neg_plus_only_cough_breath",
                               "neg_plus_only_voice_breath",
                               "neg_plus_only_voice_cough",
                               "pos_plus_only_cough_breath",
                               "pos_plus_only_voice_breath",
                               "pos_plus_only_voice_cough",
                               "neg_train_only_cough_breath",
                               "neg_train_only_voice_breath",
                               "neg_train_only_voice_cough",
                               "pos_train_only_cough_breath",
                               "pos_train_only_voice_breath",
                               "pos_train_only_voice_cough",
                               "neg_test_only_cough_breath",
                               "neg_test_only_voice_breath",
                               "neg_test_only_voice_cough",
                               "pos_test_only_cough_breath",
                               "pos_test_only_voice_breath",
                               "pos_test_only_voice_cough",
                               "neg_devel_only_cough_breath",
                               "neg_devel_only_voice_breath",
                               "neg_devel_only_voice_cough",
                               "pos_devel_only_cough_breath",
                               "pos_devel_only_voice_breath",
                               "pos_devel_only_voice_cough",
                               "neg_web_only_cough_breath",
                               "neg_web_only_voice_breath",
                               "neg_web_only_voice_cough",
                               "pos_web_only_cough_breath",
                               "pos_web_only_voice_breath",
                               "pos_web_only_voice_cough"]
FOLDER_MODALITIES["triple"] = ["neg_plus",
                               "pos_plus",
                               "neg_train",
                               "pos_train",
                               "neg_test",
                               "pos_test",
                               "neg_devel",
                               "pos_devel",
                               "neg_web",
                               "pos_web"]


def get_dataset_info(tfrecords_folder,
                     data_focus,
                     use_modality_train,
                     homogeneous_batches):
    partitions = ["train",
                  "devel",
                  "test"]

    if data_focus == "core":  # The folders containing all three modalities. The fixed ones for devel and test.
        subfolders = ["pos_train", "neg_train",
                      "pos_devel", "neg_devel",
                      "pos_test", "neg_test"]
    elif data_focus == "all":  # Adding folders that have at least one modality; only positives.
        subfolders = ["pos_train", "neg_train",
                      "pos_devel", "neg_devel",
                      "pos_test", "neg_test",
                      "pos_train_only_voice", "pos_train_only_breath", "pos_train_only_cough",
                      "pos_train_only_voice_breath", "pos_train_only_voice_cough", "pos_train_only_cough_breath"]
    elif data_focus == "all+web":  # Also adding web positives.
        subfolders = ["pos_train", "neg_train",
                      "pos_devel", "neg_devel",
                      "pos_test", "neg_test",
                      "pos_train_only_voice", "pos_train_only_breath", "pos_train_only_cough",
                      "pos_train_only_voice_breath", "pos_train_only_voice_cough", "pos_train_only_cough_breath",
                      "pos_web", "pos_web_only_breath", "pos_web_only_cough", "pos_web_only_voice",
                      "pos_web_only_cough_breath", "pos_web_only_voice_breath", "pos_web_only_voice_cough"]
    else:
        raise ValueError

    subfolders_eff = list()
    for s_f in subfolders:
        to_add = False
        for modality, use in use_modality_train.items():
            if use:
                if "only" not in s_f:
                    to_add = True
                else:
                    if modality in s_f:
                        to_add = True
        if to_add:
            subfolders_eff.append(s_f)
    subfolders = subfolders_eff

    path_list_dict = dict()
    partition_size_dict = dict()
    print("Folders with relevant input data:")
    for partition in partitions:
        path_list_dict[partition] = collections.defaultdict(list)
        partition_size_dict[partition] = collections.defaultdict(int)
        for modality_combination in ["single", "double", "triple"]:
            partition_eff = partition
            subfolders_eff = [f for f in subfolders if (partition_eff in f) and (f in FOLDER_MODALITIES[modality_combination])]
            for f in subfolders:
                if ("web" in f) and (partition_eff == "train") and (f in FOLDER_MODALITIES[modality_combination]):
                    subfolders_eff.append(f)
            print(modality_combination, subfolders_eff)

            all_path_list = list()

            if len(subfolders_eff) == 0:
                continue

            for s_f in subfolders_eff:
                to_extend = os.listdir(tfrecords_folder + "/" + s_f)
                to_extend = filter_names(to_extend,
                                         pos_variations=None,
                                         neg_variations=None)
                to_extend = [tfrecords_folder + "/" + s_f + "/" + name for name in to_extend]
                all_path_list.extend(to_extend)

            for path in all_path_list:
                has_voice = False
                has_cough = False
                has_breath = False

                availability_str = ""

                if "only" not in path:
                    has_voice = True
                    has_cough = True
                    has_breath = True
                else:
                    if "voice" in path:
                        has_voice = True
                    if "cough" in path:
                        has_cough = True
                    if "breath" in path:
                        has_breath = True

                # if has_voice and use_modality_train["voice"]:
                if has_voice:
                    availability_str += "_voice"
                # if has_cough and use_modality_train["cough"]:
                if has_cough:
                    availability_str += "_cough"
                # if has_breath and use_modality_train["breath"]:
                if has_breath:
                    availability_str += "_breath"

                # Homogeneous batches.
                if homogeneous_batches:
                    if "pos" in path:
                        asthma_str = "_pos"
                    else:
                        asthma_str = "_neg"
                else:
                    asthma_str = ""

                path_list_dict[partition][modality_combination + availability_str + asthma_str].append(path)
                partition_size_dict[partition][modality_combination + availability_str + asthma_str] += 1

    return path_list_dict, partition_size_dict


def get_name_to_metadata(tf_names):
    name_to_metadata = dict()
    for name in [
                 "cough_logmel_spectrogram_support",
                 "breath_logmel_spectrogram_support",
                 "voice_logmel_spectrogram_support",
                 "cough_logmel_spectrogram",
                 "breath_logmel_spectrogram",
                 "voice_logmel_spectrogram",
                 # "segment_id",
                 # "version_id",
                 "asthma",
                 "age",
                 "sex",
                 "smoking",
                 "language",
                 "covid_tested",
                 "recording_source",
                 "copd",
                 "longterm",
                 "lung",
                 "pulmonary",
                 "drycough",
                 "wetcough",
                 "sorethroat",
                 "shortbreath",
                 ]:
        name_to_metadata[name] = dict()

    name_to_metadata["cough_logmel_spectrogram_support"]["numpy_shape"] = (None, 1)
    name_to_metadata["breath_logmel_spectrogram_support"]["numpy_shape"] = (None, 1)
    name_to_metadata["voice_logmel_spectrogram_support"]["numpy_shape"] = (None, 1)
    name_to_metadata["cough_logmel_spectrogram"]["numpy_shape"] = (None, 128)
    name_to_metadata["breath_logmel_spectrogram"]["numpy_shape"] = (None, 128)
    name_to_metadata["voice_logmel_spectrogram"]["numpy_shape"] = (None, 128)
    # name_to_metadata["segment_id"]["numpy_shape"] = (1, )
    # name_to_metadata["version_id"]["numpy_shape"] = (1, )
    name_to_metadata["asthma"]["numpy_shape"] = (1, )
    name_to_metadata["age"]["numpy_shape"] = (10, )
    name_to_metadata["sex"]["numpy_shape"] = (4, )
    name_to_metadata["smoking"]["numpy_shape"] = (8, )
    name_to_metadata["language"]["numpy_shape"] = (12, )
    name_to_metadata["covid_tested"]["numpy_shape"] = (12, )
    name_to_metadata["recording_source"]["numpy_shape"] = (3, )
    name_to_metadata["copd"]["numpy_shape"] = (1, )
    name_to_metadata["longterm"]["numpy_shape"] = (1, )
    name_to_metadata["lung"]["numpy_shape"] = (1, )
    name_to_metadata["pulmonary"]["numpy_shape"] = (1, )
    name_to_metadata["drycough"]["numpy_shape"] = (1, )
    name_to_metadata["wetcough"]["numpy_shape"] = (1, )
    name_to_metadata["sorethroat"]["numpy_shape"] = (1, )
    name_to_metadata["shortbreath"]["numpy_shape"] = (1, )

    name_to_metadata["cough_logmel_spectrogram_support"]["variable_type"] = "support"
    name_to_metadata["breath_logmel_spectrogram_support"]["variable_type"] = "support"
    name_to_metadata["voice_logmel_spectrogram_support"]["variable_type"] = "support"
    name_to_metadata["cough_logmel_spectrogram"]["variable_type"] = "x"
    name_to_metadata["breath_logmel_spectrogram"]["variable_type"] = "x"
    name_to_metadata["voice_logmel_spectrogram"]["variable_type"] = "x"
    # name_to_metadata["segment_id"]["variable_type"] = "id"
    # name_to_metadata["version_id"]["variable_type"] = "id"
    name_to_metadata["asthma"]["variable_type"] = "y"
    name_to_metadata["age"]["variable_type"] = "y"
    name_to_metadata["sex"]["variable_type"] = "y"
    name_to_metadata["smoking"]["variable_type"] = "y"
    name_to_metadata["language"]["variable_type"] = "y"
    name_to_metadata["covid_tested"]["variable_type"] = "y"
    name_to_metadata["recording_source"]["variable_type"] = "y"
    name_to_metadata["copd"]["variable_type"] = "y"
    name_to_metadata["longterm"]["variable_type"] = "y"
    name_to_metadata["lung"]["variable_type"] = "y"
    name_to_metadata["pulmonary"]["variable_type"] = "y"
    name_to_metadata["drycough"]["variable_type"] = "y"
    name_to_metadata["wetcough"]["variable_type"] = "y"
    name_to_metadata["sorethroat"]["variable_type"] = "y"
    name_to_metadata["shortbreath"]["variable_type"] = "y"

    name_to_metadata = process_metadata(name_to_metadata)

    # name_to_metadata = {k: name_to_metadata[k] for k in tf_names}  # TODO

    return name_to_metadata


def get_config_dict_from_yaml(file_name):
    # Read the parameters from the YAML file.
    stream = open(YAML_CONFIGURATION_FOLDER + "/" + file_name + ".yaml", 'r')
    CONFIG_DICT = yaml.safe_load(stream)
    stream.close()

    use_modality_train = dict()
    use_modality_train["voice"] = False
    use_modality_train["breath"] = False
    use_modality_train["cough"] = False

    for input_type in CONFIG_DICT["model_configuration"]["input_type_list"]:
        if "voice" in input_type:
            use_modality_train["voice"] = True
        if "breath" in input_type:
            use_modality_train["breath"] = True
        if "cough" in input_type:
            use_modality_train["cough"] = True

    # Get the list of TFRECORDS file paths per partition.
    PATH_LIST_DICT, \
    PARTITIONS_SIZE_DICT = get_dataset_info(TFRECORDS_FOLDER,
                                            CONFIG_DICT["data_focus"],
                                            use_modality_train,
                                            CONFIG_DICT["homogeneous_batches"])

    CONFIG_DICT["use_modality_train"] = use_modality_train

    CONFIG_DICT["tfrecords_folder"] = TFRECORDS_FOLDER
    CONFIG_DICT["output_folder"] = OUTPUT_FOLDER

    CONFIG_DICT["path_list_dict"] = PATH_LIST_DICT

    CONFIG_DICT["model_configuration"]["name_to_metadata"] = get_name_to_metadata(CONFIG_DICT["model_configuration"]["input_type_list"] +
                                                                                  CONFIG_DICT["model_configuration"]["output_type_list"])
    CONFIG_DICT["results_summary_path"] = OUTPUT_FOLDER + "/" + CONFIG_DICT["method_string"] + "/results_summary"
    CONFIG_DICT["items_summary_path"] = OUTPUT_FOLDER + "/" + CONFIG_DICT["method_string"] + "/items_summary"

    CONFIG_DICT["monitor_target_to_measures"] = {"asthma":
                                                     ["au_pr",],}
    CONFIG_DICT["target_to_task_type"] = {"asthma":
                                              "binary_classification",
                                          "age": "multiclass_classification",
                                          "sex": "multiclass_classification",
                                          "smoking": "multiclass_classification",
                                          "language": "multiclass_classification",
                                          "covid_tested": "multiclass_classification",
                                          "recording_source": "multiclass_classification",
                                          "copd": "binary_classification",
                                          "longterm": "binary_classification",
                                          "lung": "binary_classification",
                                          "pulmonary": "binary_classification",
                                          "drycough": "binary_classification",
                                          "wetcough": "binary_classification",
                                          "sorethroat": "binary_classification",
                                          "shortbreath": "binary_classification"}

    CONFIG_DICT["output_channel_targets"] = ["asthma",]

    import covid19_sounds.neurips21.losses as losses
    import covid19_sounds.neurips21.evaluation as evaluation
    import covid19_sounds.neurips21.architecture as architecture

    CONFIG_DICT["losses_module"] = losses
    CONFIG_DICT["evaluation_module"] = evaluation
    CONFIG_DICT["architecture_module"] = architecture

    return CONFIG_DICT


def get_config_dict_from_yaml_path(file_path):
    # Read the parameters from the YAML file.
    stream = open(file_path, 'r')
    CONFIG_DICT = yaml.load(stream)
    stream.close()

    CONFIG_DICT["model_configuration"]["name_to_metadata"] = get_name_to_metadata(
        CONFIG_DICT["model_configuration"]["input_type_list"] +
        CONFIG_DICT["model_configuration"]["output_type_list"])
    return CONFIG_DICT
