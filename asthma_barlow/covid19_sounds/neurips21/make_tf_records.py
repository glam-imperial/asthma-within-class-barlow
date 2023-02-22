import os
import collections
import random
import statistics

import pandas as pd
import librosa
import numpy as np

from covid19_sounds.neurips21 import fairness
from covid19_sounds.neurips21 import preprocessing_utils

from common import normalise, tfrecord_creator
import covid19_sounds.neurips21.configuration as configuration
from common.data_sample import Sample

#####################################################################################################################
# Definitions.
#####################################################################################################################
# We first preprocess android and ios recordings because we have the user information.
RECORDING_SOURCES = ["android",
                     "ios"]

COLUMN_NAMES = ["Folder Name",
                "Age",
                "Sex",
                "Medhistory",
                "Smoking",
                "Language",
                "Uid",
                "Date",
                "Symptoms",
                "Covid-Tested",
                "Hospitalized",
                "Voice filename",
                "Cough filename",
                "Breath filename",
                "Voice check",
                "Cough check",
                "Breath check",
                "Sampling Rate"]

#####################################################################################################################
# Read metadata files.
#####################################################################################################################
METADATA_FILEPATH = dict()
for recording_source in ["web", "android", "ios"]:
    METADATA_FILEPATH[recording_source] = preprocessing_utils.DATA_FOLDER + "/all_metadata" + "/results_raw_20210426_lan_yamnet_" + recording_source + "_noloc.csv"

#####################################################################################################################
# ANDROID + IOS
# Find asthmatic and non-asthmatic user related metadata.
#####################################################################################################################
# Dictionary from user to a list of metadata. Done in order to clean-up metadata.
# It's for android and ios users only.
user_metadata = collections.defaultdict(list)

# Count users per case.
asthma_user_counts = dict()
asthma_user_counts["total"] = 0
asthma_user_counts["all_three"] = 0
asthma_user_counts["voice"] = 0
asthma_user_counts["cough"] = 0
asthma_user_counts["breath"] = 0

non_asthma_user_counts = dict()
non_asthma_user_counts["total"] = 0
non_asthma_user_counts["all_three"] = 0
non_asthma_user_counts["voice"] = 0
non_asthma_user_counts["cough"] = 0
non_asthma_user_counts["breath"] = 0

# Append users per case.
asthma_users = dict()
asthma_users["total"] = list()
asthma_users["all_three"] = list()
asthma_users["voice"] = list()
asthma_users["cough"] = list()
asthma_users["breath"] = list()

non_asthma_users = dict()
non_asthma_users["total"] = list()
non_asthma_users["all_three"] = list()
non_asthma_users["voice"] = list()
non_asthma_users["cough"] = list()
non_asthma_users["breath"] = list()

# Exploration of metadata types for alternative strings, typos.
# age_types = set()
# sex_types = set()
# med_history_types = set()
# smoking_types = set()
# language_types = set()
# symptoms_types = set()
# covid_tested_types = set()
# hospitalised_types = set()

# Read metadata.
metadata_df = dict()
for recording_source in RECORDING_SOURCES:
    metadata_df[recording_source] = pd.read_csv(METADATA_FILEPATH[recording_source],
                                                delimiter=";")
    for index, row in metadata_df[recording_source].iterrows():
        med_history_list = row["Medhistory"].split(",")
        age = row["Age"]
        sex = row["Sex"]
        smoking = row["Smoking"]
        language = row["Language"]
        symptoms_list = row["Symptoms"].split(",")
        covid_tested = row["Covid-Tested"]
        hospitalised = row["Hospitalized"]

        metadata_u = dict()
        metadata_u["age"] = age
        metadata_u["sex"] = sex
        metadata_u["smoking"] = smoking
        metadata_u["language"] = language
        metadata_u["med_history"] = med_history_list
        metadata_u["symptoms"] = symptoms_list
        metadata_u["covid_tested"] = covid_tested
        metadata_u["hospitalised"] = hospitalised
        metadata_u["recording_source"] = recording_source

        user_metadata[row["Uid"]].append(metadata_u)

        if "asthma" in med_history_list:
            has_asthma = True
            # if row["Uid"] in non_asthma_users["total"]:
            #     raise ValueError
        else:
            has_asthma = False
            # if row["Uid"] in asthma_users["total"]:
            #     raise ValueError

        # age_types.add(age)
        # sex_types.add(sex)
        # for mh in med_history_list:
        #     med_history_types.add(mh)
        # smoking_types.add(smoking)
        # language_types.add(language)
        # for symptom in symptoms_list:
        #     symptoms_types.add(symptom)
        # covid_tested_types.add(covid_tested)
        # hospitalised_types.add(hospitalised)
        if has_asthma:
            asthma_user_counts,\
            asthma_users = preprocessing_utils.identify_users_with_clean_recordings(row,
                                                                                    asthma_user_counts,
                                                                                    asthma_users,
                                                                                    recording_source)
        else:
            non_asthma_user_counts, \
            non_asthma_users = preprocessing_utils.identify_users_with_clean_recordings(row,
                                                                                        non_asthma_user_counts,
                                                                                        non_asthma_users,
                                                                                        recording_source)

print("Asthmatic rows.")
print("At least 1 modality:", asthma_user_counts["total"])
print("All 3 modalities:", asthma_user_counts["all_three"])
print("At least voice:", asthma_user_counts["voice"])
print("At least cough:", asthma_user_counts["cough"])
print("At least breath:", asthma_user_counts["breath"])

print("Non-asthmatic rows.")
print("At least 1 modality:", non_asthma_user_counts["total"])
print("All 3 modalities:", non_asthma_user_counts["all_three"])
print("At least voice:", non_asthma_user_counts["voice"])
print("At least cough:", non_asthma_user_counts["cough"])
print("At least breath:", non_asthma_user_counts["breath"])

asthma_user_set = dict()
non_asthma_user_set = dict()
asthma_user_set["total"] = set(asthma_users["total"])
asthma_user_set["all_three"] = set(asthma_users["all_three"])
asthma_user_set["voice"] = set(asthma_users["voice"])
asthma_user_set["cough"] = set(asthma_users["cough"])
asthma_user_set["breath"] = set(asthma_users["breath"])
non_asthma_user_set["total"] = set(non_asthma_users["total"])
non_asthma_user_set["all_three"] = set(non_asthma_users["all_three"])
non_asthma_user_set["voice"] = set(non_asthma_users["voice"])
non_asthma_user_set["cough"] = set(non_asthma_users["cough"])
non_asthma_user_set["breath"] = set(non_asthma_users["breath"])

print("Asthmatic users.")
print("At least 1 modality:", len(asthma_user_set["total"]))
print("All 3 modalities:", len(asthma_user_set["all_three"]))
print("At least voice:", len(asthma_user_set["voice"]))
print("At least cough:", len(asthma_user_set["cough"]))
print("At least breath:", len(asthma_user_set["breath"]))

print("Non-asthmatic users.")
print("At least 1 modality:", len(non_asthma_user_set["total"]))
print("All 3 modalities:", len(non_asthma_user_set["all_three"]))
print("At least voice:", len(non_asthma_user_set["voice"]))
print("At least cough:", len(non_asthma_user_set["cough"]))
print("At least breath:", len(non_asthma_user_set["breath"]))


both_have_asthma_and_not = asthma_user_set["total"].intersection(non_asthma_user_set["total"])
if len(both_have_asthma_and_not) == 0:
    print("There are no users that have once claimed to have asthma and once not to.")
else:
    print("There are some users that have once claimed to have asthma and once not to.")
    print(both_have_asthma_and_not)
    print(len(both_have_asthma_and_not))

print("Find a common set of users among the 5 cases.")
common_asthma_user_set = asthma_user_set["total"].intersection(asthma_user_set["all_three"],
                                                               asthma_user_set["voice"],
                                                               asthma_user_set["cough"],
                                                               asthma_user_set["breath"])
assert len(common_asthma_user_set) == len(asthma_user_set["all_three"])

common_non_asthma_user_set = non_asthma_user_set["total"].intersection(non_asthma_user_set["all_three"],
                                                                       non_asthma_user_set["voice"],
                                                                       non_asthma_user_set["cough"],
                                                                       non_asthma_user_set["breath"])
assert len(common_non_asthma_user_set) == len(non_asthma_user_set["all_three"])

print("Common asthmatic:", len(common_asthma_user_set))
print("Common non-asthmatic:", len(common_non_asthma_user_set))

# Print different types; including typos, variations. Has been used to form TYPES_STATS in preprocessing_utils.py
# print(age_types)
# print(sex_types)
# print(med_history_types)
# print(smoking_types)
# print(language_types)
# print(symptoms_types)
# print(covid_tested_types)
# print(hospitalised_types)

# print(types_counts)
# Asthma
# {'hospitalised': defaultdict(<class 'int'>, {'pnts': 19, 'no': 2512, 'yes': 24}), 'age': defaultdict(<class 'int'>, {'40-49': 514, '70-79': 41, '50-59': 285, '30-39': 727, '90-': 2, 'pnts': 46, '80-89': 3, '60-69': 121, '0-19': 213, '20-29': 602, None: 1}), 'symptoms': defaultdict(<class 'int'>, {'runnyblockednose': 469, None: 695, 'tightness': 431, 'drycough': 1053, 'pnts': 51, 'shortbreath': 552, 'muscleache': 278, 'smelltasteloss': 117, 'fever': 167, 'chills': 173, 'sorethroat': 523, 'wetcough': 507, 'dizziness': 166, 'headache': 484}), 'language': defaultdict(<class 'int'>, {'el': 33, 'es': 136, 'ro': 2, 'de': 107, 'en': 1584, 'it': 477, 'fr': 62, 'None': 1, 'pt': 132, 'ru': 21}), 'sex': defaultdict(<class 'int'>, {'pnts': 22, None: 1, 'Other': 13, 'Female': 1162, 'Male': 1357}), 'med_history': defaultdict(<class 'int'>, {None: 9, 'hbp': 284, 'stroke': 41, 'lung': 70, 'organ': 24, 'angina': 36, 'diabetes': 98, 'valvular': 47, 'long': 72, 'otherHeart': 58, 'hiv': 50, 'cystic': 44, 'asthma': 2555, 'longterm': 93, 'pnts': 6, 'heart': 47, 'pulmonary': 41, 'copd': 100, 'cancer': 40}), 'covid_tested': defaultdict(<class 'int'>, {'neverThinkHadCOVIDNever': 1257, 'negativeLast14': 74, 'neverThinkHadCOVIDLast14': 35, 'pnts': 72, 'negativeOver14': 108, 'yes': 4, None: 4, 'last14': 8, 'no': 23, 'never': 198, 'positiveOver14': 12, 'neverThinkHadCOVIDNow': 174, 'positiveLast14': 61, 'neverThinkHadCOVIDOver14': 138, 'over14': 2, 'negativeNever': 385}), 'smoking': defaultdict(<class 'int'>, {'21+': 14, '11to20': 205, 'never': 1394, None: 3, 'ex': 470, 'pnts': 48, 'ecig': 31, '1to10': 297, 'ltOnce': 93})}
# Non Asthma
# {'covid_tested': defaultdict(<class 'int'>, {None: 42, 'never': 2353, 'last14': 52, 'neverThinkHadCOVIDNever': 15034, 'pnts': 655, 'yes': 18, 'positiveOver14': 239, 'positiveLast14': 702, 'negativeNever': 3426, 'neverThinkHadCOVIDNow': 1940, 'neverThinkHadCOVIDLast14': 544, 'negativeOver14': 1324, 'over14': 23, 'neverThinkHadCOVIDOver14': 1439, 'negativeLast14': 772, 'no': 167}), 'sex': defaultdict(<class 'int'>, {'pnts': 378, None: 15, 'Male': 17822, 'Other': 63, 'Female': 10452}), 'symptoms': defaultdict(<class 'int'>, {None: 12006, 'never': 1, 'chills': 1364, 'drycough': 9113, 'wetcough': 3932, 'fever': 1551, 'pnts': 677, 'shortbreath': 2098, 'headache': 4020, 'smelltasteloss': 1026, 'muscleache': 2386, 'dizziness': 1156, 'runnyblockednose': 3796, 'tightness': 2123, 'sorethroat': 4655}), 'med_history': defaultdict(<class 'int'>, {None: 23086, 'asthma': 571, 'cystic': 76, 'hbp': 2522, 'heart': 196, 'otherHeart': 358, 'pnts': 878, 'diabetes': 741, 'lung': 271, 'stroke': 153, 'pulmonary': 70, 'organ': 43, 'long': 373, 'hiv': 166, 'longterm': 742, 'copd': 255, 'valvular': 188, 'cancer': 164, 'angina': 138}), 'age': defaultdict(<class 'int'>, {'90-': 18, '80-89': 64, '0-19': 2668, '70-79': 484, '60-69': 1497, '20-29': 6874, '30-39': 7558, '40-49': 5718, 'pnts': 666, '50-59': 3180, None: 3}), 'hospitalised': defaultdict(<class 'int'>, {None: 4, 'pnts': 237, 'no': 28237, 'yes': 252}), 'language': defaultdict(<class 'int'>, {'it': 7395, 'hi': 2, 'de': 1103, 'zh': 23, 'ro': 75, 'en': 14916, 'pt': 1514, 'fr': 644, 'None': 1, 'ru': 727, 'es': 1800, 'el': 530}), 'smoking': defaultdict(<class 'int'>, {None: 19, 'never': 15623, 'ecig': 330, 'ltOnce': 923, 'pnts': 655, 'ex': 5078, '21+': 230, '1to10': 3336, '11to20': 2536})}

#####################################################################################################################
# Make metadata array for the common, all-three modality core user set.
#####################################################################################################################
# Make a list of all metadata types and give them a numerical index.
TYPES = list()
TYPE_NAME_COLUMN_INDEX = dict()
counter = 0
for metadata_type in sorted(preprocessing_utils.TYPES_STATS.keys()):
    for metadata_value in sorted(set(list(preprocessing_utils.TYPES_STATS[metadata_type].values()))):
        TYPES.append(metadata_value)
        TYPE_NAME_COLUMN_INDEX[metadata_value] = counter
        counter += 1

print(TYPES)
print(TYPE_NAME_COLUMN_INDEX)

# Initialise metadata array.
all_three_user_metadata = {k: v for k, v in user_metadata.items() if (k in common_asthma_user_set) or (k in common_non_asthma_user_set)}
metadata_array = np.empty(shape=(len(all_three_user_metadata.keys()), len(TYPES)), dtype=np.float32)
user_ids_sorted = sorted(all_three_user_metadata.keys())

# Populate metadata array.
# First check user metadata consistency. Each time a user made a contribution, they may have submitted different metadata.
# In some cases, a correction needs to be made. And we use the most commonly submitted info by the same user. (filtered_v)
consistency_dict = collections.defaultdict(list)
for uid_i, uid in enumerate(user_ids_sorted):
    metadata_list = all_three_user_metadata[uid]

    # In these cases, we use the filtered value, even though someone may presumably change category.
    # Even though the category may change, the most commonly submitted info is useful in the data partitioning.
    for metadata_name in ["age",
                          "sex",
                          "smoking",
                          "language"]:
        consistency, filtered_v, count_v = preprocessing_utils.check_consistency(metadata_list, metadata_name)
        consistency_dict[metadata_name].append(consistency)

        metadata_array[uid_i, TYPE_NAME_COLUMN_INDEX[filtered_v]] = 1.0

    # In these cases, we use the filtered value. There is a potential issue here:
    # Some users may be COVID-tested for half of the measurements and not tested for the other.
    # We treat a user as representative of the most common type, and disregard the other in terms of partitioning.
    # (But the correct labels per measurement are stored in the tf records files at the end, for purposes of training and evaluation.)
    # This means that the users in the partitions can still contain measurements with different metadata than the ones assumed here.
    # Since this study is focused on explaining-away all other metadata that is not asthma, we believe that this is 100% in line with out assumptions.
    for metadata_name in ["covid_tested",
                          "hospitalised",
                          "recording_source"]:
        consistency, filtered_v, count_v = preprocessing_utils.check_consistency(metadata_list, metadata_name)
        consistency_dict[metadata_name].append(consistency)

        for filtered_v, count in count_v.items():
            metadata_array[uid_i, TYPE_NAME_COLUMN_INDEX[filtered_v]] = count

    # Here, the filtered value is in [0, 1] per medical condition name. Either doesn't have, or have.
    # We require 100% levels of consistency to add a user as having a medical condition.
    # (Even though presumably they may be diagnosed in-between measurements.)
    # E.g., asthma is 100% consistent for all users (if printed, you can verify).
    consistency_list = preprocessing_utils.check_consistency_list(metadata_list, "med_history")
    for m_name_symptom, consistency, filtered_v, count_v in consistency_list:
        consistency_dict[m_name_symptom].append(consistency)

        if filtered_v == 1:
            metadata_array[uid_i, TYPE_NAME_COLUMN_INDEX[m_name_symptom]] = 1.0

    # Not the same treatment for symptoms of COVID -- perfectly possible to have different symptoms per measurement.
    consistency_list = preprocessing_utils.check_consistency_list(metadata_list, "symptoms")
    for m_name_symptom, consistency, filtered_v, count_v in consistency_list:
        consistency_dict[m_name_symptom].append(consistency)

        metadata_array[uid_i, TYPE_NAME_COLUMN_INDEX[m_name_symptom]] = count_v[1]

# Print this to see levels of consistency. They are all highly consistent.
# for k, v in consistency_dict.items():
#     print(k)
#     print(statistics.mean(v))

# print(metadata_array)

# Binarise metadata array.
metadata_array[metadata_array > 0.0] = 1.0

metadata_df = pd.DataFrame(data=metadata_array,    # values
                           index=np.arange(metadata_array.shape[0]),    # 1st column as index
                           columns=TYPES)

#### User partitioning with fair metadata percentage mixture preservation.
# There are many more non-asthmatic users, and so we
asthma_global_index_train,\
asthma_global_index_devel,\
asthma_global_index_test, \
non_asthma_global_index_train,\
non_asthma_global_index_plus, \
non_asthma_global_index_devel,\
non_asthma_global_index_test = fairness.partitioning(metadata_df)

# Check if any indices were lost.
# bbb = set(asthma_global_index_train).union(set(asthma_global_index_devel),
#                                            set(asthma_global_index_test),
#                                            set(non_asthma_global_index_train),
#                                            set(non_asthma_global_index_plus),
#                                            set(non_asthma_global_index_devel),
#                                            set(non_asthma_global_index_test))
# print(len(bbb))
# print(sorted(list(bbb)))

#####################################################################################################################
# WEB
# (similar preprocessing for the web-app collected data as with the android + ios data.)
# There is only 1 user ID for web collection. We must use all WEB users on a single partition.
# Either as training, or as an evaluation set.
#####################################################################################################################
web_user_metadata = collections.defaultdict(list)

# Count users per case.
web_asthma_user_counts = dict()
web_asthma_user_counts["total"] = 0
web_asthma_user_counts["all_three"] = 0
web_asthma_user_counts["voice"] = 0
web_asthma_user_counts["cough"] = 0
web_asthma_user_counts["breath"] = 0

web_non_asthma_user_counts = dict()
web_non_asthma_user_counts["total"] = 0
web_non_asthma_user_counts["all_three"] = 0
web_non_asthma_user_counts["voice"] = 0
web_non_asthma_user_counts["cough"] = 0
web_non_asthma_user_counts["breath"] = 0

# Append users per case.
web_asthma_users = dict()
web_asthma_users["total"] = list()
web_asthma_users["all_three"] = list()
web_asthma_users["voice"] = list()
web_asthma_users["cough"] = list()
web_asthma_users["breath"] = list()

web_non_asthma_users = dict()
web_non_asthma_users["total"] = list()
web_non_asthma_users["all_three"] = list()
web_non_asthma_users["voice"] = list()
web_non_asthma_users["cough"] = list()
web_non_asthma_users["breath"] = list()

# Read metadata file.
web_metadata_df = pd.read_csv(METADATA_FILEPATH["web"],
                              delimiter=";")
# web_counter = 0
for index, row in web_metadata_df.iterrows():
    med_history_list = row["Medhistory"].split(",")
    age = row["Age"]
    sex = row["Sex"]
    smoking = row["Smoking"]
    language = row["Language"]
    symptoms_list = row["Symptoms"].split(",")
    covid_tested = row["Covid-Tested"]
    hospitalised = row["Hospitalized"]

    metadata_u = dict()
    metadata_u["age"] = age
    metadata_u["sex"] = sex
    metadata_u["smoking"] = smoking
    metadata_u["language"] = language
    metadata_u["symptoms_list"] = symptoms_list
    metadata_u["covid_tested"] = covid_tested
    metadata_u["hospitalised"] = hospitalised
    metadata_u["recording_source"] = "web"

    # uid = row["Uid"] + repr(web_counter)
    uid = row["Uid"]  # for web data collection, we do not have user identifiers. They are all under a single ID.
    # web_counter += 1

    web_user_metadata[uid].append(metadata_u)

    if "asthma" in med_history_list:
        has_asthma = True
    else:
        has_asthma = False

    # age_types.add(age)
    # sex_types.add(sex)
    # for mh in med_history_list:
    #     med_history_types.add(mh)
    # smoking_types.add(smoking)
    # language_types.add(language)
    # for symptom in symptoms_list:
    #     symptoms_types.add(symptom)
    # covid_tested_types.add(covid_tested)
    # hospitalised_types.add(hospitalised)
    if has_asthma:
        web_asthma_user_counts,\
        web_asthma_users = preprocessing_utils.identify_users_with_clean_recordings(row,
                                                                                    web_asthma_user_counts,
                                                                                    web_asthma_users,
                                                                                    "web")
    else:
        web_non_asthma_user_counts, \
        web_non_asthma_users = preprocessing_utils.identify_users_with_clean_recordings(row,
                                                                                        web_non_asthma_user_counts,
                                                                                        web_non_asthma_users,
                                                                                        "web")

# The counts are interesting.
print("WEB")
print("Asthmatic rows.")
print("At least 1 modality:", web_asthma_user_counts["total"])
print("All 3 modalities:", web_asthma_user_counts["all_three"])
print("At least voice:", web_asthma_user_counts["voice"])
print("At least cough:", web_asthma_user_counts["cough"])
print("At least breath:", web_asthma_user_counts["breath"])

print("Non-asthmatic rows.")
print("At least 1 modality:", web_non_asthma_user_counts["total"])
print("All 3 modalities:", web_non_asthma_user_counts["all_three"])
print("At least voice:", web_non_asthma_user_counts["voice"])
print("At least cough:", web_non_asthma_user_counts["cough"])
print("At least breath:", web_non_asthma_user_counts["breath"])

# No web user information; just one user ID.
web_asthma_user_set = dict()
web_non_asthma_user_set = dict()
web_asthma_user_set["total"] = set(web_asthma_users["total"])
web_asthma_user_set["all_three"] = set(web_asthma_users["all_three"])
web_asthma_user_set["voice"] = set(web_asthma_users["voice"])
web_asthma_user_set["cough"] = set(web_asthma_users["cough"])
web_asthma_user_set["breath"] = set(web_asthma_users["breath"])
web_non_asthma_user_set["total"] = set(web_non_asthma_users["total"])
web_non_asthma_user_set["all_three"] = set(web_non_asthma_users["all_three"])
web_non_asthma_user_set["voice"] = set(web_non_asthma_users["voice"])
web_non_asthma_user_set["cough"] = set(web_non_asthma_users["cough"])
web_non_asthma_user_set["breath"] = set(web_non_asthma_users["breath"])

print("Asthmatic users.")
print("At least 1 modality:", len(web_asthma_user_set["total"]))
print("All 3 modalities:", len(web_asthma_user_set["all_three"]))
print("At least voice:", len(web_asthma_user_set["voice"]))
print("At least cough:", len(web_asthma_user_set["cough"]))
print("At least breath:", len(web_asthma_user_set["breath"]))

print("Non-asthmatic users.")
print("At least 1 modality:", len(web_non_asthma_user_set["total"]))
print("All 3 modalities:", len(web_non_asthma_user_set["all_three"]))
print("At least voice:", len(web_non_asthma_user_set["voice"]))
print("At least cough:", len(web_non_asthma_user_set["cough"]))
print("At least breath:", len(web_non_asthma_user_set["breath"]))

web_both_have_asthma_and_not = web_asthma_user_set["total"].intersection(web_non_asthma_user_set["total"])
if len(web_both_have_asthma_and_not) == 0:
    print("There are no users that have once claimed to have asthma and once not to.")
else:
    print("There are some users that have once claimed to have asthma and once not to.")
    print(web_both_have_asthma_and_not)
    print(len(web_both_have_asthma_and_not))

print("Find a common set of users among the 5 cases.")
web_common_asthma_user_set = web_asthma_user_set["total"].intersection(web_asthma_user_set["all_three"],
                                                                       web_asthma_user_set["voice"],
                                                                       web_asthma_user_set["cough"],
                                                                       web_asthma_user_set["breath"])
assert len(web_common_asthma_user_set) == len(web_asthma_user_set["all_three"])

web_common_non_asthma_user_set = web_non_asthma_user_set["total"].intersection(web_non_asthma_user_set["all_three"],
                                                                               web_non_asthma_user_set["voice"],
                                                                               web_non_asthma_user_set["cough"],
                                                                               web_non_asthma_user_set["breath"])
assert len(web_common_non_asthma_user_set) == len(web_non_asthma_user_set["all_three"])
print("Common asthmatic:", len(web_common_asthma_user_set))
print("Common non-asthmatic:", len(web_common_non_asthma_user_set))


#####################################################################################################################
# This is a python generator that yields Data Samples for tf record storage.
#####################################################################################################################
def get_generator(dataset_dict,
                  audio_folder):
    cough_size_list = list()
    breath_size_list = list()
    voice_size_list = list()

    counter = 0
    user_counter = 0
    uid_to_numerical = dict()
    # web_counter = 0

    # wav2vec2_model = preprocessing_utils.get_wav2vec2_model()

    metadata_df = dict()
    for recording_source in ["android", "ios", "web"]:
        metadata_df[recording_source] = pd.read_csv(METADATA_FILEPATH[recording_source],
                                                    delimiter=";")
        for index, row in metadata_df[recording_source].iterrows():
            med_history_list = row["Medhistory"].split(",")
            age = row["Age"]
            sex = row["Sex"]
            smoking = row["Smoking"]
            language = row["Language"]
            symptoms_list = row["Symptoms"].split(",")
            covid_tested = row["Covid-Tested"]
            hospitalised = row["Hospitalized"]

            voice_filename = row["Voice filename"]
            cough_filename = row["Cough filename"]
            breath_filename = row["Breath filename"]

            folder_name = row["Folder Name"]

            if recording_source == "web":
                # uid = row["Uid"] + repr(web_counter)
                uid = row["Uid"]
                # web_counter += 1
            else:
                uid = row["Uid"]

            if uid not in dataset_dict.keys():
                continue

            if uid not in uid_to_numerical.keys():
                uid_to_numerical[uid] = user_counter
                user_counter += 1

            partition = dataset_dict[uid]
            print(partition)

            has_asthma = "asthma" in med_history_list

            # Check if a) metadata file claims that modality exists and b) if sound file exists.
            has_voice = preprocessing_utils.modality_exists(row,
                                                            "voice",
                                                            recording_source)

            has_cough = preprocessing_utils.modality_exists(row,
                                                            "cough",
                                                            recording_source)

            has_breath = preprocessing_utils.modality_exists(row,
                                                             "breath",
                                                             recording_source)

            if "web" in partition:
                if has_asthma:
                    partition_eff = "pos_web"
                else:
                    partition_eff = "neg_web"
            else:
                partition_eff = partition

            if has_voice and has_cough and has_breath:
                pass
            # elif (not has_voice) and (not has_cough) and (not has_breath):
            #     continue
            else:
                partition_eff += "_only"
                if has_voice:
                    partition_eff += "_voice"
                if has_cough:
                    partition_eff += "_cough"
                if has_breath:
                    partition_eff += "_breath"

            if recording_source == "web":
                subfolder_name = audio_folder + "/" + "form-app-users" + "/" + folder_name
                cough_filename = cough_filename[:-4] + "wav"
                breath_filename = breath_filename[:-4] + "wav"
                voice_filename = voice_filename[:-4] + "wav"
            else:
                subfolder_name = audio_folder + "/" + uid + "/" + folder_name

            # Make DataSample.
            name = uid + "_" + repr(counter)

            id_dict = collections.OrderedDict()
            id_dict["user_id"] = uid_to_numerical[uid]
            id_dict["recording_id"] = counter

            print(counter, recording_source)
            counter += 1

            if "neg_plus" in partition:
                print("Exit", partition)
                continue

            if has_cough:
                try:
                    cough = librosa.load(subfolder_name + "/" + cough_filename, sr=16000)[0]
                except FileNotFoundError:
                    cough = librosa.load(subfolder_name + "/" + cough_filename[:-4] + ".wav", sr=16000)[0]
                cough_x_dict = preprocessing_utils.get_features_and_stats(cough)
                # cough_x_dict = preprocessing_utils.get_features_and_stats(cough, wav2vec2_model)
                print(cough_x_dict["logmel_spectrogram"].shape)
                # print(cough_x_dict["wav2vec_embeddings"].shape)
                cough_size_list.append(cough.size)
            else:
                cough = None
            if has_breath:
                try:
                    breath = librosa.load(subfolder_name + "/" + breath_filename, sr=16000)[0]
                except FileNotFoundError:
                    breath = librosa.load(subfolder_name + "/" + breath_filename[:-4] + ".wav", sr=16000)[0]
                breath_x_dict = preprocessing_utils.get_features_and_stats(breath)
                # breath_x_dict = preprocessing_utils.get_features_and_stats(breath, wav2vec2_model)
                print(breath_x_dict["logmel_spectrogram"].shape)
                # print(breath_x_dict["wav2vec_embeddings"].shape)
                breath_size_list.append(breath.size)
            else:
                breath = None
            if has_voice:
                try:
                    voice = librosa.load(subfolder_name + "/" + voice_filename, sr=16000)[0]
                except FileNotFoundError:
                    voice = librosa.load(subfolder_name + "/" + voice_filename[:-4] + ".wav", sr=16000)[0]
                voice_x_dict = preprocessing_utils.get_features_and_stats(voice)
                # voice_x_dict = preprocessing_utils.get_features_and_stats(voice, wav2vec2_model)
                print(voice_x_dict["logmel_spectrogram"].shape)
                # print(voice_x_dict["wav2vec_embeddings"].shape)
                voice_size_list.append(voice.size)
            else:
                voice = None

            x_dict = dict()
            if has_cough:
                x_dict["cough_logmel_spectrogram"] = cough_x_dict["logmel_spectrogram"]
                # x_dict["cough_waveform"] = cough_x_dict["waveform"]
                # x_dict["cough_wav2vec_embeddings"] = cough_x_dict["wav2vec_embeddings"]

            if has_breath:
                x_dict["breath_logmel_spectrogram"] = breath_x_dict["logmel_spectrogram"]
                # x_dict["breath_waveform"] = breath_x_dict["waveform"]
                # x_dict["breath_wav2vec_embeddings"] = breath_x_dict["wav2vec_embeddings"]

            if has_voice:
                x_dict["voice_logmel_spectrogram"] = voice_x_dict["logmel_spectrogram"]
                # x_dict["voice_waveform"] = voice_x_dict["waveform"]
                # x_dict["voice_wav2vec_embeddings"] = voice_x_dict["wav2vec_embeddings"]

            y_dict = dict()

            y_dict["asthma"] = np.zeros((1,), dtype=np.float32)
            if has_asthma:
                y_dict["asthma"][0] = np.float32(1.0)
            else:
                y_dict["asthma"][0] = np.float32(0.0)

            y_dict["recording_source"] = np.zeros((3,), dtype=np.float32)
            if recording_source == "android":
                y_dict["recording_source"][0] = 1.0
            elif recording_source == "ios":
                y_dict["recording_source"][1] = 1.0
            elif recording_source == "web":
                y_dict["recording_source"][2] = 1.0
            else:
                raise ValueError

            y_dict["age"] = preprocessing_utils.get_label(age, "age")
            y_dict["sex"] = preprocessing_utils.get_label(sex, "sex")
            y_dict["smoking"] = preprocessing_utils.get_label(smoking, "smoking")
            y_dict["language"] = preprocessing_utils.get_label(language, "language")
            y_dict["covid_tested"] = preprocessing_utils.get_label(covid_tested, "covid_tested")

            y_dict["copd"] = preprocessing_utils.get_label(med_history_list, "copd", "med_history")
            y_dict["longterm"] = preprocessing_utils.get_label(med_history_list, "longterm", "med_history")
            y_dict["lung"] = preprocessing_utils.get_label(med_history_list, "lung", "med_history")
            y_dict["pulmonary"] = preprocessing_utils.get_label(med_history_list, "pulmonary", "med_history")

            y_dict["drycough"] = preprocessing_utils.get_label(symptoms_list, "drycough", "symptoms")
            y_dict["wetcough"] = preprocessing_utils.get_label(symptoms_list, "wetcough", "symptoms")
            y_dict["sorethroat"] = preprocessing_utils.get_label(symptoms_list, "sorethroat", "symptoms")
            y_dict["shortbreath"] = preprocessing_utils.get_label(symptoms_list, "shortbreath", "symptoms")

            support_dict = dict()
            if has_cough:
                # support_dict["cough_wav2vec_embeddings_support"] = np.ones(
                #     (x_dict["cough_wav2vec_embeddings"].shape[0], 1),
                #     dtype=np.float32)
                # if cough_x_dict["unsupport"] > 0:
                #     support_dict["cough_wav2vec_embeddings_support"][-cough_x_dict["unsupport"]:, :] = 0.0
                # print(x_dict["cough_wav2vec_embeddings"].shape)
                support_dict["cough_logmel_spectrogram_support"] = np.ones((x_dict["cough_logmel_spectrogram"].shape[0], 1),
                                                                           dtype=np.float32)
                if cough_x_dict["unsupport"] > 0:
                    support_dict["cough_logmel_spectrogram_support"][-cough_x_dict["unsupport"]:, :] = 0.0
                print(x_dict["cough_logmel_spectrogram"].shape)
            if has_breath:
                # support_dict["breath_wav2vec_embeddings_support"] = np.ones(
                #     (x_dict["breath_wav2vec_embeddings"].shape[0], 1),
                #     dtype=np.float32)
                # if breath_x_dict["unsupport"] > 0:
                #     support_dict["breath_wav2vec_embeddings_support"][-breath_x_dict["unsupport"]:, :] = 0.0
                # print(x_dict["breath_wav2vec_embeddings"].shape)
                support_dict["breath_logmel_spectrogram_support"] = np.ones(
                    (x_dict["breath_logmel_spectrogram"].shape[0], 1),
                    dtype=np.float32)
                if breath_x_dict["unsupport"] > 0:
                    support_dict["breath_logmel_spectrogram_support"][-breath_x_dict["unsupport"]:, :] = 0.0
                print(x_dict["breath_logmel_spectrogram"].shape)
            if has_voice:
                # support_dict["voice_wav2vec_embeddings_support"] = np.ones(
                #     (x_dict["voice_wav2vec_embeddings"].shape[0], 1),
                #     dtype=np.float32)
                # if voice_x_dict["unsupport"] > 0:
                #     support_dict["voice_wav2vec_embeddings_support"][-voice_x_dict["unsupport"]:, :] = 0.0
                # print(x_dict["voice_wav2vec_embeddings"].shape)
                support_dict["voice_logmel_spectrogram_support"] = np.ones((x_dict["voice_logmel_spectrogram"].shape[0], 1),
                                                                           dtype=np.float32)
                if voice_x_dict["unsupport"] > 0:
                    support_dict["voice_logmel_spectrogram_support"][-voice_x_dict["unsupport"]:, :] = 0.0
                print(x_dict["voice_logmel_spectrogram"].shape)

            sample = Sample(name=name,
                            id_dict=id_dict,
                            partition=partition_eff,
                            x_dict=x_dict,
                            y_dict=y_dict,
                            support_dict=support_dict,
                            is_time_continuous=False,
                            custom_stats=None)

            yield sample
    print(statistics.mean(cough_size_list), statistics.stdev(cough_size_list))
    print(statistics.mean(breath_size_list), statistics.stdev(breath_size_list))
    print(statistics.mean(voice_size_list), statistics.stdev(voice_size_list))


#####################################################################################################################
# Naive dataset.
#####################################################################################################################
common_asthma_user_set = sorted(common_asthma_user_set)
common_non_asthma_user_set = sorted(common_non_asthma_user_set)
web_common_asthma_user_set = sorted(web_common_asthma_user_set)
web_common_non_asthma_user_set = sorted(web_common_non_asthma_user_set)

# random.seed(0)
# random.shuffle(common_asthma_user_set)
# random.shuffle(common_non_asthma_user_set)

user_to_partition = dict()
# Naive dataset partitioning.
# for user in common_asthma_user_set[0:1080]:
#     user_to_partition[user] = "train"
# for user in common_asthma_user_set[1080:1580]:
#     user_to_partition[user] = "devel"
# for user in common_asthma_user_set[1580:2080]:
#     user_to_partition[user] = "test"
# for user in common_non_asthma_user_set[0:2160]:
#     user_to_partition[user] = "train"
# for user in common_non_asthma_user_set[2160:3160]:
#     user_to_partition[user] = "devel"
# for user in common_non_asthma_user_set[3160:4160]:
#     user_to_partition[user] = "test"

# Fair dataset partitioning.
user_ids_sorted = np.array(user_ids_sorted)

for user in asthma_user_set["total"]:
    user_to_partition[user] = "pos_train"
for user in non_asthma_user_set["total"]:
    user_to_partition[user] = "neg_plus"

for user in user_ids_sorted[asthma_global_index_train]:
    user_to_partition[user] = "pos_train"
for user in user_ids_sorted[asthma_global_index_devel]:
    user_to_partition[user] = "pos_devel"
for user in user_ids_sorted[asthma_global_index_test]:
    user_to_partition[user] = "pos_test"
for user in user_ids_sorted[non_asthma_global_index_train]:
    user_to_partition[user] = "neg_train"
for user in user_ids_sorted[non_asthma_global_index_plus]:
    user_to_partition[user] = "neg_plus"
for user in user_ids_sorted[non_asthma_global_index_devel]:
    user_to_partition[user] = "neg_devel"
for user in user_ids_sorted[non_asthma_global_index_test]:
    user_to_partition[user] = "neg_test"

# Use web recordings. # Only one "user" for web, they get "pos" or "neg" tag depending on row metadata.
# for user in web_common_asthma_user_set[0:178]:
for user in web_common_asthma_user_set[:]:
    user_to_partition[user] = "web"
# for user in web_common_non_asthma_user_set[0:178]:
for user in web_common_non_asthma_user_set[:]:
    user_to_partition[user] = "web"

#####################################################################################################################
# Make tf records.
#####################################################################################################################
# Make all folders for tf records to be stored in.
# "pos_train" implies positive training sample instances with all 3 modalities present.
# "neg_devel_only_voice" implies negative development samples instances where only the voice modality is present.
# for partition in ["train", "train_only_voice", "train_only_cough", "train_only_breath",
#                   "train_only_voice_cough", "train_only_voice_breath", "train_only_cough_breath",
#                   "plus_only_voice", "plus_only_cough", "plus_only_breath",
#                   "plus_only_voice_cough", "plus_only_voice_breath", "plus_only_cough_breath",
#                   "devel", "test",
#                   "web", "web_only_voice", "web_only_cough", "web_only_breath",
#                   "web_only_voice_cough", "web_only_voice_breath", "web_only_cough_breath"]:
for partition in ["train", "plus", "devel", "test", "web"]:
    for label in ["pos", "neg"]:
        # This is where the positive and negative WAV clips, as well as the TF RECORDS will be stored.
        partition_eff = label + "_" + partition
        tfrecords_folder = configuration.TFRECORDS_FOLDER + '/' + partition_eff + '/'
        if not os.path.exists(tfrecords_folder):
            os.makedirs(tfrecords_folder)
        for modality_combo in ["only_voice", "only_cough", "only_breath", "only_voice_cough", "only_voice_breath",
                               "only_cough_breath", "only"]:
            # This is where the positive and negative WAV clips, as well as the TF RECORDS will be stored.
            partition_eff = label + "_" + partition + "_" + modality_combo
            tfrecords_folder = configuration.TFRECORDS_FOLDER + '/' + partition_eff + '/'
            if not os.path.exists(tfrecords_folder):
                os.makedirs(tfrecords_folder)

# Generator that reads the data samples one by one.
generator = get_generator(user_to_partition,
                          preprocessing_utils.AUDIO_FOLDER)
# Data normalisation generator.
normaliser = normalise.Normaliser(sample_iterable=generator,
                                  normalisation_scope="sample")
normalised_sample_generator = normaliser.generate_normalised_samples()

# Storing data as tf records.
tfrecord_creator = tfrecord_creator.TFRecordCreator(tf_records_folder=configuration.TFRECORDS_FOLDER,
                                                    sample_iterable=normalised_sample_generator,
                                                    are_test_labels_available=True,
                                                    is_continuous_time=False)
tfrecord_creator.create_tfrecords()
