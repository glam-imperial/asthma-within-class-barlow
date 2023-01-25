import collections
from os.path import exists

import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np
import torch

import tensorflow as tf

DATA_FOLDER = "/data/Downloads/COVIDSounds/NeurIPS2021-data"
AUDIO_FOLDER = DATA_FOLDER + "/covid19_data_0426/covid19_data_0426"

# Dictionary for cleanup of metadata types.
TYPES_STATS = collections.OrderedDict()
TYPES_STATS["age"] = collections.OrderedDict()
TYPES_STATS["sex"] = collections.OrderedDict()
TYPES_STATS["med_history"] = collections.OrderedDict()
TYPES_STATS["smoking"] = collections.OrderedDict()
TYPES_STATS["language"] = collections.OrderedDict()
TYPES_STATS["symptoms"] = collections.OrderedDict()
TYPES_STATS["covid_tested"] = collections.OrderedDict()
TYPES_STATS["hospitalised"] = collections.OrderedDict()
TYPES_STATS["recording_source"] = collections.OrderedDict()
# {nan, 'pnts', '60-69', '90-', '70-79', '40-49', '0-19', '30-39', '50-59', 'Prefer not to say', '20-29', 'Unter 20', '16-19', '80-89'}
TYPES_STATS["age"]["0-19"] = "0-19"
TYPES_STATS["age"]["16-19"] = "0-19"
TYPES_STATS["age"]["Unter 20"] = "0-19"
TYPES_STATS["age"]["20-29"] = "20-29"
TYPES_STATS["age"]["30-39"] = "30-39"
TYPES_STATS["age"]["40-49"] = "40-49"
TYPES_STATS["age"]["50-59"] = "50-59"
TYPES_STATS["age"]["60-69"] = "60-69"
TYPES_STATS["age"]["70-79"] = "70-79"
TYPES_STATS["age"]["80-89"] = "80-89"
TYPES_STATS["age"]["90-"] = "90-"
TYPES_STATS["age"]["pnts"] = "pnts_age"
TYPES_STATS["age"]["Prefer not to say"] = "pnts_age"
# TYPES_STATS["age"][np.nan] = None
TYPES_STATS["age"][np.nan] = "pnts_age"
# {'Female', nan, 'Other', 'pnts', 'Male'}
TYPES_STATS["sex"]["Female"] = "Female"
TYPES_STATS["sex"]["Male"] = "Male"
TYPES_STATS["sex"]["Other"] = "Other"
TYPES_STATS["sex"]["pnts"] = "pnts_sex"
# TYPES_STATS["sex"][np.nan] = None
TYPES_STATS["sex"][np.nan] = "pnts_sex"
# {'None', '', 'pnts', 'longterm', 'hbp', 'organ', 'asthma', 'cystic', 'angina', 'stroke', 'pulmonary', 'otherHeart', 'diabetes', 'hiv', 'valvular', 'lung', 'heart', 'copd', 'cancer', 'long'}
TYPES_STATS["med_history"]["angina"] = "angina"
TYPES_STATS["med_history"]["asthma"] = "asthma"
TYPES_STATS["med_history"]["cancer"] = "cancer"
TYPES_STATS["med_history"]["copd"] = "copd"
TYPES_STATS["med_history"]["cystic"] = "cystic"
TYPES_STATS["med_history"]["diabetes"] = "diabetes"
TYPES_STATS["med_history"]["hbp"] = "hbp"
TYPES_STATS["med_history"]["heart"] = "heart"
TYPES_STATS["med_history"]["hiv"] = "hiv"
TYPES_STATS["med_history"]["longterm"] = "longterm"
TYPES_STATS["med_history"]["long"] = "longterm"
TYPES_STATS["med_history"]["lung"] = "lung"
TYPES_STATS["med_history"]["otherHeart"] = "otherHeart"
TYPES_STATS["med_history"]["organ"] = "organ"
TYPES_STATS["med_history"]["pulmonary"] = "pulmonary"
TYPES_STATS["med_history"]["stroke"] = "stroke"
TYPES_STATS["med_history"]["valvular"] = "valvular"
TYPES_STATS["med_history"]["None"] = "none_med_history"
TYPES_STATS["med_history"][""] = "none_med_history"
TYPES_STATS["med_history"]["pnts"] = "pnts_med_history"
# {'ex', nan, 'pnts', 'never', 'ecig', '21+', '11to20', '1to10', 'ltOnce'}
TYPES_STATS["smoking"]["never"] = "never"
TYPES_STATS["smoking"]["ex"] = "ex"
TYPES_STATS["smoking"]["ecig"] = "ecig"
TYPES_STATS["smoking"]["ltOnce"] = "ltOnce"
TYPES_STATS["smoking"]["1to10"] = "1to10"
TYPES_STATS["smoking"]["11to20"] = "11to20"
TYPES_STATS["smoking"]["21+"] = "21+"
TYPES_STATS["smoking"]["pnts"] = "pnts_smoking"
# TYPES_STATS["smoking"][np.nan] = None
TYPES_STATS["smoking"][np.nan] = "pnts_smoking"
# {'None', 'pt', 'hi', 'it', 'es', 'fr', 'de', 'en', 'ro', 'ru', 'zh', 'el'}
TYPES_STATS["language"]["pt"] = "pt"
TYPES_STATS["language"]["hi"] = "hi"
TYPES_STATS["language"]["it"] = "it"
TYPES_STATS["language"]["es"] = "es"
TYPES_STATS["language"]["fr"] = "fr"
TYPES_STATS["language"]["de"] = "de"
TYPES_STATS["language"]["en"] = "en"
TYPES_STATS["language"]["ro"] = "ro"
TYPES_STATS["language"]["ru"] = "ru"
TYPES_STATS["language"]["zh"] = "zh"
TYPES_STATS["language"]["el"] = "el"
TYPES_STATS["language"]["None"] = "none_language"
# {'None', 'smelltasteloss', '', 'pnts', 'never', 'muscleache', 'drycough', 'runnyblockednose', 'runny', 'dizziness', 'shortbreath', 'wetcough', 'tightness', 'chills', 'tighness', 'sorethroat', 'fever', 'headache'}
TYPES_STATS["symptoms"]["drycough"] = "drycough"
TYPES_STATS["symptoms"]["wetcough"] = "wetcough"
TYPES_STATS["symptoms"]["sorethroat"] = "sorethroat"
# TYPES_STATS["symptoms"]["runnyblockednose"] = "runnyblockednose"
TYPES_STATS["symptoms"]["runnyblockednose"] = "sorethroat"
# TYPES_STATS["symptoms"]["runny"] = "runnyblockednose"
TYPES_STATS["symptoms"]["runny"] = "sorethroat"
TYPES_STATS["symptoms"]["tightness"] = "tightness"
TYPES_STATS["symptoms"]["tighness"] = "tightness"
TYPES_STATS["symptoms"]["smelltasteloss"] = "smelltasteloss"
TYPES_STATS["symptoms"]["fever"] = "fever"
TYPES_STATS["symptoms"]["chills"] = "chills"
TYPES_STATS["symptoms"]["shortbreath"] = "shortbreath"
TYPES_STATS["symptoms"]["dizziness"] = "dizziness"
TYPES_STATS["symptoms"]["headache"] = "headache"
TYPES_STATS["symptoms"]["muscleache"] = "muscleache"
TYPES_STATS["symptoms"]["None"] = "none_symptoms"
TYPES_STATS["symptoms"][""] = "none_symptoms"
# TYPES_STATS["symptoms"]["never"] = "never"
TYPES_STATS["symptoms"]["never"] = "none_symptoms"
TYPES_STATS["symptoms"]["pnts"] = "pnts_symptoms"
# {nan, 'pnts', 'positiveOver14', 'never', 'neverThinkHadCOVIDLast14', 'neverThinkHadCOVIDNow', 'last14', 'negativeLast14', 'neverThinkHadCOVIDOver14', 'positiveLast14', 'negativeNever', 'no', 'yes', 'neverThinkHadCOVIDNever', 'negativeOver14', 'LocalizedStringKey(key: "Prefer not to say", hasFormatting: false, arguments: [])', 'over14'}
# TYPES_STATS["covid_tested"]["positiveLast14"] = "positiveLast14"
# TYPES_STATS["covid_tested"]["last14"] = "positiveLast14"
# TYPES_STATS["covid_tested"]["positiveOver14"] = "positiveOver14"
# TYPES_STATS["covid_tested"]["over14"] = "positiveOver14"
# TYPES_STATS["covid_tested"]["yes"] = "yes"
# TYPES_STATS["covid_tested"]["negativeLast14"] = "negativeLast14"
# TYPES_STATS["covid_tested"]["negativeOver14"] = "negativeOver14"
# TYPES_STATS["covid_tested"]["negativeNever"] = "negativeNever"
# TYPES_STATS["covid_tested"]["never"] = "never"
# TYPES_STATS["covid_tested"]["no"] = "never"
# TYPES_STATS["covid_tested"]["neverThinkHadCOVIDNever"] = "never"
# TYPES_STATS["covid_tested"]["neverThinkHadCOVIDNow"] = "never"
# TYPES_STATS["covid_tested"]["neverThinkHadCOVIDLast14"] = "never"
# TYPES_STATS["covid_tested"]["neverThinkHadCOVIDOver14"] = "never"
# TYPES_STATS["covid_tested"]["pnts"] = "pnts_covid_tested"
# TYPES_STATS["covid_tested"]['LocalizedStringKey(key: "Prefer not to say", hasFormatting: false, arguments: [])'] = "pnts_covid_tested"
# # TYPES_STATS["covid_tested"][np.nan] = None
# TYPES_STATS["covid_tested"][np.nan] = "pnts_covid_tested"
TYPES_STATS["covid_tested"]["positiveLast14"] = "positiveLast14"
TYPES_STATS["covid_tested"]["last14"] = "positiveLast14"
TYPES_STATS["covid_tested"]["positiveOver14"] = "positiveOver14"
TYPES_STATS["covid_tested"]["over14"] = "positiveOver14"
TYPES_STATS["covid_tested"]["yes"] = "yes"
TYPES_STATS["covid_tested"]["negativeLast14"] = "negativeLast14"
TYPES_STATS["covid_tested"]["negativeOver14"] = "negativeOver14"
TYPES_STATS["covid_tested"]["negativeNever"] = "negativeNever"
TYPES_STATS["covid_tested"]["never"] = "never"
TYPES_STATS["covid_tested"]["no"] = "never"
TYPES_STATS["covid_tested"]["neverThinkHadCOVIDNever"] = "neverThinkHadCOVIDNever"
TYPES_STATS["covid_tested"]["neverThinkHadCOVIDNow"] = "neverThinkHadCOVIDNow"
TYPES_STATS["covid_tested"]["neverThinkHadCOVIDLast14"] = "neverThinkHadCOVIDLast14"
TYPES_STATS["covid_tested"]["neverThinkHadCOVIDOver14"] = "neverThinkHadCOVIDOver14"
TYPES_STATS["covid_tested"]["pnts"] = "pnts_covid_tested"
TYPES_STATS["covid_tested"]['LocalizedStringKey(key: "Prefer not to say", hasFormatting: false, arguments: [])'] = "pnts_covid_tested"
# TYPES_STATS["covid_tested"][np.nan] = None
TYPES_STATS["covid_tested"][np.nan] = "pnts_covid_tested"
# {nan, 'pnts', 'no', 'yes'}
TYPES_STATS["hospitalised"]["no"] = "no"
TYPES_STATS["hospitalised"]["yes"] = "yes"
TYPES_STATS["hospitalised"]["pnts"] = "pnts_hospitalised"
# TYPES_STATS["hospitalised"][np.nan] = None
TYPES_STATS["hospitalised"][np.nan] = "pnts_hospitalised"
TYPES_STATS["recording_source"]["android"] = "android"
TYPES_STATS["recording_source"]["ios"] = "ios"
TYPES_STATS["recording_source"]["web"] = "web"


# TYPES = ["age",
#          "sex",
#          "angina",
#          # "asthma",
#          "cancer",
#          "copd",
#          "cystic",
#          "diabetes",
#          "hbp",
#          "heart",
#          "hiv",
#          "longterm",
#          "lung",
#          "otherHeart",
#          "organ",
#          "pulmonary",
#          "stroke",
#          "valvular",
#          "none_med_history",
#          "smoking",
#          "language",
#          "hospitalised",
#          "android",
#          "ios",
#          "web",
#          "drycough",
#          "sorethroat",
#          "tightness",
#          "smelltasteloss",
#          "fever",
#          "chills",
#          "shortbreath",
#          "dizziness",
#          "headache",
#          "muscleache",
#          "symptoms_none",
#          "positiveLast14",
#          "positiveOver14",
#          "yes",
#          "negativeLast14",
#          "negativeOver14",
#          "negativeNever",
#          "never"]


def get_label(metadata, label_name, modality_type=None):
    if modality_type is None:
        # number_of_classes = len(TYPES_STATS[label_name])
        number_of_classes = len(set(list(TYPES_STATS[label_name].values())))
        print(label_name, number_of_classes)
        label_value = np.zeros((number_of_classes,), dtype=np.float32)
        metadata_clean = TYPES_STATS[label_name][metadata]
        for c, label in enumerate(sorted(set(list(TYPES_STATS[label_name].values())))):
            if label == metadata_clean:
                label_value[c] = 1.0
                break
    else:
        label_value = np.zeros((1,), dtype=np.float32)
        flag = False
        for m in metadata:
            if m in TYPES_STATS[modality_type].keys():
                if TYPES_STATS[modality_type][m] == label_name:
                    flag = True
                    break
        if flag:
            label_value[0] = np.float32(1.0)
        else:
            label_value[0] = np.float32(0.0)

    return label_value


def modality_exists(row, modality, recording_source):
    # Passes check.
    if modality == "voice":
        passes_check = row["Voice check"] == "v"
    elif modality == "cough":
        passes_check = row["Cough check"] == "c"
    elif modality == "breath":
        passes_check = row["Breath check"] == "b"
    else:
        raise ValueError("Invalid modality choice.")

    # File exists.
    voice_filename = row["Voice filename"]
    cough_filename = row["Cough filename"]
    breath_filename = row["Breath filename"]
    uid = row["Uid"]
    folder_name = row["Folder Name"]
    if recording_source == "web":
        subfolder_name = AUDIO_FOLDER + "/" + "form-app-users" + "/" + folder_name
        cough_filename = cough_filename[:-4] + "wav"
        breath_filename = breath_filename[:-4] + "wav"
        voice_filename = voice_filename[:-4] + "wav"
    else:
        subfolder_name = AUDIO_FOLDER + "/" + uid + "/" + folder_name

    if modality == "voice":
        filename = voice_filename
    elif modality == "cough":
        filename = cough_filename
    elif modality == "breath":
        filename = breath_filename
    else:
        raise ValueError("Invalid modality choice.")

    filepath = subfolder_name + "/" + filename
    file_exists = exists(filepath)
    if not file_exists:
        filepath = subfolder_name + "/" + filename[:-4] + ".wav"
        file_exists = exists(filepath)

    return passes_check and file_exists


def identify_users_with_clean_recordings(row, user_counts_dict, users_dict, recording_source):
    voice_exists = modality_exists(row,
                                   "voice",
                                   recording_source)
    cough_exists = modality_exists(row,
                                   "cough",
                                   recording_source)
    breath_exists = modality_exists(row,
                                    "breath",
                                    recording_source)

    has_valid_recordings = False
    if voice_exists and cough_exists and breath_exists:
        user_counts_dict["all_three"] += 1
        users_dict["all_three"].append(row["Uid"])
        has_valid_recordings = True
    if voice_exists:
        user_counts_dict["voice"] += 1
        users_dict["voice"].append(row["Uid"])
        has_valid_recordings = True
    if cough_exists:
        user_counts_dict["cough"] += 1
        users_dict["cough"].append(row["Uid"])
        has_valid_recordings = True
    if breath_exists:
        user_counts_dict["breath"] += 1
        users_dict["breath"].append(row["Uid"])
        has_valid_recordings = True
    if has_valid_recordings:
        user_counts_dict["total"] += 1
        users_dict["total"].append(row["Uid"])

    return user_counts_dict, users_dict


def check_consistency(m_list, m_name):
    count_dict = collections.defaultdict(float)
    for m in m_list:
        v = m[m_name]
        v_clean = TYPES_STATS[m_name][v]
        count_dict[v_clean] += 1.0
    distribution = np.empty((1, len(count_dict.keys())), dtype=np.float32)
    for kk, k in enumerate(sorted(count_dict.keys())):
        distribution[0, kk] = count_dict[k]

    max_v = np.max(distribution)

    consistency = max_v / np.sum(distribution)

    kk_max = np.argmax(distribution[0])

    filtered_v_clean = sorted(count_dict.keys())[kk_max]

    for kk, k in enumerate(sorted(count_dict.keys())):
        count_dict[k] = count_dict[k] / max_v

    return consistency, filtered_v_clean, count_dict


def check_consistency_symptom(m_list, m_name, m_symptom):
    count_dict = collections.defaultdict(float)
    for m in m_list:
        v = m[m_name]
        v_clean = [TYPES_STATS[m_name][vv] for vv in v]
        if m_symptom in v_clean:
            count_dict[1] += 1.0
        else:
            count_dict[0] += 1.0
    distribution = np.empty((1, len(count_dict.keys())), dtype=np.float32)
    for kk, k in enumerate(sorted(count_dict.keys())):
        distribution[0, kk] = count_dict[k]

    max_v = np.max(distribution)

    consistency = max_v / np.sum(distribution)

    kk_max = np.argmax(distribution[0])

    filtered_v_clean = sorted(count_dict.keys())[kk_max]

    for kk, k in enumerate(sorted(count_dict.keys())):
        count_dict[k] = count_dict[k] / max_v

    return consistency, filtered_v_clean, count_dict


def check_consistency_list(m_list, m_name):
    consistency_list = list()
    for m_name_symptom in sorted(TYPES_STATS[m_name].values()):
        consistency, filtered_v_clean, count_dict = check_consistency_symptom(m_list, m_name, m_name_symptom)
        consistency_list.append((m_name_symptom, consistency, filtered_v_clean, count_dict))

    return consistency_list


def get_wav2vec2_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h", cache_dir="/data/huggingface")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h", cache_dir="/data/huggingface")

    return processor, model


def get_wav2vec_embeddings(waveform, wav2vec2_model):
    # waveform_norm = (waveform - waveform.mean()) / (waveform.std() + 1e-5)
    #
    # if waveform_norm.size > 246000:
    #     waveform_norm = waveform_norm[:246000]
    # elif waveform_norm.size < 246000:
    #     waveform_norm = np.hstack([waveform_norm, np.zeros((246000 - waveform_norm.size, ), dtype=waveform_norm.dtype)])
    #
    # wav2vec2_model = get_wav2vec2_model()
    # wav2vec2_features = wav2vec2_model(np.vstack([waveform_norm.reshape((1, -1)), waveform_norm.reshape((1, -1))]))
    # print(wav2vec2_features.shape)
    # wav2vec2_features = wav2vec2_features.reshape((768, 32))

    # wav2vec2_processor, wav2vec2_model = get_wav2vec2_model()
    wav2vec2_processor, wav2vec2_model = wav2vec2_model
    input_values = wav2vec2_processor(waveform, return_tensors = "pt", padding = "longest", sampling_rate=16000)  # Batch size 1

    with torch.no_grad():
        outputs = wav2vec2_model(**input_values)
    last_hidden_states = outputs.last_hidden_state.cpu().detach().numpy()
    print("wav2vec2 emb: ", last_hidden_states)
    print("wav2vec2 emb: ", last_hidden_states.shape)
    last_hidden_states = last_hidden_states.reshape((-1, 1024))

    return last_hidden_states


def get_features_and_stats(waveform, wav2vec2_model):

    # waveform_norm = waveform * (0.7079 / waveform.max())
    # maxv = float(np.iinfo(np.int16).max)
    # waveform_norm = (waveform_norm * maxv).astype(np.float32)

    wav2vec_embeddings = get_wav2vec_embeddings(waveform, wav2vec2_model)

    # print(waveform.dtype, waveform.max())
    # waveform_norm = (waveform / np.iinfo(np.int16).max).astype(np.float32)
    # waveform_norm = waveform * (0.7079 / waveform.max())
    # maxv = np.iinfo(np.int16).max
    # waveform_norm = (waveform_norm * maxv).astype(np.float32)

    # waveform_std = waveform.std()
    # if waveform_std == 0.0:
    #     waveform_std = 1.0
    #
    # waveform_norm = (waveform - waveform.mean()) / waveform_std
    waveform_norm = waveform

    # spectrogram = np.abs(librosa.stft(waveform_norm, n_fft=400, hop_length=10 * 16)) ** 1.0
    # logmel_spectrogram = librosa.power_to_db(
    #     librosa.feature.melspectrogram(y=waveform_norm,
    #                                    sr=16000,
    #                                    S=spectrogram,
    #                                    n_mels=128))#,
    #                                    # fmin=125,
    #                                    # fmax=7500))
    # mfcc = librosa.feature.mfcc(waveform_norm,
    #                             sr=16000,
    #                             n_mfcc=80,
    #                             S=logmel_spectrogram)

    # spectrogram = spectrogram.transpose()[:-1, :]
    # logmel_spectrogram = logmel_spectrogram.transpose()[:-1, :]
    # mfcc = mfcc.transpose()[:-1, :]

    # print(logmel_spectrogram)
    # print(logmel_spectrogram.shape)

    # if logmel_spectrogram.shape[0] % 96 != 0:
    #     unsupport = int(96 - logmel_spectrogram.shape[0] % 96)
    #     logmel_spectrogram = np.vstack([logmel_spectrogram,
    #                                     np.zeros((96 - logmel_spectrogram.shape[0] % 96, 128), dtype=np.float32)])
    # else:
    #     unsupport = 0
    unsupport = 0

    x_dict = dict()
    # x_dict["waveform"] = waveform_norm
    # x_dict["logmel_spectrogram"] = logmel_spectrogram
    # x_dict["mfcc"] = mfcc
    x_dict["wav2vec_embeddings"] = wav2vec_embeddings

    x_dict["unsupport"] = unsupport

    return x_dict

# has_valid_recordings = False
#             if (row["Voice check"] == "v") and (row["Cough check"] == "c") and (row["Breath check"] == "b"):
#                 asthma_user_counts["all_three"] += 1
#                 asthma_users["all_three"].append(row["Uid"])
#                 has_valid_recordings = True
#             if row["Voice check"] == "v":
#                 asthma_user_counts["voice"] += 1
#                 asthma_users["voice"].append(row["Uid"])
#                 has_valid_recordings = True
#             if row["Cough check"] == "c":
#                 asthma_user_counts["cough"] += 1
#                 asthma_users["cough"].append(row["Uid"])
#                 has_valid_recordings = True
#             if row["Breath check"] == "b":
#                 asthma_user_counts["breath"] += 1
#                 asthma_users["breath"].append(row["Uid"])
#                 has_valid_recordings = True
#             if has_valid_recordings:
#                 asthma_user_counts["total"] += 1
#                 asthma_users["total"].append(row["Uid"])
