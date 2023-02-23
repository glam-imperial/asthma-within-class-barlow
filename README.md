# asthma-within-class-barlow

Code to replicate results for "POSITIVE-PAIR REDUNDANCY REDUCTION REGULARISATION FOR SPEECH-BASED ASTHMA DIAGNOSIS PREDICTION" from ICASSP 2023

In this paper, we predict whether someone has been diagnosed with asthma, using audio-based (verbal and non-verbal) voice recordings.

A regularising auxiliary task is used in the loss value calculation that is based on reducing redundancy between same-class (asthma or non-asthma) audio representations, as in self-supervised domain adaptation techniques, and we use the Barlow Twins method [1].

The dataset used in this paper that contains asthma diagnosis labels was introduced in [2]. Our core model, which also serves as a baseline is a finetuned on this dataset, pre-trained VGGish model [3]. The pre-training weights, and implementation of the core model can be found in [4].

- [1] https://arxiv.org/abs/2103.03230
- [2] https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/e2c0be24560d78c5e599c2a9c9d0bbd2-Abstract-round2.html
- [3] https://research.google/pubs/pub45611/
- [4] https://pypi.org/project/vggish-keras/

(Partially uploaded -- cleanup and complete uploading imminent)

## Preprocessing the dataset.

Access to the data is given by the authors of [2].

- Open covid19_sounds/neurips21/configuration.py, and edit the PROJECT_FOLDER and DATA_FOLDER accordingly.
- Run covid19_sounds/neurips21/make_tf_records.py. This will read the data, perform fair partitioning as described in our paper, and store the data as tf_records, for faster training.

## Running experiments.

Get the pre-trained weights of the VGGish following [4].

Should be at common/models/vggish_keras/model/audioset_weights.h5 (https://drive.google.com/file/d/1mhqXZ8CANgHyepum7N4yrjiyIg6qaMe6/view)

- Replicate experiments by running covid19_sounds/neurips21/train.py. Include the methods you want by commenting/uncommenting accordingly.
- The configurations of each of these experiments can be found in YAML files in the covid19_sounds/neurips21/experiment_configurations folder.
- Once the selected experiments are finished, you can read a summarised version, including standard deviations by running covid19_sounds/neurips21/summarise_results_v2.py
