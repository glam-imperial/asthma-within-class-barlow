import math

import numpy as np
import numpy.random as random

from common.batch_generator import BatchGenerator


class HeterogeneousBatchGenerator:
    def __init__(self,
                 tf_records,
                 is_training,
                 partition,
                 are_test_labels_available,
                 name_to_metadata,
                 input_type_list,
                 output_type_list,
                 batch_size,
                 buffer_size,
                 path_list_dict,
                 use_autopad=False,
                 augmentation_configuration=None):
        self.tf_records = tf_records
        self.is_training = is_training
        self.partition = partition
        self.are_test_labels_available = are_test_labels_available
        self.name_to_metadata = name_to_metadata
        self.input_type_list = input_type_list
        self.output_type_list = output_type_list
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.path_list_dict = path_list_dict
        self.use_autopad = use_autopad
        self.augmentation_configuration = augmentation_configuration

        self.batch_generators = dict()

        print(self.path_list_dict.keys())

        use_modality_partition = dict()
        use_modality_partition["voice"] = False
        use_modality_partition["breath"] = False
        use_modality_partition["cough"] = False
        for input_type in self.input_type_list:
            if "voice" in input_type:
                use_modality_partition["voice"] = True
            if "breath" in input_type:
                use_modality_partition["breath"] = True
            if "cough" in input_type:
                use_modality_partition["cough"] = True

        self.sizes = dict()
        self.steps_per_epoch = dict()
        for modality_combination, path_list in self.path_list_dict.items():
            name_to_metadata_eff = dict()
            for name, metadata in name_to_metadata.items():
                if "logmel_spectrogram" not in name:
                # if "wav2vec_embeddings" not in name:
                    name_to_metadata_eff[name] = metadata
                else:
                    if ("voice" in name) and (use_modality_partition["voice"]) and ("voice" in modality_combination):
                        name_to_metadata_eff[name] = metadata
                    if ("breath" in name) and (use_modality_partition["breath"]) and ("breath" in modality_combination):
                        name_to_metadata_eff[name] = metadata
                    if ("cough" in name) and (use_modality_partition["cough"]) and ("cough" in modality_combination):
                        name_to_metadata_eff[name] = metadata

            num_available_modalities = 0

            input_type_list_eff = list()
            modality_combination_eff_2 = ""
            for input_type in self.input_type_list:
                if ("voice" in input_type) and (use_modality_partition["voice"]) and ("voice" in modality_combination):
                    input_type_list_eff.append(input_type)
                    if "support" not in input_type:
                        num_available_modalities += 1
                        modality_combination_eff_2 += "_voice"
                if ("breath" in input_type) and (use_modality_partition["breath"]) and ("breath" in modality_combination):
                    input_type_list_eff.append(input_type)
                    if "support" not in input_type:
                        num_available_modalities += 1
                        modality_combination_eff_2 += "_breath"
                if ("cough" in input_type) and (use_modality_partition["cough"]) and ("cough" in modality_combination):
                    input_type_list_eff.append(input_type)
                    if "support" not in input_type:
                        num_available_modalities += 1
                        modality_combination_eff_2 += "_cough"

            print(modality_combination)
            print(input_type_list_eff)
            print(num_available_modalities)

            if num_available_modalities == 1:
                modality_combination_eff = "single"
            elif num_available_modalities == 2:
                modality_combination_eff = "double"
            elif num_available_modalities == 3:
                modality_combination_eff = "triple"
            else:
                raise ValueError

            if "pos" in modality_combination:
                asthma_str = "_pos"
            elif "neg" in modality_combination:
                asthma_str = "_neg"
            else:
                asthma_str = ""

            # modality_combination_eff = modality_combination_eff + modality_combination[6:]
            modality_combination_eff = modality_combination_eff + modality_combination_eff_2 + asthma_str
            counter = 0
            if modality_combination_eff + repr(counter) not in self.batch_generators.keys():
                modality_combination_eff = modality_combination_eff + repr(counter)
            else:
                counter = 1
                while True:
                    if modality_combination_eff + repr(counter) in self.batch_generators.keys():
                        counter += 1
                    else:
                        modality_combination_eff = modality_combination_eff + repr(counter)
                        break

            # dataset, \
            # iterator, \
            # next_element, \
            # init_op
            self.batch_generators[modality_combination_eff] =\
                BatchGenerator(tf_records_folder=self.tf_records,
                               is_training=self.is_training,
                               partition=self.partition,
                               are_test_labels_available=self.are_test_labels_available,
                               name_to_metadata=name_to_metadata_eff,
                               input_type_list=input_type_list_eff,
                               output_type_list=self.output_type_list,
                               batch_size=self.batch_size,
                               buffer_size=15 * self.batch_size,
                               path_list=path_list,
                               augmentation_configuration=augmentation_configuration).get_tf_dataset()
            self.sizes[modality_combination_eff] = len(path_list)
            self.steps_per_epoch[modality_combination_eff] = math.ceil(len(path_list) / batch_size)

    def get_tf_dataset(self):
        print(self.sizes)
        print(self.steps_per_epoch)
        return self.batch_generators

    def heterogeneous_generation(self, sess, shuffle):
        num_modality_combinations = len(self.sizes.keys())
        modality_combinations = sorted(self.steps_per_epoch.keys())

        init_op_list = [self.batch_generators[k][3] for k in modality_combinations]
        next_element_list = [self.batch_generators[k][2] for k in modality_combinations]

        for init_op in init_op_list:
            sess.run(init_op)

        counts = np.zeros((num_modality_combinations,), dtype=np.float32)
        total_steps_per_epoch = 0
        for t, modality_combination in enumerate(modality_combinations):
            counts[t] = self.steps_per_epoch[modality_combination]
            total_steps_per_epoch += self.steps_per_epoch[modality_combination]
        c = 0
        for step in range(total_steps_per_epoch):
            probabilities = counts / counts.sum()
            if shuffle:
                while True:
                    c = random.choice(num_modality_combinations, p=probabilities)
                    if counts[c] > 0:
                        break
                counts[c] -= 1
            else:
                if counts[c] > 0:
                    pass
                else:
                    c += 1
                counts[c] -= 1

            modality_combination = modality_combinations[c]

            # TODO: hom vs not hom
            # yield modality_combination[:6], sess.run(next_element_list[c])
            yield modality_combination[:-5], sess.run(next_element_list[c])
            # yield modality_combination[:-1], sess.run(next_element_list[c])
