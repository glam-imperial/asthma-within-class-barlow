import numpy as np
import scipy.stats as spst
from skmultilearn.model_selection import IterativeStratification


# Stratified partitioning of a metadata array.
def fair_split(data_array,  # Metadata array. Similar to an one-hot multilabel label array.
               global_index,  # The metadata array refers to the core set with all 3 modalities. The global index maps the core set to the index of all users.
               sample_distribution_per_fold):  # Split percentages.
    skf = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=sample_distribution_per_fold)
    for index_0, index_1 in skf.split(data_array, data_array):
        data_array_0, data_array_1 = data_array[index_0], data_array[index_1]
        global_index_0, global_index_1 = global_index[index_0], global_index[index_1]
    return data_array_0, data_array_1, global_index_0, global_index_1


def partitioning(metadata_df):
    asthma_index = list(metadata_df.columns).index("asthma")
    # print(asthma_index)

    copd_index = list(metadata_df.columns).index("copd")
    # print(copd_index)

    metadata_array = metadata_df.values
    print(metadata_array[:, asthma_index].sum(), metadata_array.shape[0] - metadata_array[:, asthma_index].sum())

    global_index = np.arange(metadata_array.shape[0], dtype=np.int32)
    asthma_global_index = global_index[metadata_array[:, asthma_index] == 1.0]
    non_asthma_global_index = global_index[metadata_array[:, asthma_index] == 0.0]

    asthma_metadata_array = metadata_array[metadata_array[:, asthma_index] == 1.0, :]
    non_asthma_metadata_array = metadata_array[metadata_array[:, asthma_index] == 0.0, :]
    print(asthma_metadata_array.shape[0])
    print(non_asthma_metadata_array.shape[0])

    print(asthma_metadata_array[:, copd_index].mean())
    print(non_asthma_metadata_array[:, copd_index].mean())
    # 0.029326923
    # 0.005541467

    # Asthmatic.
    asthma_y_train_devel,\
    asthma_y_test,\
    asthma_global_index_train_devel,\
    asthma_global_index_test = fair_split(asthma_metadata_array,
                                          asthma_global_index,
                                          [0.75, 0.25])

    # print(asthma_y_test)
    print(asthma_y_test.shape)
    # print(asthma_y_test.mean(axis=0))

    asthma_y_train, \
    asthma_y_devel, \
    asthma_global_index_train, \
    asthma_global_index_devel = fair_split(asthma_y_train_devel,
                                           asthma_global_index_train_devel,
                                           [0.67, 0.33])

    # print(asthma_y_train)
    # print(asthma_y_devel)
    # print(asthma_y_test)
    print(asthma_y_train.shape)
    print(asthma_y_devel.shape)
    print(asthma_y_test.shape)
    # print(asthma_y_train.mean(axis=0))
    # print(asthma_y_devel.mean(axis=0))
    # print(asthma_y_test.mean(axis=0))

    print(spst.pearsonr(asthma_y_train.mean(axis=0), asthma_y_devel.mean(axis=0)))
    print(spst.pearsonr(asthma_y_test.mean(axis=0), asthma_y_devel.mean(axis=0)))
    print(spst.pearsonr(asthma_y_test.mean(axis=0), asthma_y_train.mean(axis=0)))

    # Non asthmatic.
    non_asthma_y_devel_test_plus, \
    non_asthma_y_train, \
    non_asthma_global_index_devel_test_plus, \
    non_asthma_global_index_train = fair_split(non_asthma_metadata_array,
                                               non_asthma_global_index,
                                               [0.9, 0.1])

    # print(non_asthma_y_train)
    print(non_asthma_y_train.shape)
    # print(non_asthma_y_train.mean(axis=0))

    non_asthma_y_plus, \
    non_asthma_y_devel_test, \
    non_asthma_global_index_plus, \
    non_asthma_global_index_devel_test = fair_split(non_asthma_y_devel_test_plus,
                                                    non_asthma_global_index_devel_test_plus,
                                                    [0.89, 0.11])

    non_asthma_y_devel, \
    non_asthma_y_test, \
    non_asthma_global_index_devel, \
    non_asthma_global_index_test = fair_split(non_asthma_y_devel_test,
                                              non_asthma_global_index_devel_test,
                                              [0.5, 0.5])

    # print(non_asthma_y_train)
    # print(non_asthma_y_devel)
    # print(non_asthma_y_test)
    print(non_asthma_y_train.shape)
    print(non_asthma_y_plus.shape)
    print(non_asthma_y_devel.shape)
    print(non_asthma_y_test.shape)
    # print(non_asthma_y_train.mean(axis=0))
    # print(non_asthma_y_plus.mean(axis=0))
    # print(non_asthma_y_devel.mean(axis=0))
    # print(non_asthma_y_test.mean(axis=0))

    print(spst.pearsonr(non_asthma_y_train.mean(axis=0), non_asthma_y_devel.mean(axis=0)))
    print(spst.pearsonr(non_asthma_y_train.mean(axis=0), non_asthma_y_plus.mean(axis=0)))
    print(spst.pearsonr(non_asthma_y_test.mean(axis=0), non_asthma_y_devel.mean(axis=0)))
    print(spst.pearsonr(non_asthma_y_test.mean(axis=0), non_asthma_y_train.mean(axis=0)))
    # Order 2.
    # (0.9998976929267321, 8.462437013672502e-144)
    # (0.9999410646606713, 5.080887849777093e-153)
    # (0.9999258278752994, 3.553190822160839e-149)

    # Order 1.
    # (0.9999462489654338, 1.4669177730218683e-154)
    # (0.9999590491705374, 4.156507419621779e-159)
    # (0.9999197163054795, 7.489542294140859e-148)

    # print(asthma_global_index_train)
    # print(asthma_global_index_devel)
    # print(asthma_global_index_test)
    # print(non_asthma_global_index_train)
    # print(non_asthma_global_index_devel)
    # print(non_asthma_global_index_test)

    return asthma_global_index_train, asthma_global_index_devel, asthma_global_index_test,\
           non_asthma_global_index_train, non_asthma_global_index_plus, non_asthma_global_index_devel, non_asthma_global_index_test
