import tensorflow as tf

from common.models.audio_core_blocks import SEResNet, \
    SEResNetish
from common.models.vggish_keras import vggish as vgk
from common.models.top_blocks import FeedForwardBlock
from covid19_sounds.neurips21.model import model_call


def get_model(name_to_metadata,
              model_configuration):
    core_model = model_configuration["core_model"]
    core_model_configuration = model_configuration["core_model_configuration"]

    top_model = model_configuration["top_model"]
    top_model_configuration = model_configuration["top_model_configuration"]

    input_type_list = model_configuration["input_type_list"]
    output_type_list = model_configuration["output_type_list"]

    # We need to create architectures for all possible kinds of input combination.
    logmel_spectrogram_padded_shape = name_to_metadata["voice" + "_logmel_spectrogram"]["padded_shape"]
    logmel_spectrogram_padded_shape = [e for e in logmel_spectrogram_padded_shape]
    logmel_spectrogram_padded_shape[-1] = 64
    logmel_spectrogram_padded_shape = tuple(logmel_spectrogram_padded_shape)
    logmel_spectrogram_support_padded_shape = name_to_metadata["voice" + "_logmel_spectrogram_support"]["padded_shape"]

    modality_availabilities = ["single_voice",
                               "single_breath",
                               "single_cough",
                               "double_voice_breath",
                               "double_voice_cough",
                               "double_breath_cough",
                               "triple_voice_breath_cough"]

    input_layers_dict = dict()
    for modality_availability in modality_availabilities:
        input_layers_dict[modality_availability] = list()
        if "single" in modality_availability:
            m_total = 1
        elif "double" in modality_availability:
            m_total = 2
        elif "triple" in modality_availability:
            m_total = 3
        else:
            raise ValueError
        for m in range(m_total):
            input_layers_dict[modality_availability].append(
                tf.keras.Input(shape=logmel_spectrogram_padded_shape))
            input_layers_dict[modality_availability].append(
                tf.keras.Input(shape=logmel_spectrogram_support_padded_shape))

    custom_objects = dict()

    # Define core model. ["SEResNet", "SeResNetish", "audioset-VGGish"]
    if core_model == "SEResNet":
        core_model_configuration_effective = {k: v for k, v in core_model_configuration.items()}
        core_model_keras = list()
        for i, modality in enumerate(["voice", "breath", "cough"]):
            core_model_configuration_effective["name"] = "seresnet_core_" + modality
            core_model_keras.append(SEResNet(**core_model_configuration))
        embedd = None
    elif core_model == "SEResNetish":
        core_model_configuration_effective = {k: v for k, v in core_model_configuration.items()}
        core_model_keras = list()
        embedd = list()
        for i, modality in enumerate(["voice", "breath", "cough"]):
            core_model_configuration_effective["name"] = "resnetish_core_" + modality
            core_model_keras.append(SEResNetish(**core_model_configuration_effective))
            embedd.append(FeedForwardBlock(layer_units=[4096, 4096],
                                           outputs_list=[128, ],
                                           name="ff_embed_" + modality))
    elif core_model == "audioset-VGGish":
        core_model_keras = list()
        for i, modality in enumerate(["voice", "breath", "cough"]):
            core_model_keras.append(vgk.VGGish(pump=None,
                                               input_shape=(96, 64, 1),
                                               include_top=True,
                                               pooling=None,
                                               weights="audioset",
                                               name="vggish_core_" + modality,
                                               compress=False))
        embedd = None
    elif core_model == "VGGish":
        core_model_keras = vgk.VGGish(pump=None,
                                      input_shape=(96, 64, 1),
                                      include_top=True,
                                      pooling=None,
                                      weights=None,
                                      name="vggish_core",
                                      compress=False)
        embedd = None
    else:
        raise ValueError("Invalid core_model type.")

    # Top model.
    if top_model == "FeedForwardBlock":
        feed_forward_block = list()
        top_model_configuration_effective = {k: v for k, v in top_model_configuration.items()}
        for m_i in range(1):
            top_model_configuration_effective["name"] = "ff_" + repr(m_i)
            feed_forward_block.append(FeedForwardBlock(**top_model_configuration_effective))
    else:
        raise ValueError("Invalid global_pooling type.")

    # cdpl = FeedForwardBlock(layer_units=[128, ],
    #                         outputs_list=[128, ],
    #                         name="cdpl")
    cdpl = None

    # Propagate inputs.
    # Same core model per modality.
    net_train_dict, \
    net_test_dict = model_call(input_layers_dict,
                               core_model,
                               core_model_keras,
                               embedd,
                               feed_forward_block,
                               cdpl)

    prediction_train = dict()
    prediction_test = dict()

    for modality_availability in modality_availabilities:

        prediction_train[modality_availability] = dict()
        prediction_train[modality_availability]["embedding"] = net_train_dict[modality_availability + "_embedding"]
        for o_i, output_type in enumerate(output_type_list):
            prediction_train[modality_availability][output_type] = net_train_dict[modality_availability][o_i]
            prediction_train[modality_availability][output_type + "_prob"] = tf.nn.sigmoid(net_train_dict[modality_availability][o_i])

        prediction_test[modality_availability] = dict()
        prediction_test[modality_availability]["embedding"] = net_test_dict[modality_availability + "_embedding"]
        for o_i, output_type in enumerate(output_type_list):
            prediction_test[modality_availability][output_type] = net_test_dict[modality_availability][o_i]
            prediction_test[modality_availability][output_type + "_prob"] = tf.nn.sigmoid(net_test_dict[modality_availability][o_i])

    keras_model_train = dict()
    keras_model_test = dict()

    for modality_availability in modality_availabilities:
        model_outputs = list()
        for o_i, output_type in enumerate(output_type_list):
            model_outputs.append(prediction_train[modality_availability][output_type])

        keras_model_train[modality_availability] = tf.keras.Model(
            inputs=input_layers_dict[modality_availability],
            outputs=model_outputs)

    for modality_availability in modality_availabilities:
        model_outputs = list()
        for o_i, output_type in enumerate(output_type_list):
            model_outputs.append(prediction_test[modality_availability][output_type])

        keras_model_test[modality_availability] = tf.keras.Model(
            inputs=input_layers_dict[modality_availability],
            outputs=model_outputs)

    keras_model_test["triple_voice_breath_cough"].summary()

    other_outputs = dict()

    return prediction_train, prediction_test,\
           keras_model_train, keras_model_test,\
           other_outputs, custom_objects
