import tensorflow as tf


def model_call(input_layers_dict,
               core_model,
               core_model_keras,
               embedd,
               attention_global_pooling,
               feed_forward_block,
               meta_cleaner=None,
               cdpl=None):
    # We use a dedicated model parameterisation per modality.
    modality_availabilities = ["single_voice",
                               "single_breath",
                               "single_cough",
                               "double_voice_breath",
                               "double_voice_cough",
                               "double_breath_cough",
                               "triple_voice_breath_cough"]
    # In case we just use 1 model (shared parameters across modalities)
    # modality_availabilities = ["single",
    #                            "double",
    #                            "triple"]
    output_dict_train = dict()
    output_dict_test = dict()
    for modality_availability in modality_availabilities:
        net_per_modality = list()

        # 1 shared model.
        # if modality_availability == "single":
        #     num_modalities = 1
        # elif modality_availability == "double":
        #     num_modalities = 2
        # elif modality_availability == "triple":
        #     num_modalities = 3
        # else:
        #     raise ValueError

        # 3 models.
        if "single" in modality_availability:
            num_modalities = 1
        elif "double" in modality_availability:
            num_modalities = 2
        elif "triple" in modality_availability:
            num_modalities = 3
        else:
            raise ValueError

        modalities = list()
        if "voice" in modality_availability:
            modalities.append(0)
        if "breath" in modality_availability:
            modalities.append(1)
        if "cough" in modality_availability:
            modalities.append(2)

        for m_i, m in enumerate(modalities):
            net_train = tf.expand_dims(input_layers_dict[modality_availability][m_i * 2], axis=3)
            net_support_train = input_layers_dict[modality_availability][(m_i * 2) + 1]
            net_per_modality_i = core_model_call(net_train,
                                                 net_support_train,
                                                 core_model,
                                                 core_model_keras,
                                                 embedd,
                                                 attention_global_pooling,
                                                 m)
            net_per_modality.append(net_per_modality_i)

        # Fuse 1 to 3 modalities.
        # net_train_fused = tf.concat(net_per_modality, axis=1)
        net_per_modality = [tf.expand_dims(n, axis=2) for n in net_per_modality]
        net_train_fused = tf.concat(net_per_modality, axis=2)
        net_train_fused = tf.reduce_max(net_train_fused, axis=2)

        # Avg. modality representations did not work as well as max.
        # net_train_fused = net_per_modality[0]
        # if len(net_per_modality) > 1:
        #     for to_add in net_per_modality[1:]:
        #         net_train_fused = net_train_fused + to_add
        # net_train_fused = net_train_fused / float(num_modalities)

        # Meta cleaner.
        if meta_cleaner is not None:
            cleaner_attention = meta_cleaner(net_train_fused, training=False)[0]
            cleaner_attention = tf.nn.softmax(cleaner_attention, axis=0)
            net_train_fused = tf.multiply(cleaner_attention, net_train_fused)
            net_train_fused = tf.reduce_sum(net_train_fused, axis=0, keepdims=True)

        # Make prediction.
        # net_train_fused = feed_forward_block[num_modalities - 1](net_train_fused, training=False)
        output_dict_train[modality_availability + "_embedding"] = net_train_fused
        # output_dict_train[modality_availability + "_embedding"] = cdpl(net_train_fused, training=False)[0]
        logits_train = feed_forward_block[0](net_train_fused, training=False)
        output_dict_train[modality_availability] = logits_train

        # net_per_modality = [tf.expand_dims(n, axis=2) for n in net_per_modality]
        net_test_fused = tf.concat(net_per_modality, axis=2)
        net_test_fused = tf.reduce_max(net_test_fused, axis=2)

        # Make prediction.
        # net_test_fused = feed_forward_block[num_modalities - 1](net_test_fused, training=False)
        output_dict_test[modality_availability + "_embedding"] = net_test_fused
        # output_dict_test[modality_availability + "_embedding"] = cdpl(net_test_fused, training=False)[0]
        logits_test = feed_forward_block[0](net_test_fused, training=False)
        output_dict_test[modality_availability] = logits_test

    return output_dict_train, output_dict_test
    # return output_dict_test, output_dict_test


def core_model_call(net_train,
                    net_support_train,
                    core_model,
                    core_model_keras,
                    embedd,
                    attention_global_pooling,
                    m_i):
    net_shape = [tf.shape(net_train)[k] for k in range(4)]

    # ["SEResNet", "SeResNetish", "audioset-VGGish", "VGGish", "VGGish-alt"]
    if core_model in ["SEResNet", ]:
        net_train = core_model_keras[m_i](net_train, training=False)

        # Global pooling per modality.
        net_support_train = tf.expand_dims(net_support_train, 3)

        net_support_train = tf.nn.max_pool(net_support_train, ksize=(2, 1), strides=(2, 1), padding="VALID")
        net_support_train = tf.nn.max_pool(net_support_train, ksize=(2, 1), strides=(2, 1), padding="VALID")
        net_support_train = tf.nn.max_pool(net_support_train, ksize=(2, 1), strides=(2, 1), padding="VALID")
        net_support_train = tf.nn.max_pool(net_support_train, ksize=(2, 1), strides=(2, 1), padding="VALID")
        net_support_train = tf.nn.max_pool(net_support_train, ksize=(2, 1), strides=(2, 1), padding="VALID")
        net_support_train = tf.reshape(net_support_train, (net_shape[0], -1, 1, 1))

        net_train = tf.reshape(net_train, (net_shape[0], -1, 512, 1))
        net_train = tf.multiply(net_train, net_support_train)
        net_train = tf.multiply(tf.reduce_sum(net_train, axis=1, keepdims=True), 1.0 / tf.reduce_sum(net_support_train, axis=1, keepdims=True))
        net_train = tf.reshape(net_train, (net_shape[0], 512))
        # net_train = tf.reduce_mean(net_train, axis=1)
        # net_train = tf.reshape(net_train, (net_shape[0], -1, 128, 1))
        # net_train, \
        # attention_weights_train = attention_global_pooling(net_train,
        #                                                    training=False,
        #                                                    support=None)
        net_per_modality = net_train
    elif core_model in ["SEResNetish", "audioset-VGGish", "VGGish", "VGGish-alt"]:
        net_train = tf.reshape(net_train, (-1, 96, 64, 1))

        net_train = core_model_keras[m_i](net_train, training=False)
        if core_model in ["SEResNetish", "VGGish-alt"]:
            if core_model == "VGGish-alt":
                # net_train = tf.reshape(net_train, (net_shape[0], -1))
                net_train = tf.keras.layers.Flatten()(net_train)
            elif core_model == "SEResNetish":
                # net_train = tf.reshape(net_train, (net_shape[0], -1))
                net_train = tf.keras.layers.Flatten()(net_train)
            else:
                raise ValueError
            net_train = embedd[m_i](net_train, training=False)[0]
            net_train = tf.nn.relu(net_train)

        # Global pooling per modality.
        net_support_train = tf.expand_dims(net_support_train, 3)
        net_support_train = tf.reshape(net_support_train, (-1, 96, 1, 1))
        net_support_train = tf.reduce_mean(net_support_train, axis=1, keepdims=True)
        net_support_train = tf.reshape(net_support_train, (net_shape[0], -1, 1, 1))

        net_train = tf.reshape(net_train, (net_shape[0], -1, 128))

        net_train = tf.reshape(net_train, (net_shape[0], -1, 128, 1))

        # net_train = tf.reduce_mean(net_train, axis=1)

        net_train = tf.multiply(net_train, net_support_train)
        net_train = tf.multiply(tf.reduce_sum(net_train, axis=1, keepdims=True), 1.0 / tf.reduce_sum(net_support_train, axis=1, keepdims=True))
        net_train = tf.reshape(net_train, (net_shape[0], 128))

        # net_train = tf.reduce_mean(net_train, axis=1)

        # net_train = tf.reshape(net_train, (net_shape[0], -1, 128, 1))
        # net_train, \
        # attention_weights_train = attention_global_pooling(net_train,
        #                                                    training=False,
        #                                                    support=None)
        net_per_modality = net_train
    elif core_model in ["FF", ]:
        net_train = tf.reshape(net_train, (-1, 1024))
        net_train = core_model_keras[m_i](net_train, training=False)[0]
        net_train = tf.nn.relu(net_train)

        # Global pooling per modality.
        net_support_train = tf.expand_dims(net_support_train, 3)
        net_support_train = tf.reshape(net_support_train, (net_shape[0], -1, 1, 1))

        net_train = tf.reshape(net_train, (net_shape[0], -1, 128, 1))

        net_train = tf.multiply(net_train, net_support_train)
        net_train = tf.multiply(tf.reduce_sum(net_train, axis=1, keepdims=True), 1.0 / tf.reduce_sum(net_support_train, axis=1, keepdims=True))
        net_train = tf.reshape(net_train, (net_shape[0], 128))

        net_per_modality = net_train
    else:
        raise ValueError("Invalid core_model type.")

    return net_per_modality
