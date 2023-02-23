import tensorflow as tf


class FeedForwardBlock(tf.keras.layers.Layer):
    def __init__(self,
                 layer_units,
                 outputs_list,
                 regularisation_factor=None,
                 **kwargs):
        super(FeedForwardBlock, self).__init__(**kwargs)
        self.layer_units = layer_units
        self.outputs_list = outputs_list
        self.regularisation_factor = regularisation_factor

        self.layer_list = list()
        if self.regularisation_factor is not None:
            l2 = tf.keras.regularizers.l2(l=self.regularisation_factor)
            print("regularise l2")
        else:
            l2 = None
        for units in self.layer_units:
            self.layer_list.append(tf.keras.layers.Dense(units,
                                                         activation='relu',
                                                         use_bias=True,
                                                         kernel_regularizer=l2,
                                                         bias_regularizer=l2))

        self.dense_layer_list = list()
        for t, output_units in enumerate(self.outputs_list):
            self.dense_layer_list.append(tf.keras.layers.Dense(output_units,
                                                               activation=None,
                                                               use_bias=True,
                                                               kernel_regularizer=l2,
                                                               bias_regularizer=l2))

    def get_config(self):
        config = super().get_config()
        config.update({
            'layer_units': self.layer_units,
            'outputs_list': self.outputs_list,
            'regularisation_factor': self.regularisation_factor,
        })
        return config

    def call(self, x, training):
        net = self.layer_list[0](x, training=training)
        for l in self.layer_list[1:]:
            net = l(net, training=training)

        prediction_single = list()
        for t_i, t in enumerate(range(len(self.outputs_list))):
            # if t_i > 0:
            #     net_o = -0.1 * net + tf.stop_gradient(1.1 * net)
            # else:
            #     net_o = net
            o = self.dense_layer_list[t](net, training=training)
            prediction_single.append(o)

        return prediction_single


# @tf.custom_gradient
# def reverse_grad_layer(x):
#     def grad(dy):
#         return -dy
#     return tf.identity(x), grad
