import tensorflow as tf

from common.models.conv_blocks import ResBlock, ConvBlock


class SEResNet(tf.keras.layers.Layer):
    def __init__(self,
                 block_filter_list,
                 block_multiplier_list,
                 use_se,
                 use_tc,
                 use_ds,
                 pool_type,
                 **kwargs):
        super(SEResNet, self).__init__(**kwargs)
        self.block_filter_list = block_filter_list
        self.block_multiplier_list = block_multiplier_list
        self.use_se = use_se
        self.use_tc = use_tc
        self.use_ds = use_ds
        self.pool_type = pool_type

        if self.use_tc:
            self.max_pool_size = (2, 1)
            self.max_pool_strides = (2, 1)

        else:
            self.max_pool_size = (2, 2)
            self.max_pool_strides = (2, 2)

        if self.pool_type in ["attentive", ]:
            raise NotImplementedError
        elif self.pool_type in ["max", ]:
            self.pooling_class = tf.keras.layers.MaxPool2D
        else:
            raise ValueError("Invalid max pool type.")

        self.layer_list = list()

    def build(self, input_shape):
        for b_i, b_filters in enumerate(self.block_filter_list):
            # block_filter_list = [64, 64, 128, 256, 512, 1024]
            # block_multiplier_list = [1, 2, 2, 2, 2, 1]
            layer_number = self.block_multiplier_list[b_i]
            if b_i == 0:
                for l_i in range(layer_number):
                    self.layer_list.append(ConvBlock(filters=b_filters,
                                                     use_bias=False,
                                                     max_pool_size=self.max_pool_size,
                                                     max_pool_strides=self.max_pool_strides,
                                                     num_layers=2,
                                                     use_max_pool=False,
                                                     use_se=False,
                                                     use_ds=False,
                                                     pool_type=self.pool_type,
                                                     ratio=4))

                self.layer_list.append(self.pooling_class(pool_size=self.max_pool_size,
                                                          strides=self.max_pool_strides,
                                                          padding='valid'))
            elif b_i == len(self.block_filter_list) - 1:
                for l_i in range(layer_number):
                    self.layer_list.append(ConvBlock(filters=b_filters,
                                                     use_bias=False,
                                                     max_pool_size=self.max_pool_size,
                                                     max_pool_strides=self.max_pool_strides,
                                                     num_layers=2,
                                                     use_max_pool=False,
                                                     use_se=False,
                                                     use_ds=False,
                                                     pool_type=self.pool_type,
                                                     ratio=4))
            else:
                for l_i in range(layer_number):
                    self.layer_list.append(ResBlock(b_filters,
                                                    use_se=self.use_se,
                                                    use_ds=self.use_ds))
                self.layer_list.append(self.pooling_class(pool_size=self.max_pool_size,
                                                          strides=self.max_pool_strides,
                                                          padding='valid'))
        super(SEResNet, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'block_filter_list': self.block_filter_list,
            'block_multiplier_list': self.block_multiplier_list,
            'use_se': self.use_se,
            'use_tc': self.use_tc,
            'use_ds': self.use_ds,
            'pool_type': self.pool_type,
        })
        return config

    def call(self, x, training):
        x_shape = tf.shape(x)
        if self.use_tc:
            net = tf.reshape(x,
                             (x_shape[0],
                              x_shape[1],
                              1,
                              x_shape[2]))
        else:
            net = x
            # net = tf.reshape(x,
            #                  (x_shape[0],
            #                   x_shape[1],
            #                   x_shape[2],
            #                   1))

        net = self.layer_list[0](net, training=training)
        for l in self.layer_list[1:]:
            net = l(net, training=training)

        net_shape = tf.shape(net)
        if self.use_tc:
            net = tf.reshape(net,
                             (net_shape[0],
                              net_shape[1],
                              1,
                              net_shape[3]))

        else:
            net = tf.reshape(net,
                             (net_shape[0],
                              net_shape[1],
                              1,
                              net_shape[2] * net_shape[3]))

        return net


class SEResNetish(tf.keras.layers.Layer):
    def __init__(self,
                 block_filter_list,
                 block_multiplier_list,
                 use_se,
                 use_tc,
                 use_ds,
                 pool_type,
                 regularisation_factor=None,
                 **kwargs):
        super(SEResNetish, self).__init__(**kwargs)
        self.block_filter_list = block_filter_list
        self.block_multiplier_list = block_multiplier_list
        self.use_se = use_se
        self.use_tc = use_tc
        self.use_ds = use_ds
        self.pool_type = pool_type
        self.regularisation_factor = regularisation_factor

        if self.use_tc:
            self.max_pool_size = (2, 1)
            self.max_pool_strides = (2, 1)

        else:
            self.max_pool_size = (2, 2)
            self.max_pool_strides = (2, 2)

        if self.pool_type in ["attentive", ]:
            raise NotImplementedError
        elif self.pool_type in ["max", ]:
            self.pooling_class = tf.keras.layers.MaxPool2D
        else:
            raise ValueError("Invalid max pool type.")

        self.layer_list = list()

    def build(self, input_shape):
        for b_i, b_filters in enumerate(self.block_filter_list):
            # block_filter_list = [64, 64, 128, 256, 512, 1024]
            # block_multiplier_list = [1, 2, 2, 2, 2, 1]
            layer_number = self.block_multiplier_list[b_i]
            if b_i == 0:
                self.layer_list.append(ConvBlock(filters=b_filters,
                                                 use_bias=False,
                                                 max_pool_size=self.max_pool_size,
                                                 max_pool_strides=self.max_pool_strides,
                                                 num_layers=layer_number,
                                                 use_max_pool=False,
                                                 use_se=False,
                                                 use_ds=False,
                                                 pool_type=self.pool_type,
                                                 ratio=4))

                self.layer_list.append(self.pooling_class(pool_size=self.max_pool_size,
                                                          strides=self.max_pool_strides,
                                                          padding='same'))
            elif b_i == len(self.block_filter_list) - 1:
                self.layer_list.append(ConvBlock(filters=b_filters,
                                                 use_bias=False,
                                                 max_pool_size=self.max_pool_size,
                                                 max_pool_strides=self.max_pool_strides,
                                                 num_layers=layer_number,
                                                 use_max_pool=False,
                                                 use_se=False,
                                                 use_ds=False,
                                                 pool_type=self.pool_type,
                                                 ratio=4))

                self.layer_list.append(self.pooling_class(pool_size=self.max_pool_size,
                                                          strides=self.max_pool_strides,
                                                          padding='same'))
            else:
                self.layer_list.append(ResBlock(b_filters,
                                                use_se=self.use_se,
                                                use_ds=self.use_ds))
                self.layer_list.append(self.pooling_class(pool_size=self.max_pool_size,
                                                          strides=self.max_pool_strides,
                                                          padding='same'))
        super(SEResNetish, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'block_filter_list': self.block_filter_list,
            'block_multiplier_list': self.block_multiplier_list,
            'use_se': self.use_se,
            'use_tc': self.use_tc,
            'use_ds': self.use_ds,
            'pool_type': self.pool_type,
            'regularisation_factor': self.regularisation_factor,
        })
        return config

    def call(self, x, training):
        # x_shape = tf.shape(x)
        # if self.use_tc:
        #     net = tf.reshape(x,
        #                      (x_shape[0],
        #                       x_shape[1],
        #                       1,
        #                       x_shape[2]))
        # else:
        #     net = tf.reshape(x,
        #                      (x_shape[0],
        #                       x_shape[1],
        #                       x_shape[2],
        #                       1))
        net = x

        net = self.layer_list[0](net, training=training)
        for l in self.layer_list[1:]:
            net = l(net, training=training)

        # net_shape = tf.shape(net)
        # if self.use_tc:
        #     net = tf.reshape(net,
        #                      (net_shape[0],
        #                       net_shape[1],
        #                       1,
        #                       net_shape[3]))
        #
        # else:
        #     net = tf.reshape(net,
        #                      (net_shape[0],
        #                       net_shape[1],
        #                       1,
        #                       net_shape[2] * net_shape[3]))

        return net
