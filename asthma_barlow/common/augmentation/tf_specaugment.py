# Based on: https://github.com/DemisEom/SpecAugment/blob/7f1435963b37ac8f9e4de9e44d754ecc41eaba85/SpecAugment/spec_augment_tensorflow.py#L46
import random

import tensorflow as tf
# from tensorflow_addons.image import sparse_image_warp


def frequency_masking(mel_spectrogram, frequency_masking_para, frequency_mask_num):
    # fbank_size = mel_spectrogram.get_shape().as_list()
    fbank_size = tf.shape(mel_spectrogram)
    _, v = fbank_size[0], fbank_size[1]

    for i in range(frequency_mask_num):
        # f = random.randint(0, frequency_masking_para)
        f = tf.random.uniform([], minval=0, maxval=frequency_masking_para, dtype=tf.int32)
        # f0 = random.randint(0, v-f)
        f0 = tf.random.uniform([], minval=0, maxval=v-f, dtype=tf.int32)

        mask = tf.concat([tf.ones(shape=(1, f0), dtype=tf.float32),
                          tf.zeros(shape=(1, f), dtype=tf.float32),
                          tf.ones(shape=(1, v - f - f0), dtype=tf.float32),
                          ], 1)
        mel_spectrogram = tf.multiply(mel_spectrogram, mask)
    return tf.cast(mel_spectrogram, tf.float32)


def time_masking(mel_spectrogram, time_masking_para, time_mask_num):
    # fbank_size = mel_spectrogram.get_shape().as_list()
    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[0], fbank_size[1]

    for i in range(time_mask_num):
        # t = random.randint(0, time_masking_para)
        t = tf.random.uniform([], minval=0, maxval=time_masking_para, dtype=tf.int32)
        # t0 = random.randint(0, n - t)
        t0 = tf.random.uniform([], minval=0, maxval=n-t, dtype=tf.int32)

        mask = tf.concat([tf.ones(shape=(t0, 1), dtype=tf.float32),
                          tf.zeros(shape=(t, 1), dtype=tf.float32),
                          tf.ones(shape=(n - t - t0, 1), dtype=tf.float32),
                          ], 0)
        mel_spectrogram = tf.multiply(mel_spectrogram, mask)
    return tf.cast(mel_spectrogram, tf.float32)


# def sparse_warp(mel_spectrogram, time_warping_para=80):
#     fbank_size = mel_spectrogram.get_shape().as_list()
#     n, v = fbank_size[0], fbank_size[1]
#
#     pt = tf.random.uniform([], time_warping_para, n-time_warping_para, tf.int32)  # radnom point along the time axis
#     src_ctr_pt_freq = tf.range(v // 2)  # control points on freq-axis
#     src_ctr_pt_time = tf.ones_like(src_ctr_pt_freq) * pt  # control points on time-axis
#     src_ctr_pts = tf.stack((src_ctr_pt_time, src_ctr_pt_freq), -1)
#     src_ctr_pts = tf.cast(src_ctr_pts, dtype=tf.float32)
#
#     # Destination
#     w = tf.random.uniform([], -time_warping_para, time_warping_para, tf.int32)  # distance
#     dest_ctr_pt_freq = src_ctr_pt_freq
#     dest_ctr_pt_time = src_ctr_pt_time + w
#     dest_ctr_pts = tf.stack((dest_ctr_pt_time, dest_ctr_pt_freq), -1)
#     dest_ctr_pts = tf.cast(dest_ctr_pts, dtype=tf.float32)
#
#     # warp
#     source_control_point_locations = tf.expand_dims(src_ctr_pts, 0)  # (1, v//2, 2)
#     dest_control_point_locations = tf.expand_dims(dest_ctr_pts, 0)  # (1, v//2, 2)
#
#     warped_image, _ = sparse_image_warp(mel_spectrogram,
#                                         source_control_point_locations,
#                                         dest_control_point_locations)
#     return warped_image
