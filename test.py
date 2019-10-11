# coding=utf-8
import sys
import argparse
import cv2
import numpy as np
from scipy import misc
import tensorflow as tf


from nets import nets_factory
from preprocessing import preprocessing_factory


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_A', type=str, default='./img/CT-MRI/source17_1.tif', help='absolute path for image A.')
    parser.add_argument('--image_B', type=str, default='./img/CT-MRI/source17_2.tif', help='absolute path for image B.')
    parser.add_argument('--model_dir', type=str, default='./models/model.ckpt-2475',
                        help='Path where .ckpt file is saved')
    parser.add_argument('--gpus', type=str, default="3")

    return parser.parse_args(argv)


class CNNModel:
    def __init__(self, model_dir):
        self.images_placeholder = None
        self.H3_images = None
        self.L3_images = None
        self.Weight = None
        self.Bias = None
        self.build_model(model_dir)

    @staticmethod
    def load_checkpoint(sess, model_path):
        saver_restore = tf.train.Saver(tf.global_variables())
        saver_restore.restore(sess, model_path)

    def build_model(self, model_dir):
        model_name, preprocessing_name = 'fusion', 'fusion'
        image_height, image_width = 256, 256

        # Select the model
        network_fn = nets_factory.get_network_fn(model_name,
                                                 num_classes=None,
                                                 is_training=False)
        # Define image placeholder and preprocessing
        image_heights = image_height or network_fn.default_image_size
        image_widths = image_width or network_fn.default_image_size
        self.images_placeholder = tf.placeholder(dtype=tf.uint8, shape=(None, None, None, 3), name="images")
        preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)
        images = tf.map_fn(lambda x: preprocessing_fn(x, image_heights, image_widths),
                           self.images_placeholder, dtype=tf.float32)

        # Get logits tensor
        logits, endpoints = network_fn(images)
        H3_images = endpoints['fusion/conv_H3']
        L3_images = endpoints['fusion/conv_L3']
        self.Weight = tf.get_default_graph().get_tensor_by_name('InceptionV3/conv_C1/weights:0')
        self.Bias = tf.get_default_graph().get_tensor_by_name('InceptionV3/conv_C1/BatchNorm/beta:0')
        self.Mean = tf.get_default_graph().get_tensor_by_name('InceptionV3/conv_C1/BatchNorm/moving_mean:0')
        self.Std = tf.get_default_graph().get_tensor_by_name('InceptionV3/conv_C1/BatchNorm/moving_variance:0')
        self.H3_images = H3_images
        self.L3_images = L3_images

    # calculate prediction for samples
    def cal_predict(self, sess, image_A_path, image_B_path):

        image_A = cv2.imread(image_A_path)
        image_B = cv2.imread(image_B_path)
        re_image_A = cv2.resize(image_A, (256, 256))
        re_image_B = cv2.resize(image_B, (256, 256))
        expand_A = np.expand_dims(re_image_A, axis=0)
        expand_B = np.expand_dims(re_image_B, axis=0)

        H3_image_A = sess.run(self.H3_images, feed_dict={self.images_placeholder: expand_A})
        H3_image_A = H3_image_A.squeeze()
        L3_image_A = sess.run(self.L3_images, feed_dict={self.images_placeholder: expand_A})
        L3_image_A = L3_image_A.squeeze()
        H3_image_B = sess.run(self.H3_images, feed_dict={self.images_placeholder: expand_B})
        H3_image_B = H3_image_B.squeeze()
        L3_image_B = sess.run(self.L3_images, feed_dict={self.images_placeholder: expand_B})
        L3_image_B = L3_image_B.squeeze()

        # height light
        H3_image_A_1 = cv2.filter2D(H3_image_A**2, -1, np.ones((5, 5))/24, borderType=cv2.BORDER_REFLECT)
        H3_image_A_2 = cv2.filter2D(H3_image_A, -1, np.ones((5, 5)), borderType=cv2.BORDER_REFLECT) ** 2 / (25*24)
        val_H3_image_A = np.sqrt(np.maximum(H3_image_A_1 - H3_image_A_2, 0))

        H3_image_B_1 = cv2.filter2D(H3_image_B ** 2, -1, np.ones((5, 5)) / 24, borderType=cv2.BORDER_REFLECT)
        H3_image_B_2 = cv2.filter2D(H3_image_B, -1, np.ones((5, 5)), borderType=cv2.BORDER_REFLECT) ** 2 / (25 * 24)
        val_H3_image_B = np.sqrt(np.maximum(H3_image_B_1 - H3_image_B_2, 0))

        height_map = np.where(val_H3_image_A > val_H3_image_B, H3_image_A, H3_image_B)

        # lower light
        E_L3_image_A = cv2.filter2D(L3_image_A * L3_image_A, -1, np.ones((5, 5)), borderType=cv2.BORDER_REFLECT)
        E_L3_image_B = cv2.filter2D(L3_image_B * L3_image_B, -1, np.ones((5, 5)), borderType=cv2.BORDER_REFLECT)

        # fill border
        multi_A_B = np.ones((L3_image_A.shape[0], L3_image_A.shape[1]))
        L3_image_A_expand = cv2.copyMakeBorder(L3_image_A, 2, 2, 2, 2, cv2.BORDER_REFLECT)
        L3_image_B_expand = cv2.copyMakeBorder(L3_image_B, 2, 2, 2, 2, cv2.BORDER_REFLECT)

        for i in range(2, L3_image_A_expand.shape[0] - 4):
            for j in range(2, L3_image_A_expand.shape[1] - 4):
                multi_A_B[i][j] = np.sum(L3_image_A_expand[i-2: i+2, j-2: j+2] * L3_image_B_expand[i-2: i+2, j-2: j+2])

        MAB = multi_A_B / (E_L3_image_A + E_L3_image_B)

        W_L = 23.0 / 6.0 - (10.0/3) * MAB
        W_S = (10.0/3) * MAB - 17.0 / 6.0
        lower_map = np.where(MAB < 0.85, np.where(E_L3_image_A > E_L3_image_B, L3_image_A, L3_image_B),
                             np.where(E_L3_image_A > E_L3_image_B, W_L*L3_image_A + W_S*L3_image_B,
                                      W_S*L3_image_A + W_L*L3_image_B))
        input_map = np.stack([height_map, lower_map], axis=-1)

        # compute fusion image
        weight = sess.run(self.Weight, feed_dict={self.images_placeholder: expand_A})
        weight = weight.squeeze()
        biases = sess.run(self.Bias, feed_dict={self.images_placeholder: expand_A})
        mean = sess.run(self.Mean, feed_dict={self.images_placeholder: expand_A})
        var = sess.run(self.Std, feed_dict={self.images_placeholder: expand_A})

        # convert input map by batch normalize
        input_map = (input_map - mean) / np.sqrt(var)

        kernel = 5
        result_map = np.ones((input_map.shape[0], input_map.shape[1]))
        input_map_expand = cv2.copyMakeBorder(input_map, kernel/2, kernel/2, kernel/2, kernel/2, cv2.BORDER_REFLECT)
        for i in range(kernel/2, input_map_expand.shape[0] - 2 * (kernel/2)):
            for j in range(kernel/2, input_map_expand.shape[1] - 2 * (kernel/2)):
                result_map[i-kernel/2][j-kernel/2] = np.sum(input_map_expand[i-kernel/2: i - kernel/2 + kernel, j-kernel/2:j - kernel/2 + kernel, :] * weight) + biases

        result_map = ((result_map * 0.5 + 0.5) * 255).astype(np.uint8)
        re_original_map = misc.imresize(result_map, (image_A.shape[0], image_A.shape[1]))

        cv2.imshow('fusion_image', re_original_map)
        cv2.waitKey(0)


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # setting models
        cnn_model = CNNModel(args.model_dir)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1, allow_growth=True,
                                    visible_device_list=args.gpus)
        config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)

        with tf.Session(config=config) as sess:
            cnn_model.load_checkpoint(sess, args.model_dir)

            cnn_model.cal_predict(sess, args.image_A, args.image_B)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))