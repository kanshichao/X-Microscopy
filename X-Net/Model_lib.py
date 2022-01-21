import tensorflow as tf
import constant as con
import opts as tools
import numpy as np

class pix2pix_generator(object):
    def __init__(self):
        self.base_dim = 32

        self.d_bn1 = tools.batch_norm(name='dis_bn1')
        self.d_bn2 = tools.batch_norm(name='dis_bn2')
        self.d_bn3 = tools.batch_norm(name='dis_bn3')

        self.d_bn1_1 = tools.batch_norm(name='dis1_bn1')
        self.d_bn2_1 = tools.batch_norm(name='dis1_bn2')
        self.d_bn3_1 = tools.batch_norm(name='dis1_bn3')

        self.d_bn1_2 = tools.batch_norm(name='dis2_bn1')
        self.d_bn2_2 = tools.batch_norm(name='dis2_bn2')
        self.d_bn3_2 = tools.batch_norm(name='dis2_bn3')

        self.d_bn1_3 = tools.batch_norm(name='dis3_bn1')
        self.d_bn2_3 = tools.batch_norm(name='dis3_bn2')
        self.d_bn3_3 = tools.batch_norm(name='dis3_bn3')

        self.d_bn1_4 = tools.batch_norm(name='dis4_bn1')
        self.d_bn2_4 = tools.batch_norm(name='dis4_bn2')
        self.d_bn3_4 = tools.batch_norm(name='dis4_bn3')

        self.d_bn1_f = tools.batch_norm(name='disf_bn1')
        self.d_bn2_f = tools.batch_norm(name='disf_bn2')
        self.d_bn3_f = tools.batch_norm(name='disf_bn3')

        self.g_bn_e2 = tools.batch_norm(name='gab_bn_e2')
        self.g_bn_e3 = tools.batch_norm(name='gab_bn_e3')
        self.g_bn_e4 = tools.batch_norm(name='gab_bn_e4')
        self.g_bn_e5 = tools.batch_norm(name='gab_bn_e5')
        self.g_bn_e6 = tools.batch_norm(name='gab_bn_e6')
        self.g_bn_e7 = tools.batch_norm(name='gab_bn_e7')
        self.g_bn_e8 = tools.batch_norm(name='gab_bn_e8')
        self.g_bn_e9 = tools.batch_norm(name='g_bn_e9')
        self.g_bn_e10 = tools.batch_norm(name='g_bn_e10')
        self.g_bn_e11 = tools.batch_norm(name='g_bn_e11')
        self.g_bn_e12 = tools.batch_norm(name='g_bn_e12')
        self.g_bn_e13 = tools.batch_norm(name='g_bn_e13')
        self.g_bn_e14 = tools.batch_norm(name='g_bn_e14')
        self.g_bn_e15 = tools.batch_norm(name='g_bn_e15')
        self.g_bn_e16 = tools.batch_norm(name='g_bn_e16')

        self.gu_bn_e2 = tools.batch_norm(name='gabu_bn_e2')
        self.gu_bn_e3 = tools.batch_norm(name='gabu_bn_e3')
        self.gu_bn_e4 = tools.batch_norm(name='gabu_bn_e4')
        self.gu_bn_e5 = tools.batch_norm(name='gabu_bn_e5')
        self.gu_bn_e6 = tools.batch_norm(name='gabu_bn_e6')
        self.gu_bn_e7 = tools.batch_norm(name='gabu_bn_e7')
        self.gu_bn_e8 = tools.batch_norm(name='gabu_bn_e8')
        self.gu_bn_e9 = tools.batch_norm(name='gu_bn_e9')
        self.gu_bn_e10 = tools.batch_norm(name='gu_bn_e10')
        self.gu_bn_e11 = tools.batch_norm(name='gu_bn_e11')
        self.gu_bn_e12 = tools.batch_norm(name='gu_bn_e12')
        self.gu_bn_e13 = tools.batch_norm(name='gu_bn_e13')
        self.gu_bn_e14 = tools.batch_norm(name='gu_bn_e14')
        self.gu_bn_e15 = tools.batch_norm(name='gu_bn_e15')
        self.gu_bn_e16 = tools.batch_norm(name='gu_bn_e16')

        self.g_bn_e17_1 = tools.batch_norm(name='g_bn_e17_1')
        self.gu_bn_e17_2 = tools.batch_norm(name='gu_bn_e17_2')

        self.g_bn_e18_1 = tools.batch_norm(name='g_bn_e18_1')
        self.gu_bn_e18_2 = tools.batch_norm(name='gu_bn_e18_2')

        self.g_bn_attd1_1 = tools.batch_norm(name='g_bn_attd1_1')
        self.g_bn_attd1_2 = tools.batch_norm(name='g_bn_attd1_2')

        self.g_bn_attd2_1 = tools.batch_norm(name='g_bn_attd2_1')
        self.g_bn_attd2_2 = tools.batch_norm(name='g_bn_attd2_2')

        self.g_bn_attd3_1 = tools.batch_norm(name='g_bn_attd3_1')
        self.g_bn_attd3_2 = tools.batch_norm(name='g_bn_attd3_2')

        self.g_bn_attd4_1 = tools.batch_norm(name='g_bn_attd4_1')
        self.g_bn_attd4_2 = tools.batch_norm(name='g_bn_attd4_2')

        self.g_bn_attd5_1 = tools.batch_norm(name='g_bn_attd5_1')
        self.g_bn_attd5_2 = tools.batch_norm(name='g_bn_attd5_2')

        self.g_bn_attd6_1 = tools.batch_norm(name='g_bn_attd6_1')
        self.g_bn_attd6_2 = tools.batch_norm(name='g_bn_attd6_2')

        self.g_bn_attd7_1 = tools.batch_norm(name='g_bn_attd7_1')
        self.g_bn_attd7_2 = tools.batch_norm(name='g_bn_attd7_2')

        self.g_bn_attd8_1 = tools.batch_norm(name='g_bn_attd8_1')
        self.g_bn_attd8_2 = tools.batch_norm(name='g_bn_attd8_2')

        self.gu_bn_attd1_1 = tools.batch_norm(name='gu_bn_attd1_1')
        self.gu_bn_attd1_2 = tools.batch_norm(name='gu_bn_attd1_2')

        self.gu_bn_attd2_1 = tools.batch_norm(name='gu_bn_attd2_1')
        self.gu_bn_attd2_2 = tools.batch_norm(name='gu_bn_attd2_2')

        self.gu_bn_attd3_1 = tools.batch_norm(name='gu_bn_attd3_1')
        self.gu_bn_attd3_2 = tools.batch_norm(name='gu_bn_attd3_2')

        self.gu_bn_attd4_1 = tools.batch_norm(name='gu_bn_attd4_1')
        self.gu_bn_attd4_2 = tools.batch_norm(name='gu_bn_attd4_2')

        self.gu_bn_attd5_1 = tools.batch_norm(name='gu_bn_attd5_1')
        self.gu_bn_attd5_2 = tools.batch_norm(name='gu_bn_attd5_2')

        self.gu_bn_attd6_1 = tools.batch_norm(name='gu_bn_attd6_1')
        self.gu_bn_attd6_2 = tools.batch_norm(name='gu_bn_attd6_2')

        self.gu_bn_attd7_1 = tools.batch_norm(name='gu_bn_attd7_1')
        self.gu_bn_attd7_2 = tools.batch_norm(name='gu_bn_attd7_2')

        self.gu_bn_attd8_1 = tools.batch_norm(name='gu_bn_attd8_1')
        self.gu_bn_attd8_2 = tools.batch_norm(name='gu_bn_attd8_2')


        self.gf_bn_ef_18_1 = tools.batch_norm(name='gf_bn_ef_18_1')
        self.gf_bn_ef_18_2 = tools.batch_norm(name='gf_bn_ef_18_2')

        self.g_bn_d1 = tools.batch_norm(name='g_bn_d1')
        self.g_bn_d2 = tools.batch_norm(name='g_bn_d2')
        self.g_bn_d3 = tools.batch_norm(name='g_bn_d3')
        self.g_bn_d4 = tools.batch_norm(name='g_bn_d4')
        self.g_bn_d5 = tools.batch_norm(name='g_bn_d5')
        self.g_bn_d6 = tools.batch_norm(name='g_bn_d6')
        self.g_bn_d7 = tools.batch_norm(name='g_bn_d7')
        self.g_bn_d8 = tools.batch_norm(name='g_bn_d8')

        self.gu_bn_d1 = tools.batch_norm(name='gu_bn_d1')
        self.gu_bn_d2 = tools.batch_norm(name='gu_bn_d2')
        self.gu_bn_d3 = tools.batch_norm(name='gu_bn_d3')
        self.gu_bn_d4 = tools.batch_norm(name='gu_bn_d4')
        self.gu_bn_d5 = tools.batch_norm(name='gu_bn_d5')
        self.gu_bn_d6 = tools.batch_norm(name='gu_bn_d6')
        self.gu_bn_d7 = tools.batch_norm(name='gu_bn_d7')
        self.gu_bn_d8 = tools.batch_norm(name='gu_bn_d8')

        self.gf_bn_attd9_1 = tools.batch_norm(name='gf_bn_d9_1')
        self.gf_bn_attd9_2 = tools.batch_norm(name='gf_bn_d9_2')


    def discriminator(self, input_data, batch_size=None, reuse=False):

        if batch_size is None:
            batch_size = con.FLAGS.batch_size
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = tools.lrelu(tools.conv2d(input_data, self.base_dim, k_h=3, k_w=3, d_h=2, d_w=2, name='dis_h0_conv'),leak=0)
            # h0 is (128 x 128 x self.df_dim)
            h1 = tools.lrelu(self.d_bn1(tools.conv2d(h0, self.base_dim*2,  k_h=3, k_w=3, d_h=2, d_w=2, name='dis_h1_conv')),leak=0)
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = tools.lrelu(self.d_bn2(tools.conv2d(h1, self.base_dim*4,  k_h=3, k_w=3, d_h=2, d_w=2, name='dis_h2_conv')),leak=0)
            # h2 is (32x 32 x self.df_dim*4)
            h3 = tools.lrelu(self.d_bn3(tools.conv2d(h2, self.base_dim*8,  k_h=3, k_w=3, d_h=1, d_w=1, name='dis_h3_conv')),leak=0)
            # h3 is (16 x 16 x self.df_dim*8)
            h3 = tools.spp_layer(h3, levels=4, name='dis_SPP_layer', pool_type='max_pool')
            h4 = tools.linear(tf.reshape(h3, [batch_size, -1]), 1, 'dis_h3_lin')
            return tf.nn.sigmoid(h4), h4

    def inference(self, input_data_A, input_data_B,batch_size=None, h=None, w=None,reuse = False):
        """
        The forward process of network.
        :param input_data:  Batch used to for training, always in size of [batch_size, h, w, 3]
        :param batch_size:  1 for evaluation and custom number for training.
        :param h: height of the image
        :param w: width of the image
        :return: The result processed by this generator
        """
        if h is None:
            h = con.FLAGS.input_image_height
        if w is None:
            w = con.FLAGS.input_image_width
        if batch_size is None:
            batch_size = con.FLAGS.batch_size
        with tf.variable_scope('generator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # else:
                # assert tf.get_variable_scope().reuse == False
            # image is (h x w x input_c_dim)
            # img_input = tf.split(input_data, 2, 3)

            e1 = tools.conv2d(input_data_A, self.base_dim,  k_h=5, k_w=5, d_h=2, d_w=2, name='gab_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(tools.conv2d(tools.lrelu(e1, leak=0), self.base_dim * 2, k_h=5, k_w=5, d_h=2, d_w=2,name='gab_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(tools.conv2d(tools.lrelu(e2, leak=0), self.base_dim * 4, k_h=3, k_w=3, d_h=2, d_w=2,name='gab_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(tools.conv2d(tools.lrelu(e3, leak=0), self.base_dim * 8, k_h=3, k_w=3, d_h=2, d_w=2,name='gab_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(tools.conv2d(tools.lrelu(e4, leak=0), self.base_dim * 8, k_h=3, k_w=3, d_h=2, d_w=2,name='gab_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(tools.conv2d(tools.lrelu(e5, leak=0), self.base_dim * 8, k_h=3, k_w=3, d_h=2, d_w=2,name='gab_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(tools.conv2d(tools.lrelu(e6, leak=0), self.base_dim * 8, k_h=3, k_w=3, d_h=2, d_w=2,name='gab_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(tools.conv2d(tools.lrelu(e7, leak=0), self.base_dim * 8, k_h=3, k_w=3, d_h=2, d_w=2,name='gab_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            eu1 = tools.conv2d(input_data_B, self.base_dim, k_h=5, k_w=5, d_h=2, d_w=2,name='gabu_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            eu2 = self.gu_bn_e2(tools.conv2d(tools.lrelu(eu1, leak=0), self.base_dim * 2,k_h=5, k_w=5, d_h=2, d_w=2, name='gabu_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            eu3 = self.gu_bn_e3(tools.conv2d(tools.lrelu(eu2, leak=0), self.base_dim * 4,k_h=3, k_w=3, d_h=2, d_w=2,name='gabu_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            eu4 = self.gu_bn_e4(tools.conv2d(tools.lrelu(eu3, leak=0), self.base_dim * 8,k_h=3, k_w=3, d_h=2, d_w=2,name='gabu_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            eu5 = self.gu_bn_e5(tools.conv2d(tools.lrelu(eu4, leak=0), self.base_dim * 8,k_h=3, k_w=3, d_h=2, d_w=2,name='gabu_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            eu6 = self.gu_bn_e6(tools.conv2d(tools.lrelu(eu5, leak=0), self.base_dim * 8,k_h=3, k_w=3, d_h=2, d_w=2,name='gabu_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            eu7 = self.gu_bn_e7(tools.conv2d(tools.lrelu(eu6, leak=0), self.base_dim * 8,k_h=3, k_w=3, d_h=2, d_w=2,name='gabu_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            eu8 = self.gu_bn_e8(tools.conv2d(tools.lrelu(eu7, leak=0), self.base_dim * 8,k_h=3, k_w=3, d_h=2, d_w=2,name='gabu_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            ef_AB = tf.concat([e8,eu8],3)

            att_d1 = tools.lrelu(self.g_bn_attd1_1(
                tools.conv2d(ef_AB, self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='g_attd1_1_conv')), leak=0)
            att_convd1_1 = tools.conv2d(e7, self.base_dim * 8, k_h=2, k_w=2, d_h=2, d_w=2, name='g_att_convd1_1')
            att_d1 = tools.conv2d(att_d1, self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='g_attd1_2_conv')
            att_add_d1 = tools.conv2d(tools.lrelu(tf.add(att_convd1_1, att_d1), leak=0), self.base_dim * 8, k_h=1,
                                      k_w=1, d_h=1, d_w=1, name='g_att_add_d1_conv')
            # att_add_d1 = tf.nn.sigmoid(att_add_d1)
            att_add_d1 = tools.lrelu(att_add_d1,leak=0)
            att_add_up_d1 = tools.deconv2d(att_add_d1,
                                           [batch_size, int((h - 1) / 128) + 1, int((w - 1) / 128) + 1, self.base_dim * 8],k_h=3, k_w=3, d_h=2, d_w=2,
                                           name='g_att_add_up_d1_conv')
            att_add_up_d1 = tf.nn.sigmoid(att_add_up_d1)
            att1 = self.g_bn_attd1_2(
                tools.conv2d(tf.multiply(e7, att_add_up_d1), self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1,
                             name='g_attd1_3_conv'))

            self.d1, self.d1_w, self.d1_b = tools.deconv2d(tools.lrelu(ef_AB, leak=0),
                                                           [batch_size, int((h - 1) / 128) + 1, int((w - 1) / 128) + 1,
                                                            self.base_dim * 8],k_h=3, k_w=3, d_h=2, d_w=2,
                                                           name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7, eu7, att1], 3)
            e9 = self.g_bn_e9(tools.conv2d(tools.lrelu(d1, leak=0.2), self.base_dim * 8, k_h=3,
                                           k_w=3, d_h=1, d_w=1, name='g_e9_conv'))
            d1 = tf.add(e7, e9)
            #  d1 is (2 x 2 x self.gf_dim*8*2)

            att_d2 = tools.lrelu(self.g_bn_attd2_1(
                tools.conv2d(d1, self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='g_attd2_1_conv')), leak=0)
            att_convd2_1 = tools.conv2d(e6, self.base_dim * 8, k_h=2, k_w=2, d_h=2, d_w=2, name='g_att_convd2_1')
            att_d2 = tools.conv2d(att_d2, self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='g_attd2_2_conv')
            att_add_d2 = tools.conv2d(tools.lrelu(tf.add(att_convd2_1, att_d2), leak=0), self.base_dim * 8, k_h=1,
                                      k_w=1, d_h=1, d_w=1, name='g_att_add_d2_conv')
            # att_add_d2 = tf.nn.sigmoid(att_add_d2)
            att_add_d2 = tools.lrelu(att_add_d2,leak=0)
            att_add_up_d2 = tools.deconv2d(att_add_d2,
                                           [batch_size, int((h - 1) / 64) + 1, int((w - 1) / 64) + 1, self.base_dim * 8],k_h=3, k_w=3, d_h=2, d_w=2,
                                           name='g_att_add_up_d2_conv')
            att_add_up_d2 = tf.nn.sigmoid(att_add_up_d2)
            att2 = self.g_bn_attd2_2(
                tools.conv2d(tf.multiply(e6, att_add_up_d2), self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1,
                             name='g_attd2_3_conv'))

            self.d2, self.d2_w, self.d2_b = tools.deconv2d(tools.lrelu(d1, leak=0),
                                                           [batch_size, int((h - 1) / 64) + 1, int((w - 1) / 64) + 1,
                                                            self.base_dim * 8],k_h=3, k_w=3, d_h=2, d_w=2,
                                                           name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6, eu6, att2], 3)
            e10 = self.g_bn_e10(tools.conv2d(tools.lrelu(d2, leak=0.2), self.base_dim * 8, k_h=3,
                                             k_w=3, d_h=1, d_w=1, name='g_e10_conv'))
            d2 = tf.add(e6, e10)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            att_d3 = tools.lrelu(self.g_bn_attd3_1(
                tools.conv2d(d2, self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='g_attd3_1_conv')), leak=0)
            att_convd3_1 = tools.conv2d(e5, self.base_dim * 8, k_h=2, k_w=2, d_h=2, d_w=2, name='g_att_convd3_1')
            att_d3 = tools.conv2d(att_d3, self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='g_attd3_2_conv')
            att_add_d3 = tools.conv2d(tools.lrelu(tf.add(att_convd3_1, att_d3), leak=0), self.base_dim * 8, k_h=1,
                                      k_w=1, d_h=1, d_w=1, name='g_att_add_d3_conv')
            # att_add_d3 = tf.nn.sigmoid(att_add_d3)
            att_add_d3 = tools.lrelu(att_add_d3,leak=0)
            att_add_up_d3 = tools.deconv2d(att_add_d3,
                                           [batch_size, int((h - 1) / 32) + 1, int((w - 1) / 32) + 1, self.base_dim * 8],k_h=3, k_w=3, d_h=2, d_w=2,
                                           name='g_att_add_up_d3_conv')
            att_add_up_d3 = tf.nn.sigmoid(att_add_up_d3)
            att3 = self.g_bn_attd3_2(
                tools.conv2d(tf.multiply(e5, att_add_up_d3), self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1,
                             name='g_attd3_3_conv'))

            self.d3, self.d3_w, self.d3_b = tools.deconv2d(tools.lrelu(d2, leak=0),
                                                           [batch_size, int((h - 1) / 32) + 1, int((w - 1) / 32) + 1,
                                                            self.base_dim * 8],k_h=3, k_w=3, d_h=2, d_w=2,
                                                           name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5, eu5, att3], 3)
            e11 = self.g_bn_e11(tools.conv2d(tools.lrelu(d3, leak=0.2), self.base_dim * 8, k_h=3,
                                             k_w=3, d_h=1, d_w=1, name='g_e11_conv'))
            d3 = tf.add(e5, e11)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            att_d4 = tools.lrelu(self.g_bn_attd4_1(
                tools.conv2d(d3, self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='g_attd4_1_conv')), leak=0)
            att_convd4_1 = tools.conv2d(e4, self.base_dim * 8, k_h=2, k_w=2, d_h=2, d_w=2, name='g_att_convd4_1')
            att_d4 = tools.conv2d(att_d4, self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='g_attd4_2_conv')
            att_add_d4 = tools.conv2d(tools.lrelu(tf.add(att_convd4_1, att_d4), leak=0), self.base_dim * 8, k_h=1,
                                      k_w=1, d_h=1, d_w=1, name='g_att_add_d4_conv')
            # att_add_d4 = tf.nn.sigmoid(att_add_d4)
            att_add_d4 = tools.lrelu(att_add_d4,leak=0)
            att_add_up_d4 = tools.deconv2d(att_add_d4,
                                           [batch_size, int((h - 1) / 16) + 1, int((w - 1) / 16) + 1, self.base_dim * 8],k_h=3, k_w=3, d_h=2, d_w=2,
                                           name='g_att_add_up_d4_conv')
            att_add_up_d4 = tf.nn.sigmoid(att_add_up_d4)
            att4 = self.g_bn_attd4_2(
                tools.conv2d(tf.multiply(e4, att_add_up_d4), self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1,
                             name='g_attd4_3_conv'))

            self.d4, self.d4_w, self.d4_b = tools.deconv2d(tools.lrelu(d3, leak=0),
                                                           [batch_size, int((h - 1) / 16) + 1, int((w - 1) / 16) + 1,
                                                            self.base_dim * 8],k_h=3, k_w=3, d_h=2, d_w=2,
                                                           name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4, eu4, att4], 3)
            e12 = self.g_bn_e12(tools.conv2d(tools.lrelu(d4, leak=0.2), self.base_dim * 8, k_h=3,
                                             k_w=3, d_h=1, d_w=1, name='g_e12_conv'))
            d4 = tf.add(e4, e12)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            att_d5 = tools.lrelu(self.g_bn_attd5_1(
                tools.conv2d(d4, self.base_dim * 4, k_h=1, k_w=1, d_h=1, d_w=1, name='g_attd5_1_conv')), leak=0)
            att_convd5_1 = tools.conv2d(e3, self.base_dim * 4, k_h=2, k_w=2, d_h=2, d_w=2, name='g_att_convd5_1')
            att_d5 = tools.conv2d(att_d5, self.base_dim * 4, k_h=1, k_w=1, d_h=1, d_w=1, name='g_attd5_2_conv')
            att_add_d5 = tools.conv2d(tools.lrelu(tf.add(att_convd5_1, att_d5), leak=0), self.base_dim * 4, k_h=1,
                                      k_w=1, d_h=1, d_w=1, name='g_att_add_d5_conv')
            # att_add_d5 = tf.nn.sigmoid(att_add_d5)
            att_add_d5 = tools.lrelu(att_add_d5,leak=0)
            att_add_up_d5 = tools.deconv2d(att_add_d5,
                                           [batch_size, int((h - 1) / 8) + 1, int((w - 1) / 8) + 1, self.base_dim * 4],k_h=3, k_w=3, d_h=2, d_w=2,
                                           name='g_att_add_up_d5_conv')
            att_add_up_d5 = tf.nn.sigmoid(att_add_up_d5)
            att5 = self.g_bn_attd5_2(
                tools.conv2d(tf.multiply(e3, att_add_up_d5), self.base_dim * 4, k_h=1, k_w=1, d_h=1, d_w=1,
                             name='g_attd5_3_conv'))

            self.d5, self.d5_w, self.d5_b = tools.deconv2d(tools.lrelu(d4, leak=0),
                                                           [batch_size, int((h - 1) / 8) + 1, int((w - 1) / 8) + 1,
                                                            self.base_dim * 4],k_h=3, k_w=3, d_h=2, d_w=2,
                                                           name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3, eu3, att5], 3)
            e13 = self.g_bn_e13(tools.conv2d(tools.lrelu(d5, leak=0.2), self.base_dim * 4, k_h=3,
                                             k_w=3, d_h=1, d_w=1, name='g_e13_conv'))
            d5 = tf.add(e3, e13)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            att_d6 = tools.lrelu(self.g_bn_attd6_1(
                tools.conv2d(d5, self.base_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1, name='g_attd6_1_conv')), leak=0)
            att_convd6_1 = tools.conv2d(e2, self.base_dim * 2, k_h=2, k_w=2, d_h=2, d_w=2, name='g_att_convd6_1')
            att_d6 = tools.conv2d(att_d6, self.base_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1, name='g_attd6_2_conv')
            att_add_d6 = tools.conv2d(tools.lrelu(tf.add(att_convd6_1, att_d6), leak=0), self.base_dim * 2, k_h=1,
                                      k_w=1, d_h=1, d_w=1, name='g_att_add_d6_conv')
            # att_add_d6 = tf.nn.sigmoid(att_add_d6)
            att_add_d6 = tools.lrelu(att_add_d6,leak=0)
            att_add_up_d6 = tools.deconv2d(att_add_d6,
                                           [batch_size, int((h - 1) / 4) + 1, int((w - 1) / 4) + 1, self.base_dim * 2],k_h=3, k_w=3, d_h=2, d_w=2,
                                           name='g_att_add_up_d6_conv')
            att_add_up_d6 = tf.nn.sigmoid(att_add_up_d6)
            att6 = self.g_bn_attd6_2(
                tools.conv2d(tf.multiply(e2, att_add_up_d6), self.base_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1,
                             name='g_attd6_3_conv'))

            self.d6, self.d6_w, self.d6_b = tools.deconv2d(tools.lrelu(d5, leak=0),
                                                           [batch_size, int((h - 1) / 4) + 1, int((w - 1) / 4) + 1,
                                                            self.base_dim * 2],k_h=3, k_w=3, d_h=2, d_w=2,
                                                           name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.nn.dropout(d6, 0.5)
            d6 = tf.concat([d6, e2, eu2, att6], 3)
            e14 = self.g_bn_e14(tools.conv2d(tools.lrelu(d6, leak=0.2), self.base_dim * 2, k_h=3,
                                             k_w=3, d_h=1, d_w=1, name='g_e14_conv'))
            d6 = tf.add(e2, e14)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            att_d7 = tools.lrelu(self.g_bn_attd7_1(
                tools.conv2d(d6, self.base_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='g_attd7_1_conv')), leak=0)
            att_convd7_1 = tools.conv2d(e1, self.base_dim, k_h=2, k_w=2, d_h=2, d_w=2, name='g_att_convd7_1')
            att_d7 = tools.conv2d(att_d7, self.base_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='g_attd7_2_conv')
            att_add_d7 = tools.conv2d(tools.lrelu(tf.add(att_convd7_1, att_d7), leak=0), self.base_dim, k_h=1,
                                      k_w=1, d_h=1, d_w=1, name='g_att_add_d7_conv')
            # att_add_d7 = tf.nn.sigmoid(att_add_d7)
            att_add_d7 = tools.lrelu(att_add_d7,leak=0)
            att_add_up_d7 = tools.deconv2d(att_add_d7,
                                           [batch_size, int((h - 1) / 2) + 1, int((w - 1) / 2) + 1,
                                            self.base_dim], k_h=5, k_w=5, d_h=2, d_w=2,name='g_att_add_up_d7_conv')
            att_add_up_d7 = tf.nn.sigmoid(att_add_up_d7)
            att7 = self.g_bn_attd7_2(
                tools.conv2d(tf.multiply(e1, att_add_up_d7), self.base_dim, k_h=1, k_w=1, d_h=1, d_w=1,
                             name='g_attd7_3_conv'))

            self.d7, self.d7_w, self.d7_b = tools.deconv2d(tools.lrelu(d6, leak=0),
                                                           [batch_size, int((h - 1) / 2) + 1, int((w - 1) / 2) + 1,
                                                            self.base_dim],k_h=5, k_w=5, d_h=2, d_w=2,
                                                           name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1, eu1, att7], 3)
            e15 = self.g_bn_e15(tools.conv2d(tools.lrelu(d7, leak=0.2), self.base_dim, k_h=3,
                                             k_w=3, d_h=1, d_w=1, name='g_e15_conv'))
            d7 = tf.add(e1, e15)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            att_d8 = tools.lrelu(self.g_bn_attd8_1(
                tools.conv2d(d7, 3, k_h=1, k_w=1, d_h=1, d_w=1, name='g_attd8_1_conv')), leak=0)
            att_convd8_1 = tools.conv2d(input_data_A, 3, k_h=2, k_w=2, d_h=2, d_w=2, name='g_att_convd8_1')
            att_d8 = tools.conv2d(att_d8, 3, k_h=1, k_w=1, d_h=1, d_w=1, name='g_attd8_2_conv')
            att_add_d8 = tools.conv2d(tools.lrelu(tf.add(att_convd8_1, att_d8), leak=0), 3, k_h=1,
                                      k_w=1, d_h=1, d_w=1, name='g_att_add_d8_conv')
            # att_add_d8 = tf.nn.sigmoid(att_add_d8)
            att_add_d8 = tools.lrelu(att_add_d8,leak=0)
            att_add_up_d8 = tools.deconv2d(att_add_d8,
                                           [batch_size, int((h + 0) / 1), int((w + 0) / 1),
                                            3], k_h=5, k_w=5, d_h=2, d_w=2,name='g_att_add_up_d8_conv')
            att_add_up_d8 = tf.nn.sigmoid(att_add_up_d8)
            att8 = self.g_bn_attd8_2(
                tools.conv2d(tf.multiply(input_data_A, att_add_up_d8), 3, k_h=1, k_w=1, d_h=1, d_w=1,
                             name='g_attd8_3_conv'))

            self.d8, self.d8_w, self.d8_b = tools.deconv2d(tools.lrelu(d7, leak=0),
                                                           [batch_size, h, w, self.base_dim], k_h=5, k_w=5, d_h=2, d_w=2,name='g_d8', with_w=True)
            d8 = self.g_bn_d8(self.d8)
            d8 = tf.concat([d8, input_data_A, input_data_B, att8], 3)
            self.e16 = self.g_bn_e16(tools.conv2d(tools.lrelu(d8, leak=0), self.base_dim, k_h=3, k_w=3,
                                                  d_h=1, d_w=1, name="g_e16_conv"))

            d9_1 = self.g_bn_e17_1(tools.conv2d(tools.lrelu(self.e16, leak=0.2), 3, k_h=1, k_w=1,
                                                 d_h=1, d_w=1, name="g_e17_1_conv"))
            d9_1 = tf.add(input_data_A,d9_1)

            self.super_result1 = tools.lrelu(d9_1,leak=0)
            # self.super_result1 = tf.nn.sigmoid(d9_1)

            att_du1 = tools.lrelu(self.gu_bn_attd1_1(
                tools.conv2d(ef_AB, self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='gu_attd1_1_conv')), leak=0)
            att_convdu1_1 = tools.conv2d(eu7, self.base_dim * 8, k_h=2, k_w=2, d_h=2, d_w=2, name='gu_att_convd1_1')
            att_du1 = tools.conv2d(att_du1, self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='gu_attd1_2_conv')
            att_add_du1 = tools.conv2d(tools.lrelu(tf.add(att_convdu1_1, att_du1), leak=0), self.base_dim * 8, k_h=1,
                                       k_w=1, d_h=1, d_w=1, name='gu_att_add_d1_conv')
            # att_add_du1 = tf.nn.sigmoid(att_add_du1)
            att_add_du1 = tools.lrelu(att_add_du1,leak=0)
            att_add_up_du1 = tools.deconv2d(att_add_du1,
                                            [batch_size, int((h - 1) / 128) + 1, int((w - 1) / 128) + 1, self.base_dim * 8],
                                            k_h=3, k_w=3, d_h=2, d_w=2,name='gu_att_add_up_d1_conv')
            att_add_up_du1 = tf.nn.sigmoid(att_add_up_du1)
            attu1 = self.gu_bn_attd1_2(
                tools.conv2d(tf.multiply(eu7, att_add_up_du1), self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1,
                             name='g_attdu1_3_conv'))

            self.du1, self.du1_w, self.du1_b = tools.deconv2d(tools.lrelu(ef_AB, leak=0),
                                                              [batch_size, int((h - 1) / 128) + 1, int((w - 1) / 128) + 1,
                                                               self.base_dim * 8],k_h=3, k_w=3, d_h=2, d_w=2,
                                                              name='gu_d1', with_w=True)
            du1 = tf.nn.dropout(self.gu_bn_d1(self.du1), 0.5)
            du1 = tf.concat([du1, e7, eu7, attu1], 3)
            eu9 = self.gu_bn_e9(tools.conv2d(tools.lrelu(du1, leak=0.2), self.base_dim * 8, k_h=3,
                                             k_w=3, d_h=1, d_w=1, name='gu_e9_conv'))
            du1 = tf.add(eu7, eu9)
            #  d1 is (2 x 2 x self.gf_dim*8*2)

            att_du2 = tools.lrelu(self.gu_bn_attd2_1(
                tools.conv2d(du1, self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='gu_attd2_1_conv')), leak=0)
            att_convdu2_1 = tools.conv2d(eu6, self.base_dim * 8, k_h=2, k_w=2, d_h=2, d_w=2, name='gu_att_convd2_1')
            att_du2 = tools.conv2d(att_du2, self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='gu_attd2_2_conv')
            att_add_du2 = tools.conv2d(tools.lrelu(tf.add(att_convdu2_1, att_du2), leak=0), self.base_dim * 8, k_h=1,
                                       k_w=1, d_h=1, d_w=1, name='gu_att_add_d2_conv')
            # att_add_du2 = tf.nn.sigmoid(att_add_du2)
            att_add_du2 = tools.lrelu(att_add_du2,leak=0)
            att_add_up_du2 = tools.deconv2d(att_add_du2,
                                            [batch_size, int((h - 1) / 64) + 1, int((w - 1) / 64) + 1, self.base_dim * 8],
                                            k_h=3, k_w=3, d_h=2, d_w=2,name='gu_att_add_up_d2_conv')
            att_add_up_du2 = tf.nn.sigmoid(att_add_up_du2)
            attu2 = self.gu_bn_attd2_2(
                tools.conv2d(tf.multiply(eu6, att_add_up_du2), self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1,
                             name='gu_attd2_3_conv'))

            self.du2, self.du2_w, self.du2_b = tools.deconv2d(tools.lrelu(du1, leak=0),
                                                              [batch_size, int((h - 1) / 64) + 1, int((w - 1) / 64) + 1,
                                                               self.base_dim * 8],k_h=3, k_w=3, d_h=2, d_w=2,
                                                              name='gu_d2', with_w=True)
            du2 = tf.nn.dropout(self.gu_bn_d2(self.du2), 0.5)
            du2 = tf.concat([du2, e6, eu6, attu2], 3)
            eu10 = self.gu_bn_e10(tools.conv2d(tools.lrelu(du2, leak=0.2), self.base_dim * 8, k_h=3,
                                               k_w=3, d_h=1, d_w=1, name='gu_e10_conv'))
            du2 = tf.add(eu6, eu10)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            att_du3 = tools.lrelu(self.gu_bn_attd3_1(
                tools.conv2d(du2, self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='gu_attd3_1_conv')), leak=0)
            att_convdu3_1 = tools.conv2d(eu5, self.base_dim * 8, k_h=2, k_w=2, d_h=2, d_w=2, name='gu_att_convd3_1')
            att_du3 = tools.conv2d(att_du3, self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='gu_attd3_2_conv')
            att_add_du3 = tools.conv2d(tools.lrelu(tf.add(att_convdu3_1, att_du3), leak=0), self.base_dim * 8, k_h=1,
                                       k_w=1, d_h=1, d_w=1, name='gu_att_add_d3_conv')
            # att_add_du3 = tf.nn.sigmoid(att_add_du3)
            att_add_du3 = tools.lrelu(att_add_du3,leak=0)
            att_add_up_du3 = tools.deconv2d(att_add_du3,
                                            [batch_size, int((h - 1) / 32) + 1, int((w - 1) / 32) + 1, self.base_dim * 8],
                                            k_h=3, k_w=3, d_h=2, d_w=2,name='gu_att_add_up_d3_conv')
            att_add_up_du3 = tf.nn.sigmoid(att_add_up_du3)
            attu3 = self.gu_bn_attd3_2(
                tools.conv2d(tf.multiply(eu5, att_add_up_du3), self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1,
                             name='gu_attd3_3_conv'))

            self.du3, self.du3_w, self.du3_b = tools.deconv2d(tools.lrelu(du2, leak=0),
                                                              [batch_size, int((h - 1) / 32) + 1, int((w - 1) / 32) + 1,
                                                               self.base_dim * 8],k_h=3, k_w=3, d_h=2, d_w=2,
                                                              name='gu_d3', with_w=True)
            du3 = tf.nn.dropout(self.gu_bn_d3(self.du3), 0.5)
            du3 = tf.concat([du3, e5, eu5, attu3], 3)
            eu11 = self.gu_bn_e11(tools.conv2d(tools.lrelu(du3, leak=0.2), self.base_dim * 8, k_h=3,
                                               k_w=3, d_h=1, d_w=1, name='gu_e11_conv'))
            du3 = tf.add(eu5, eu11)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            att_du4 = tools.lrelu(self.gu_bn_attd4_1(
                tools.conv2d(du3, self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='gu_attd4_1_conv')), leak=0)
            att_convdu4_1 = tools.conv2d(eu4, self.base_dim * 8, k_h=2, k_w=2, d_h=2, d_w=2, name='gu_att_convd4_1')
            att_du4 = tools.conv2d(att_du4, self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='gu_attd4_2_conv')
            att_add_du4 = tools.conv2d(tools.lrelu(tf.add(att_convdu4_1, att_du4), leak=0), self.base_dim * 8, k_h=1,
                                       k_w=1, d_h=1, d_w=1, name='gu_att_add_d4_conv')
            # att_add_du4 = tf.nn.sigmoid(att_add_du4)
            att_add_du4 = tools.lrelu(att_add_du4,leak=0)
            att_add_up_du4 = tools.deconv2d(att_add_du4,
                                            [batch_size, int((h - 1) / 16) + 1, int((w - 1) / 16) + 1, self.base_dim * 8],
                                            k_h=3, k_w=3, d_h=2, d_w=2,name='gu_att_add_up_d4_conv')
            att_add_up_du4 = tf.nn.sigmoid(att_add_up_du4)
            attu4 = self.gu_bn_attd4_2(
                tools.conv2d(tf.multiply(eu4, att_add_up_du4), self.base_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1,
                             name='gu_attd4_3_conv'))

            self.du4, self.du4_w, self.du4_b = tools.deconv2d(tools.lrelu(du3, leak=0),
                                                              [batch_size, int((h - 1) / 16) + 1, int((w - 1) / 16) + 1,
                                                               self.base_dim * 8],k_h=3, k_w=3, d_h=2, d_w=2,
                                                              name='gu_d4', with_w=True)
            du4 = self.gu_bn_d4(self.du4)
            du4 = tf.concat([du4, e4, eu4, attu4], 3)
            eu12 = self.gu_bn_e12(tools.conv2d(tools.lrelu(du4, leak=0.2), self.base_dim * 8, k_h=3,
                                               k_w=3, d_h=1, d_w=1, name='gu_e12_conv'))
            du4 = tf.add(eu4, eu12)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            att_du5 = tools.lrelu(self.gu_bn_attd5_1(
                tools.conv2d(du4, self.base_dim * 4, k_h=1, k_w=1, d_h=1, d_w=1, name='gu_attd5_1_conv')), leak=0)
            att_convdu5_1 = tools.conv2d(eu3, self.base_dim * 4, k_h=2, k_w=2, d_h=2, d_w=2, name='gu_att_convd5_1')
            att_du5 = tools.conv2d(att_du5, self.base_dim * 4, k_h=1, k_w=1, d_h=1, d_w=1, name='gu_attd5_2_conv')
            att_add_du5 = tools.conv2d(tools.lrelu(tf.add(att_convdu5_1, att_du5), leak=0), self.base_dim * 4, k_h=1,
                                       k_w=1, d_h=1, d_w=1, name='gu_att_add_d5_conv')
            # att_add_du5 = tf.nn.sigmoid(att_add_du5)
            att_add_du5 = tools.lrelu(att_add_du5,leak=0)
            att_add_up_du5 = tools.deconv2d(att_add_du5,
                                            [batch_size, int((h - 1) / 8) + 1, int((w - 1) / 8) + 1, self.base_dim * 4],
                                            k_h=3, k_w=3, d_h=2, d_w=2,name='gu_att_add_up_d5_conv')
            att_add_up_du5 = tf.nn.sigmoid(att_add_up_du5)
            attu5 = self.gu_bn_attd5_2(
                tools.conv2d(tf.multiply(eu3, att_add_up_du5), self.base_dim * 4, k_h=1, k_w=1, d_h=1, d_w=1,
                             name='gu_attd5_3_conv'))

            self.du5, self.du5_w, self.du5_b = tools.deconv2d(tools.lrelu(du4, leak=0),
                                                              [batch_size, int((h - 1) / 8) + 1, int((w - 1) / 8) + 1,
                                                               self.base_dim * 4],k_h=3, k_w=3, d_h=2, d_w=2,
                                                              name='gu_d5', with_w=True)
            du5 = self.gu_bn_d5(self.du5)
            du5 = tf.concat([du5, e3, eu3, attu5], 3)
            eu13 = self.gu_bn_e13(tools.conv2d(tools.lrelu(du5, leak=0.2), self.base_dim * 4, k_h=3,
                                               k_w=3, d_h=1, d_w=1, name='gu_e13_conv'))
            du5 = tf.add(eu3, eu13)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            att_du6 = tools.lrelu(self.gu_bn_attd6_1(
                tools.conv2d(du5, self.base_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1, name='gu_attd6_1_conv')), leak=0)
            att_convdu6_1 = tools.conv2d(eu2, self.base_dim * 2, k_h=2, k_w=2, d_h=2, d_w=2, name='gu_att_convd6_1')
            att_du6 = tools.conv2d(att_du6, self.base_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1, name='gu_attd6_2_conv')
            att_add_du6 = tools.conv2d(tools.lrelu(tf.add(att_convdu6_1, att_du6), leak=0), self.base_dim * 2, k_h=1,
                                       k_w=1, d_h=1, d_w=1, name='gu_att_add_d6_conv')
            # att_add_du6 = tf.nn.sigmoid(att_add_du6)
            att_add_du6 = tools.lrelu(att_add_du6,leak=0)
            att_add_up_du6 = tools.deconv2d(att_add_du6,
                                            [batch_size, int((h - 1) / 4)  + 1, int((w - 1) / 4) + 1, self.base_dim * 2],
                                            k_h=3, k_w=3, d_h=2, d_w=2,name='gu_att_add_up_d6_conv')
            att_add_up_du6 = tf.nn.sigmoid(att_add_up_du6)
            attu6 = self.gu_bn_attd6_2(
                tools.conv2d(tf.multiply(eu2, att_add_up_du6), self.base_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1,
                             name='gu_attd6_3_conv'))

            self.du6, self.du6_w, self.du6_b = tools.deconv2d(tools.lrelu(du5, leak=0),
                                                              [batch_size, int((h - 1) / 4) + 1, int((w - 1) / 4) + 1,
                                                               self.base_dim * 2],k_h=3, k_w=3, d_h=2, d_w=2,
                                                              name='gu_d6', with_w=True)
            du6 = self.gu_bn_d6(self.du6)
            du6 = tf.nn.dropout(du6, 0.5)
            du6 = tf.concat([du6, e2, eu2, attu6], 3)
            eu14 = self.gu_bn_e14(tools.conv2d(tools.lrelu(du6, leak=0.2), self.base_dim * 2, k_h=3,
                                               k_w=3, d_h=1, d_w=1, name='gu_e14_conv'))
            du6 = tf.add(eu2, eu14)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            att_du7 = tools.lrelu(self.gu_bn_attd7_1(
                tools.conv2d(du6, self.base_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='gu_attd7_1_conv')), leak=0)
            att_convdu7_1 = tools.conv2d(eu1, self.base_dim, k_h=2, k_w=2, d_h=2, d_w=2, name='gu_att_convd7_1')
            att_du7 = tools.conv2d(att_du7, self.base_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='gu_attd7_2_conv')
            att_add_du7 = tools.conv2d(tools.lrelu(tf.add(att_convdu7_1, att_du7), leak=0), self.base_dim, k_h=1,
                                       k_w=1, d_h=1, d_w=1, name='gu_att_add_d7_conv')
            # att_add_du7 = tf.nn.sigmoid(att_add_du7)
            att_add_du7 = tools.lrelu(att_add_du7,leak=0)
            att_add_up_du7 = tools.deconv2d(att_add_du7,
                                            [batch_size, int((h - 1) / 2) + 1, int((w - 1) / 2) + 1,
                                             self.base_dim],k_h=5, k_w=5, d_h=2, d_w=2, name='gu_att_add_up_d7_conv')
            att_add_up_du7 = tf.nn.sigmoid(att_add_up_du7)
            attu7 = self.gu_bn_attd7_2(
                tools.conv2d(tf.multiply(eu1, att_add_up_du7), self.base_dim, k_h=1, k_w=1, d_h=1, d_w=1,
                             name='gu_attd7_3_conv'))

            self.du7, self.du7_w, self.du7_b = tools.deconv2d(tools.lrelu(du6, leak=0),
                                                              [batch_size, int((h - 1) / 2) + 1, int((w - 1) / 2) + 1,
                                                               self.base_dim],k_h=5, k_w=5, d_h=2, d_w=2,
                                                              name='gu_d7', with_w=True)
            du7 = self.gu_bn_d7(self.du7)
            du7 = tf.concat([du7, e1, eu1, attu7], 3)
            eu15 = self.gu_bn_e15(tools.conv2d(tools.lrelu(du7, leak=0.2), self.base_dim, k_h=3,
                                               k_w=3, d_h=1, d_w=1, name='gu_e15_conv'))
            du7 = tf.add(eu1, eu15)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            att_du8 = tools.lrelu(self.gu_bn_attd8_1(
                tools.conv2d(du7, 3, k_h=1, k_w=1, d_h=1, d_w=1, name='gu_attd8_1_conv')), leak=0)
            att_convdu8_1 = tools.conv2d(input_data_B, 3, k_h=2, k_w=2, d_h=2, d_w=2,
                                         name='gu_att_convd8_1')
            att_du8 = tools.conv2d(att_du8, 3, k_h=1, k_w=1, d_h=1, d_w=1, name='gu_attd8_2_conv')
            att_add_du8 = tools.conv2d(tools.lrelu(tf.add(att_convdu8_1, att_du8), leak=0), 3, k_h=1,
                                       k_w=1, d_h=1, d_w=1, name='gu_att_add_d8_conv')
            # att_add_du8 = tf.nn.sigmoid(att_add_du8)
            att_add_du8 = tools.lrelu(att_add_du8,leak=0)
            att_add_up_du8 = tools.deconv2d(att_add_du8,
                                            [batch_size, int((h + 0) / 1), int((w + 0) / 1),
                                             3], k_h=5, k_w=5, d_h=2, d_w=2,name='gu_att_add_up_d8_conv')
            att_add_up_du8 = tf.nn.sigmoid(att_add_up_du8)
            attu8 = self.gu_bn_attd8_2(
                tools.conv2d(tf.multiply(input_data_B, att_add_up_du8), 3, k_h=1, k_w=1, d_h=1, d_w=1,
                             name='gu_attd8_3_conv'))

            self.du8, self.du8_w, self.du8_b = tools.deconv2d(tools.lrelu(du7, leak=0),
                                                              [batch_size, h, w, self.base_dim], k_h=5, k_w=5, d_h=2, d_w=2, name='gu_d8',
                                                              with_w=True)
            du8 = self.gu_bn_d8(self.du8)
            du8 = tf.concat([du8, input_data_A, input_data_B, attu8], 3)
            self.eu16 = self.gu_bn_e16(tools.conv2d(tools.lrelu(du8, leak=0), self.base_dim, k_h=3, k_w=3,
                                                    d_h=1, d_w=1, name="gu_e16_conv"))

            d9_2 = self.gu_bn_e17_2(tools.conv2d(tools.lrelu(self.eu16, leak=0.2), 3, k_h=1, k_w=1,
                                                  d_h=1, d_w=1, name="gu_e17_2_conv"))
            d9_2 = tf.add(input_data_B, d9_2)

            self.super_result2 = tools.lrelu(d9_2,leak=0)
            # self.super_result2 = tf.nn.sigmoid(d9_2)

            super_add = tf.add(self.super_result1,self.super_result2)
            super_concat = tf.concat([self.super_result1, self.super_result2, super_add], 3)


            att_add_du9_1 = tools.conv2d(tools.lrelu(super_concat, leak=0), 3, k_h=1,
                                       k_w=1, d_h=1, d_w=1, name='gf_att_add_d9_1_conv')
            att_add_up_du9_1 = tf.nn.sigmoid(att_add_du9_1)
            attu9_1 = self.gf_bn_attd9_1(
                tools.conv2d(tf.multiply(self.super_result1, att_add_up_du9_1), 3, k_h=1, k_w=1, d_h=1, d_w=1,
                             name='gf_attd9_1_conv'))

            att_add_du9_2 = tools.conv2d(tools.lrelu(super_concat, leak=0), 3, k_h=1,
                                         k_w=1, d_h=1, d_w=1, name='gf_att_add_d9_2_conv')
            att_add_up_du9_2 = tf.nn.sigmoid(att_add_du9_2)
            attu9_2 = self.gf_bn_attd9_2(
                tools.conv2d(tf.multiply(self.super_result2, att_add_up_du9_2), 3, k_h=1, k_w=1, d_h=1, d_w=1,
                             name='gf_attd9_2_conv'))

            super_concat = tf.concat([self.super_result1,  attu9_1, self.super_result2, attu9_2], 3)

            self.eu18 = self.gf_bn_ef_18_1(tools.conv2d(tools.lrelu(super_concat, leak=0), self.base_dim, k_h=3, k_w=3,
                                                    d_h=1, d_w=1, name="gf_bn_ef_18_1_conv"))
            d10 = self.gf_bn_ef_18_2(tools.conv2d(tools.lrelu(self.eu18, leak=0), 3, k_h=1, k_w=1,
                                                 d_h=1, d_w=1, name="gf_bn_ef_18_2_conv"))
            # d10 = tf.add(0.5*super_add,0.5*d10)
            # self.super_result = tf.nn.sigmoid(d10)
            self.super_result = tools.lrelu(d10,leak=0)

            return self.super_result1,self.super_result2,self.super_result
