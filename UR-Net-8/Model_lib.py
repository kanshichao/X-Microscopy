import tensorflow as tf
import constant as con
import opts as tools

class pix2pix_generator(object):
    def __init__(self):
        self.base_dim = 64
        self.d_bn1 = tools.batch_norm(name='d_bn1')
        self.d_bn2 = tools.batch_norm(name='d_bn2')
        self.d_bn3 = tools.batch_norm(name='d_bn3')

        self.g_bn_e0 = tools.batch_norm(name='g_bn_e0')
        self.g_bn_e1 = tools.batch_norm(name='g_bn_e1')
        self.g_bn_e2 = tools.batch_norm(name='g_bn_e2')
        self.g_bn_e3 = tools.batch_norm(name='g_bn_e3')
        self.g_bn_e4 = tools.batch_norm(name='g_bn_e4')
        self.g_bn_e5 = tools.batch_norm(name='g_bn_e5')
        self.g_bn_e6 = tools.batch_norm(name='g_bn_e6')
        self.g_bn_e7 = tools.batch_norm(name='g_bn_e7')
        self.g_bn_e8 = tools.batch_norm(name='g_bn_e8')
        self.g_bn_e9 = tools.batch_norm(name='g_bn_e9')
        self.g_bn_e10 = tools.batch_norm(name='g_bn_e10')
        self.g_bn_e11 = tools.batch_norm(name='g_bn_e11')
        self.g_bn_e12 = tools.batch_norm(name='g_bn_e12')
        self.g_bn_e13 = tools.batch_norm(name='g_bn_e13')
        self.g_bn_e14 = tools.batch_norm(name='g_bn_e14')
        self.g_bn_e15 = tools.batch_norm(name='g_bn_e15')
        self.g_bn_e16 = tools.batch_norm(name='g_bn_e16')

        self.g_bn_d1 = tools.batch_norm(name='g_bn_d1')
        self.g_bn_d2 = tools.batch_norm(name='g_bn_d2')
        self.g_bn_d3 = tools.batch_norm(name='g_bn_d3')
        self.g_bn_d4 = tools.batch_norm(name='g_bn_d4')
        self.g_bn_d5 = tools.batch_norm(name='g_bn_d5')
        self.g_bn_d6 = tools.batch_norm(name='g_bn_d6')
        self.g_bn_d7 = tools.batch_norm(name='g_bn_d7')
        self.g_bn_d8 = tools.batch_norm(name='g_bn_d8')

    def discriminator(self, input_data, batch_size=None, reuse=False):

        if batch_size is None:
            batch_size = con.FLAGS.batch_size
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = tools.lrelu(tools.conv2d(input_data, self.base_dim, name='d_h0_conv'),leak=0)
            # h0 is (128 x 128 x self.df_dim)
            h1 = tools.lrelu(self.d_bn1(tools.conv2d(h0, self.base_dim*2, name='d_h1_conv')),leak=0)
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = tools.lrelu(self.d_bn2(tools.conv2d(h1, self.base_dim*4, name='d_h2_conv')),leak=0)
            # h2 is (32x 32 x self.df_dim*4)
            h3 = tools.lrelu(self.d_bn3(tools.conv2d(h2, self.base_dim*8, d_h=1, d_w=1, name='d_h3_conv')),leak=0)
            # h3 is (16 x 16 x self.df_dim*8)
            h3 = tools.spp_layer(h3, levels=4, name='SPP_layer', pool_type='max_pool')
            h4 = tools.linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4


    def inference(self, input_data, batch_size=None, h=None, w=None,reuse = False):
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
            # self.e0 = self.g_bn_e0(tools.conv2d(input_data, 3,  k_h=3, k_w=3, d_h=1, d_w=1,  name='g_e0_conv'))

            e1 = tools.conv2d(input_data, self.base_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(tools.conv2d(tools.lrelu(e1,leak=0), self.base_dim * 2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(tools.conv2d(tools.lrelu(e2,leak=0), self.base_dim * 4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(tools.conv2d(tools.lrelu(e3,leak=0), self.base_dim * 8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(tools.conv2d(tools.lrelu(e4,leak=0), self.base_dim * 8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(tools.conv2d(tools.lrelu(e5,leak=0), self.base_dim * 8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(tools.conv2d(tools.lrelu(e6,leak=0), self.base_dim * 8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(tools.conv2d(tools.lrelu(e7,leak=0), self.base_dim * 8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = tools.deconv2d(tools.lrelu(e8,leak=0),
                    [batch_size, int((h - 1)/128) + 1,int((w - 1)/128) + 1, self.base_dim * 8],
                    name='g_d1',with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            e9 = self.g_bn_e9(tools.conv2d(tools.lrelu(d1,leak=0.2), self.base_dim * 8, k_h=3,
                                           k_w=3, d_h=1, d_w=1, name='g_e9_conv'))
            d1 = tf.add(e7, e9)
            #  d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = tools.deconv2d(tools.lrelu(d1,leak=0),
                    [batch_size, int((h - 1)/64) + 1, int((w - 1)/64) + 1, self.base_dim * 8],
                    name='g_d2',with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            e10 = self.g_bn_e10(tools.conv2d(tools.lrelu(d2,leak=0.2), self.base_dim * 8, k_h=3,
                                             k_w=3, d_h=1, d_w=1, name='g_e10_conv'))
            d2 = tf.add(e6, e10)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = tools.deconv2d(tools.lrelu(d2,leak=0),
                    [batch_size, int((h - 1)/32) + 1, int((w - 1)/32) + 1, self.base_dim * 8],
                    name='g_d3',with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            e11 = self.g_bn_e11(tools.conv2d(tools.lrelu(d3,leak=0.2), self.base_dim * 8, k_h=3,
                                             k_w=3, d_h=1, d_w=1, name='g_e11_conv'))
            d3 = tf.add(e5, e11)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = tools.deconv2d(tools.lrelu(d3,leak=0),
                    [batch_size, int((h - 1)/16) + 1, int((w - 1)/16) + 1, self.base_dim * 8],
                    name='g_d4',with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            e12 = self.g_bn_e12(tools.conv2d(tools.lrelu(d4,leak=0.2), self.base_dim * 8, k_h=3,
                                             k_w=3, d_h=1, d_w=1, name='g_e12_conv'))
            d4 = tf.add(e4, e12)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = tools.deconv2d(tools.lrelu(d4,leak=0),
                    [batch_size, int((h - 1)/8) + 1, int((w - 1)/8) + 1, self.base_dim * 4],
                    name='g_d5',with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            e13 = self.g_bn_e13(tools.conv2d(tools.lrelu(d5,leak=0.2), self.base_dim * 4, k_h=3,
                                             k_w=3, d_h=1, d_w=1, name='g_e13_conv'))
            d5 = tf.add(e3, e13)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = tools.deconv2d(tools.lrelu(d5,leak=0),
                    [batch_size, int((h - 1)/4) + 1, int((w -1)/4) + 1, self.base_dim * 2],
                    name='g_d6',with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            e14 = self.g_bn_e14(tools.conv2d(tools.lrelu(d6,leak=0.2), self.base_dim * 2, k_h=3,
                                             k_w=3, d_h=1, d_w=1, name='g_e14_conv'))
            d6 = tf.add(e2, e14)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = tools.deconv2d(tools.lrelu(d6,leak=0),
                    [batch_size, int((h - 1)/2) + 1, int((w - 1)/2) + 1, self.base_dim],
                    name='g_d7',with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            e15 = self.g_bn_e15(tools.conv2d(tools.lrelu(d7,leak=0.2), self.base_dim, k_h=3,
                                             k_w=3,d_h=1, d_w=1, name='g_e15_conv'))
            d7 = tf.add(e1, e15)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = tools.deconv2d(tools.lrelu(d7,leak=0),
                    [batch_size, h, w, self.base_dim], name='g_d8',with_w=True)
            d8 = self.g_bn_d8(self.d8)
            d8 = tf.concat([d8, input_data], 3)
            self.e16 = self.g_bn_e16(tools.conv2d(tools.lrelu(d8,leak=0.2), 3, k_h=3, k_w=3,
                                                  d_h=1, d_w=1, name="g_e16_conv"))
            self.d9 = tf.add(input_data, self.e16)
            # self.d9 = tools.conv2d(self.e16, 3,  k_h=3, k_w=3, d_h=1, d_w=1, name='g_e17_conv')
            return tools.lrelu(self.d9, leak=0.0), -self.e16
            # return tf.nn.sigmoid(self.d9), -self.e16

