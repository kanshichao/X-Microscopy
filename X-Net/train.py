#encoding=utf-8
from __future__ import division
import time
from glob import glob
from six.moves import xrange
from utils import *
from evaluate import *
import os
import opts as tools
from logger import setup_logger
import cv2

class pix2pix(object):
    def __init__(self, sess,pix_model,h = 968, w = 774,batch_size = 1,
                 base_dim=64, L1_lambda=100,
                 input_c_dim=3, output_c_dim=3, dataset_name='haze',
                 checkpoint_dir=None,withbn = True,base_count = 0, Model = 'my', logger=None,logger_val=None,logger_best=None,logger_test=None,best_checkpoint_dir = None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.h = h
        self.w = w
        self.base_dim = base_dim
        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
        self.L1_lambda = L1_lambda
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.withbn = withbn
        self.checkpoint_dir = checkpoint_dir
        self.dataset_name = dataset_name
        self.base_count = base_count
        self.pix_model = pix_model
        self.Model = Model
        self.build_model()
        self.logger = logger
        self.logger_val = logger_val
        self.best_ssim = 0.0
        # self.best_check = './experiment_2/check_best_new/'
        self.best_check = best_checkpoint_dir
        self.logger_best = logger_best
        self.logger_test = logger_test
        self.key_90 = 1
        self.key_91 = 1
    def build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.h,self.w,
                                         self.input_c_dim*2 + self.output_c_dim],
                                        name='real_A_and_B_images')
        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim*2]
        self.real_super = self.real_data[:, :, :, self.input_c_dim*2:self.input_c_dim*2 + self.output_c_dim]

        if self.withbn:
            self.fake_super1, self.fake_super2, self.fake_super = self.pix_model.inference(
                self.real_A,self.real_B,batch_size=self.batch_size,h=self.h,w = self.w,reuse=False)


        self.real_AA = tf.concat([self.real_A, self.real_B,self.real_super], 3)
        self.fake_AA = tf.concat([self.real_A, self.real_B,self.fake_super1], 3)

        self.real_BB = tf.concat([self.real_A,self.real_B, self.real_super], 3)
        self.fake_BB = tf.concat([self.real_A,self.real_B, self.fake_super2], 3)

        self.real_fusion = tf.concat([self.real_A,self.real_B,self.real_super],3)
        self.fake_fusion = tf.concat([self.real_A,self.real_B, self.fake_super], 3)

        self.DA, self.DA_logits = self.pix_model.discriminator(self.real_AA, batch_size=self.batch_size, reuse=False)
        self.DA_, self.DA_logits_ = self.pix_model.discriminator(self.fake_AA, batch_size=self.batch_size, reuse=True)
        self.DB, self.DB_logits = self.pix_model.discriminator(self.real_BB, batch_size=self.batch_size, reuse=True)
        self.DB_, self.DB_logits_ = self.pix_model.discriminator(self.fake_BB, batch_size=self.batch_size, reuse=True)
        self.Df, self.Df_logits = self.pix_model.discriminator(self.real_fusion, batch_size=self.batch_size,
                                                               reuse=True)
        self.Df_, self.Df_logits_ = self.pix_model.discriminator(self.fake_fusion, batch_size=self.batch_size,
                                                                     reuse=True)

        self.disf_sum = tf.summary.histogram("disf", self.Df)
        self.disf__sum = tf.summary.histogram("disf_", self.Df_)

        self.disa_sum = tf.summary.histogram("disA", self.DA)
        self.disa__sum = tf.summary.histogram("disA_", self.DA_)

        self.disb_sum = tf.summary.histogram("disB", self.DB)
        self.disb__sum = tf.summary.histogram("disB_", self.DB_)


        self.fusion_sum = tf.summary.image("fusion", self.fake_super)

        self.fake_A_sum = tf.summary.image("fake_A", self.fake_super1)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_super2)

        self.disf_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Df_logits, labels=tf.ones_like(self.Df)))
        self.disf_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Df_logits_, labels=tf.zeros_like(self.Df_)))
        self.disa_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DA_logits, labels=tf.ones_like(self.DA)))
        self.disa_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DA_logits_, labels=tf.zeros_like(self.DA_)))

        self.disb_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DB_logits, labels=tf.ones_like(self.DB)))
        self.disb_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DB_logits_, labels=tf.zeros_like(self.DB_)))

        self.gf_loss_1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Df_logits_, labels=tf.ones_like(self.Df_)))
        self.gf_loss_2 = self.L1_lambda * tf.reduce_mean(tf.abs(self.real_super - self.fake_super)) * 1.0
                       # + self.L1_lambda * (1 - tf.image.psnr(self.real_super, self.fake_super, max_val=1.0)[0] / 60.) * 1.0
        self.gf_loss_3 = self.L1_lambda * (1 - tf.image.ssim_multiscale(self.real_super, self.fake_super, max_val=1.0)[0]) * 1.0

        self.ga_loss_1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DA_logits_, labels=tf.ones_like(self.DA_)))
        self.ga_loss_2 = self.L1_lambda * tf.reduce_mean(tf.abs(self.real_super - self.fake_super1)) * 1.0
                       #  + self.L1_lambda * (1 - tf.image.psnr(self.real_super, self.fake_super1, max_val=1.0)[0] / 60.) * 1.0
        self.ga_loss_3 = self.L1_lambda * (1 - tf.image.ssim_multiscale(self.real_super, self.fake_super1, max_val=1.0)[0])*1.0

        self.gb_loss_1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DB_logits_, labels=tf.ones_like(self.DB_)))
        self.gb_loss_2 = self.L1_lambda * tf.reduce_mean(tf.abs(self.real_super - self.fake_super2)) * 1.0
                        # + self.L1_lambda * (1 - tf.image.psnr(self.real_super, self.fake_super2, max_val=1.0)[0] / 60.) * 1.0
        self.gb_loss_3 = self.L1_lambda * (1 - tf.image.ssim_multiscale(self.real_super, self.fake_super2, max_val=1.0)[0]) * 1.0

        self.disf_loss_real_sum = tf.summary.scalar("disf_loss_real", self.disf_loss_real)
        self.disf_loss_fake_sum = tf.summary.scalar("disf_loss_fake", self.disf_loss_fake)

        self.disa_loss_real_sum = tf.summary.scalar("disa_loss_real", self.disa_loss_real)
        self.disa_loss_fake_sum = tf.summary.scalar("disa_loss_fake", self.disa_loss_fake)

        self.disb_loss_real_sum = tf.summary.scalar("disb_loss_real", self.disb_loss_real)
        self.disb_loss_fake_sum = tf.summary.scalar("disb_loss_fake", self.disb_loss_fake)

        self.disf_loss = 1.*(self.disf_loss_real + self.disf_loss_fake)
        self.disa_loss = 1.*(self.disa_loss_real + self.disa_loss_fake)
        self.disb_loss = 1.*(self.disb_loss_real + self.disb_loss_fake)


        t_vars = tf.trainable_variables()

        total_parameters = 0
        for vari in  t_vars:
            shape = vari.get_shape()
            vari_para = 1
            for vari_dim in shape:
                vari_para *= vari_dim.value
            total_parameters += vari_para
        print('total trainable parameters:{}'.format(total_parameters))

        self.dis_vars = [var for var in t_vars if 'dis_' in var.name]



        self.gf_vars = [var for var in t_vars if 'gf_' in var.name or 'g_' in var.name
                        or 'gu_' in var.name or 'gab_' in var.name or 'gabu_' in var.name]
        # self.gf_vars = [var for var in t_vars if 'gf_' in var.name]

        self.ga_vars = [var for var in t_vars if 'g_' in var.name or 'gab_' in var.name
                        or 'gabu_' in var.name]
        # self.ga_vars = [var for var in t_vars if 'g_' in var.name or 'gab_' in var.name]

        self.gb_vars = [var for var in t_vars if 'gu_' in var.name  or 'gab_' in var.name
                        or 'gabu_' in var.name]
        # self.gb_vars = [var for var in t_vars if 'gu_' in var.name or 'gabu_' in var.name]

        self.gf_loss = 1. * (self.gf_loss_1 + self.gf_loss_2 + 1 * self.gf_loss_3)
        self.ga_loss = 1. * (self.ga_loss_1 + self.ga_loss_2 + 1 * self.ga_loss_3)
        self.gb_loss = 1. * (self.gb_loss_1 + self.gb_loss_2 + 1 * self.gb_loss_3)

        self.gf_loss_sum = tf.summary.scalar("gf_loss", self.gf_loss)
        self.ga_loss_sum = tf.summary.scalar("ga_loss", self.ga_loss)
        self.gb_loss_sum = tf.summary.scalar("gb_loss", self.gb_loss)

        self.disf_loss_sum = tf.summary.scalar("disf_loss", self.disf_loss)
        self.disa_loss_sum = tf.summary.scalar("disa_loss", self.disa_loss)
        self.disb_loss_sum = tf.summary.scalar("disb_loss", self.disb_loss)

        self.saver = tf.train.Saver(tf.global_variables())
        self.saver_1 = tf.train.Saver(tf.global_variables())

    def train(self, args):
        # for i in range(4):
        """Train pix2pix"""
        if self.base_count > 0:
            lr_t = args.lr * np.sqrt(1 - np.power(0.999, self.base_count)) / (
                    1 - np.power(args.beta1, self.base_count))
        else:
            lr_t = args.lr

        lr_weight = 1.0

        disf_optim = tf.train.AdamOptimizer(lr_t*lr_weight, beta1=args.beta1) \
            .minimize(self.disf_loss, var_list=self.dis_vars)
        gf_optim = tf.train.AdamOptimizer(lr_t*lr_weight, beta1=args.beta1) \
            .minimize(self.gf_loss, var_list=self.gf_vars)

        disa_optim = tf.train.AdamOptimizer(lr_t*lr_weight, beta1=args.beta1) \
            .minimize(self.disa_loss, var_list=self.dis_vars)
        ga_optim = tf.train.AdamOptimizer(lr_t*lr_weight, beta1=args.beta1) \
            .minimize(self.ga_loss, var_list=self.ga_vars)

        disb_optim = tf.train.AdamOptimizer(lr_t*lr_weight, beta1=args.beta1) \
            .minimize(self.disb_loss, var_list=self.dis_vars)
        gb_optim = tf.train.AdamOptimizer(lr_t*lr_weight, beta1=args.beta1) \
            .minimize(self.gb_loss, var_list=self.gb_vars)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        self.gf_sum = tf.summary.merge([self.disf__sum,
                                       self.fusion_sum, self.disf_loss_fake_sum, self.gf_loss_sum])
        self.disf_sum = tf.summary.merge([self.disf_sum, self.disf_loss_real_sum, self.disf_loss_sum])

        self.ga_sum = tf.summary.merge([self.disa__sum,
                                        self.fake_A_sum, self.disa_loss_fake_sum, self.ga_loss_sum])
        self.disa_sum = tf.summary.merge([self.disa_sum, self.disa_loss_real_sum, self.disa_loss_sum])

        self.gb_sum = tf.summary.merge([self.disb__sum,
                                        self.fake_B_sum, self.disb_loss_fake_sum, self.gb_loss_sum])
        self.disb_sum = tf.summary.merge([self.disb_sum, self.disb_loss_real_sum, self.disb_loss_sum])

        self.writer = tf.summary.FileWriter("./logs_tublin", self.sess.graph)

        counter = 1
        start_time = time.time()

        for epoch in xrange(0,args.epoch):
            # if i < 3:
            #     if epoch == 10:
            #         break
            # data = glob('./datasets/{}/train_wild/*.tif'.format(self.dataset_name))
            data = glob('{}/example-data-training samples/*'.format(self.dataset_name))
            # data = glob('/home/ksc/anet-models/different_bits/{}/*'.format(args.dataset_name))
            # print(data)
            np.random.shuffle(data)
            batch_idxs = min(len(data), args.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):

                batch_files = data[idx * self.batch_size:(idx + 1) * self.batch_size]

                batch = [load_data(batch_file,axis=2) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                # Update D network
                for ss in range(0,1):
                    _, summary_stra = self.sess.run([disa_optim, self.disa_sum],
                                               feed_dict={self.real_data: batch_images})
                    self.writer.add_summary(summary_stra, counter)
                    _, summary_strb = self.sess.run([disb_optim, self.disb_sum],
                                                feed_dict={self.real_data: batch_images})
                    self.writer.add_summary(summary_strb, counter)

                    for hh in range(0,1):
                        _, summary_strf = self.sess.run([disf_optim, self.disf_sum],
                                               feed_dict={self.real_data: batch_images})
                        self.writer.add_summary(summary_strf, counter)

                # Update G network
                for ss in range(0, 6):
                    # Run g_optim more iter to make sure that d_loss does not go to zero (different from paper)

                    if np.mod(ss, 3) > 1:
                    # for hh in range(0,2):

                        _, summary_stra = self.sess.run([ga_optim, self.ga_sum],
                                                   feed_dict={self.real_data: batch_images})
                        self.writer.add_summary(summary_stra, counter)

                    if np.mod(ss, 3) > 1:

                        _, summary_strb = self.sess.run([gb_optim, self.gb_sum],
                                                    feed_dict={self.real_data: batch_images})
                        self.writer.add_summary(summary_strb, counter)

                    if np.mod(ss,3) >= 0:

                        _, summary_strf = self.sess.run([gf_optim, self.gf_sum],
                                                   feed_dict={self.real_data: batch_images})
                        self.writer.add_summary(summary_strf, counter)


                errDa_fake = self.disa_loss_fake.eval({self.real_data: batch_images})
                errDa_real = self.disa_loss_real.eval({self.real_data: batch_images})
                errGa = self.ga_loss.eval({self.real_data: batch_images})
                errDb_fake = self.disb_loss_fake.eval({self.real_data: batch_images})
                errDb_real = self.disb_loss_real.eval({self.real_data: batch_images})
                errGb = self.gb_loss.eval({self.real_data: batch_images})

                errD_fakef = self.disf_loss_fake.eval({self.real_data: batch_images})
                errD_realf = self.disf_loss_real.eval({self.real_data: batch_images})
                errGf = self.gf_loss.eval({self.real_data: batch_images})


                counter += 1

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f,\nda_: %.8f, ga_: %.8f,\ndb_: %.8f, gb_: %.8f,\ndf_: %.8f, gf_: %.8f,\n"
                      % (epoch, idx, batch_idxs, time.time() - start_time, errDa_fake + errDa_real, errGa,errDb_fake + errDb_real, errGb,errD_fakef + errD_realf, errGf))
                print("da_fake:{}\n da_real:{}\n".format(errDa_fake, errD_realf))
                self.logger.info("Epoch: [%2d] [%4d/%4d] time: %4.4f,\nda_: %.8f, ga_: %.8f,\ndb_: %.8f, gb_: %.8f,\ndf_: %.8f, gf_: %.8f,\n"
                      % (epoch, idx, batch_idxs, time.time() - start_time, errDa_fake + errDa_real, errGa,errDb_fake + errDb_real, errGb,errD_fakef + errD_realf, errGf))
                self.logger.info("da_fake:{}\n da_real:{}\n".format(errDa_fake, errD_realf))
                if np.mod(counter, 60) == 1:
                    self.sample(args.sample_dir,args.abf_dir, epoch, idx)
                if np.mod(counter, 50) == 1:
                    print("~~~~~~~~~save~~~~~~~mode~~~~")
                    self.saver.save(self.sess,args.checkpoint_dir,counter)
                # if np.mod(counter, 2) == 1:
                #     print("~~~~~~~~~save~~~~~~~mode~~~~")
                #     self.saver.save(self.sess,args.checkpoint_dir,counter)
        self.writer.close()

    def train_different_size(self, args, batch_images,epoch,idx,counter,first_save):
        """Train pix2pix"""
        lr_t = args.lr * np.sqrt(1-np.power(0.999,self.base_count+counter))/(1-np.power(args.beta1,self.base_count+counter))
        disf_optim = tf.train.AdamOptimizer(lr_t, beta1=args.beta1) \
            .minimize(self.disf_loss, var_list=self.dis_vars)
        gf_optim = tf.train.AdamOptimizer(lr_t, beta1=args.beta1) \
            .minimize(self.gf_loss, var_list=self.gf_vars)

        disa_optim = tf.train.AdamOptimizer(lr_t, beta1=args.beta1) \
            .minimize(self.disa_loss, var_list=self.dis_vars)
        ga_optim = tf.train.AdamOptimizer(lr_t, beta1=args.beta1) \
            .minimize(self.ga_loss, var_list=self.ga_vars)

        disb_optim = tf.train.AdamOptimizer(lr_t, beta1=args.beta1) \
            .minimize(self.disb_loss, var_list=self.dis_vars)
        gb_optim = tf.train.AdamOptimizer(lr_t, beta1=args.beta1) \
            .minimize(self.gb_loss, var_list=self.gb_vars)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        self.gf_sum = tf.summary.merge([self.disf__sum,
                                        self.fusion_sum, self.disf_loss_fake_sum, self.gf_loss_sum])
        self.disf_sum = tf.summary.merge([self.disf_sum, self.disf_loss_real_sum, self.disf_loss_sum])

        self.ga_sum = tf.summary.merge([self.disa__sum,
                                        self.fake_A_sum, self.disa_loss_fake_sum, self.ga_loss_sum])
        self.disa_sum = tf.summary.merge([self.disa_sum, self.disa_loss_real_sum, self.disa_loss_sum])

        self.gb_sum = tf.summary.merge([self.disb__sum,
                                        self.fake_B_sum, self.disb_loss_fake_sum, self.gb_loss_sum])
        self.disb_sum = tf.summary.merge([self.disb_sum, self.disb_loss_real_sum, self.disb_loss_sum])

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        start_time = time.time()
        # Update D network
        _, summary_stra = self.sess.run([disa_optim, self.disa_sum],
                                        feed_dict={self.real_data: batch_images})
        self.writer.add_summary(summary_stra, counter)
        _, summary_strb = self.sess.run([disb_optim, self.disb_sum],
                                        feed_dict={self.real_data: batch_images})
        self.writer.add_summary(summary_strb, counter)

        for hh in range(0, 1):
            _, summary_strf = self.sess.run([disf_optim, self.disf_sum],
                                        feed_dict={self.real_data: batch_images})
            self.writer.add_summary(summary_strf, counter)

        # Update G network
        for ss in range(0, 4):
            # Run g_optim more iter to make sure that d_loss does not go to zero (different from paper)
            if np.mod(ss, 2) >= 0:
                _, summary_stra = self.sess.run([ga_optim, self.ga_sum],
                                            feed_dict={self.real_data: batch_images})
                self.writer.add_summary(summary_stra, counter)
            if np.mod(ss, 2) >= 0:
                _, summary_strb = self.sess.run([gb_optim, self.gb_sum],
                                            feed_dict={self.real_data: batch_images})
                self.writer.add_summary(summary_strb, counter)
            if np.mod(ss, 2) >= 0:
                _, summary_strf = self.sess.run([gf_optim, self.gf_sum],
                                            feed_dict={self.real_data: batch_images})
                self.writer.add_summary(summary_strf, counter)

        errDa_fake = self.disa_loss_fake.eval({self.real_data: batch_images})
        errDa_real = self.disa_loss_real.eval({self.real_data: batch_images})
        errGa = self.ga_loss.eval({self.real_data: batch_images})
        errDb_fake = self.disb_loss_fake.eval({self.real_data: batch_images})
        errDb_real = self.disb_loss_real.eval({self.real_data: batch_images})
        errGb = self.gb_loss.eval({self.real_data: batch_images})

        errD_fakef = self.disf_loss_fake.eval({self.real_data: batch_images})
        errD_realf = self.disf_loss_real.eval({self.real_data: batch_images})
        errGf = self.gf_loss.eval({self.real_data: batch_images})

        counter += 1

        print("Epoch: [%2d] [%4d] time: %4.4f,\nda_: %.8f, ga_: %.8f,\ndb_: %.8f, gb_: %.8f,\ndf_: %.8f, gf_: %.8f,\n"
                    % (epoch, idx, time.time() - start_time, errDa_fake + errDa_real, errGa,
                       errDb_fake + errDb_real, errGb, errD_fakef + errD_realf, errGf))
        print("da_fake:{}\n da_real:{}\n".format(errDa_fake,errD_realf))
        if np.mod(counter, 10) == 1:
            self.sample(args.sample_dir, args.abf_dir, epoch, idx)
        if ckpt and ckpt.model_checkpoint_path:
            if first_save == False:
                os.remove(args.checkpoint_dir+'-'+str(counter-1)+'.data-00000-of-00001')
                os.remove(args.checkpoint_dir + '-' + str(counter - 1) + '.index')
                os.remove(args.checkpoint_dir + '-' + str(counter - 1) + '.meta')
            self.saver.save(self.sess,args.checkpoint_dir,counter)
        self.writer.close()


    def sample(self, sample_dir,abf_dir,epoch,idx):
        # data = np.random.choice(glob('./datasets/{}/val_wild/*.tif'.format(self.dataset_name)), self.batch_size)
        # for sample_file in data:
        #     sample = load_val_image(sample_file)
        #     #sample_file = sample_file.replace('val_wild', 'val_sparse')
        #     sample_file = sample_file.replace('val_wild', 'val_wild')
        #     sample_1 = load_val_image(sample_file)
        #     #sample_file = sample_file.replace('val_sparse', 'val_gt')
        #     sample_file = sample_file.replace('val_wild', 'val_gt')
        #     sample_gt = load_val_image(sample_file)
        #     img_name = txt_wrap_by('/', '.', sample_file)
        data = np.random.choice(
            glob('{}/example-data-training samples/*'.format(self.dataset_name)),
            self.batch_size)
        # data = np.random.choice(
        #     glob('/home/ksc/anet-models/different_bits/{}/*'.format(self.dataset_name)),
        #     self.batch_size)
        for sample_file in data:
            print(sample_file)
            sample = load_wf_image_ksc(sample_file,reg=True)
            sample_1 = load_sparse_image_ksc(sample_file,reg=True)
            # sample = sample_1
            # sample_1 = sample
            sample_gt = load_gt_image_ksc(sample_file,reg=True)

            img_name = txt_wrap_by('/', '.', sample_file)
            break
        sample_images = np.array(sample).astype(np.float32)
        sample_images_1 = np.array(sample_1).astype(np.float32)
        sample_gt_1 = np.array(sample_gt).astype(np.float32)
        # sample_images_input = np.concatenate((sample_images,sample_images_1),axis=2)

        graph = tf.Graph()
        with graph.as_default():
            fusion_image_placeholder = tf.placeholder(tf.float32,
                                                     shape=[1, sample_images.shape[0], sample_images.shape[1],
                                                        3])
            fusion_image_placeholder_1 = tf.placeholder(tf.float32,
                                                      shape=[1, sample_images_1.shape[0], sample_images_1.shape[1],
                                                             3])
            # target_image_placeholder = tf.placeholder(tf.float32,shape=[1,sample_gt_1.shape[0],sample_gt_1.shape[1],
            #                                                  3])

            if self.withbn:
                super_a,super_b,super_fusion = self.pix_model.inference(
                    fusion_image_placeholder,fusion_image_placeholder_1, batch_size=1,
                    h=sample_images.shape[0], w=sample_images.shape[1], reuse=False)
                # ssim_a = tf.image.ssim_multiscale(target_image_placeholder, super_a, max_val=1.0)[0]
            saver = tf.train.Saver(tf.global_variables())
            with tf.Session(graph=graph, config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False,
                    gpu_options=tf.GPUOptions(allow_growth=True,
                                                per_process_gpu_memory_fraction=1,
                                                visible_device_list="0"))) as sess:
                ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                x = np.zeros((1,sample_images.shape[0], 5, 3), dtype=np.float32) + 255. #65535.0
                samplea,sampleb,sample_fusion= sess.run(
                        [super_a,super_b,super_fusion],feed_dict={fusion_image_placeholder:[sample_images],fusion_image_placeholder_1:[sample_images_1]})

                # ssim_b = tf.image.ssim_multiscale(sample_gt, super_b, max_val=1.0)[0]
                # ssim_fusion = tf.image.ssim_multiscale(sample_gt, super_fusion, max_val=1.0)[0]
                # ssim_a, ssim_b, ssim_fusion = tf.Session.run(
                #     [ssim_a, ssim_b, ssim_fusion], feed_dict=[samplea, sampleb, sample_fusion, sample_gt]
                #
                # print("ssim_a: {} ".format(ssim_a))

                print("~~~~~~~~~~~~~~~~~~~~~~~~the size = {}".format(samplea.shape))
                print(np.shape(sample_gt),np.shape(x[0]),np.shape(samplea[0]),np.shape(sampleb[0]),np.shape(sample_fusion[0]))
                im_result = np.concatenate((sample_gt,x[0],samplea[0],x[0],sampleb[0],x[0],sample_fusion[0]), axis=1)
                #eval(im_result,"train_{:04d}_{:01d}".format(epoch+1815,idx))
                image = np.array(im_result).astype(np.float32)
                image_a = np.array(samplea[0]).astype(np.float32)
                image_b = np.array(sampleb[0]).astype(np.float32)
                image_f = np.array(sample_fusion[0]).astype(np.float32)
                save_images_val(image,'{}/train_{:04d}_{:01d}_{}.tif'.format(sample_dir, epoch, idx,img_name))
                # save_images_val(image_a,
                #                 './{}/a/train_{:04d}_{:01d}_{}_{}.tif'.format(abf_dir, epoch, idx, 'w1',img_name))
                # save_images_val(image_b,
                #                 './{}/b/train_{:04d}_{:01d}_{}_{}.tif'.format(abf_dir, epoch, idx, 'w2',img_name))
                # save_images_val(image_f,
                #                 './{}/f/train_{:04d}_{:01d}_{}_{}.tif'.format(abf_dir, epoch, idx, 'f',img_name))
                # ssim_aa = sess.run(
                #     [ssim_a],feed_dict={target_image_placeholder: [sample_gt_1],super_a:super_a}
                # )
                # print("ssim_a = "+ssim_a)
                path = '{}/train_{:04d}_{:01d}_{}.tif'.format(sample_dir, epoch, idx,img_name)
                img = scipy.misc.imread(path).astype(np.float32)
                eval(img,self.logger_val,"train_{:04d}_{:01d}__{}".format(epoch,idx,img_name),epoch)
            sess.close()

    def test_end(self, args):
        sample_files = sorted(glob('/media/ksc/code/example-data-2022.12.12/tubulin-models/example-data-tubulin-model-WF+MU-SRM-to-SRM/{}/*'.format(self.dataset_name)))
        # sample_files = glob('/media/ksc/data/below-100-100/*')
        # sample_files = glob('/home/ksc/anet-models/different_bits')
        # sample_files = glob(
        #     '/media/ksc/code/tubulin-model-data/{}/model-2-training-samples/*'.format(self.dataset_name))
        count = 0
        print(sample_files)
        for sample_file in sample_files:
            print(sample_file)
            dir = sample_file + '/'
            filelist = get_filelist(dir, [])
            for fl in filelist:
                if 'wf.tif' in fl:
                # if 'wf/1-1.tif' in fl:
                    print(fl)
                    sample_file_img = fl
                    # sample_sparse_img = fl[:-6] + '1-5.tif'
                    # sample_gt_img = fl[:-6] + 'perfect.tif'
                    for sparse_count in range(1, 2):
                        graph = tf.Graph()
                        with graph.as_default():
                            count = count+1
                            print("sampling image ", count)
                            if count <0:
                                continue
                            image1 = scipy.misc.imread(sample_file_img).astype('uint8')

                            if image1.ndim==2:
                                img_AA = np.zeros((image1.shape[0], image1.shape[1], 3))
                                img_AA[:, :, 0] = image1
                                img_AA[:, :, 1] = image1
                                img_AA[:, :, 2] = image1
                                image1 = img_AA

                            # sampled sparse
                            # sample_sparse_img = fl[:-6] + '1-{}.tif'.format(sparse_count)
                            # image2 = scipy.misc.imread(sample_sparse_img).astype('uint8')
                            # image2[:, :, 1] = image2[:, :, 0]
                            # image2[:, :, 2] = image2[:, :, 0]


                            #generated sparse
                            sample_sparse_img = fl.replace(fl.split('/')[-1], '') + '/MU-SRM/MU-SRM.tif'
                            image2 = scipy.misc.imread(sample_sparse_img).astype('uint8')

                            # sparse + sparse
                            # image1 = image2

                            # wf + wf
                            # image2 = image1


                            image1 = image1 / 255.
                            image2 = image2 / 255.

                            image1 = np.array(image1).astype(np.float32)
                            image2 = np.array(image2).astype(np.float32)

                            fusion_image_placeholder = tf.placeholder(tf.float32,
                                                          shape=[1, image1.shape[0], image1.shape[1],
                                                                 3])
                            fusion_image_placeholder_1 = tf.placeholder(tf.float32,
                                                            shape=[1, image2.shape[0], image2.shape[1],
                                                                   3])
                            if self.withbn:
                                super_a,super_b, super_fusion = self.pix_model.inference(
                                    fusion_image_placeholder,fusion_image_placeholder_1, batch_size=1,
                                    h=image2.shape[0], w=image2.shape[1], reuse=False)

                            saver = tf.train.Saver(tf.global_variables())
                            with tf.Session(graph=graph, config=tf.ConfigProto(
                                allow_soft_placement=True,
                                log_device_placement=False,
                                gpu_options=tf.GPUOptions(allow_growth=True,
                                                  per_process_gpu_memory_fraction=1,
                                                  visible_device_list="0"))) as sess:
                                ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
                                if ckpt and ckpt.model_checkpoint_path:
                                    saver.restore(sess, ckpt.model_checkpoint_path)

                                start = time.time()
                                samplea,sampleb, sample_fusion = sess.run(
                                    [super_a,super_b, super_fusion],
                                    feed_dict={fusion_image_placeholder: [image1],fusion_image_placeholder_1:[image2]})
                                end = time.time()
                                print('run time: {}'.format(end-start))

                                image_a = np.array(samplea[0]).astype(np.float32)
                                image_b = np.array(sampleb[0]).astype(np.float32)
                                image_f = np.array(sample_fusion[0]).astype(np.float32)

                                # folder = 'wf+sparse-to-SRM-desired_output/'
                                # folder = 'sparse+sparse-to-SRM-desired_output/'
                                folder = 'wf+MU-SRM-to-SRM-desired_output/'
                                # folder = ''

                                if not os.path.exists(fl.replace(fl.split('/')[-1], '') + folder):
                                    os.makedirs(fl.replace(fl.split('/')[-1], '') + folder)

                                img_name_ak1 = fl.replace(fl.split('/')[-1],'') + folder + 'XN1'.format(fl.split('/')[-1])
                                img_name_ak2 = fl.replace(fl.split('/')[-1],'') + folder + 'XN2'.format(fl.split('/')[-1])
                                img_name_ak3 = fl.replace(fl.split('/')[-1], '') + folder + 'XN3'.format(
                                    fl.split('/')[-1])
                                save_images_val(image_a,
                                    '{}.tif'.format(img_name_ak1))
                                save_images_val(image_b,
                                    '{}.tif'.format(img_name_ak2))
                                save_images_val(image_f,
                                    '{}.tif'.format(img_name_ak3))
                                sess.close()

