from __future__ import division
import time
from glob import glob
from six.moves import xrange
from utils import *
import tensorflow as tf
import os
from evaluate import  eval
class pix2pix(object):
    def __init__(self, sess,pix_model,h = 256, w = 256,batch_size = 1,
                 base_dim=64, L1_lambda=100,
                 input_c_dim=3, output_c_dim=3, dataset_name='haze',
                 checkpoint_dir=None,withbn = True,base_count = 0, Model = 'my', logger=None,logger_val=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
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

    def build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size,self.h,self.w,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')
        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        if self.withbn:
            self.fake_B,self.hazemap = self.pix_model.inference(self.real_A,batch_size=self.batch_size,h=self.h,w =self.w,reuse=False)

        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)


        self.D, self.D_logits = self.pix_model.discriminator(self.real_AB, batch_size = self.batch_size, reuse=False)
        self.D_, self.D_logits_ = self.pix_model.discriminator(self.fake_AB, batch_size = self.batch_size, reuse=True)


        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)

        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)


        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))

        self.g_loss_1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))*1.0
        self.g_loss_2 = self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B)) * 1.0
        # self.g_loss_3 = self.L1_lambda * (1 - tf.image.psnr(self.real_B, self.fake_B, max_val=1.0)[0] / 60.) * 1.0
        self.g_loss_4 = self.L1_lambda * (1 - tf.image.ssim_multiscale(self.real_B, self.fake_B, max_val=1.0)[0])*1.0

        # self.g_loss_5 = self.L1_lambda*tf.reduce_mean(tf.abs(self.real_A-tf.exp(self.Ig)-self.fake_B+tf.exp(self.Jg)))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)


        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss = 1.0 * (self.g_loss_1 + self.g_loss_2 + self.g_loss_4)


        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)


        t_vars = tf.trainable_variables()

        total_parameters = 0
        for vari in t_vars:
            shape = vari.get_shape()
            vari_para = 1
            for vari_dim in shape:
                vari_para *= vari_dim.value
            total_parameters += vari_para
        print('total trainable parameters:{}'.format(total_parameters))

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, args):
        if self.base_count > 0:
            lr_t = args.lr * np.sqrt(1 - np.power(0.999, self.base_count)) / (
                    1 - np.power(args.beta1, self.base_count))
        else:
            lr_t = args.lr


        d_optim = tf.train.AdamOptimizer(lr_t, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(lr_t, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        self.g_sum = tf.summary.merge([self.d__sum,
                                       self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        for epoch in range(args.epoch):
            # data = glob('./datasets/{}/train/*.tif'.format(self.dataset_name))
            data = glob('{}/example-data-training samples/*'.format(self.dataset_name))
            # np.random.shuffle(data)
            batch_idxs = min(len(data), args.train_size) // self.batch_size

            # print(data)

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx * self.batch_size:(idx + 1) * self.batch_size]

                batch = [load_data(batch_file,axis=2) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                # Update D network
                for ss in range(0,2):
                # if np.mod(counter, 1) == 0:
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={self.real_data: batch_images})
                    self.writer.add_summary(summary_str, counter)

                # Update G network
                for ss in range(0, 15):
                    # Run g_optim more iter to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.real_data: batch_images})
                    self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.real_data: batch_images})
                errD_real = self.d_loss_real.eval({self.real_data: batch_images})
                errG = self.g_loss.eval({self.real_data: batch_images})
                counter += 1

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))
                self.logger.info("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))
                if np.mod(counter, 50) == 1:
                    self.sample(args.sample_dir, args.abf_dir,epoch, idx)

                if np.mod(counter, 45) == 1:
                    self.saver.save(self.sess,args.checkpoint_dir,counter)
                self.writer.close()

    def train_different_size(self, args, batch_images,epoch,idx,counter,first_save):
        lr_t = args.lr * np.sqrt(1-np.power(0.999,self.base_count+counter))/(1-np.power(args.beta1,self.base_count+counter))
        d_optim = tf.train.AdamOptimizer(lr_t, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(lr_t, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        self.g_sum = tf.summary.merge([self.d__sum,
                                       self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        start_time = time.time()
        # Update D network
        if np.mod(counter,1) == 0:
            _, summary_str = self.sess.run([d_optim, self.d_sum],
                                    feed_dict={self.real_data: batch_images})
            self.writer.add_summary(summary_str, counter)
        # Update G network
        for ss in range(0, 5):
            # Run g_optim more iter to make sure that d_loss does not go to zero (different from paper)
            _, summary_str = self.sess.run([g_optim, self.g_sum],
                                        feed_dict={self.real_data: batch_images})
            self.writer.add_summary(summary_str, counter)

        errD_fake = self.d_loss_fake.eval({self.real_data: batch_images})
        errD_real = self.d_loss_real.eval({self.real_data: batch_images})
        errG = self.g_loss.eval({self.real_data: batch_images})
        counter += 1
        print("Epoch: [%2d] [%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                % (epoch, idx, time.time() - start_time, errD_fake + errD_real, errG))

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
        # data = np.random.choice(glob('./datasets/{}/val/*.tif'.format(self.dataset_name)), self.batch_size)
        data = np.random.choice(glob('{}/example-data-training samples/*'.format(self.dataset_name)),self.batch_size)
        for sample_file in data:
            print(sample_file)
            sample = load_val_image_ksc(sample_file)
            image1 = load_gt_image_ksc(sample_file)
            img_name = txt_wrap_by('/', '.', sample_file)
            break
        sample_images = np.array(sample).astype(np.float32)

        graph = tf.Graph()
        with graph.as_default():
            hazed_image_placeholder = tf.placeholder(tf.float32,
                                                     shape=[1, image1.shape[0], image1.shape[1],
                                                        3])
            if self.withbn:
                dehaze_image,hazemap = self.pix_model.inference(hazed_image_placeholder,batch_size=1, h=image1.shape[0], w=image1.shape[1])

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

                sample,hm = sess.run(
                        [dehaze_image,hazemap],feed_dict={hazed_image_placeholder:[sample_images]})
                # sample_1 = sample[0].transpose(2, 0, 1)
                # R_0 = sample_1[0]
                # R_1 = sample_1[1]
                # R_2 = sample_1[2]
                # R_avg = (R_0 + R_1 + R_2) / 3.0
                # new_img = np.zeros((3, 512, 512), dtype=np.int)
                # new_img[0] = new_img[0] + R_avg
                # new_img[1] = new_img[1]-1
                # new_img[2] = new_img[2]-1
                # new_img = new_img.transpose(1, 2, 0)
                x = np.zeros((1, image1.shape[0], 5, 3), dtype=np.float32) + 255.0
                if(image1.shape[-1]==4):
                    im_result = np.concatenate(
                            (sample_images, x[0],image1[:,:,:3], x[0],sample[0][0]), axis=1)
                    image_a = np.array(sample[0][0]).astype(np.float32)
                    print("image_shape={}".format('4'))
                else:
                    # im_result = np.concatenate(
                    #         (sample_images,x[0], image1, x[0],sample[0]), axis=1)
                    im_result = np.concatenate(
                        (sample_images, x[0], image1, x[0],sample[0]), axis=1)
                    print("image_shape={}".format('3'))
                    image_a = np.array(sample[0]).astype(np.float32)
                image = np.array(im_result).astype(np.float32)

                save_images_val(image,
                                './{}/train_{:04d}_{:01d}_{}.tif'.format(sample_dir, epoch, idx,img_name))


                # save_images_val(image_a,
                #                 './{}/a/train_{:04d}_{:01d}_{}_{}.tif'.format(abf_dir, epoch, idx, 'w1',img_name))
                path = '{}/train_{:04d}_{:01d}_{}.tif'.format(sample_dir, epoch, idx, img_name)
                img = scipy.misc.imread(path).astype(np.float)
                eval(img, self.logger_val, "train_{:04d}_{:01d}__{}".format(epoch, idx, img_name), epoch)

            sess.close()

    def test_ksc(self, args):
        # sample_files = glob('./datasets/{}/test_train/wild/*.tif'.format(self.dataset_name))
        sample_files = glob('{}/example-data-Test-samples/*'.format(self.dataset_name))
        # sample_files = glob('/media/ksc/code/tubulin-model-data/{}/model-3-training-samples/*'.format(self.dataset_name))
        count = 0
        # print(sample_files)
        for sample_file in sample_files:
            dir = sample_file + '/'
            filelist = get_filelist(dir, [])
            # print(filelist)
            for fl in filelist:
                if 'wf.tif' in fl:
                # if 'wf/1-1.tif' in fl:
                    sample_file_img = fl
                    # print(fl)
                    graph = tf.Graph()
                    with graph.as_default():
                        count = count+1
                        print("sampling image ", count)

                        image1 = scipy.misc.imread(sample_file_img)
                        # image1 = image1[2016:,:,:]
                        image1 = image1 /255.
                        if image1.ndim==2:
                            img_AA = np.zeros((image1.shape[0], image1.shape[1], 3))
                            img_AA[:, :, 0] = image1
                            img_AA[:, :, 1] = image1
                            img_AA[:, :, 2] = image1
                            image1 = img_AA

                        hazed_image_placeholder = tf.placeholder(tf.float32,
                                                     shape=[1, image1.shape[0], image1.shape[1],
                                                            3])
                        img_name =fl[:-6]

                        img_name = img_name + 'wf-to-generated-sparse/'
                        if not os.path.exists(img_name):
                            os.makedirs(img_name)
                        img_name = img_name + 'generated-sparse.tif'
                        if self.withbn:
                            dehaze_image,hazemap = self.pix_model.inference(hazed_image_placeholder,batch_size=1, h=image1.shape[0], w=image1.shape[1])

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
                            sample,hm = sess.run(
                            [dehaze_image,hazemap], feed_dict={hazed_image_placeholder: [image1]})

                            image = np.array(sample[0]).astype(np.float32)

                            save_images_val(image,
                                    '{}'.format(img_name))
                            sess.close()

