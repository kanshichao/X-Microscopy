from __future__ import division
import time
from glob import glob
from six.moves import xrange
from utils import *
import tensorflow as tf
import os
import cv2
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
            # data = glob('/media/ksc/code/tubulin-model-data/{}/model-3-training-samples/*'.format(self.dataset_name))
            # data = glob('/media/ksc/code/2022.06.16--test-superresolution/{}/finetune-samples/*'.format(self.dataset_name))
            data = glob('/media/ksc/code/2022.06.16--test-superresolution/{}/training-samples/*'.format(self.dataset_name))
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
                for ss in range(0, 10):
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
        # data = np.random.choice(glob('/media/ksc/code/tubulin-model-data/{}/model-3-training-samples/*'.format(self.dataset_name)),self.batch_size)
        # data = np.random.choice(glob('/media/ksc/code/2022.06.16--test-superresolution/{}/finetune-samples/*'.format(self.dataset_name)),self.batch_size)
        data = np.random.choice(glob('/media/ksc/code/2022.06.16--test-superresolution/{}/training-samples/*'.format(self.dataset_name)),self.batch_size)
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

    def test(self, args):
        sample_files = glob('./datasets/{}/test_train/wild/*.tif'.format(self.dataset_name))
        count = 0
        for sample_file in sample_files:
            graph = tf.Graph()
            with graph.as_default():
                count = count+1
                print("sampling image ", count)

                image1 = scipy.misc.imread(sample_file)
                # image1 = image1[2016:,:,:]
                image1 = image1 /255.
                hazed_image_placeholder = tf.placeholder(tf.float32,
                                                     shape=[1, image1.shape[0], image1.shape[1],
                                                            3])
                # img_name = split_name('/','.',sample_file)
                img_name = sample_file + ''
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
                                './{}/{}'.format(args.test_dir, img_name))
                sess.close()
    def getVarianceMean(self,scr, winSize):
        if scr is None or winSize is None:
            print("The input parameters of getVarianceMean Function error")
            return -1

        if winSize % 2 == 0:
            print("The window size should be singular")
            return -1

        copyBorder_map = cv2.copyMakeBorder(scr, winSize // 2, winSize // 2, winSize // 2, winSize // 2,
                                            cv2.BORDER_REPLICATE)
        shape = np.shape(scr)

        local_mean = np.zeros_like(scr)
        local_std = np.zeros_like(scr)

        for i in range(shape[0]):
            for j in range(shape[1]):
                temp = copyBorder_map[i:i + winSize, j:j + winSize]
                local_mean[i, j], local_std[i, j] = cv2.meanStdDev(temp)
                if local_std[i, j] <= 0:
                    local_std[i, j] = 1e-8

        return local_mean, local_std

    def adaptContrastEnhancement(self,scr, winSize, maxCg):
        if scr is None or winSize is None or maxCg is None:
            print("The input parameters of ACE Function error")
            return -1

        YUV_img = cv2.cvtColor(scr, cv2.COLOR_BGR2YUV)  ##转换通道
        Y_Channel = YUV_img[:, :, 0]
        shape = np.shape(Y_Channel)

        meansGlobal = cv2.mean(Y_Channel)[0]

        ##这里提供使用boxfilter 计算局部均质和方差的方法
        #    localMean_map=cv2.boxFilter(Y_Channel,-1,(winSize,winSize),normalize=True)
        #    localVar_map=cv2.boxFilter(np.multiply(Y_Channel,Y_Channel),-1,(winSize,winSize),normalize=True)-np.multiply(localMean_map,localMean_map)
        #    greater_Zero=localVar_map>0
        #    localVar_map=localVar_map*greater_Zero+1e-8
        #    localStd_map = np.sqrt(localVar_map)

        localMean_map, localStd_map = self.getVarianceMean(Y_Channel, winSize)

        for i in range(shape[0]):
            for j in range(shape[1]):

                cg = 0.2 * meansGlobal / localStd_map[i, j]
                if cg > maxCg:
                    cg = maxCg
                elif cg < 1:
                    cg = 1

                temp = Y_Channel[i, j].astype(float)
                temp = max(0, min(localMean_map[i, j] + cg * (temp - localMean_map[i, j]), 255))

                #            Y_Channel[i,j]=max(0,min(localMean_map[i,j]+cg*(Y_Channel[i,j]-localMean_map[i,j]),255))
                Y_Channel[i, j] = temp

        YUV_img[:, :, 0] = Y_Channel

        dst = cv2.cvtColor(YUV_img, cv2.COLOR_YUV2BGR)

        return dst

    def test_ksc(self, args):
        from image_utils import RandomRotate, CenterCropNumpy, RandomCropNumpy, PoissonSubsampling, \
            AddGaussianPoissonNoise, GaussianBlurring, AddGaussianNoise, ElasticTransform
        from PIL import Image
        from PIL import ImageEnhance
        # sample_files = glob('./datasets/{}/test_train/wild/*.tif'.format(self.dataset_name))
        # sample_files = glob('/media/ksc/code/2022.06.16--test-superresolution/{}/2022.11.09-test/*'.format(self.dataset_name))
        sample_files = glob('/media/ksc/code/example-data-2022.12.12/tubulin-models/example-data-tubulin-model-WF+MU-SRM-to-SRM/{}/*'.format(self.dataset_name))
        # sample_files = glob('/media/ksc/code/2022.06.16--test-superresolution/{}/finetune-samples/*'.format(self.dataset_name))
        # sample_files = glob('/media/ksc/code/tubulin-model-data/{}/model-3-training-samples/*'.format(self.dataset_name))
        # sample_files = glob('/media/ksc/code/control_background_remove')

        iBlur = GaussianBlurring(sigma=1.5)
        iPoisson = PoissonSubsampling(peak=['lognormal', -0.5, 0.001])
        iBG = AddGaussianPoissonNoise(sigma=25, peak=0.06)

        look_up_table = np.empty((1, 256), np.uint8)
        gamma = 0.4
        for i in range(256):
            look_up_table[0, i] = np.clip(np.power(i / 255.0, gamma) * 255.0, 0, 255)
        kernel = np.ones((3, 3), np.int8)
        count = 0
        print(sample_files)
        for sample_file in sample_files:
            dir = sample_file + '/'
            filelist = get_filelist(dir, [])
            # print(filelist)
            for fl in filelist:
                # if '/' in fl and '.tif' in fl and ('test-before' not in fl and 'perfect' not in fl and 'sparse-generated' not in fl and 'wf+generated_sparse-to-SRM-desired_output' not in fl):
                # if '/1/wf/1-1.tif' in fl or '/2/wf/1-1.tif' in fl or '/3/wf/1-1.tif' in fl or '/4/wf/1-1.tif' in fl:
                # if '/11/crop/' in fl and ('test-before' not in fl and 'perfect' not in fl and 'sparse-generated' not in fl and 'wf+generated_sparse-to-SRM-desired_output' not in fl):
                if 'wf.tif' in fl:
                    sample_file_img = fl
                    # print(fl)
                    graph = tf.Graph()
                    with graph.as_default():
                        count = count+1
                        print("sampling image ", count)

                        image1 = scipy.misc.imread(sample_file_img)

                        if args.scale!=1.0:
                            image0 = image1
                            image1 = cv2.resize(image0, (image0.shape[1] * args.scale, image0.shape[0] *args.scale))
                            # image1 = cv2.dilate(image1, kernel)
                            image1 = iBlur(image1)
                        # image1 = iBG(image1)

                        # image1 = cv2.LUT(image1, look_up_table)
                        # image1 = image1*1.0
                        # image1 = cv2.erode(image1,kernel,iterations=2)
                        # image1 = cv2.dilate(image1,kernel,iterations=1)haoxi

                        # image1 = cv2.resize(image1, (image0.shape[1]//2, image0.shape[0]//2))
                        # image1 = cv2.resize(image1,(image0.shape[1] *2, image0.shape[0] *2))

                        # image1 = scipy.misc.imresize(image0, size=(image0.shape[0] //3, image0.shape[1] //3))
                        # image1[image1<20] = 0
                        # image1 = image1[2016:,:,:]

                        if args.scale>2.0:
                            # image1 = self.adaptContrastEnhancement(image1, 11, 5)
                            enh = ImageEnhance.Contrast(Image.fromarray(image1))
                            contrast = 1.5
                            image1 =np.array(enh.enhance(contrast))
                        image1 = image1 /255.
                        if args.scale > 2.0:
                            image1 = np.power(image1, 1.4)


                        # image1 = image1 / 127.5 - 1.
                        if image1.ndim==2:
                            img_AA = np.zeros((image1.shape[0], image1.shape[1], 3))
                            img_AA[:, :, 0] = image1
                            img_AA[:, :, 1] = image1
                            img_AA[:, :, 2] = image1
                            image1 = img_AA
                        # img_AA = np.zeros((image1.shape[0], image1.shape[1], 3))
                        # img_AA[:, :, 0] = image1[:,:,1]
                        # img_AA[:, :, 1] = image1[:,:,1]
                        # img_AA[:, :, 2] = image1[:,:,1]
                        # image1 = img_AA
                        # # image1[image1 < 0.01] = 0


                        hazed_image_placeholder = tf.placeholder(tf.float32,
                                                     shape=[1, image1.shape[0], image1.shape[1],
                                                            3])
                        img_name =fl.replace(''+fl.split('/')[-1],'')
                        # img_name = fl[:-10]
                        # print(img_name)

                        img_name = img_name + 'MU-SRM/'
                        if not os.path.exists(img_name):
                            os.makedirs(img_name)
                        img_name = img_name + 'MU-SRM.tif'
                        # img_name = img_name + fl.split('/')[-1]
                        # img_name1 = img_name + 'wf'+fl.split('/')[-1]
                        if self.withbn:
                            dehaze_image,hazemap = self.pix_model.inference(hazed_image_placeholder,batch_size=1, h=image1.shape[0], w=image1.shape[1])

                        print(args.checkpoint_dir)

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
                            start = time.clock()
                            sample,hm = sess.run(
                            [dehaze_image,hazemap], feed_dict={hazed_image_placeholder: [image1]})
                            end = time.clock()
                            print(end - start)

                            image = np.array(sample[0]).astype(np.float32)

                            # image[image < 0.1] = 0

                            if args.scale != 1.0:
                                image = cv2.resize(image, (image0.shape[1], image0.shape[0]))
                                # image = cv2.resize(image, (1024, 1024))
                                image = image*1.0

                            # image = scipy.misc.imresize(image, size=(image0.shape[0], image0.shape[1]))
                            if args.scale > 2.0:
                                image = self.adaptContrastEnhancement((image+1.0)*127.5, 11, 5) / 127.5 - 1.0

                            save_images_val(image,
                                    '{}'.format(img_name))
                            # save_images_val(image1,
                            #                 '{}'.format(img_name1))
                            sess.close()

    def test_large(self, args):
        sample_file = '/media/ksc/code/test_large/Export_1002-20_RGB_New.tif'
        grid = 4096
        if True:
            print(sample_file)
            image_ori = scipy.misc.imread(sample_file)
            m,n,ss = image_ori.shape
            print(m, n)
            spare_im = np.zeros((m,n,3))
            count = 0
            img_name = '/media/ksc/code/test_large/Export_1002-20_RGB_New_sparse.tif'
            for i in range(int((m) / grid)):
                for j in range(int((n)/grid)):
                    graph = tf.Graph()
                    with graph.as_default():
                        count = count + 1
                        print("sampling image ", count)

                        if grid*(i+1) <= m and grid*(j+1)<=n:
                            image1 = image_ori[grid*i:grid*(i+1),grid*j:grid*(j+1),:]
                        else:
                            image1 = image_ori[grid*(i+1):,grid*(j+1):,:]
                        image1 = image1 / 255.
                        if image1.ndim == 2:
                            img_AA = np.zeros((image1.shape[0], image1.shape[1], 3))
                            img_AA[:, :, 0] = image1
                            img_AA[:, :, 1] = image1
                            img_AA[:, :, 2] = image1
                            image1 = img_AA

                        hazed_image_placeholder = tf.placeholder(tf.float32,
                                                                 shape=[1, image1.shape[0], image1.shape[1],
                                                                        3])
                        if self.withbn:
                            dehaze_image, hazemap = self.pix_model.inference(hazed_image_placeholder, batch_size=1,
                                                                             h=image1.shape[0], w=image1.shape[1])

                        saver = tf.train.Saver(tf.global_variables())
                        with tf.Session(graph=graph, config=tf.ConfigProto(
                                allow_soft_placement=True,
                                log_device_placement=False)) as sess:
                        # with tf.Session(graph=graph, config=tf.ConfigProto(
                        #             allow_soft_placement=True,
                        #             log_device_placement=False,
                        #             gpu_options=tf.GPUOptions(allow_growth=True,
                        #                                       per_process_gpu_memory_fraction=1,
                        #                                       visible_device_list="0"))) as sess:
                            ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
                            if ckpt and ckpt.model_checkpoint_path:
                                saver.restore(sess, ckpt.model_checkpoint_path)
                            sample, hm = sess.run(
                                [dehaze_image, hazemap], feed_dict={hazed_image_placeholder: [image1]})

                            image = np.array(sample[0]).astype(np.float32)

                            # save_images_val(image,
                            #                 '{}'.format('/media/ksc/code/test_large/sub/{}.tif'.format(count)))
                            if grid * (i + 1) <= m and grid * (j + 1) <= n:
                                spare_im[grid * i:grid * (i + 1), grid * j:grid * (j + 1), :] = image
                            else:
                                spare_im[grid * (i + 1):, grid * (j + 1):, :] = image
                            sess.close()
            scipy.misc.imsave(img_name,spare_im)
            # save_images_val(spare_im,'{}'.format(img_name))
