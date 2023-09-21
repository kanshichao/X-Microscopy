import argparse
import os
import tensorflow as tf
from Model_lib import pix2pix_generator
from train import pix2pix
from glob import glob
from utils import *
from logger import setup_logger
from PIL import Image
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='test-example-data', help='name of the dataset')
# parser.add_argument('--dataset_name', dest='dataset_name', default='EB1', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=1000, help='# of epoch')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')#default=512,1024,256
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--phase', dest='phase', default='test', help='train, test')
parser.add_argument('--scale', dest='scale', default=1.0, help='1.0 for most models, 3.0 or laminB1')
# parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='/media/ksc/code/2022.06.16--test-superresolution/microtubule/test/wf-to-generated-sparse/', help='models are saved here')
# parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='/media/ksc/code/2022.06.16--test-superresolution/2022.11.09-LaminB1-bottom/original-model/UR-NET-8/', help='models are saved here')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='/media/ksc/code/example-data-2022.12.12/tubulin-models/example-data-tubulin-model-WF+MU-SRM-to-SRM/model/UR-Net-8/', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./experiment_model_54/base_val_finetune/', help='sample are saved here')
parser.add_argument('--abf_dir', dest='abf_dir', default='datasets', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./experiment_model_54/datasets/test_train/save/', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=5000, help='weight on L1 term in objective')
parser.add_argument('--withbn', dest='withbn', type=bool, default=True, help='model with or without bn, if withbn=True, the batch_size must be 1, else the batch_size can be set larger than 1, only True is supported now')#True
parser.add_argument('--same_input_size', dest='same_input_size', type=bool, default=True, help='mode input with same or different size')
parser.add_argument('--Model', dest='Model', default='my', help='which mode to choose, only my is supported now')
parser.add_argument('--log_dir', default='experiment_model_54/runs/logs_generated/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    batch_size = 1
    logger = setup_logger("test_40wild_train", args.log_dir, filename='{}_log.txt'.format(
        'test_wild_train_experiment_1'))
    logger_val = setup_logger("test_40wild_val", args.log_dir, filename='{}_log.txt'.format(
        'test_wild_experiment_1'))
    # os.environ['CUDA_DEVICE_ORDE'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if args.same_input_size:
        base_count = 0
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                pix_model = pix2pix_generator()
                haze_model = pix2pix(sess,pix_model,h = args.fine_size, w = args.fine_size,
                        batch_size = batch_size, L1_lambda=args.L1_lambda, dataset_name=args.dataset_name,
                        input_c_dim=args.input_nc, output_c_dim=args.output_nc,
                        checkpoint_dir=args.checkpoint_dir,withbn = args.withbn,
                                     base_count = base_count,Model = args.Model,logger=logger,logger_val=logger_val)
                if args.phase == 'train':
                    haze_model.train(args)
                else:
                    haze_model.test_ksc(args)
            sess.close()
    if args.phase == 'train' and not args.same_input_size:
        counter = 1
        first_save = True
        base_count = 94312
        # base_count = 10
        for epoch in range(args.epoch):
            # data = glob('./datasets/{}/train/*.tif'.format(args.dataset_name))
            # data = glob('/media/ksc/code/tubulin-model-data/{}/model-3-training-samples/*'.format(args.dataset_name))
            # data = glob('/media/ksc/code/2022.06.16--test-superresolution/{}/finetune-samples/*'.format(args.dataset_name))
            data = glob('/media/ksc/code/2022.06.16--test-superresolution/{}/training-samples/*'.format(args.dataset_name))
            # data = glob('/media/ksc/code/tubulin-model-data/tubulin-model-3/finetune-samples/*')
            # np.random.shuffle(data)
            batch_idxs = min(len(data), args.train_size) // 1
            for idx in range(0, batch_idxs):
                batch_files = data[idx * 1:(idx + 1) * 1]
                for image_path in batch_files:
                    print(image_path)
                    # img_A = imread(image_path)
                    # if 'train' in image_path:
                    #     image_path = image_path.replace('train', 'train_gt')
                    # if 'val' in image_path:
                    #     image_path = image_path.replace('val', 'val_gt')
                    # img_B = imread(image_path)

                    # image_path_train = image_path + '/wf.tif'
                    # img_A = imread(image_path_train)
                    #
                    # img_AA = np.zeros((img_A.shape[0], img_A.shape[1], 3))
                    # img_AA[:, :, 0] = img_A
                    # img_AA[:, :, 1] = img_A
                    # img_AA[:, :, 2] = img_A
                    # img_A = img_AA
                    #
                    # new_n = random.randint(4, 6)
                    # image_path_val = image_path + '/1-' + str(new_n) + '.tif'
                    # img_B = imread(image_path_val)
                    #
                    # for i in range(0, 2):
                    #     new_n = random.randint(4, 6)
                    #     image_path_val = image_path + '/1-' + str(new_n) + '.tif'
                    #     img_B1 = imread(image_path_val)
                    #     img_B[:, :, i + 1] = img_B1[:, :, 0]

                    image_path_train = image_path + '/wf/1-1.tif'
                    img_A = imread(image_path_train)

                    # new_n = random.randint(1, 30)
                    # image_path_val = image_path + '/sparse/1-' + str(new_n) + '.tif'
                    # img_B1 = imread(image_path_val)
                    if img_A.ndim==2:
                        img_AA = np.zeros((img_A.shape[0],img_A.shape[1],3))
                        img_AA[:,:,0] = img_A
                        img_AA[:, :, 1] = img_A
                        img_AA[:, :, 2] = img_A
                        img_A = img_AA
                    num_ge = 10
                    new_n = random.randint(1, num_ge)
                    # image_path_val = image_path + '/dense/1-' + str(new_n) + '.tif'
                    image_path_val = image_path + '/sparse/1-' + str(new_n) + '.tif'
                    img_B = imread(image_path_val)

                    for i in range(0, 2):
                        new_n = random.randint(1, num_ge)
                        # image_path_val = image_path + '/dense/1-' + str(new_n) + '.tif'
                        image_path_val = image_path + '/sparse/1-' + str(new_n) + '.tif'
                        img_B1 = imread(image_path_val)
                        img_B[:, :, i + 1] = img_B1[:, :, 0]

                    # print(np.shape(img_B))

                    if args.phase == 'train' and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)
                    h = img_A.shape[0]
                    w = img_A.shape[1]
                    h1 = int(np.ceil(np.random.uniform(1e-2, 25)))
                    w1 = int(np.ceil(np.random.uniform(1e-2, 25)))
                    img_A = img_A[h1:h-h1, w1:w-w1,:]
                    img_B = img_B[h1:h-h1, w1:w-w1, :]

                    img_A = img_A / 255.
                    img_B = img_B / 255.
                    batch = [np.concatenate((img_A, img_B), axis=2)]
                    batch_images = np.array(batch).astype(np.float32)
                    tf.reset_default_graph()
                    graph = tf.Graph()
                    with graph.as_default():
                        with tf.Session(graph=graph) as sess:
                            pix_model = pix2pix_generator()
                            haze_model = pix2pix(sess,pix_model,h = img_A.shape[0], w = img_A.shape[1],
                                    batch_size = 1, L1_lambda=args.L1_lambda, dataset_name=args.dataset_name,
                                    input_c_dim=args.input_nc, output_c_dim=args.output_nc,
                                    checkpoint_dir=args.checkpoint_dir,withbn = args.withbn,
                                                 base_count=base_count,Model = args.Model,logger=logger,logger_val=logger_val)
                            haze_model.train_different_size(args,batch_images,epoch,idx,counter,first_save=first_save)
                        counter = counter + 1
                        first_save = False
                        sess.close()

    # elif args.phase == 'train':
    #     counter = 1
    #     first_save = True
    #     base_count = 94312
    #     tf.reset_default_graph()
    #     graph = tf.Graph()
    #     with graph.as_default():
    #         with tf.Session(graph=graph) as sess:
    #             pix_model = pix2pix_generator()
    #             fusion_model = pix2pix(sess, pix_model, h=968, w=774,
    #                                    batch_size=batch_size, L1_lambda=args.L1_lambda, dataset_name=args.dataset_name,
    #                                    input_c_dim=args.input_nc, output_c_dim=args.output_nc,
    #                                    checkpoint_dir=args.checkpoint_dir, withbn=args.withbn,
    #                                    base_count=base_count, Model=args.Model, logger=logger, logger_val=logger_val
    #                                    )
    #             fusion_model.train_different_size_1(args,counter)
    #
    # elif args.same_input_size==False:
    #     graph = tf.Graph()
    #     base_count = 0
    #     with graph.as_default():
    #         with tf.Session(graph=graph) as sess:
    #             pix_model = pix2pix_generator()
    #             haze_model = pix2pix(sess, pix_model, h=args.fine_size, w=args.fine_size,
    #                                 batch_size=1, L1_lambda=args.L1_lambda,
    #                                 dataset_name=args.dataset_name,
    #                                 input_c_dim=args.input_nc, output_c_dim=args.output_nc,
    #                                 checkpoint_dir=args.checkpoint_dir, withbn=args.withbn,
    #                                  base_count=base_count,Model = args.Model)
    #             haze_model.test(args)
    #         sess.close()

if __name__ == '__main__':
    tf.app.run()
