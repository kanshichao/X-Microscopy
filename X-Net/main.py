import argparse
import os
import tensorflow as tf
from Model_lib import pix2pix_generator
from train import pix2pix
from glob import glob
from utils import *
import tensorflow as tf
from logger import setup_logger
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='test-example-data', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=1000,help='# of epoch')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--phase', dest='phase', default='test', help='train, test')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='/media/ksc/code/example-data-2022.12.12/tubulin-models/example-data-tubulin-model-WF+MU-SRM-to-SRM/model/X-Net/', help='models are saved here')
parser.add_argument('--best_checkpoint_dir', dest='best_checkpoint_dir', default='./experiment_xl_sparse/check_best_new/', help='best models are saved here')
parser.add_argument('--fine_checkpoint_dir', dest='checkpoint_dir', default='./experiment_xl_sparse/fine_checkpoint/', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./experiment_xl_sparse/base_val_finetune/', help='sample are saved here')
parser.add_argument('--abf_dir', dest='abf_dir', default='./base_super/sample_1129/', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./experiment_xl_sparse/test_0219/', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=1000, help='weight on L1 term in objective')
parser.add_argument('--withbn', dest='withbn', type=bool, default=True, help='model with or without fusion')
parser.add_argument('--same_input_size', dest='same_input_size', type=bool, default=True, help='mode input with same or different size')
parser.add_argument('--Model', dest='Model', default='my', help='which mode to choose, my or other')
parser.add_argument('--log_dir', default='experiment_xl_sparse/runs/logs_xl_sparse/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    if args.withbn == True:
        batch_size = 1
        args.Model = 'my'
    else:
        batch_size = 1
    logger = setup_logger("test_ge_train", args.log_dir, filename='{}_log.txt'.format(
        'train_ge'))
    logger_val = setup_logger("test_ge", args.log_dir, filename='{}_log.txt'.format(
        'val_ge'))
    logger_best = setup_logger("best_val", args.log_dir, filename='{}_log.txt'.format(
        'best_val'))
    logger_best = setup_logger("best_val", args.log_dir, filename='{}_log.txt'.format(
        'best_val'))
    logger_test = setup_logger("tests_val", args.log_dir, filename='{}_train.txt'.format(
        'test_0107'))
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if args.same_input_size:
        base_count = 0
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph,config=config) as sess:
                pix_model = pix2pix_generator()
                fusion_model = pix2pix(sess,pix_model,h = args.fine_size, w = args.fine_size,
                        batch_size = batch_size, L1_lambda=args.L1_lambda, dataset_name=args.dataset_name,
                        input_c_dim=args.input_nc, output_c_dim=args.output_nc,
                        checkpoint_dir=args.checkpoint_dir,withbn = args.withbn,
                                     base_count = base_count,Model = args.Model,logger=logger,logger_val=logger_val,logger_best=logger_best,logger_test=logger_test,
                                       best_checkpoint_dir = args.best_checkpoint_dir)
                if args.phase == 'train':
                    fusion_model.train(args)
                    # fusion_model.train_ge(args)
                else:
                    fusion_model.test_end(args)
            sess.close()
    elif args.phase == 'train' and not args.same_input_size:
        counter = 1
        first_save = True
        base_count = 94312
        for epoch in range(args.epoch):
            # data = glob('./datasets/{}/train_wild/*.tif'.format(args.dataset_name))
            data = glob('{}/example-data-training samples/*'.format(args.dataset_name))
            # data = glob('/home/ksc/anet-models/different_bits/{}/*'.format(args.dataset_name))
            # data = glob('/media/ksc/code/tubulin-model-data/tubulin-model-3/finetune-samples/*')
            # np.random.shuffle(data)
            # print(data)
            # print(data)
            batch_idxs = min(len(data), args.train_size) // 1
            for idx in range(0, batch_idxs):
                batch_files = data[idx * 1:(idx + 1) * 1]
                for image_path in batch_files:
                    print(image_path)

                    image_path_train = image_path + '/wf/1-1.tif'
                    img_A = imread(image_path_train)
                    if img_A.ndim==2:
                        img_AA = np.zeros((img_A.shape[0], img_A.shape[1], 3))
                        img_AA[:, :, 0] = img_A
                        img_AA[:, :, 1] = img_A
                        img_AA[:, :, 2] = img_A
                        img_A = img_AA

                    # select_prob = random.randint(1, 2)
                    num_de = 10
                    select_prob = 2
                    if select_prob == 1:
                        new_n = random.randint(1, num_de)
                        image_path_val = image_path + '/sparse/1-' + str(new_n) + '.tif'
                        img_B = imread(image_path_val)
                        if img_B.ndim == 2:
                            img_BB = np.zeros((img_B.shape[0], img_B.shape[1], 3))
                            img_BB[:, :, 0] = img_B
                            img_B = img_BB
                        img_B[:,:,1] = img_B[:,:,0]
                        img_B[:,:,2] = img_B[:,:,0]
                    else:
                        image_path_val = image_path + '/sparse-generated/1sparse-generated.tif'
                        img_B = imread(image_path_val)

                    # img_B = img_A

                    # img_A = img_B

                    new_n = random.randint(1, num_de)
                    image_path_gt = image_path + '/dense/1-' + str(new_n) + '.tif'
                    img_gt = imread(image_path_gt)
                    if img_gt.ndim == 2:
                        img_BB = np.zeros((img_gt.shape[0], img_gt.shape[1], 3))
                        img_BB[:, :, 0] = img_gt
                        img_gt = img_BB
                    for rani in range(2):
                        new_n = random.randint(1, num_de)
                        image_path_gt = image_path + '/dense/1-' + str(new_n) + '.tif'
                        img_gt1 = imread(image_path_gt)
                        if img_gt1.ndim == 2:
                            img_BB = np.zeros((img_gt1.shape[0], img_gt1.shape[1], 3))
                            img_BB[:, :, 0] = img_gt1
                            img_gt1 = img_BB
                        img_gt[:,:,rani+1] = img_gt1[:,:,0]


                    if args.phase == 'train' and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)
                        img_gt = np.fliplr(img_gt)
                    h = img_A.shape[0]
                    w = img_A.shape[1]
                    h1 = int(np.ceil(np.random.uniform(1e-2, 25)))
                    w1 = int(np.ceil(np.random.uniform(1e-2, 25)))
                    img_A = img_A[h1:h-h1, w1:w-w1, :]
                    img_B = img_B[h1:h-h1, w1:w-w1, :]
                    img_gt = img_gt[h1:h - h1, w1:w - w1, :]

                    img_A = img_A / 255.
                    img_B = img_B / 255.
                    img_gt = img_gt / 255.

                    batch = [np.concatenate((img_A, img_B, img_gt), axis=2)]
                    batch_images = np.array(batch).astype(np.float32)
                    tf.reset_default_graph()
                    graph = tf.Graph()
                    with graph.as_default():
                        with tf.Session(graph=graph, config=config) as sess:
                            pix_model = pix2pix_generator()
                            fusion_model = pix2pix(sess,pix_model,h = img_A.shape[0], w = img_A.shape[1],
                                    batch_size = 1, L1_lambda=args.L1_lambda, dataset_name=args.dataset_name,
                                    input_c_dim=args.input_nc, output_c_dim=args.output_nc,
                                    checkpoint_dir=args.checkpoint_dir,withbn = args.withbn,
                                    base_count=base_count,Model = args.Model,logger=logger,logger_val=logger_val,logger_best=logger_best,logger_test=logger_test,
                                       best_checkpoint_dir = args.best_checkpoint_dir)
                            fusion_model.train_different_size(args,batch_images,epoch,idx,counter,first_save=first_save)
                        counter = counter + 1
                        first_save = False
                        sess.close()
    else:
        tf.reset_default_graph()
        graph = tf.Graph()
        base_count = 0
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                pix_model = pix2pix_generator()
                fusion_model = pix2pix(sess, pix_model, h=args.fine_size, w=args.fine_size,
                                    batch_size=1, L1_lambda=args.L1_lambda,
                                    dataset_name=args.dataset_name,
                                    input_c_dim=args.input_nc, output_c_dim=args.output_nc,
                                    checkpoint_dir=args.checkpoint_dir, withbn=args.withbn,
                                     base_count=base_count,Model = args.Model)
                fusion_model.test_end(args)
            sess.close()

if __name__ == '__main__':
    tf.app.run()
