#encoding=utf-8
from __future__ import division
import scipy.misc
import numpy as np
import tensorflow as tf
from PIL import Image as im
import os
import random
def load_data(image_path, flip=True, is_test=False,axis = 1,reg=False):
    # print(image_path)
    # img_A, img_B, img_gt = load_image(image_path)
    img_A, img_B, img_gt = load_image_random_ksc(image_path)
    img_A, img_B, img_gt = preprocess_A_and_B(img_A, img_B, img_gt,flip=flip, is_test=is_test)
    if reg:
        img_A = img_A / 255.
        img_B = img_B / 255.
        img_gt = img_gt / 255.
    else:
        img_A = img_A / 255.
        img_B = img_B / 255.
        img_gt = img_gt / 255.
    # img_A = img_A / 65535.
    # img_B = img_B / 65535.
    # img_gt = img_gt / 65535.
    img_AB = np.concatenate((img_A, img_B, img_gt), axis=axis)
    return img_AB

def load_ge_data(image_path, flip=True, is_test=False,axis = 1,reg=False):
    print(image_path)
    img_A, img_B, img_gt = load_ge_image(image_path)
    img_A, img_B, img_gt = preprocess_A_and_B(img_A, img_B, img_gt,flip=flip, is_test=is_test)
    if reg:
        img_A = img_A / 255.
        img_B = img_B / 255.
        img_gt = img_gt / 255.
    else:
        img_A = img_A / 255.
        img_B = img_B / 255.
        img_gt = img_gt / 255.
    img_AB = np.concatenate((img_A, img_B,img_gt), axis=axis)
    return img_AB

def load_image_test(image_path):
    img = imread(image_path)
    width = img.shape[1]
    h = 4
    # img_A = img[:,width//h:width//h*(h-3),:]
    # img_A = img[:,width//h*(h-2):width//h*(h-1),:]
    # img_B = img[:, width // h * (h - 1):width // h * (h - 0), :]
    img_A = img[:, 0: (width - 15) // h * (h - 3), :]
    img_B = img[:, (width - 15) // h * (h - 3) + 5:(width - 15) // h * (h - 2) + 5, :]
    img_C = img[:, (width - 15) // h * (h - 2) + 10:(width - 15) // h * (h - 1) + 10, :]
    img_D = img[:, (width - 15) // h * (h - 1) + 15:(width - 15) // h * (h - 0) + 15, :]
    #img_B = img[:, width // h * (h - 1):width // h * (h - 0), :]
    return img_A,img_B,img_C,img_D#,img_B1,img_B2

def load_gt_image(image_path,reg=True):
    if 'train' in image_path:
        image_path = image_path.replace('train_wild','train_gt')
    if 'val' in image_path:
        image_path = image_path.replace('val_wild','val_gt')
    # image_path = image_path.split('_')[0] + '_' + image_path.split('_')[1] + '.jpg'
    image = imread(image_path)
    image = scipy.misc.imresize(image, [256, 256])
    if image.ndim==2:
        image = image
    else:
        image = image[:,:,:3]
    if reg:
        image = image / 255.
    else:
        image = image / 255.
    return image

def load_gt_image_ksc(image_path,reg=True):
    num_ge = 10
    new_n = random.randint(1, num_ge)
    image_path_val = image_path + '/dense/1-' + str(new_n) + '.tif'
    image = imread(image_path_val)
    if image.ndim == 2:
        img_BB = np.zeros((image.shape[0], image.shape[1], 3))
        img_BB[:, :, 0] = image
        image = img_BB
    for i in range(2):
        new_n = random.randint(1, num_ge)
        image_path_val = image_path + '/dense/1-' + str(new_n) + '.tif'
        img_B1 = imread(image_path_val)
        if img_B1.ndim == 2:
            img_BB = np.zeros((img_B1.shape[0], img_B1.shape[1], 3))
            img_BB[:, :, 0] = img_B1
            img_B1 = img_BB
        image[:, :, i + 1] = img_B1[:, :, 0]
    if reg:
        image = image / 255.
    else:
        image = image / 127.5 - 1.
    # if reg:
    #     image = image / 65535.
    # else:
    #     image = image / 65535. * 2. - 1.
    return image

def load_sparse_image_ksc(image_path,reg=True):
    # select_prob = random.randint(1, 2)
    num_ge = 10
    select_prob = 2
    if select_prob == 1:
        new_n = random.randint(1, num_ge)
        image_path_val = image_path + '/sparse/1-' + str(new_n) + '.tif'
        image = imread(image_path_val)
        if image.ndim == 2:
            img_BB = np.zeros((image.shape[0], image.shape[1], 3))
            img_BB[:,:,0] = image
            image = img_BB
        image[:, :, 1] = image[:, :, 0]
        image[:, :, 2] = image[:, :, 0]
        # for i in range(2):
        #     new_n = random.randint(1, 30)
        #     image_path_val = image_path + '/sparse/1-' + str(new_n) + '.tif'
        #     img_B1 = imread(image_path_val)
        #     image[:, :, i + 1] = img_B1[:, :, 0]
    else:
        image_path_val = image_path + '/sparse-generated/1sparse-generated.tif'
        image = imread(image_path_val)
        # image[:, :, 1] = image[:, :, 0]
        # image[:, :, 2] = image[:, :, 0]


    # image_path = image_path + '/wf/1-1.tif'
    # image = imread(image_path)
    #
    # if image.ndim == 2:
    #     img_AA = np.zeros((image.shape[0], image.shape[1], 3))
    #     img_AA[:, :, 0] = image
    #     img_AA[:, :, 1] = image
    #     img_AA[:, :, 2] = image
    #     image = img_AA

    if reg:
        image = image / 255.
    else:
        image = image / 127.5 - 1.
    # if reg:
    #     image = image / 65535.
    # else:
    #     image = image / 65535.*2. - 1.
    return image

def load_val_image(image_path,reg=True):
    image = imread(image_path)
    image = scipy.misc.imresize(image, [256, 256])
    if image.shape[2]>3:
        image = image[:,:,:3]
    if reg:
        image = image / 255.
    else:
        image = image / 255.
    return image

def load_wf_image_ksc(image_path,reg=True):
    image_path = image_path + '/wf/1-1.tif'
    image = imread(image_path)

    if image.ndim==2:
        img_AA = np.zeros((image.shape[0], image.shape[1], 3))
        img_AA[:, :, 0] = image
        img_AA[:, :, 1] = image
        img_AA[:, :, 2] = image
        image = img_AA

    if reg:
        image = image / 255.
    else:
        image = image / 127.5 - 1.
    # if reg:
    #     image = image / 65535.
    # else:
    #     image = image / 65535.*2. - 1.
    return image


def load_image(image_path):
    img_A = imread(image_path)
    if 'train' in image_path:
        image_path1 = image_path.replace('train_wild','train_sparse')
        # image_path1 = image_path.replace('train_wild', 'train_wild')
        #image_path1 = image_path.replace('train_wild', 'train_sparse')
        #image_path1 = image_path1
    if 'val' in image_path:
        # image_path1 = image_path.replace('val_wild','val_sparse')
        #image_path1 = image_path.replace('val_wild', 'val_wild')
        image_path1 = image_path.replace('val_wild', 'val_sparse')
        #image_path1 = image_path
    img_B = imread(image_path1)
    if 'train' in image_path:
        image_path1 = image_path.replace('train_wild','train_gt')
    if 'val' in image_path:
        image_path1 = image_path.replace('val_wild','val_gt')
    img_gt = imread(image_path1)
    if img_A.shape[2]>3:
        img_A = img_A[:,:,:3]
    if img_B.shape[2]>3:
        img_B = img_B[:,:,:3]
    if img_gt.shape[2]>3:
        img_gt = img_gt[:,:,:3]
    return img_A,img_B,img_gt

def load_ge_image(image_path):
    img_A = imread(image_path)
    file_basename = os.path.splitext(image_path)  # ['','.tif']
    # print(file_basename)
    # print(file_basename[0].split('000'))
    number = int(file_basename[0].split('000')[-1])  # 提取序号
    n = (number - 1) // 30 + 1  # 找到组号
    new_n = 1 + (n - 1) * 30
    image_path_new = file_basename[0].split('000')[0] + '000' + str(new_n) + '.tif'
    if 'train' in image_path:
        image_path1 = image_path_new.replace('train_wild','train_ge_sparse')
        print(image_path1)
        # image_path1 = image_path.replace('train_wild', 'train_wild')
        #image_path1 = image_path.replace('train_wild', 'train_sparse')
        #image_path1 = image_path1
    if 'val' in image_path:
        image_path1 = image_path_new.replace('val_wild','val_sparse')
        #image_path1 = image_path.replace('val_wild', 'val_wild')
        # image_path1 = image_path.replace('val_wild', 'val_sparse')
        #image_path1 = image_path
    img_B = imread(image_path1)
    if 'train' in image_path:
        image_path1 = image_path.replace('train_wild','train_gt')
    if 'val' in image_path:
        image_path1 = image_path.replace('val_wild','val_gt')
    img_gt = imread(image_path1)
    if img_A.shape[2]>3:
        img_A = img_A[:,:,:3]
    if img_B.shape[2]>3:
        img_B = img_B[:,:,:3]
    if img_gt.shape[2]>3:
        img_gt = img_gt[:,:,:3]
    return img_A,img_B,img_gt

def load_image_random_ksc(image_path):
    image_path_train = image_path+'/wf/1-1.tif'
    img_A = imread(image_path_train)

    # print(img_A.ndim)

    if img_A.ndim==2:
        img_AA = np.zeros((img_A.shape[0], img_A.shape[1], 3))
        img_AA[:, :, 0] = img_A
        img_AA[:, :, 1] = img_A
        img_AA[:, :, 2] = img_A
        img_A = img_AA

    # select_prob = random.randint(1,2)
    num_ge = 10
    select_prob = 2
    if select_prob == 1:
        new_n = random.randint(1, num_ge)
        image_path_val = image_path + '/sparse/1-' + str(new_n) + '.tif'
        img_B = imread(image_path_val)
        if img_B.ndim == 2:
            img_BB = np.zeros((img_B.shape[0], img_B.shape[1], 3))
            img_BB[:,:,0] = img_B
            img_B = img_BB
        img_B[:, :, 1] = img_B[:, :, 0]
        img_B[:, :, 2] = img_B[:, :, 0]
        # for i in range(2):
        #     new_n = random.randint(1, 30)
        #     image_path_val = image_path + '/sparse/1-' + str(new_n) + '.tif'
        #     img_B1 = imread(image_path_val)
        #     img_B[:,:,i+1] = img_B1[:,:,0]
    else:
        image_path_val = image_path + '/sparse-generated/1sparse-generated.tif'
        img_B = imread(image_path_val)
        # img_B[:,:,1] = img_B[:,:,0]
        # img_B[:, :, 2] = img_B[:, :, 0]
    # img_B = img_A
    # img_A = img_B
    new_n = random.randint(1, num_ge)
    image_path_gt = image_path + '/dense/1-' + str(new_n) + '.tif'
    img_gt = imread(image_path_gt)
    if img_gt.ndim==2:
        img_gta = np.zeros((img_gt.shape[0], img_gt.shape[1], 3))
        img_gta[:,:,0] = img_gt
        img_gt = img_gta
    for i in range(2):
        new_n = random.randint(1, num_ge)
        image_path_gt = image_path + '/dense/1-' + str(new_n) + '.tif'
        img_B1 = imread(image_path_gt)
        if img_B1.ndim == 2:
            img_BB = np.zeros((img_B1.shape[0], img_B1.shape[1], 3))
            img_BB[:, :, 0] = img_B1
            img_B1 = img_BB
        img_gt[:, :, i + 1] = img_B1[:, :, 0]
    return img_A,img_B,img_gt


def load_image_random(image_path):
    img_A = imread(image_path)
    if 'train' in image_path:
        #image_path1 = image_path.replace('train_wild','train_sparse')
        #image_path1 = image_path.replace('train_wild', 'train_wild')
        image_path1 = image_path.replace('train_wild', 'train_sparse')
        #image_path1 = image_path1
    if 'val' in image_path:
        #image_path1 = image_path.replace('val_wild','val_sparse')
        #image_path1 = image_path.replace('val_wild', 'val_wild')
        image_path1 = image_path.replace('val_wild', 'val_sparse')
        #image_path1 = image_path
    img_B = imread(image_path1)
    file_basename = os.path.splitext(image_path)  # ['','.tif']
    # print(file_basename)
    # print(file_basename[0].split('000'))
    number = int(file_basename[0].split('000')[-1])  # 提取序号
    n = (number - 1) // 30 + 1  # 找到组号
    new_n = []
    img_RRR = []
    img_new = []
    for i in range(3):
        new_n.append(random.randint(1, 30) + (n - 1) * 30)
        new_path = file_basename[0].split('000')[0] + '000' + str(new_n[i]) + '.tif'
        if 'train' in new_path:
            image_path = new_path.replace('train_wild', 'train_gt')
            print(image_path)
        if 'val' in new_path:
            image_path = new_path.replace('val_wild', 'val_gt')
        img_RRR.append(scipy.misc.imread(image_path))
        # img_B = imread(image_path)
    for i in range(3):
        img_new.append(img_RRR[i].transpose(2, 0, 1))
    R_0 = img_new[0][0]
    R_1 = img_new[1][0]
    R_2 = img_new[2][0]
    new_img = np.zeros((3, R_0.shape[0], R_0.shape[1]), dtype=np.float)
    new_img[0] = new_img[0] + R_0
    new_img[1] = new_img[1] + R_1
    new_img[2] = new_img[2] + R_2
    img_gt = new_img.transpose(1, 2, 0)

    # new_n = random.randint(1, 30) + (n - 1) * 30
    # image_path_new = file_basename[0].split('000')[0] + '000' + str(new_n) + '.tif'
    # if 'train' in image_path_new:
    #     image_path1 = image_path_new.replace('train_wild','train_gt')
    #     print(image_path1)
    # if 'val' in image_path:
    #     image_path1 = image_path.replace('val_wild','val_gt')
    # img_gt = imread(image_path1)
    if img_A.shape[2]>3:
        img_A = img_A[:,:,:3]
    if img_B.shape[2]>3:
        img_B = img_B[:,:,:3]
    if img_gt.shape[2]>3:
        img_gt = img_gt[:,:,:3]
    return img_A,img_B,img_gt

# 286+250+512
# 544 368 + 176
def preprocess_A_and_B(img_A, img_B,img_gt,load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
        img_gt = scipy.misc.imresize(img_gt, [fine_size, fine_size])
    else:
        if load_size > 0:
            height = img_A.shape[0]
            width = img_A.shape[1]
            pad = 50
            h0 = int(np.ceil(np.random.uniform(1e-2, pad)))
            h1 = int(np.ceil(np.random.uniform(1e-2, pad)))
            w0 = int(np.ceil(np.random.uniform(1e-2, pad)))
            w1 = int(np.ceil(np.random.uniform(1e-2, pad)))
            img_A = img_A[h0:height-h1,w0:width-w1,:]
            img_B = img_B[h0:height-h1,w0:width-w1,:]
            img_gt = img_gt[h0:height-h1,w0:width-w1,:]
            img_A = scipy.misc.imresize(img_A, [load_size, load_size])
            img_B = scipy.misc.imresize(img_B, [load_size, load_size])
            img_gt = scipy.misc.imresize(img_gt, [load_size, load_size])
            h1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
            w1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
            if img_A.ndim==2:
                img_A = img_A[h1:h1 + fine_size, w1:w1 + fine_size]
            else:
                img_A = img_A[h1:h1 + fine_size, w1:w1 + fine_size, :]
            if img_B.ndim==2:
                img_B = img_B[h1:h1 + fine_size, w1:w1 + fine_size]
            else:
                img_B = img_B[h1:h1 + fine_size, w1:w1 + fine_size, :]
            if img_gt.ndim==2:
                img_gt = img_gt[h1:h1 + fine_size, w1:w1 + fine_size]
            else:
                img_gt = img_gt[h1:h1 + fine_size, w1:w1 + fine_size,:]
        else:
            height = img_A.shape[0]
            width = img_A.shape[1]
            if height < width:
                mv = (width - height) / 2
                w0 = int(np.ceil(np.random.uniform(1e-2, mv)))
                if img_A.ndim==2:
                    img_A = img_A[:, w0:w0 + height]
                else:
                    img_A = img_A[:,w0:w0+height,:]
                if img_B.ndim==2:
                    img_B = img_B[:, w0:w0 + height, :]
                else:
                    img_B = img_B[:, w0:w0 + height,:]
                if img_gt.ndim==2:
                    img_gt = img_gt[:, w0:w0 + height]
                else:
                    img_gt = img_gt[:, w0:w0 + height, :]
            else:
                mv = (height - width) / 2
                h0 = int(np.ceil(np.random.uniform(1e-2, mv)))
                if img_A.ndim==2:
                    img_A = img_A[h0:h0 + width, :]
                else:
                    img_A = img_A[h0:h0+width, :, :]
                if img_B.ndim==2:
                    img_B = img_B[h0:h0 + width, :]
                else:
                    img_B = img_B[h0:h0+width, :, :]
                if img_gt.ndim==2:
                    img_gt = img_gt[h0:h0 + width, :]
                else:
                    img_gt = img_gt[h0:h0 + width, :, :]
            height = img_A.shape[0]
            if height < fine_size:
                img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
                img_B = scipy.misc.imresize(img_B, [fine_size,  fine_size])
                img_gt = scipy.misc.imresize(img_gt, [fine_size, fine_size])
            else:
                mv = (height-fine_size) / 2
                h0 = int(np.ceil(np.random.uniform(1e-2, mv)))
                w0 = int(np.ceil(np.random.uniform(1e-2, mv)))
                if img_A.ndim==2:
                    img_A = img_A[h0:h0 + fine_size, w0:w0 + fine_size]
                else:
                    img_A = img_A[h0:h0+fine_size, w0:w0 + fine_size, :]
                if img_B.ndim==2:
                    img_B = img_B[h0:h0 + fine_size, w0:w0 + fine_size, :]
                else:
                    img_B = img_B[h0:h0+fine_size, w0:w0 + fine_size, :]
                if img_gt.ndim==2:
                    img_gt = img_gt[h0:h0 + fine_size, w0:w0 + fine_size]
                else:
                    img_gt = img_gt[h0:h0 + fine_size, w0:w0 + fine_size, :]
        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
            img_gt = np.fliplr(img_gt)
    if img_A.ndim==2:
        img_A = img_A[:,:,None]
    if img_B.ndim==2:
        img_B = img_B[:,:,None]
    if img_gt.ndim==2:
        img_gt = img_gt[:,:,None]
    return img_A, img_B, img_gt

def save_images_val(images, image_path,reg=True,mode='RGB'):
    if reg:
        images = images * 255.
    else:
        images = (images + 1.) * 255. / 2.

    # images[:, :, 0] = (images[:, :, 0] + images[:, :, 1] + images[:, :, 2]) / 3.0
    # images[:, :, 1] = 0
    # images[:, :, 2] = 0

    images[:,:,0] = np.max(images, 2)
    # images = images[:,:,0]
    images[:, :, 1] = 0
    images[:, :, 2] = 0

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    images = tf.saturate_cast(images, tf.uint8)
    images = images.eval(session = sess)
    result_image = im.fromarray(images,mode=mode)
    result_image.save(image_path)

# def save_images_test(images, sess, image_path):
#     images  = (images + 1.) * 127.5
#     result_image = tf.saturate_cast(images, tf.uint8)
#     arr1 = sess.run(result_image)
#     result_image = im.fromarray(arr1, 'RGB')
#     result_image.save(image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def find_last(string, str):
    last_position = -1
    while True:
        position = string.find(str,last_position+1)
        if position == -1:
            return last_position
        last_position = position

def txt_wrap_by(start_str,end,html):
    start = find_last(html,start_str)
    if start >=0:
        start += len(start_str)
        end = html.find(end,start)
        if end >= 0:
            return html[start:end].strip()

def split_name(start_str,end,sample_file):
    basename = sample_file.split(start_str)[-1]
    name = basename.split(end)[0]+'.'+basename.split(end)[1]
    return name

def save_single_dense(path):
    img = scipy.misc.imread(path)
    sample = img.transpose(2, 0, 1)
    R_0 = sample[0]
    R_1 = sample[1]
    R_2 = sample[2]
    R_avg = (R_0 + R_1 + R_2) / 3.0
    new_img = np.zeros((3, 512, 512), dtype=np.int8)
    new_img[0] = new_img[0] + R_avg
    new_img = new_img.transpose(1, 2, 0)
    result_image = im.fromarray(new_img, 'RGB')
    path = os.path.splitext(path)
    print(path)

    image_path = "base_super/sample_1129/single_dense/" + path[0].split('/')[-1] + '.tif'
    print(image_path)
    result_image.save(image_path)
    # scipy.misc.imsave(new_img,image_path)

def save_single_img(path):
    img = scipy.misc.imread(path)
    width = img.shape[1]
    h=4
    img_A = img[:, 0: (width - 15) // h * (h - 3), :]
    img_B = img[:, (width - 15) // h * (h - 3) + 5:(width - 15) // h * (h - 2) + 5, :]
    img_C = img[:, (width - 15) // h * (h - 2) + 10:(width - 15) // h * (h - 1) + 10, :]
    img_D = img[:, (width - 15) // h * (h - 1) + 15:(width - 15) // h * (h - 0) + 15, :]
    sample_b = img_B.transpose(2, 0, 1)
    R_0_b = sample_b[0]
    R_1_b = sample_b[1]
    R_2_b = sample_b[2]
    R_avg_b = (R_0_b + R_1_b + R_2_b) / 3.0
    new_b = np.zeros((3, 512, 512), dtype=np.uint8)
    new_b[0] = new_b[0] + R_avg_b
    new_b = new_b.transpose(1, 2, 0)

    sample_c = img_C.transpose(2, 0, 1)
    R_0_c = sample_c[0]
    R_1_c = sample_c[1]
    R_2_c = sample_c[2]
    R_avg_c = (R_0_c + R_1_c + R_2_c) / 3.0
    # R_max = np.max(sample_c,0)
    new_c = np.zeros((3, 512, 512), dtype=np.uint8)
    new_c[0] = new_c[0] + R_avg_c
    new_c = new_c.transpose(1, 2, 0)

    sample_d = img_D.transpose(2, 0, 1)
    R_0_d = sample_d[0]
    R_1_d = sample_d[1]
    R_2_d = sample_d[2]
    R_avg_d = (R_0_d + R_1_d + R_2_d) / 3.0
    new_d = np.zeros((3, 512, 512), dtype=np.uint8)
    new_d[0] = new_d[0] + R_avg_d
    new_d = new_d.transpose(1, 2, 0)

    x = np.zeros((512, 5, 3), dtype=np.uint8) + 255
    im_result = np.concatenate(
        (img_A, x, new_b, x, new_c, x, new_d), axis=1)
    # im_result = np.concatenate(
    #     (img_A,new_b), axis=1)
    print(im_result.shape)
    result_image = im.fromarray(im_result, 'RGB')
    path = os.path.splitext(path)
    image_path = "base_super/sample_1129/single_img/" + path[0].split('/')[-1] + '.tif'
    result_image.save(image_path)

def save_single_img_max(path):
    img = scipy.misc.imread(path)
    width = img.shape[1]
    h=4
    img_A = img[:, 0: (width - 15) // h * (h - 3), :]
    img_B = img[:, (width - 15) // h * (h - 3) + 5:(width - 15) // h * (h - 2) + 5, :]
    img_C = img[:, (width - 15) // h * (h - 2) + 10:(width - 15) // h * (h - 1) + 10, :]
    img_D = img[:, (width - 15) // h * (h - 1) + 15:(width - 15) // h * (h - 0) + 15, :]
    sample_b = img_B.transpose(2, 0, 1)

    R_argmax_b= np.max(sample_b, 0)
    new_b = np.zeros((3, 512, 512), dtype=np.uint8)
    new_b[0] = new_b[0] + R_argmax_b
    new_b = new_b.transpose(1, 2, 0)

    sample_c = img_C.transpose(2, 0, 1)
    R_argmax_c = np.max(sample_c, 0)
    # R_max = np.max(sample_c,0)
    new_c = np.zeros((3, 512, 512), dtype=np.uint8)
    new_c[0] = new_c[0] + R_argmax_c
    new_c = new_c.transpose(1, 2, 0)

    sample_d = img_D.transpose(2, 0, 1)
    R_argmax_d = np.max(sample_d, 0)
    new_d = np.zeros((3, 512, 512), dtype=np.uint8)
    new_d[0] = new_d[0] + R_argmax_d
    new_d = new_d.transpose(1, 2, 0)

    x = np.zeros((512, 5, 3), dtype=np.uint8) + 255
    im_result = np.concatenate(
        (img_A, x, new_b, x, new_c, x, new_d), axis=1)
    # im_result = np.concatenate(
    #     (img_A,new_b), axis=1)
    print(im_result.shape)
    result_image = im.fromarray(im_result, 'RGB')
    path = os.path.splitext(path)
    image_path = "base_super/sample_1129/single_img/" + path[0].split('/')[-1] + '.tif'
    result_image.save(image_path)



def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def get_filelist(dir, Filelist):
    newDir = dir
    if os.path.isfile(dir):
        Filelist.append(dir)
        # Filelist.append(os.path.basename(dir))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Filelist)

    return Filelist

# if __name__ == '__main__':
#     n = split_name('/','.','experiment_1/test_1225_10/gt//U3.73-11.tif')
#     print(n)