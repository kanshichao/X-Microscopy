#encoding=utf-8
from __future__ import division
import scipy.misc
import numpy as np
import tensorflow as tf
from PIL import Image as im
import random
import os
def load_data(image_path, flip=True, is_test=False,axis = 1,reg=True):
    print(image_path)
    img_A, img_B = load_image_random_ksc(image_path)
    img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)
    if reg:
        img_A = img_A / 255.
        img_B = img_B / 255.
    else:
        img_A = img_A / 127.5 - 1.
        img_B = img_B / 127.5 - 1.
    img_AB = np.concatenate((img_A, img_B), axis=axis)
    return img_AB

def split_name(start_str,end,sample_file):
    basename = sample_file.split(start_str)[-1]
    name = basename.split(end)[0]+'.'+basename.split(end)[1]
    return name

def load_image_test(image_path):
    img = imread(image_path)
    width = img.shape[1]
    h = 3
    # img_A = img[:,width//h:width//h*(h-3),:]
    # img_B = img[:,width//h*(h-3):width//h*(h-2),:]
    img_A = img[:, width // h * (h - 2):width // h * (h - 1), :]
    img_B = img[:, width // h * (h - 1):width // h * (h - 0), :]
    return img_A,img_B#,img_B1,img_B2

def load_gt_image(image_path,reg=False):
    if 'train' in image_path:
        image_path = image_path.replace('train','train_gt')
    if 'val' in image_path:
        image_path = image_path.replace('val','val_gt')
    image = imread(image_path)
    if reg:
        image = image / 255.
    else:
        image = image / 127.5 - 1.
    return image

def load_gt_image_ksc(image_path,reg=True):
    num_ge = 10
    new_n = random.randint(1, num_ge)
    # image_path_gt = image_path + '/dense/1-' + str(new_n) + '.tif'
    image_path_gt = image_path + '/sparse/1-' + str(new_n) + '.tif'
    image = imread(image_path_gt)
    for i in range(0,2):
        new_n = random.randint(1, num_ge)
        # image_path_gt = image_path + '/dense/1-' + str(new_n) + '.tif'
        image_path_gt = image_path + '/sparse/1-' + str(new_n) + '.tif'
        image1 = imread(image_path_gt)
        image[:,:,i+1] = image1[:,:,0]
    if reg:
        image = image / 255.
    else:
        image = image / 127.5 - 1.
    return image

def load_val_image(image_path,reg=True):
    image = imread(image_path)
    if reg:
        image = image / 255.
    else:
        image = image / 127.5 - 1.
    return image

def load_val_image_ksc(image_path,reg=True):
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
    return image

def load_image(image_path):
    img_A = imread(image_path)
    if 'train' in image_path:
        image_path = image_path.replace('train','train_gt')
    if 'val' in image_path:
        image_path = image_path.replace('val','val_gt')
    img_B = imread(image_path)
    return img_A,img_B
def load_image_random(image_path):
    img_A = imread(image_path)
    file_basename = os.path.splitext(image_path) #['','.tif']
    #print(file_basename)
    #print(file_basename[0].split('000'))
    number = int(file_basename[0].split('000')[-1]) #提取序号
    n = (number - 1) // 30 + 1  #找到组号
    new_n = random.randint(1,30)+(n-1)*30
    image_path_new = file_basename[0].split('000')[0]+'000'+str(new_n)+'.tif'
    if 'train' in image_path_new:
        image_path = image_path_new.replace('train','train_gt')
        print(image_path)
    if 'val' in image_path_new:
        image_path = image_path_new.replace('val','val_gt')
    img_B = imread(image_path)
    return img_A,img_B

def load_image_random_ksc(image_path):
    num_ge = 10
    new_n = random.randint(1,num_ge)
    # image_path_val = image_path + '/dense/1-' + str(new_n) + '.tif'
    image_path_val = image_path+'/sparse/1-'+str(new_n)+'.tif'
    img_B = imread(image_path_val)
    for i in range(0,2):
        new_n = random.randint(1, num_ge)
        # image_path_val = image_path + '/dense/1-' + str(new_n) + '.tif'
        jimage_path_val = image_path + '/sparse/1-' + str(new_n) + '.tif'
        img_B1 = imread(image_path_val)
        img_B[:,:,i+1] = img_B1[:,:,0]

    image_path_train = image_path + '/wf/1-1.tif'
    img_A = imread(image_path_train)

    # new_n = random.randint(1, 30)
    # image_path_val = image_path+'/sparse/1-'+str(new_n)+'.tif'
    # img_B1 = imread(image_path_val)

    if img_A.ndim==2:
        img_AA = np.zeros((img_A.shape[0], img_A.shape[1], 3))
        img_AA[:, :, 0] = img_A
        img_AA[:, :, 1] = img_A
        img_AA[:, :, 2] = img_A
        img_A = img_AA
    return img_A,img_B

# def load_image_random(image_path):
#     img_A = imread(image_path)
#     file_basename = os.path.splitext(image_path)  # ['','.tif']
#     # print(file_basename)
#     # print(file_basename[0].split('000'))
#     number = int(file_basename[0].split('000')[-1])  # 提取序号
#     n = (number - 1) // 30 + 1  # 找到组号
#     new_n = []
#     img_RRR = []
#     img_new = []
#     for i in range(3):
#         new_n.append(random.randint(1, 30) + (n - 1) * 30)
#         new_path = file_basename[0].split('000')[0] + '000' + str(new_n[i]) + '.tif'
#         if 'train' in new_path:
#             image_path = new_path.replace('train','train_gt')
#             print(image_path)
#         if 'val' in new_path:
#             image_path = new_path.replace('val','val_gt')
#         img_RRR.append(scipy.misc.imread(image_path))
#         # img_B = imread(image_path)
#     for i in range(3):
#         img_new.append(img_RRR[i].transpose(2, 0, 1))
#     R_0 = img_new[0][0]
#     R_1 = img_new[1][0]
#     R_2 = img_new[2][0]
#     new_img = np.zeros((3, R_0.shape[0], R_0.shape[1]), dtype=np.float)
#     new_img[0] = new_img[0] + R_0
#     new_img[1] = new_img[1] + R_1
#     new_img[2] = new_img[2] + R_2
#     img_B = new_img.transpose(1, 2, 0)
#     return img_A,img_B

def preprocess_A_and_B(img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False):#load_size=286+250,512,1024 ,368
    if is_test:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size]) #,fine_size+176
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
    else:
        height = img_A.shape[0]
        width = img_A.shape[1]
        pad = 50
        h0 = int(np.ceil(np.random.uniform(1e-2, pad)))
        h1 = int(np.ceil(np.random.uniform(1e-2, pad)))
        w0 = int(np.ceil(np.random.uniform(1e-2, pad)))
        w1 = int(np.ceil(np.random.uniform(1e-2, pad)))
        img_A = img_A[h0:height - h1, w0:width - w1, :]
        img_B = img_B[h0:height - h1, w0:width - w1, :]

        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size,:]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size,:]

        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
    return img_A, img_B

def save_images_val(images, image_path,reg=True):
    if reg:
        images = images * 255.
    else:
        images = (images + 1.) * 127.5

    # images[:,:,0] = (images[:,:,0] + images[:,:,1] + images[:,:,2]) / 3.0
    # images[:,:,1] = 0
    # images[:,:,2] = 0

    # if '1-1' in image_path:
    #     images[:, :, 1] = np.max(images, 2)
    #     images[:,:,0] = 0
    #     images[:,:,2] = 0
    # if '1-2' in image_path:
    #     images[:, :, 0] = np.max(images, 2)
    #     images[:, :, 1] = 0
    #     images[:, :, 2] = 0

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    images = tf.saturate_cast(images, tf.uint8)
    images = images.eval(session = sess)
    result_image = im.fromarray(images,'RGB')
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