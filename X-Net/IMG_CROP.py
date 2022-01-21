import cv2
from glob import glob
from utils import get_filelist
import os
from libtiff import TIFF
from PIL import Image
class UseCv:
    def __init__(self):
        self.path1 = '/media/ksc/code/tubulin-model-data/tubulin-model-3/new_testdata/'
        self.example = '5'
        self.save_folder = '/cut/'
        self.sample_files = sorted(glob(self.path1+self.example+'/*'))
    def cut(self):
        h1 = 300
        w1 = 0
        for sample_file in self.sample_files:
            dir = sample_file + '/'
            filelist = get_filelist(dir, [])
            for fl in filelist:
                if 'perfect' in fl and '.tif' in fl:
                    img_1 = cv2.imread(fl,flags=cv2.IMREAD_COLOR)
                    img_1 = img_1[h1:,w1:]
                    break
        bbox = cv2.selectROI(img_1,False)
        print(self.sample_files)
        for sample_file in self.sample_files:
            dir = sample_file + '/'
            filelist = get_filelist(dir, [])
            for fl in filelist:
                if 'dense/1-1.tif' in fl and '.tif' in fl:
                    save_path = self.path1 + self.save_folder + self.example + '/dense/'
                elif 'perfect/1-1.tif' in fl and '.tif' in fl:
                    save_path = self.path1 + self.save_folder + self.example + '/perfect/'
                elif 'sparse/1-1.tif' in fl and '.tif' in fl:
                    save_path = self.path1 + self.save_folder + self.example + '/sparse/'
                elif 'wf/1-1.tif' in fl and '.tif' in fl:
                    save_path = self.path1 + self.save_folder + self.example + '/wf/'
                if '1-1.tif' in fl:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    # img_1 = cv2.imread(fl, flags=cv2.IMREAD_COLOR)
                    print(fl)
                    img_1 = Image.open(fl,mode='r')
                    print(fl)
                    str = fl.split('/')
                    im_name = str[-1]
                    # cut_p = img_1[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
                    cut_p = img_1.crop((w1+bbox[0],h1+bbox[1], w1+bbox[0]+bbox[2],h1+bbox[1]+bbox[3]))
                    # cv2.imwrite(save_path+im_name,cut_p)
                    print(save_path,im_name)
                    cut_p.save(save_path+im_name)
                    # img_1 = TIFF.open(save_path+im_name, mode='w')
                    # img_1.write_image(cut_p, compression=None)
                    # img_1.close()

if __name__=='__main__':
    UseCv().cut()
