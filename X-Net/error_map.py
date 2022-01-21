#encoding=utf-8
from logger import setup_logger
from utils import *
from glob import glob
from skimage import measure as m
import os
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib as mlp

sample_files = sorted(glob('/media/ksc/code/Figure 4E/AK3 is better than perfect/'))
print(sample_files)

for sample_file in sample_files:
    dir = sample_file + '/'
    filelist = get_filelist(dir, [])
    for fl in filelist:
        if 'error_map' in fl:
            if os.path.exists(fl):
                os.remove(fl)
count = 0
for sample_file in sample_files:
    dir = sample_file + '/'
    filelist = get_filelist(dir, [])
    for fl in filelist:
        print(fl)
        if 'perfect.tif' in fl:
            print(fl)
            image1 = scipy.misc.imread(fl)
            compare_files = sorted(glob(fl[:-11]+'*'))
            count += 1
            for cfl in compare_files:
                # if not ('wf.tif' in cfl or 'log_scores.txt' in cfl):
                if  'ak3' in cfl:
                    image2 = scipy.misc.imread(cfl)

                    image1 = image1.astype(np.float32)
                    image2 = image2.astype(np.float32)

                    emap = np.abs(image1-image2)
                    # emap[:,:,1] = emap[:,:,0]
                    # emap[:,:,2] = emap[:,:,0]
                    emap = emap[:,:,0]
                    # color = ['blue','cyan','green','Lime','purple','black','orange','cyan']
                    # cmap = mlp.colors.ListedColormap(color)
                    plt.imshow(emap,interpolation='lanczos')
                    plt.xticks(fontproperties='Arial', weight='bold')
                    plt.yticks(fontproperties='Arial', weight='bold')
                    cbar = plt.colorbar(shrink=0.7)
                    font = {'family':'Times New Roman',
                            'weight': 'bold'}
                    # cbar.ax.tick_params(labelsize=13)
                    # cbar.set_ticklabels([0,50,100])
                    image_path =cfl.split('.tif')
                    plt.savefig(image_path[0]+'_error_map.png')
                    plt.show()
                    # result_image = Image.fromarray(emap, 'RGB')
                    # image_path =cfl.split('.tif')
                    # result_image.save(image_path[0]+'_error_map.tif')