from logger import setup_logger
from utils import *
from glob import glob
from skimage import measure as m
import os

sample_files = sorted(glob('/media/ksc/code/tubulin-model-data/tubulin-model-3/model-3-TEST/*'))

# for sample_file in sample_files:
#     dir = sample_file + '/'
#     filelist = get_filelist(dir, [])
#     for fl in filelist:
#         if 'log_scores_wf+U-SRM-to-SRM.txt' in fl:
#             if os.path.exists(fl):
#                 os.remove(fl)
count = 0
for sample_file in sample_files:
    dir = sample_file + '/'

    filelist = get_filelist(dir, [])
    for fl in filelist:
        # print(fl)
        if 'perfect.tif' in fl and 'post/' not in fl:
            print(fl)
            image1 = scipy.misc.imread(fl)
            compare_files = sorted(glob(fl[:-11]+'wf+MR-SRM-to-SRM-desired_output/*'))
            logger = setup_logger("evaluate{}".format(count), fl[:-11], filename='log_scores_wf+MU-SRM-to-SRM.txt')
            count += 1
            for cfl in compare_files:
                # if not ('wf.tif' in cfl or 'log_scores.txt' in cfl):
                if not '.txt' in cfl and 'ak' in cfl:
                    image2 = scipy.misc.imread(cfl)

                    image1 = image1.astype(np.float32)
                    image2 = image2.astype(np.float32)
                    # psnr = m.compare_psnr(image1, image2, data_range=255)
                    ssim= m.compare_ssim(image1, image2, K1=0.01, K2=0.03, win_size=11, data_range=255,
                                                multichannel=True)
                    im_name = cfl.split('/')
                    logger.info('{}  ,  ssim  : {}'.format(im_name[-1], ssim))


