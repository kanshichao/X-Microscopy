from logger import setup_logger
from utils import *
from glob import glob
from skimage import measure as m
import os

sample_files = sorted(glob('/media/ksc/code/example-data-2022.12.12/tubulin-models/example-data-tubulin-model-WF+MU-SRM-to-SRM/test-example-data/*'))

# for sample_file in sample_files:
#     dir = sample_file + '/'
#     filelist = get_filelist(dir, [])
#     for fl in filelist:
#         if 'log_scores_wf+sparse-to-SRM.txt' in fl:
#             if os.path.exists(fl):
#                 os.remove(fl)
count = 0
for sample_file in sample_files:
    dir = sample_file + '/'

    filelist = get_filelist(dir, [])
    for fl in filelist:
        # print(fl)
        if 'F-SRM.tif' in fl and 'post/' not in fl:
            print(fl)
            image1 = scipy.misc.imread(fl)
            compare_files = sorted(glob(fl.replace(fl.split('/')[-1],'')+'/wf+MU-SRM-to-SRM-desired_output/*'))
            logger = setup_logger("evaluate{}".format(count), fl.replace(fl.split('/')[-1],''), filename='log_scores_wf+MU-SRM-to-SRM.txt')
            count += 1
            for cfl in compare_files:
                # if not ('wf.tif' in cfl or 'log_scores.txt' in cfl):
                if not '.txt' in cfl and 'XN' in cfl:
                    image2 = scipy.misc.imread(cfl)

                    image1 = image1.astype(np.float32)
                    image2 = image2.astype(np.float32)
                    # image1[:,:,1] = 0
                    # image1[:,:,2] = 0
                    # image2[:,:,1] = 0
                    # image2[:,:,2] = 0
                    # psnr = m.compare_psnr(image1, image2, data_range=255)
                    ssim= m.compare_ssim(image1, image2, K1=0.01, K2=0.03, win_size=11, data_range=255,
                                                multichannel=True)
                    im_name = cfl.split('/')
                    logger.info('{}  ,  ssim  : {}'.format(im_name[-1], ssim))


