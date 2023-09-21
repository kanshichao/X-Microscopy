import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
from glob import glob

if __name__ == '__main__':
    sample_files = sorted(glob('/media/ksc/code/tubulin-model-data/multicolor-data/EB1/*'))
    for num in sample_files:
        print(num)
        eb1_name = str(num)+'/model3-400-ak3-g-g-tubulin.tif'
        tubulin_name = str(num)+'/model3-400-ak3-g-g-tubulin.tif'
        tubulin_name = tubulin_name.replace('EB1','tubulin')
        save_name_ori = str(num) + '/model3-400-ak3-g-g-tubulin-merged'
        save_name_ori = save_name_ori.replace('EB1', 'tubulin')
        save_name = str(num)+'/model3-400-ak3-g-g-tubulin-merged'
        save_name = save_name.replace('EB1','tubulin')
        image1 = Image.open(eb1_name)
        image2 = Image.open(tubulin_name)

        image1 = np.array(image1)
        image2 = np.array(image2)
        image2[:,:,1] = image1[:,:,0]

        scipy.misc.imsave(save_name_ori + '_ori.tif', image2)
    # print(np.shape(image1),np.shape(image2))
    # image1 = image1.astype(np.float32)
    # image2 = image2.astype(np.float32)

    # plt.imshow(image2)
    # plt.axis('off')
    # plt.show()

        algi = 5
        sum0 = np.sum(np.sum(np.abs(image2[:,:,0] - image2[:,:,1])))
        print(sum0)
        m = np.shape(image1)[0]
        n = np.shape(image1)[1]
        print(m,n)
        for i in range(algi-1,algi):
            for j in range(algi-1,algi):
                image0 = np.zeros((m,n))
                for k in range(2*algi):
                    for l in range(2*algi):
                        image0[k:m-i,l:n-j] = image1[i:m-k,j:n-l,0]
                        sum1 = np.sum(np.sum(np.abs(image2[:,:,0] - image0)))
                        if sum1 <= sum0:
                            image2[:, :, 1] = image0
                            sum0 = sum1
        # plt.imshow(image2)
        # plt.axis('off')
        # plt.show()
        scipy.misc.imsave(save_name + '_align.tif', image2)
