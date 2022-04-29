import numpy as np
import tifffile as T
from skimage.morphology import remove_small_objects
from skimage import io

#Load desired file to postprocess
path_img= r'C:\Users\sonil\PycharmProjects\Semester project\Data_output\classical_unsup\out_ACWE_inhouse.tif'
img=np.divide(T.imread(path_img),np.amax(T.imread(path_img)))

img=img.astype('bool')
im_post=remove_small_objects(img, min_size=200, connectivity=1)
#Location desired to save postprocessed file
io.imsave(r'C:\Users\sonil\PycharmProjects\Semester project\Data_output\classical_unsup\out_ACWE_inhouse_postproc.tif', im_post, plugin='tifffile')
