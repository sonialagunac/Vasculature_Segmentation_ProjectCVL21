import cv2
import numpy as np

def imfill(skel_filt):
    im_th = skel_filt * 255
    im_floodfill = skel_filt * 255
    im_floodfill = im_floodfill.astype(np.uint8)
    # Mask used to flood filling.
    # The size needs to be 2 pixels larger than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    return im_out