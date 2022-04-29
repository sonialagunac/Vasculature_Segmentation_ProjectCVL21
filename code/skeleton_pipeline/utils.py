import cv2
import torch
from tqdm import tqdm
import numpy as np
from morphpoollayer import MorphPool3D
from networks.segmentation import SegmentNet3D_Resnet
from networks.utils import GradXYZ, norm_range

def imfill(skel_filt):
    im_th = skel_filt*255
    im_floodfill = skel_filt*255
    im_floodfill=im_floodfill.astype(np.uint8)
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

def forward(args,dataloader_val,dataset_val):
    # Train
    model = SegmentNet3D_Resnet().cuda()
    grad_fn = GradXYZ().cuda()
    mp3d = MorphPool3D().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # DataParallel
    model = torch.nn.DataParallel(model)
    grad_fn = torch.nn.DataParallel(grad_fn)
    mp3d = torch.nn.DataParallel(mp3d)

    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    model.module.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_pred = checkpoint['best_pred']

    print('Model loaded')
    is_best = False
    best_pred = 0

    with torch.no_grad():
        model.eval()
        iterator = tqdm(dataloader_val,
                        leave=True,
                        dynamic_ncols=True,
                        desc='Validation ::')
        input = dataset_val.img[
                dataset_val.effective_lable_idx[0][0]:dataset_val.effective_lable_idx[0][1],
                dataset_val.effective_lable_idx[1][0]:dataset_val.effective_lable_idx[1][1],
                dataset_val.effective_lable_idx[2][0]:dataset_val.effective_lable_idx[2][1]
                ]
        input_gt = dataset_val.lbl[
                   dataset_val.effective_lable_idx[0][0]:dataset_val.effective_lable_idx[0][1],
                   dataset_val.effective_lable_idx[1][0]:dataset_val.effective_lable_idx[1][1],
                   dataset_val.effective_lable_idx[2][0]:dataset_val.effective_lable_idx[2][1]
                   ]
        input_gt = input_gt // input_gt.max()
        output = np.zeros((1,
                           dataset_val.effective_lable_shape[0],
                           dataset_val.effective_lable_shape[1],
                           dataset_val.effective_lable_shape[2]))
        idx_sum = np.zeros((1,
                            dataset_val.effective_lable_shape[0],
                            dataset_val.effective_lable_shape[1],
                            dataset_val.effective_lable_shape[2]))
        for index, (data, lables) in enumerate(iterator):
            # To CUDA
            data = data.cuda()
            lables = lables.cuda()

            # Network
            seg, _ = model(data)

            # Smooth
            seg = mp3d(seg)
            seg = mp3d(seg, True)
            seg = mp3d(seg)
            seg = mp3d(seg, True)
            seg = mp3d(seg)
            seg = mp3d(seg, True)

            for batch_idx, val in enumerate(seg[:, 0]):
                out_i = index * dataloader_val.batch_size + batch_idx
                z, y, x = np.unravel_index(out_i, (dataset_val.dz, dataset_val.dy, dataset_val.dx))
                z = z * dataset_val.stride[0]
                y = y * dataset_val.stride[1]
                x = x * dataset_val.stride[2]

                idx_sum[0,
                z: z + dataset_val.lables_shape[0],
                y: y + dataset_val.lables_shape[1],
                x: x + dataset_val.lables_shape[2]] += 1

                output[0,
                z: z + dataset_val.lables_shape[0],
                y: y + dataset_val.lables_shape[1],
                x: x + dataset_val.lables_shape[2]] += val.cpu().data.numpy()
        output = output / idx_sum
        output = torch.Tensor(output).unsqueeze(0).cuda()
        input_gt = torch.Tensor(input_gt).unsqueeze(0)

        output_np = output.detach().cpu().numpy()
        input_np = input_gt.detach().cpu().numpy()

        output = norm_range(output)
    return output_np, input_np