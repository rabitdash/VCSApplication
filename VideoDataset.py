
from torch.utils.data import Dataset
import os
import torch
import scipy.io as scio
import cv2
import numpy as np
import logging

from tqdm import tqdm

logger = logging.getLogger("ColorCS VideoDataset")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

def fimage2uimage(im):
    mi = np.nanmin(im)
    ma = np.nanmax(im)
    im = (im - mi) / (ma - mi) * (np.power(2.0, 8) - 1) + 0.499999999
    im = im.astype(np.uint8)
    return im

class VideoDataset(Dataset):

    def __init__(self, path, use_cache = True,max_frame=2048):
        super(VideoDataset, self).__init__()
        self.data = []
        stem, suffix = os.path.splitext(path)
        # output_file_path = stem + f"_CS_B{args.B}.mp4"
        if use_cache is True:
            if os.path.exists(f"{stem}.mat"):
                logger.info("Cache found.")
                self.data = scio.loadmat(f"{stem}.mat")["orig"]
            else:
                logger.info("Making Cache...")
                frame_cs_mat_buffer = []
                cap = cv2.VideoCapture(path)
                SOURCE_FRAME_NUM = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                for i in tqdm(range(int(SOURCE_FRAME_NUM))):
                    if i >= max_frame:
                        logger.info("Reach max frame num")
                        break
                    ret, frame = cap.read()
                    frame_cs_mat_buffer.append(frame)
                cap.release()
                mat = np.stack(frame_cs_mat_buffer)
                logger.info("Compressing Cache.")
                scio.savemat(f"{stem}.mat", {"orig":mat}, do_compression=True)
                logger.info("Cache done.")
                self.data = mat

    def __getitem__(self, index):
        '''

        :param index:
        :return: uint8 result
        '''
        gt = self.data[index]
        gt = np.transpose(gt, [2,0,1])
        gt = torch.from_numpy(gt)
        # print(gt.shape)
        # meas = torch.from_numpy(meas['meas'] / 255)
        #
        # gt = gt.permute(2, 3, 0, 1)
        # gt = np.permute(self.data, (2,3,0,1))/255.0
        # gt = torch.from_numpy()
        # print(tran(img).shape)

        return gt

    def __len__(self):

        return len(self.data)
