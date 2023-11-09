import streamlit as st
import numpy as np
import logging
import cv2
from os.path import splitext

logger = logging.getLogger("SCI Decoder")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
from utils import generate_masks
import torch
# import scipy.io as scio
import argparse
import numpy as np
# from thop import profile
from CSNet import CSNet, save_name

VIDEO_WIDTH = 704
VIDEO_HEIGHT = 576
LR_SAMPLING_RATE = 24 * 8
HR_SAMPLING_RATE = 4
# %% Encode
parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')
# parser.add_argument('--last_train', default=26, type=int, help='pretrain model')
parser.add_argument('--checkpoint', default='./revnet_uint8.pthmdl', type=str,
                    help='pretrain model save folder name')
parser.add_argument('--max_iter', default=100, type=int, help='max epoch')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--B', default=8, type=int, help='compressive rate')
# parser.add_argument('--learning_rate', default=0.0001, type=float)
parser.add_argument('--size', default=[576, 704], type=int, help='input image resolution')
# parser.add_argument('--mode', default='noreverse', type=str, help='training mode: reverse or noreverse')
parser.add_argument('--cache', default=8, type=int, help='Compress buffer size, times of compressive rate')
args = parser.parse_args()
args = parser.parse_args()

# 兼容CPU和GPU
device = torch.device("cuda:0") if (torch.cuda.is_available()) else "cpu"
# device = "cpu"
logger.info(f"Running on {device}")

# %% mask
mask, _ = generate_masks("mask.mat")  # mask
mask = mask[:, :576, :704]
mask = mask.to(device)
# %% 对mask沿时间轴进行求和，以便normalize 测量
mask_s = np.sum(mask.cpu().numpy(), axis=0)
index = np.where(mask_s == 0)
mask_s[index] = 1
mask_s = mask_s.astype(np.uint8)
mask_s = torch.from_numpy(mask_s)
mask_s = mask_s.float().to(device)

BAYER = torch.empty(args.cache, args.size[0], args.size[1]).to(device)


def bayer_filter_tensor(imgs: torch.Tensor, color_mode="RGB"):
    # [color_channel,]
    (batch, colors, height, width) = imgs.shape

    # bayer = torch.empty((batch, height, width)).cuda()
    # legacy
    R, G, B = None, None, None
    (R, G, B) = (imgs[:, 0, :, :],
                 imgs[:, 1, :, :],
                 imgs[:, 2, :, :])
    # strided slicing for this pattern:
    #   G R
    #   B G
    BAYER[:, 0::2, 0::2] = R[:, 0::2, 0::2]  # top left
    BAYER[:, 0::2, 1::2] = G[:, 0::2, 1::2]  # top right
    BAYER[:, 1::2, 0::2] = G[:, 1::2, 0::2]  # bottom left
    BAYER[:, 1::2, 1::2] = B[:, 1::2, 1::2]  # bottom right
    return BAYER.float()


def bayer_mosaic_ndarray(img):
    (height, width) = img.shape[:2]
    (B, G, R) = cv2.split(img)
    # debayer_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerGR2RGB)
    bayer = np.empty((height, width), np.uint8)

    # strided slicing for this pattern:
    #   G R
    #   B G
    bayer[0::2, 0::2] = G[0::2, 0::2]  # top left
    bayer[0::2, 1::2] = R[0::2, 1::2]  # top right
    bayer[1::2, 0::2] = B[1::2, 0::2]  # bottom left
    bayer[1::2, 1::2] = G[1::2, 1::2]  # bottom right
    return bayer


def compress_frame(pic: np.ndarray, color_mode="RGB"):
    # RG
    # GB
    global bayer
    with torch.no_grad():
        # buffer =
        mask_t = mask[:args.B, :, :]
        pic_t = np.transpose(pic, [0, 3, 1, 2])

        # else:
        #     # 转换为RGB
        #     pic_t[:,:,:,:] = pic_t[:,:,:,:]
        pic_t = torch.from_numpy(pic_t).to(device)

        bayer_pic_t = bayer_filter_tensor(pic_t, color_mode)
        bayer_pic_t = bayer_pic_t / 255.0
        bayer_pic_t = torch.reshape(bayer_pic_t, [args.cache // args.B, args.B, args.size[0], args.size[1]])

        meas_t = torch.sum(torch.mul(bayer_pic_t, mask_t), dim=1)

        # print(meas_t.shape)
        return meas_t


def fimage2uimage(lr_rec):
    lr_rec[lr_rec > 1] = 1
    lr_rec[lr_rec < 0] = 0
    lr_rec = (lr_rec * 255).astype(np.uint8)
    return lr_rec


def decode(meas: np.ndarray, model):
    # RG
    # GB
    with torch.no_grad():
        model.mask = mask[:8, :, :]
        meas = torch.from_numpy(meas).float().to(device)
        # print(mask_s.shape)
        meas_re = torch.div(meas, mask_s)

        meas_re = torch.unsqueeze(meas_re, 0)
        meas_re = torch.unsqueeze(meas_re, 0)

        # print(meas_re.shape)
        # print(torch.cuda.memory_allocated(0) / 1024 / 1024)
        out_pic1 = model.forward(meas_re, args)
        # print("sss", out_pic1.shape)

        # print(out_pic1.shape)
        out_pic1 = out_pic1[0, :, :, :, :].cpu().numpy()
        out = out_pic1.transpose(1, 0, 2, 3)
        # out = out_pic1[0, :, :, :, :].cpu().numpy()
        # print(out.shape)
        return out



def detect_motion(m, m0):
    # judge difference
    delta = torch.abs(m[0].detach() - m0)
    # print(m0.shape) # 12x37x65
    delta = torch.sum(delta, dim=0) / 12
    # print(torch.max(delta))
    threshold = 0.1
    # delta = torch.nn.functional.relu(delta - threshold) + threshold
    delta = delta.cpu().numpy()
    # show delta and count HR rate
    for i in range(m0.shape[1]):
        for j in range(m0.shape[2]):
            if delta[i, j] < threshold:
                delta[i, j] = 0
            else:
                delta[i, j] = 1
    # print(num_hr/(36.0*64.0))

    # erode and dilate (denoise)
    kernel = np.ones((2, 2))
    delta = cv2.erode(delta, kernel, iterations=1)
    kernel = np.ones((3, 3))
    delta = cv2.dilate(delta, kernel, iterations=2)

    return delta


def SCI_Compress(source_file_path):
    vpic = st.empty()
    snap = st.empty()

    # 捕获视频
    stem = f"CS_B{args.B}"
    output_file_path = "./tmp/" + stem + ".mp4"
    print(f"Write to {output_file_path}")
    cap = cv2.VideoCapture(source_file_path)
    # 定义编解码器，创建VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    FPS = 1
    out = cv2.VideoWriter(output_file_path, fourcc, FPS, (args.size[1], args.size[0]), False)
    # （写出的文件，？？，帧率，（分辨率），是否彩色）  非彩色要把每一帧图像装换成灰度图
    out_hr = cv2.VideoWriter(f"./tmp/{stem}hr.mp4", fourcc, 24, (args.size[1], args.size[0]), True)

    m0 = None
    model = CSNet().to(device)
    model.load_state_dict(torch.load(
        "./MR_01c_res3_l1.pth", map_location=torch.device(device)))
    model.eval()

    # 帧缓冲
    frame_raw_buffer = []
    frame_cs_buffer = []
    frame_cs_mat_buffer = []
    # 仅对视频有用，统计视频总帧数
    SOURCE_FRAME_NUM = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    PROCESSED_FRAME_COUNT = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        # logging.info(f"P {PROCESSED_FRAME_COUNT}/{SOURCE_FRAME_NUM}")
        if ret == True:
            # frame = cv2.flip(frame,0)  #可以进行视频反转
            # write the flipped frame
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #换换成灰度图
            # minsize = min(frame.shape[0],frame.shape[1])
            # frame = cv2.resize(frame,(VIDEO_WIDTH,VIDEO_HEIGHT))
            if PROCESSED_FRAME_COUNT % HR_SAMPLING_RATE == 0:
                I = frame.astype(np.float32) / 255.0
                # convert to 1*3*w*h (channel first)
                test_in = torch.from_numpy(I).permute(
                    (2, 0, 1)).unsqueeze(0).to(device)

                m = model.compress(test_in)
                if m0 is None or (PROCESSED_FRAME_COUNT % LR_SAMPLING_RATE == 0):
                    # first frame: just capture
                    m0 = m[0].detach()
                delta = detect_motion(m, m0)
                # delta = cv2.dilate(delta, kernel, iterations=1)
                # mr = 0.01 + 0.2 * np.sum(delta) / (36.0 * 64)

                delta = cv2.resize(delta, (VIDEO_WIDTH + 20, VIDEO_HEIGHT + 20),
                                   interpolation=cv2.INTER_NEAREST)  # , interpolation=cv2.INTER_NEAREST)

                delta = delta[10:-10, 20:]
                delta[delta > 1] = 1
                delta[delta < 0] = 0
                delta = delta.astype(np.uint8)

                delta = cv2.cvtColor(delta, cv2.COLOR_GRAY2BGR)
                hr_img = delta * frame
                # cv2.imwrite("hr.jpg", hr_img)
                vpic.image(cv2.cvtColor(hr_img,cv2.COLOR_BGR2RGB), caption='变动区域', width=300)
                # delay -= 1
                out_hr.write(hr_img)

            # frame = cv2.resize(frame,dsize=(576,704),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
            PROCESSED_FRAME_COUNT += 1

            if PROCESSED_FRAME_COUNT % LR_SAMPLING_RATE // args.B == 0:
                if len(frame_raw_buffer) < args.cache:
                    # 继续堆
                    logger.info("Buffer append")
                    frame_raw_buffer.append(frame)
                if len(frame_raw_buffer) >= args.cache:
                    # vpic.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption='压缩视频帧', width=300)
                    logger.info(f"COMPRESSING {PROCESSED_FRAME_COUNT}/{SOURCE_FRAME_NUM}")
                    meas_t = compress_frame(np.stack(frame_raw_buffer))
                    frame_cs_buffer.append(meas_t.cpu().numpy())
                    for i in meas_t.cpu().numpy():
                        snap.image(fimage2uimage(i), caption="快照结果", width=300)
                    if len(frame_cs_buffer) > 1:
                        logger.info(f"Writing snapshots buffer...")
                        for frame in frame_cs_buffer:
                            for i in frame:
                                # print(i.shape)
                                # cv2.imshow('frame', i)

                                # i = (i * 255).astype(np.uint8)
                                out.write(fimage2uimage(i))
                        frame_cs_buffer.clear()
                    frame_raw_buffer.clear()
        else:
            logger.info(f"Writing snapshots...")

            for frame in frame_cs_buffer:
                for i in frame:
                    # i = (i * 255).astype(np.uint8)
                    out.write(fimage2uimage(i))
            print(len(frame_cs_mat_buffer))
            cap.release()

    # Release everything if job is finished
    print("release")
    out.release()
    out_hr.release()
    lr_path = output_file_path
    hr_path = f"./tmp/{stem}hr.mp4"
    output_path = package_lr_hr(lr_path, hr_path)
    return output_path


def package_lr_hr(lr_path, hr_path):
    lr_cap = cv2.VideoCapture(lr_path)
    hr_cap = cv2.VideoCapture(hr_path)

    # 定义编解码器，创建VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    FPS = 25
    # （写出的文件，？？，帧率，（分辨率），是否彩色）  非彩色要把每一帧图像装换成灰度图
    HR_SOURCE_FRAME_NUM = int(hr_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    LR_SOURCE_FRAME_NUM = int(lr_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_file_path = f"./tmp/DATA_{HR_SOURCE_FRAME_NUM}_{LR_SOURCE_FRAME_NUM}_.mat.mp4"
    out_package_vd = cv2.VideoWriter(out_file_path, fourcc, FPS, (args.size[1], args.size[0]), False)
    PROCESSED_FRAME_COUNT = 0
    for _ in range(LR_SOURCE_FRAME_NUM):
        lr_ret, lr_frame = lr_cap.read()
        out_package_vd.write(lr_frame[:, :, 0])
    for _ in range(HR_SOURCE_FRAME_NUM):
        hr_ret, hr_frame = hr_cap.read()
        hr_frame = bayer_mosaic_ndarray(hr_frame)
        out_package_vd.write(hr_frame)
    # %% diff transmission
    # while(1):
    #     lr_ret, lr_frame = lr_cap.read()
    #     if lr_ret:
    #         out_package_vd.write(lr_frame[:,:,0])
    #         PROCESSED_FRAME_COUNT += 1
    #         for _ in range(LR_SAMPLING_RATE // HR_SAMPLING_RATE):
    #             hr_ret, hr_frame = hr_cap.read()
    #             if hr_ret:
    #                 hr_frame = bayer_mosaic_ndarray(hr_frame)
    #                 out_package_vd.write(hr_frame)
    #                 PROCESSED_FRAME_COUNT += 1
    #             else:
    #                 break
    #     else:
    #         break
    # %%
    out_package_vd.release()
    return out_file_path


def unpackage_lr_hr(file_path, in_file_name):
    HR_FRAME_COUNT, LR_FRAME_COUNT = int(in_file_name.split('_')[-3]), int(in_file_name.split('_')[-2])

    out_file_path = f"./tmp/out.mp4"

    cap = cv2.VideoCapture(file_path)
    lr_count = 0
    hr_count = 0
    snap = st.empty()

    # final_conv = np.where((initial_conv < 10), initial_conv, 0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    FPS = 24
    out_vd = cv2.VideoWriter(out_file_path, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT), True)
    # VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    SOURCE_FRAME_NUM = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    PROCESSED_FRAME_COUNT = 0
    logger.info("Loading Model")
    rev_net = torch.load(
        args.checkpoint, map_location=device)
    rev_net = rev_net.module if hasattr(rev_net, "module") else rev_net
    logger.info("Load model complete")

    while (1):
        if lr_count >= LR_FRAME_COUNT:
            break
        else:
            if PROCESSED_FRAME_COUNT % LR_SAMPLING_RATE == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, lr_count)
                lr_ret, lr_frame = cap.read()
                # print(lr_frame.shape)
                ## process lr
                lr_frame = lr_frame[:, :, 0]
                lr_frame = lr_frame.astype(np.float32) / 255.0
                # lr_recon
                meas_re = decode(lr_frame, rev_net)
                lr_count += 1
            lr_index = PROCESSED_FRAME_COUNT % LR_SAMPLING_RATE // 24
            lr_recon_frame = np.transpose(meas_re[lr_index, :, :, :], (1, 2, 0))
            lr_recon_frame = fimage2uimage(lr_recon_frame)

            if PROCESSED_FRAME_COUNT % HR_SAMPLING_RATE == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, LR_FRAME_COUNT + hr_count)
                hr_ret, hr_frame = cap.read()
                hr_frame = cv2.cvtColor(hr_frame[:, :, 0], cv2.COLOR_BAYER_GR2RGB)

                delta = np.max(np.where(hr_frame > 1, 1, 0), axis=2)
                delta = delta.astype(np.uint8)
                delta = cv2.cvtColor(delta, cv2.COLOR_GRAY2BGR)
                kernel = np.ones((4, 4))
                delta = cv2.erode(delta, kernel, iterations=2)
                hr_count += 1

            out_frame = (hr_frame * delta + lr_recon_frame * (1 - delta)).astype(
                np.uint8)
            snap.image(cv2.cvtColor(out_frame,cv2.COLOR_BGR2RGB),caption='重建图像', width=300)
            out_vd.write(out_frame)
            PROCESSED_FRAME_COUNT += 1
    out_vd.release()
    return out_file_path


# %% App GUI definition


mcol1, mcol2 = st.columns(2)
with mcol1:
    st.text("视频压缩")
    raw_video = st.file_uploader("原始视频MP4", type='.mp4')
    # print(type(raw_video))
    enc_btn = st.button("压缩")
    if enc_btn:
        raw_video = raw_video.getvalue()
        with open('./tmp/raw.mp4', 'wb') as f:
            f.write(raw_video)
        CSVideo_path = SCI_Compress("./tmp/raw.mp4")
        st.info("压缩完成")
        with open(CSVideo_path, 'rb') as vf:
            st.download_button(label="数据",
                               data=vf.read(),
                               file_name=CSVideo_path)
with mcol2:
    st.text("视频快照重构")
    mat = st.file_uploader("压缩快照数据")

    recon_btn = st.button("重构")
    if recon_btn:
        mat_data = mat.getvalue()
        print(mat.name.split("_"))
        with open(f'./tmp/cs.mat', 'wb') as f:
            f.write(mat_data)
        RecVideo_path = unpackage_lr_hr('./tmp/cs.mat', mat.name)
        st.info("重构完成")
        # vf = open(CSVideo_path, 'rb').read()
        # matf =open(CSMat_path, 'rb').read()
        with open(RecVideo_path, 'rb') as vf:
            st.download_button(label="重构结果",
                               data=vf.read(),
                               file_name=RecVideo_path)
# %%
