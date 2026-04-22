import os
import threading

import numpy as np
import cv2

import util.util
from options.test_options import TestOptions
from model.pix2pixHD.models import create_model
import util.util as util
import torch
from util.multithread_video_loader import MultithreadVideoLoader
from util.image2video import Image2VideoWriter
from SMPL.smpl_regressor import SMPL_Regressor
import glm
from OffscreenRenderer.flat_renderer import FlatRenderer
from util.image_warp import zoom_in, zoom_out, shift_image_right, shift_image_down, rotate_image
from util.image_process import blur_image
#from composition.short_sleeve_composition import ShortSleeveComposer
from model.DensePose.densepose_extractor import DensePoseExtractor
import time

from SMPL.fullbody_smpl.FullBody import FullBodySMPL
from tqdm import tqdm
from composition.naive_overlay import naive_overlay, naive_overlay_alpha
from util.densepose_util import IUV2UpperBodyImg, IUV2TorsoLeg, IUV2SDP, IUV2SSDP
from threading import Thread
from util.cv2_trans_util import TemporalSmoothing


def make_pix2pix_model(name, input_nc=6, output_nc=4, model_name='pix2pixHD_RNN_RGBA', ckpt_dir=None):
    opt = TestOptions().parse(save=False, use_default=True, show_info=False)
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.name = name
    opt.input_nc = input_nc
    opt.output_nc = output_nc
    opt.isTrain = False
    opt.model = model_name
    opt.checkpoints_dir = './rtv_ckpts' if ckpt_dir is None else ckpt_dir
    opt.gpu_ids=''#load to cpu
    model = create_model(opt)
    # print(model)
    return model


class FullBodySeqFrameProcessor:
    def __init__(self, target_name_list=None, ckpt_dir=None):
        if target_name_list is None:
            target_name_list = ['coat_seq_vmssdp2ta_576']
        elif isinstance(target_name_list, str):
            target_name_list = [target_name_list]

        self.ckpt_dir = ckpt_dir
        self.target_name_list = target_name_list
        self.viton_model_list = [None for _ in self.target_name_list]
        self.viton_model = None
        self.lock = threading.Lock()
        self.smpl_regressor = SMPL_Regressor(use_bev=True, fix_body=True)
        self.full_body = FullBodySMPL()
        self.temporal_smoothing = TemporalSmoothing(c=0.9)
        self.roi_height = 576
        self.roi_width = int(self.roi_height * 0.75)
        self.densepose_extractor = DensePoseExtractor()
        self.load_all = Thread(target=self.load_all_models, args=())
        self.load_all.daemon = True
        self.load_all.start()

    def load_all_models(self):
        for i, target_name in enumerate(self.target_name_list):
            new_model = make_pix2pix_model(target_name, ckpt_dir=self.ckpt_dir)
            if self.viton_model_list[i] is None:
                self.viton_model_list[i] = new_model

    def load_one_model(self, target_name):
        new_model = make_pix2pix_model(target_name, ckpt_dir=self.ckpt_dir)
        target_id = self.target_name_list.index(target_name)
        if self.viton_model_list[target_id] is None:
            self.viton_model_list[target_id] = new_model

    def switch_to_target_garment(self, garment_id):
        with self.lock:
            old_model = self.viton_model
            if garment_id < 0:
                self.viton_model = None
            else:
                if self.viton_model_list[garment_id] is None:
                    self.load_one_model(self.target_name_list[garment_id])
                self.viton_model = self.viton_model_list[garment_id].to('cuda:0')
                if hasattr(self.viton_model, 'reset'):
                    self.viton_model.reset()
            self.temporal_smoothing = TemporalSmoothing(c=0.9)
            if old_model is not None and old_model is not self.viton_model:
                old_model = old_model.to('cpu')
                del old_model
                torch.cuda.empty_cache()

    def set_target_garment(self, garment_id):
        t = Thread(target=self.switch_to_target_garment, args=(garment_id,))
        t.daemon = True
        t.start()

    def __call__(self, roi_vm, roi_ssdp):
        if self.viton_model is None:
            return None, None
        vm_tensor = util.im2tensor(roi_vm) * 2.0 - 1.0
        vm_tensor = vm_tensor[:, [2, 1, 0], :, :]
        dp_tensor = util.im2tensor(roi_ssdp) * 2.0 - 1.0
        with self.lock:
            if self.viton_model is None:
                return None, None
            with torch.no_grad():
                target_tensor = self.viton_model.forward(torch.cat([vm_tensor, dp_tensor], 1).cuda())
        roi_target = util.tensor2im(target_tensor[0, [0, 1, 2], :, :], normalize=True, rgb=False)
        roi_alpha = ((target_tensor[0, 3, :, :].clamp(min=-1.0, max=1.0) / 2.0 + 0.5).cpu().numpy() * 255).astype(
            np.uint8)
        return roi_target, roi_alpha

    def forward(self, raw_image, isRGB=False):
        if self.viton_model is None:
            return raw_image
        height = raw_image.shape[0]
        width = raw_image.shape[1]

        smpl_param = self.smpl_regressor.forward(raw_image, False)  # 1.38
        # print(list(smpl_param.keys()))
        if smpl_param is None:
            return raw_image
        trans2roi, inv_trans = self.smpl_regressor.get_fullbody_trans2roi(smpl_param, s=1.4, new_h=self.roi_height,
                                                                          new_w=self.roi_width)
        trans2roi, inv_trans = self.temporal_smoothing(trans2roi)

        vertices = self.smpl_regressor.get_raw_verts(smpl_param)
        vertices = torch.from_numpy(vertices).unsqueeze(0)

        v = vertices

        raw_vm = self.full_body.render(v[0], height=height, width=width)
        roi_vm = cv2.warpAffine(raw_vm, trans2roi, (self.roi_width, self.roi_height), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))

        IUV = self.densepose_extractor.get_IUV(raw_image, isRGB=False)
        if IUV is None:
            IUV = np.zeros_like(raw_image)

        ssdp = IUV2SSDP(IUV)
        roi_ssdp = cv2.warpAffine(ssdp, trans2roi, (self.roi_width, self.roi_height), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))

        roi_target, roi_alpha = self.__call__(roi_vm, roi_ssdp)
        if roi_target is None or roi_alpha is None:
            return raw_image
        raw_target_img = cv2.warpAffine(roi_target, inv_trans, (raw_image.shape[1], raw_image.shape[0]),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))
        raw_alpha = cv2.warpAffine(roi_alpha, inv_trans, (raw_image.shape[1], raw_image.shape[0]),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0,))
        composed_img = naive_overlay_alpha(raw_image, raw_target_img, raw_alpha)

        return composed_img
