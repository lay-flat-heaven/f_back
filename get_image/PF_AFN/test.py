import time
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F

from options.select_struct import Selector
from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
from copytool.copy_utils import get_pairs,get_pair
from edge.mask_get import batch_edging,single_edging

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

def futurama_edge(srcs):
    for src in srcs:
        image_path = "./dataset/real_clothes"
        save_path = "./dataset/real_edge"

        image_path_full = os.path.join(image_path,src)
        save_path_full = os.path.join(save_path,src)

        single_edging(image_path_full,save_path_full)


def futurama_pair(peos, clos):

    get_pair(peos, clos, note="./pairs.txt")

def futurama_generate(peos, clos):

    opt = Selector()

    start_epoch, epoch_iter = 1, 0

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    # print(dataset_size)

    warp_model = AFWM(opt, 3)
    # print(warp_model)
    warp_model.eval()
    warp_model.to(device)
    load_checkpoint(warp_model, opt.warp_checkpoint)

    gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
    # print(gen_model)
    gen_model.eval()
    gen_model.to(device)
    load_checkpoint(gen_model, opt.gen_checkpoint)

    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    step = 0
    step_per_batch = dataset_size / opt.batchSize

    for epoch in range(1, 2):

        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            real_image = data['image']
            clothes = data['clothes']
            ##edge is extracted from the clothes image with the built-in function in python
            edge = data['edge']
            edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(int))
            clothes = clothes * edge

            flow_out = warp_model(real_image.to(device), clothes.to(device))
            warped_cloth, last_flow, = flow_out
            warped_edge = F.grid_sample(edge.to(device), last_flow.permute(0, 2, 3, 1),
                                        mode='bilinear', padding_mode='zeros', align_corners=False)

            gen_inputs = torch.cat([real_image.to(device), warped_cloth, warped_edge], 1)
            gen_outputs = gen_model(gen_inputs)
            p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
            p_rendered = torch.tanh(p_rendered)
            m_composite = torch.sigmoid(m_composite)
            m_composite = m_composite * warped_edge
            p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

            path = 'results/' + opt.name
            os.makedirs(path, exist_ok=True)
            sub_path = path + '/PFAFN'
            os.makedirs(sub_path, exist_ok=True)

            if step % 1 == 0:
                a = real_image.float().to(device)
                b = clothes.to(device)
                c = p_tryon
                combine = c[0].squeeze()
                cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
                rgb = (cv_img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                peo = peos[step]
                clo = clos[step]
                # print(peos,clos,step)
                name = peo[:-4] + "_" + clo[:-4]
                print(name)
                cv2.imwrite(sub_path + '/' + name + '.jpg', bgr)

            step += 1
            if epoch_iter >= dataset_size:
                break

def futurama_single(peos, clos):
    s = time.time()
    futurama_edge(clos)
    futurama_pair(peos,clos)
    futurama_generate(peos,clos)
    e = time.time()
    print(f"finish generating {min([len(peos),len(clos)])} pics in {e-s}s ")




def futurama_all(peos, clos):
    image_path = "./dataset/real_clothes"
    save_path = "./dataset/real_edge"
    batch_edging(image_path, save_path)


    path1 = "./dataset/real_img"
    path2 = "./dataset/real_clothes"

    doc = "./pairs.txt"

    get_pairs(path1, path2, doc)

    opt = Selector()

    start_epoch, epoch_iter = 1, 0

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    # print(dataset_size)

    warp_model = AFWM(opt, 3)
    # print(warp_model)
    warp_model.eval()
    warp_model.to(device)
    load_checkpoint(warp_model, opt.warp_checkpoint)

    gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
    # print(gen_model)
    gen_model.eval()
    gen_model.to(device)
    load_checkpoint(gen_model, opt.gen_checkpoint)

    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    step = 0
    step_per_batch = dataset_size / opt.batchSize

    for epoch in range(1, 2):

        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            real_image = data['image']
            clothes = data['clothes']
            ##edge is extracted from the clothes image with the built-in function in python
            edge = data['edge']
            edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(int))
            clothes = clothes * edge

            flow_out = warp_model(real_image.to(device), clothes.to(device))
            warped_cloth, last_flow, = flow_out
            warped_edge = F.grid_sample(edge.to(device), last_flow.permute(0, 2, 3, 1),
                                        mode='bilinear', padding_mode='zeros',align_corners=False)

            gen_inputs = torch.cat([real_image.to(device), warped_cloth, warped_edge], 1)
            gen_outputs = gen_model(gen_inputs)
            p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
            p_rendered = torch.tanh(p_rendered)
            m_composite = torch.sigmoid(m_composite)
            m_composite = m_composite * warped_edge
            p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

            path = 'results/' + opt.name
            os.makedirs(path, exist_ok=True)
            sub_path = path + '/PFAFN'
            os.makedirs(sub_path, exist_ok=True)

            if step % 1 == 0:
                a = real_image.float().to(device)
                b = clothes.to(device)
                c = p_tryon
                combine = torch.cat([a[0], b[0], c[0]], 2).squeeze()
                cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
                rgb = (cv_img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                peo = peos[step]
                clo = clos[step]
                name = peo[:-4] + "_" + clo[:-4] + ".jpg"
                print(name)
                cv2.imwrite(sub_path + '/' + name + '.jpg', bgr)

            step += 1
            if epoch_iter >= dataset_size:
                break

if __name__ == "__main__":
    peos = os.listdir("./dataset/real_img")
    clos = os.listdir("./dataset/real_clothes")
    futurama_single(peos,clos)