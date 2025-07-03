# -*- coding: UTF-8 -*-
"""
@Time : 03/07/2025 11:17
@Author : Xiaoguang Liang
@File : test_EMDModule.py
@Project : EMD_module
"""
import time

import numpy as np
import torch

from EMD_module.emd_module import EMDModule


def test_emd():
    x1 = torch.rand(20, 8192, 3).cuda()  # please normalize your point cloud to [0, 1]
    x2 = torch.rand(20, 8192, 3).cuda()
    emd = EMDModule()
    start_time = time.perf_counter()
    dis, assigment = emd(x1, x2, 0.002, 10000)  # 0.005, 50 for training
    print("Input_size: ", x1.shape)
    print("Runtime: %lfs" % (time.perf_counter() - start_time))
    print("EMD: %lf" % np.sqrt(dis.cpu()).mean())
    print("|set(assignment)|: %d" % assigment.unique().numel())
    assigment = assigment.cpu().numpy()
    assigment = np.expand_dims(assigment, -1)
    x2 = np.take_along_axis(x2, assigment, axis=1)
    d = (x1 - x2) * (x1 - x2)
    print("Verified EMD: %lf" % np.sqrt(d.cpu().sum(-1)).mean())


if __name__ == '__main__':
    test_emd()
