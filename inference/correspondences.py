from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
import sys
import matplotlib.pyplot as plt
import pymesh
import glob
sys.path.append('./auxiliary/')
from datasetFaust import *
from model import *
from utils import *
from ply import *
import os
import reconstruct

from sklearn.neighbors import NearestNeighbors

sys.path.append("./nndistance/")
from modules.nnd import NNDModule

distChamfer = NNDModule()

neigh = NearestNeighbors(1, 0.4)

def compute_correspondances(source_p, source_reconstructed_p, target_p, target_reconstructed_p):
    # inputs are all filepaths
    with torch.no_grad():
        source = pymesh.load_mesh(source_p)
        source_reconstructed = pymesh.load_mesh(source_reconstructed_p)
        target = pymesh.load_mesh(target_p)
        target_reconstructed = pymesh.load_mesh(target_reconstructed_p)

        # project on source_reconstructed
        neigh.fit(source_reconstructed.vertices)
        idx_knn = neigh.kneighbors(source.vertices, return_distance=False)

        #correspondances thought template
        closest_points = target_reconstructed.vertices[idx_knn]
        closest_points = np.mean(closest_points, 1, keepdims=False)

        # project on target
        neigh.fit(target.vertices)
        idx_knn = neigh.kneighbors(closest_points, return_distance=False)
        closest_points = target.vertices[idx_knn]
        closest_points = np.mean(closest_points, 1, keepdims=False)

        # save output
        mesh = pymesh.form_mesh(vertices=closest_points, faces=source.faces)
        pymesh.meshio.save_mesh("results/correpondences.ply", mesh, ascii=True)
        np.savetxt("results/correpondences.txt", closest_points, fmt='%1.10f')
        return

if __name__ == '__main__':
    print("computing correspondences for" + sys.argv[1] + "and" + sys.argv[2])
    reconstruct.reconstruct(sys.argv[1])
    reconstruct.reconstruct(sys.argv[2])
    compute_correspondances(sys.argv[1], sys.argv[1] + "FinalReconstruction.ply", sys.argv[2], sys.argv[2] + "FinalReconstruction.ply")
