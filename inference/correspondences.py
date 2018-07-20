from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import sys
import pymesh
sys.path.append('./auxiliary/')
from datasetFaust import *
from model import *
from utils import *
from ply import *
import reconstruct
import time
from sklearn.neighbors import NearestNeighbors
sys.path.append("./nndistance/")
from modules.nnd import NNDModule
import visdom
import global_variables



def compute_correspondances(source_p, source_reconstructed_p, target_p, target_reconstructed_p):
    """
    Given 2 meshes, and their reconstruction, compute correspondences between the 2 meshes through neireast neighbors
    :param source_p: path for source mesh
    :param source_reconstructed_p: path for source mesh reconstructed
    :param target_p: path for target mesh
    :param target_reconstructed_p: path for target mesh reconstructed
    :return: None but save a file with correspondences
    """
    # inputs are all filepaths
    with torch.no_grad():
        source = pymesh.load_mesh(source_p)
        source_reconstructed = pymesh.load_mesh(source_reconstructed_p)
        target = pymesh.load_mesh(target_p)
        target_reconstructed = pymesh.load_mesh(target_reconstructed_p)

        # project on source_reconstructed
        neigh.fit(source_reconstructed.vertices)
        idx_knn = neigh.kneighbors(source.vertices, return_distance=False)

        #correspondances throught template
        closest_points = target_reconstructed.vertices[idx_knn]
        closest_points = np.mean(closest_points, 1, keepdims=False)

        # project on target
        # neigh.fit(target.vertices)
        # idx_knn = neigh.kneighbors(closest_points, return_distance=False)
        # closest_points = target.vertices[idx_knn]
        # closest_points = np.mean(closest_points, 1, keepdims=False)

        # save output
        mesh = pymesh.form_mesh(vertices=closest_points, faces=source.faces)
        pymesh.meshio.save_mesh("results/correspondences.ply", mesh, ascii=True)
        np.savetxt("results/correspondences.txt", closest_points, fmt='%1.10f')
        return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--HR', type=int, default=1, help='Use high Resolution template for better precision in the nearest neighbor step ?')
    parser.add_argument('--nepoch', type=int, default=3000, help='number of epochs to train for during the regression step')
    parser.add_argument('--model', type=str, default = 'trained_models/sup_human_network_last.pth',  help='your path to the trained model')
    parser.add_argument('--inputA', type=str, default =  "data/example_0.ply",  help='your path to mesh 0')
    parser.add_argument('--inputB', type=str, default =  "data/example_1.ply",  help='your path to mesh 1')
    parser.add_argument('--num_points', type=int, default = 6890,  help='number of points fed to poitnet')
    parser.add_argument('--num_angles', type=int, default = 300,  help='number of angle in the search of optimal reconstruction. Set to 1, if you mesh are already facing the cannonical direction as in data/example_1.ply')
    parser.add_argument('--env', type=str, default="CODED", help='visdom environment')
    parser.add_argument('--clean', type=int, default=1, help='if 1, remove points that dont belong to any edges')
    parser.add_argument('--scale', type=int, default=0, help='if 1, scale input mesh to have same volume as the template')


    opt = parser.parse_args()
    global_variables.opt = opt
    vis = visdom.Visdom(port=8888, env=opt.env)

    distChamfer = NNDModule()

    # load network
    global_variables.network = AE_AtlasNet_Humans(num_points=opt.num_points)
    global_variables.network.cuda()
    global_variables.network.apply(weights_init)
    if opt.model != '':
        global_variables.network.load_state_dict(torch.load(opt.model))
    global_variables.network.eval()

    neigh = NearestNeighbors(1, 0.4)
    opt.manualSeed = random.randint(1, 10000) # fix seed
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    cudnn.benchmark = True

    start = time.time()
    print("computing correspondences for " + opt.inputA + " and " + opt.inputB)

    # Reconstruct meshes
    reconstruct.reconstruct(opt.inputA)
    reconstruct.reconstruct(opt.inputB)

    # Compute the correspondences through the recontruction
    compute_correspondances(opt.inputA, opt.inputA[:-4] + "FinalReconstruction.ply", opt.inputB, opt.inputB[:-4] + "FinalReconstruction.ply")
    end = time.time()
    print("ellapsed time is ", end - start, " seconds !")