from __future__ import print_function
import argparse
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
import sys
sys.path.append('./auxiliary/')
from model import *
from utils import *
from ply import *
import sys
import os
sys.path.append("./nndistance/")
from modules.nnd import NNDModule
import visdom
distChamfer = NNDModule()

try:
    from script.normalize_obj import *
except:
    print('couldnt load normalize obj')


parser = argparse.ArgumentParser()
parser.add_argument('--HR', type=int, default=1, help='Use high Res template ?')
parser.add_argument('--nepoch', type=int, default=3000, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = 'trained_models/network_last.pth',  help='your path to the trained model')
parser.add_argument('--input', type=str, default = '',  help='your path to the trained model')
parser.add_argument('--num_points', type=int, default = 6890,  help='number of points fed to poitnet')
parser.add_argument('--nb_primitives', type=int, default = 1,  help='number of primitives')
parser.add_argument('--bottleneck', type=int, default=1024, help='visdom environment')
parser.add_argument('--env', type=str, default="regress-rot", help='visdom environment')

opt = parser.parse_args()
vis = visdom.Visdom(port=8888, env=opt.env)
# print (opt)

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
# print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

# load network
network = AE_AtlasNet_Humans(num_points=opt.num_points, nb_primitives=opt.nb_primitives,
                                    bottleneck_size=opt.bottleneck)
network.cuda()

network.apply(weights_init)
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print("previous weight loaded")

val_loss = AverageValueMeter()

network.eval()

# load template at high and low resolution
mesh_ref = pymesh.load_mesh("/home/thibault/Downloads/MPI-FAUST/training/ref/reg_color_ref.ply")
mesh_ref_LR = pymesh.load_mesh("/home/thibault/Downloads/MPI-FAUST/training/ref/reg_color_ref.ply")

if opt.HR:
    mesh_ref = pymesh.load_mesh("/home/thibault/Downloads/MPI-FAUST/training/ref/reg_color_ref_HR.ply")

def regress(points):
    # saerch the latent space to optimize reconstruction using the chamfer distance
    points = Variable(points.data, requires_grad=True)
    latent_code = network.encoder(points)
    lrate = 0.001  # learning rate
    input_param = nn.Parameter(latent_code.data, requires_grad=True)
    optimizer = optim.Adam([input_param], lr=lrate)
    loss = 10
    i = 0
    while np.log(loss) > -9 and i < 3000:
        optimizer.zero_grad()
        pointsReconstructed = network.decode(input_param)  # forward pass
        dist1, dist2 = distChamfer(points.transpose(2, 1).contiguous(), pointsReconstructed)
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        loss_net.backward()
        optimizer.step()
        loss = loss_net.data[0]
        i = i + 1
    with torch.no_grad():
        if opt.HR:
            pointsReconstructed = network.decode_full(input_param)  # forward pass
        else :
            pointsReconstructed = network.decode(input_param)  # forward pass
    if np.log(loss) > -9:
        print(" sucks")
    print("loss reg : ", loss)
    return pointsReconstructed

def run(input):
    points = input.vertices
    random_sample = np.random.choice(np.shape(points)[0], size=10000)
    points = torch.from_numpy(points.astype(np.float32)).contiguous().unsqueeze(0)
    points = Variable(points)
    points = points.transpose(2, 1).contiguous()
    points = points.cuda()

    points_LR = torch.from_numpy(input.vertices[random_sample].astype(np.float32)).contiguous().unsqueeze(0)
    points_LR = Variable(points_LR)
    points_LR = points_LR.transpose(2, 1).contiguous()
    points_LR = points_LR.cuda()

    theta = 0
    bestLoss = 10
    print("size: ", points_LR.size())
    pointsReconstructed = network(points_LR)
    dist1, dist2 = distChamfer(points_LR.transpose(2, 1).contiguous(), pointsReconstructed)
    loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
    print("loss : ",  loss_net.data[0], 0,  " index ", i)

    # ---- Search best angle for reconstruction ---
    for theta in np.linspace(-np.pi/2, np.pi/2, 351):
        rot_matrix = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [- np.sin(theta), 0,  np.cos(theta)]])
        rot_matrix = Variable(torch.from_numpy(rot_matrix).float()).cuda()
        points2 = torch.matmul(rot_matrix, points_LR)
        mesh_tmp = pymesh.form_mesh(vertices=points2[0].transpose(1,0).data.cpu().numpy(), faces=network.mesh.faces)
        norma = Variable(torch.from_numpy((mesh_tmp.bbox[0] + mesh_tmp.bbox[1]) / 2).float().cuda())
        norma2 = norma.unsqueeze(1).expand(3,points2.size(2)).contiguous()
        points2[0] = points2[0] - norma2
        mesh_tmp = pymesh.form_mesh(vertices=points2[0].transpose(1,0).data.cpu().numpy(), faces=np.array([[0,0,0]]))

        pointsReconstructed = network(points2)
        dist1, dist2 = distChamfer(points2.transpose(2, 1).contiguous(), pointsReconstructed)
        norma3 = norma.unsqueeze(0).expand(pointsReconstructed.size(1), 3).contiguous()

        pointsReconstructed[0] = pointsReconstructed[0] + norma3
        rot_matrix = np.array([[np.cos(-theta), 0, np.sin(-theta)], [0, 1, 0], [- np.sin(-theta), 0,  np.cos(-theta)]])
        rot_matrix = Variable(torch.from_numpy(rot_matrix).float()).cuda()
        pointsReconstructed = torch.matmul(pointsReconstructed, rot_matrix.transpose(1,0))

        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        if loss_net < bestLoss:
            bestLoss = loss_net
            best_theta = theta
            bestPoints = pointsReconstructed
    print("best loss : ", bestLoss.data[0], best_theta, " index ", i)
    val_loss.update(bestLoss.data[0])

    if not(opt.input==""):
        opt.model = opt.input[:52]+".ply"
    if opt.HR:
        faces_tosave = network.mesh_HR.faces
    else:
        faces_tosave = network.mesh.faces
    mesh = pymesh.form_mesh(vertices=bestPoints[0].data.cpu().numpy(), faces=network.mesh.faces)
    mesh.add_attribute("red")
    mesh.add_attribute("green")
    mesh.add_attribute("blue")
    mesh.set_attribute("red", mesh_ref_LR.get_attribute("vertex_red"))
    mesh.set_attribute("green", mesh_ref_LR.get_attribute("vertex_green"))
    mesh.set_attribute("blue", mesh_ref_LR.get_attribute("vertex_blue"))
    a = mesh_ref.get_attribute("vertex_red")
    print(np.shape(a))
    #START REGRESSION
    print("start regression...")
    rot_matrix = np.array([[np.cos(best_theta), 0, np.sin(best_theta)], [0, 1, 0], [- np.sin(best_theta), 0,  np.cos(best_theta)]])
    rot_matrix = Variable(torch.from_numpy(rot_matrix).float()).cuda()
    points2 = torch.matmul(rot_matrix, points)
    mesh_tmp = pymesh.form_mesh(vertices=points2[0].transpose(1,0).data.cpu().numpy(), faces=network.mesh.faces)
    norma = Variable(torch.from_numpy((mesh_tmp.bbox[0] + mesh_tmp.bbox[1]) / 2).float().cuda())
    norma2 = norma.unsqueeze(1).expand(3,points2.size(2)).contiguous()
    points2[0] = points2[0] - norma2
    pointsReconstructed1 = regress(points2)

    norma3 = norma.unsqueeze(0).expand(pointsReconstructed1.size(1), 3).contiguous()
    rot_matrix = np.array([[np.cos(-best_theta), 0, np.sin(-best_theta)], [0, 1, 0], [- np.sin(-best_theta), 0,  np.cos(-best_theta)]])
    rot_matrix = Variable(torch.from_numpy(rot_matrix).float()).cuda()

    pointsReconstructed1[0] = pointsReconstructed1[0] + norma3
    pointsReconstructed1 = torch.matmul(pointsReconstructed1, rot_matrix.transpose(1,0))
    print(pointsReconstructed1.size())
    meshReg = pymesh.form_mesh(vertices=pointsReconstructed1[0].data.cpu().numpy(), faces=faces_tosave)
    meshReg.add_attribute("red")
    meshReg.add_attribute("green")
    meshReg.add_attribute("blue")
    meshReg.set_attribute("red", mesh_ref.get_attribute("vertex_red"))
    meshReg.set_attribute("green", mesh_ref.get_attribute("vertex_green"))
    meshReg.set_attribute("blue", mesh_ref.get_attribute("vertex_blue"))
    return mesh, meshReg


def reconstruct(input_p):
    input = pymesh.load_mesh(input_p)
    mesh, meshReg = run(input)
    pymesh.meshio.save_mesh(input_p[:-4] + "InitialGuess.ply", mesh, "red", "green", "blue", ascii=True)
    pymesh.meshio.save_mesh(input_p[:-4] + "FinalReconstruction.ply", meshReg, "red", "green", "blue", ascii=True)
