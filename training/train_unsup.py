from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import sys
sys.path.append('./aux/')
from datasetFaust import *
from datasetSMPL2 import *
from model import *
from utils import *
from ply import *
import sys
import os
import json
import datetime
import visdom
from LaplacianLoss import *
sys.path.append("./nndistance/")
from modules.nnd import NNDModule

distChamfer = NNDModule()

#options
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--nepoch', type=int, default=75, help='number of epochs to train for')
parser.add_argument('--model', type=str, default='', help='optional reload model path')
parser.add_argument('--env', type=str, default="unsup-symcorrect-ratio", help='visdom environment')
parser.add_argument('--laplace', type=int, default=1, help='regularize towords 0 curvature, or template curvature')

opt = parser.parse_args()
print(opt)

# Launch visdom for visualization
vis = visdom.Visdom(port=8888, env=opt.env)
now = datetime.datetime.now()
save_path = now.isoformat()
dir_name = os.path.join('log', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname = os.path.join(dir_name, 'log.txt')


opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
best_val_loss = 10
best_val_smpl_loss = 10
best_val_loss_id = 0
best_val_smpl_loss_id = 0
best_correspondence_loss = 10
best_correspondence_loss_id = 0

# Create train/test dataloader

## Synthetic training set
dataset = SMPL(train=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers), drop_last=True)
## Synthetic testing set
dataset_smpl_test = SMPL(train=False)
dataloader_smpl_test = torch.utils.data.DataLoader(dataset_smpl_test, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))

## Real testing set for correspondence
dataset_valCorrepondance = FAUST(train=True, correspondance=True)
dataloader_valCorrepondance = torch.utils.data.DataLoader(dataset_valCorrepondance,
                                                          batch_size=len(dataset_valCorrepondance),
                                                          shuffle=False, num_workers=int(opt.workers))
## Real testing set for reconstruction
dataset_test = FAUST(train=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))


lambda_laplace = 0.005
lambda_ratio = 0.005

print('training set', len(dataset))
print('testing set', len(dataset_test))

cudnn.benchmark = True
len_dataset = len(dataset)

# create network
network = AE_AtlasNet_Humans()


faces = network.mesh.faces
faces = [faces for i in range(opt.batchSize)]
faces = np.array(faces)
faces = Variable(torch.from_numpy(faces).cuda())
print(faces.size())
#takes cuda torch variable repeated batch time

vertices = network.mesh.vertices
vertices = [vertices for i in range(opt.batchSize)]
vertices = np.array(vertices)
vertices = Variable(torch.from_numpy(vertices).cuda())
toref = opt.laplace # regularize towards 0 or template

#Initialize Laplacian Loss
laplaceloss = LaplacianLoss(faces, vertices, toref)
print(laplaceloss.Lx)

laplaceloss(vertices)
print(laplaceloss.Lx)

print("initial : ", laplaceloss(vertices)
)
network.cuda()  # put network on GPU

# print(network)

network.apply(weights_init)  # initialization of the weight

if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")

lrate = 0.001  # learning rate
optimizer = optim.Adam(network.parameters(), lr=lrate)

# meters to record stats on learning
train_loss_L2_smpl = AverageValueMeter()
val_smpl_augmented = AverageValueMeter()
test_loss_L2_smpl = AverageValueMeter()
train_loss_correspondances = AverageValueMeter()
val_loss = AverageValueMeter()
my_val_loss = AverageValueMeter()
val_correspondance = AverageValueMeter()

with open(logname, 'a') as f:  # open and append
    f.write(str(network) + '\n')

# initialize learning curve on visdom, and color for each primitive in visdom display
train_chamfer_curve = vis.line(
    X=np.array([0]),
    Y=np.array([0]),
)
train_correspondances_curve = vis.line(
    X=np.array([0]),
    Y=np.array([0]),
)
val_curve = vis.line(
    X=np.array([0]),
    Y=np.array([1]),
)
val_correspondance_curve = vis.line(
    X=np.array([0]),
    Y=np.array([1]),
)
test_loss_L2_smpl_curve = vis.line(
    X=np.array([0]),
    Y=np.array([1]),
)
labels_generated_points = torch.Tensor(
    range(1, (opt.nb_primitives + 1) * (opt.num_points / opt.nb_primitives) + 1)).view(
    opt.num_points / opt.nb_primitives, (opt.nb_primitives + 1)).transpose(0, 1)
labels_generated_points = (labels_generated_points) % (opt.nb_primitives + 1)
labels_generated_points = labels_generated_points.contiguous().view(-1)
print(labels_generated_points)


def isReg(x):
    return (x[0:6] == "tr_reg") and int(x[7:10]) < opt.denseSupervision
mesh_ref = pymesh.load_mesh("/home/thibault/Downloads/MPI-FAUST/training/ref/reg_color_ref.ply")

def L2(path):
    my_val_loss.reset()
    with torch.no_grad():
        for i in range(1000):
            try:
                print("val L2 ", i)
                input = pymesh.load_mesh(path + str(i) + ".ply")
                points = input.vertices
                points = points - (input.bbox[0] + input.bbox[1]) / 2

                points = torch.from_numpy(points.astype(np.float32)).contiguous().unsqueeze(0)
                points = Variable(points)
                points = points.transpose(2, 1).contiguous()
                points = points.cuda()
                with torch.no_grad():
                    pointsReconstructed = network(points)
                    loss_net = torch.mean(
                        (pointsReconstructed - points.transpose(2, 1).contiguous()) ** 2)
                if i%10==8:
                    continue
                my_val_loss.update(loss_net.data[0])
            except:
                print(path, i)
                break
    log_table = {
        "my_val_loss": my_val_loss.avg,
    }
    print(log_table)
    return my_val_loss.avg


def chamfer(path):
    my_val_loss.reset()
    with torch.no_grad():
        for i in range(1001):
            try:
                print("val chamfer ", i)
                input = pymesh.load_mesh(path + str(i) + ".ply")
                points = input.vertices
                points = torch.from_numpy(points.astype(np.float32)).contiguous().unsqueeze(0)
                points = Variable(points)
                points = points.transpose(2, 1).contiguous()
                points = points.cuda()
                with torch.no_grad():
                    pointsReconstructed = network(points)
                    dist1, dist2 = distChamfer(points.transpose(2, 1).contiguous(), pointsReconstructed)
                    loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
                my_val_loss.update(loss_net.data[0])
            except:
                print(path, i)
                break
    log_table = {
        "my_val_loss": my_val_loss.avg,
    }
    print(log_table)
    return my_val_loss.avg

def init_regul(source):
    sommet_A_source = source.vertices[source.faces[:, 0]]
    sommet_B_source = source.vertices[source.faces[:, 1]]
    sommet_C_source = source.vertices[source.faces[:, 2]]
    target = []
    target.append(np.sqrt( np.sum((sommet_A_source - sommet_B_source) ** 2, axis=1)))
    target.append(np.sqrt( np.sum((sommet_B_source - sommet_C_source) ** 2, axis=1)))
    target.append(np.sqrt( np.sum((sommet_A_source - sommet_C_source) ** 2, axis=1)))
    return target

def compute_score(points, faces, target):
    score = 0
    sommet_A = points[:,faces[:, 0]]
    sommet_B = points[:,faces[:, 1]]
    sommet_C = points[:,faces[:, 2]]

    score = torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_B) ** 2, dim=2)) / target[0] -1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_B - sommet_C) ** 2, dim=2)) / target[1] -1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_C) ** 2, dim=2)) / target[2] -1)
    return torch.mean(score)

target = init_regul(network.mesh)
target = np.array(target)

target = torch.from_numpy(target).float().cuda()
print(target.size())
target = target.unsqueeze(1).expand(3,opt.batchSize,-1)
target= Variable(target)
print(target.size())


# Load all the points from the template
template_points = network.vertex.clone()
template_points = template_points.unsqueeze(0).expand(opt.batchSize, template_points.size(0), template_points.size(
    1))  # have to have two stacked template because of weird error related to batchnorm
template_points = Variable(template_points, requires_grad=False)
template_points = template_points.cuda()

# start of the learning loop
for epoch in range(opt.nepoch):
    if epoch==15:
        lrate = lrate/10  # learning rate
        optimizer = optim.Adam(network.parameters(), lr=lrate)

    # TRAIN MODE
    train_loss_L2_smpl.reset()
    train_loss_correspondances.reset()
    network.train()
    if epoch == 0:
        #initialize reconstruction to be same as template to avoid symmetry issues
        init_step = 0
        for i, data in enumerate(dataloader, 0):
            if (init_step>1000):
                break
            init_step = init_step+1
            optimizer.zero_grad()
            points, fn, idx = data

            points = Variable(points)
            points = points.transpose(2, 1).contiguous()
            points = points.cuda()

            pointsReconstructed = network(points)  # forward pass
            #compute the laplacian loss
            loss_net =  torch.mean((pointsReconstructed - template_points)**2)
            loss_net.backward()
            train_loss_L2_smpl.update(loss_net.data[0])
            optimizer.step()  # gradient update
            print('init [%d: %d/%d] train loss:  %f' % (epoch, i, len_dataset / 32,  loss_net.data[0]))

    if not epoch == -1:
        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            points, fn, idx = data

            points = Variable(points)
            points = points.transpose(2, 1).contiguous()
            points = points.cuda()

            pointsReconstructed = network(points)  # forward pass
            #compute the laplacian loss
            regul = laplaceloss(pointsReconstructed)
            dist1, dist2 = distChamfer(points.transpose(2, 1).contiguous(), pointsReconstructed)
            loss_net = (torch.mean(dist1)) + (torch.mean(dist2)) + lambda_laplace * regul + lambda_ratio* compute_score(pointsReconstructed, network.mesh.faces, target)
            loss_net.backward()
            train_loss_L2_smpl.update(loss_net.data[0])
            optimizer.step()  # gradient update

            # VIZUALIZE
            if i % 10 == 0:
                vis.scatter(X=points.transpose(2, 1).contiguous()[0].data.cpu(),
                win = 'Train_input',
                opts = dict(
                    title="Train_input",
                    markersize=2,
                ),
                )
                vis.scatter(X=pointsReconstructed[0].data.cpu(),
                Y = labels_generated_points[0:pointsReconstructed.size(1)],
                win = 'Train_output',
                opts = dict(
                    title="Train_output",
                    markersize=2,
                ),
                )

            print('[%d: %d/%d] train loss:  %f' % (epoch, i, len_dataset / 32,  loss_net.data[0]))

    # UPDATE CURVES
    if train_loss_L2_smpl.avg != 0:
        vis.updateTrace(
            X = np.array([epoch]),
            Y = np.log(np.array([train_loss_L2_smpl.avg])),
            win = train_chamfer_curve,
            name = 'L2 train smpl'
            )

    with torch.no_grad():
        #val on SMPL data
        network.eval()
        test_loss_L2_smpl.reset()
        for i, data in enumerate(dataloader_smpl_test, 0):
            points, fn, idx = data

            points = Variable(points)
            points = points.transpose(2, 1).contiguous()
            points = points.cuda()

            pointsReconstructed = network(points)  # forward pass
            loss_net = torch.mean(
                (pointsReconstructed - points.transpose(2, 1).contiguous()) ** 2)
            test_loss_L2_smpl.update(loss_net.data[0])
            # VIZUALIZE
            if i % 10 == 0:
                vis.scatter(X=points.transpose(2, 1).contiguous()[0].data.cpu(),
                            win='Test_smlp_input',
                            opts=dict(
                                title="Test_smlp_input",
                                markersize=2,
                            ),
                            )
                vis.scatter(X=pointsReconstructed[0].data.cpu(),
                            Y=labels_generated_points[0:pointsReconstructed.size(1)],
                            win='Test_smlp_output',
                            opts=dict(
                                title="Test_smlp_output",
                                markersize=2,
                            ),
                            )

            print('[%d: %d/%d] test smlp loss:  %f' % (epoch, i, len_dataset / 32, loss_net.data[0]))
        if test_loss_L2_smpl.avg != 0:
            vis.updateTrace(
                X = np.array([epoch]),
                Y = np.log(np.array([test_loss_L2_smpl.avg])),
                win = test_loss_L2_smpl_curve,
                name = ' val L2 Smpl'
            )
       
        # VALIDATION on FAUST TEST CHAMFER
        val_loss.reset()
        network.eval()
        loss = chamfer("/home/thibault/Dropbox/ECCV/quantitative_results/faust_centered_highres_bb_test/")
        val_loss.update(loss)
        if val_loss.avg != 0:
            vis.updateTrace(
                X = np.array([epoch]),
                Y = np.log(np.array([loss])),
                win = val_curve,
                name = 'Chamfer val'
            )  # UPDATE CURVES

        # VALIDATION on FAUST TRAIN CORRESPONDANCES
        val_correspondance.reset()
        network.eval()
        loss = L2("/home/thibault/Dropbox/ECCV/quantitative_results/faust_centered_bb_train/")
        val_correspondance.update(loss)

        # VALIDATION on SMPL Augmented
        val_smpl_augmented.reset()
        network.eval()
        loss = L2("/home/thibault/Dropbox/ECCV/quantitative_results/val_dataset_augmented/")
        val_smpl_augmented.update(loss)
        # VALIDATION in terms of dense correspondanc

        # UPDATE CURVES
        if val_correspondance.avg != 0:
            vis.updateTrace(
                X = np.array([epoch]),
                Y = np.log(np.array([val_correspondance.avg])),
                win = val_correspondance_curve,
                name = ' val correspondance'
            )


        #save latest network
        torch.save(network.state_dict(), '%s/network_last.pth' % (dir_name))

        # dump stats in log file
        log_table = {
            "lambda_laplace": lambda_laplace,
            "lambda_ratio": lambda_ratio,
            "val_smpl_augmented": val_smpl_augmented.avg,
            "train_loss_L2_smpl": train_loss_L2_smpl.avg,
            "train_loss_correspondances": train_loss_correspondances.avg,
            "val_loss": val_loss.avg,
            "val_correspondance": val_correspondance.avg,
            "val_smpl": test_loss_L2_smpl.avg,
            "epoch": epoch,
            "lr": lrate,
            "super_points": opt.super_points,
            "env": opt.env,
        }
        print(log_table)
        with open(logname, 'a') as f:  # open and append
            f.write('json_stats: ' + json.dumps(log_table) + '\n')
