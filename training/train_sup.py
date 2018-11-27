from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys
sys.path.append('./auxiliary/')
from datasetSMPL2 import *
from model import *
from utils import *
from ply import *
import os
import json
import datetime
import visdom

# =============PARAMETERS======================================== #
path_faust_centered_bb_test_cleaned = "/home/thibault/ssd/3DCODED_data/path_faust_centered_bb_test_cleaned/"
path_faust_centered_bb_train = "/home/thibault/ssd/3DCODED_data/faust_centered_bb_train/"
path_val_dataset_augmented = "/home/thibault/ssd/3DCODED_data/val_dataset_augmented/"

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--nepoch', type=int, default=35, help='number of epochs to train for')
parser.add_argument('--model', type=str, default='', help='optional reload model path')
parser.add_argument('--env', type=str, default="3DCODED_supervised", help='visdom environment')

opt = parser.parse_args()
print(opt)
# ========================================================== #

# =============DEFINE CHAMFER LOSS======================================== #
sys.path.append("./extension/")
import dist_chamfer as ext
distChamfer =  ext.chamferDist()
# ========================================================== #

# =============DEFINE stuff for logs ======================================== #
# Launch visdom for visualization
vis = visdom.Visdom(port=8888, env=opt.env)
now = datetime.datetime.now()
save_path = now.isoformat()
dir_name = os.path.join('log', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname = os.path.join(dir_name, 'log.txt')

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
L2curve_train_loss_L2_smpl = []
L2curve_val_smlp_augmented = []
curve_faust_centered_bb_test_cleaned = []
L2curve_faust_centered_bb_train = []
test_loss_L2_smpl_curve = []
# meters to record stats on learning
train_loss_L2_smpl = AverageValueMeter()
val_smpl_augmented = AverageValueMeter()
test_loss_L2_smpl = AverageValueMeter()
loss_faust_centered_bb_test_cleaned = AverageValueMeter()
tmp_val_loss = AverageValueMeter()
L2loss_faust_centered_bb_train = AverageValueMeter()
# ========================================================== #


# ===================CREATE DATASET================================= #
dataset = SMPL(train=True, regular = True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
dataset_smpl_test = SMPL(train=False)
dataloader_smpl_test = torch.utils.data.DataLoader(dataset_smpl_test, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
len_dataset = len(dataset)
# ========================================================== #

# ===================CREATE network================================= #
network = AE_AtlasNet_Humans()
network.cuda()  # put network on GPU
network.apply(weights_init)  # initialization of the weight
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")
# ========================================================== #

# ===================CREATE optimizer================================= #
lrate = 0.001  # learning rate
optimizer = optim.Adam(network.parameters(), lr=lrate)

with open(logname, 'a') as f:  # open and append
    f.write(str(network) + '\n')
# ========================================================== #

# =============Define eval functions L2 and chamfer ========================= #
def L2(path):
    tmp_val_loss.reset()
    with torch.no_grad():
        for i in range(1000):
            try:
                print("val L2 ", i)
                input = pymesh.load_mesh(path + str(i) + ".ply")
                points = input.vertices
                points = points - (input.bbox[0] + input.bbox[1]) / 2

                points = torch.from_numpy(points.astype(np.float32)).contiguous().unsqueeze(0)
                points = points.transpose(2, 1).contiguous()
                points = points.cuda()
                with torch.no_grad():
                    pointsReconstructed = network(points)
                    loss_net = torch.mean(
                        (pointsReconstructed - points.transpose(2, 1).contiguous()) ** 2)
                # the 8th is rotated so can't be properly reconstructed without pre processing
                if i%10==8:
                    continue
                tmp_val_loss.update(loss_net.item())
            except:
                print(path, i)
                break
    log_table = {
        "tmp_val_loss": tmp_val_loss.avg,
    }
    print(log_table)
    return tmp_val_loss.avg


def chamfer(path):
    tmp_val_loss.reset()
    with torch.no_grad():
        for i in range(1001):
            try:
                print("val chamfer ", i)
                input = pymesh.load_mesh(path + str(i) + ".ply")
                points = input.vertices
                points = torch.from_numpy(points.astype(np.float32)).contiguous().unsqueeze(0)
                points = points.transpose(2, 1).contiguous()
                points = points.cuda()
                with torch.no_grad():
                    pointsReconstructed = network(points)
                    dist1, dist2 = distChamfer(points.transpose(2, 1).contiguous(), pointsReconstructed)
                    loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
                tmp_val_loss.update(loss_net.item())
            except:
                print(path, i)
                break
    log_table = {
        "tmp_val_loss": tmp_val_loss.avg,
    }
    print(log_table)
    return tmp_val_loss.avg
# ========================================================== #



# =============start of the learning loop ======================================== #
for epoch in range(opt.nepoch):
    if epoch==25:
        lrate = lrate/10.0  # learning rate scheduled decay
        optimizer = optim.Adam(network.parameters(), lr=lrate)

    # TRAIN MODE
    train_loss_L2_smpl.reset()
    network.train()
    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        points, idx,_ = data

        points = points.transpose(2, 1).contiguous()
        points = points.cuda()

        pointsReconstructed = network.forward_idx(points, idx)  # forward pass
        loss_net = torch.mean(
                (pointsReconstructed - points.transpose(2, 1).contiguous()) ** 2)
        loss_net.backward()
        train_loss_L2_smpl.update(loss_net.item())
        optimizer.step()  # gradient update

        # VIZUALIZE
        if i % 100 == 0:
            vis.scatter(X=points.transpose(2, 1).contiguous()[0].data.cpu(),
            win = 'Train_input',
            opts = dict(
                title="Train_input",
                markersize=2,
            ),
            )
            vis.scatter(X=pointsReconstructed[0].data.cpu(),
            win = 'Train_output',
            opts = dict(
                title="Train_output",
                markersize=2,
            ),
            )

        print('[%d: %d/%d] train loss:  %f' % (epoch, i, len_dataset / 32,  loss_net.item()))

    # Validation
    with torch.no_grad():
        #val on SMPL data
        network.eval()
        test_loss_L2_smpl.reset()
        for i, data in enumerate(dataloader_smpl_test, 0):
            points, fn, idx = data

            points = points.transpose(2, 1).contiguous()
            points = points.cuda()
            pointsReconstructed = network(points)  # forward pass
            loss_net = torch.mean(
                (pointsReconstructed - points.transpose(2, 1).contiguous()) ** 2)
            test_loss_L2_smpl.update(loss_net.item())
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
                            win='Test_smlp_output',
                            opts=dict(
                                title="Test_smlp_output",
                                markersize=2,
                            ),
                            )
            print('[%d: %d/%d] test smlp loss:  %f' % (epoch, i, len_dataset / 32, loss_net.item()))
        

        # Further evaluation
        try:
            # VALIDATION on FAUST TEST CHAMFER
            loss_faust_centered_bb_test_cleaned.reset()
            network.eval()
            loss = chamfer(path_faust_centered_bb_test_cleaned)
            loss_faust_centered_bb_test_cleaned.update(loss)
            # VALIDATION on FAUST TRAIN CORRESPONDANCES
            L2loss_faust_centered_bb_train.reset()
            network.eval()
            loss = L2(path_faust_centered_bb_train)
            L2loss_faust_centered_bb_train.update(loss)
            # VALIDATION on SMPL Augmented
            val_smpl_augmented.reset()
            network.eval()
            loss = L2(path_val_dataset_augmented)
            val_smpl_augmented.update(loss)
        except:
            print("No further validation")
            
        curve_faust_centered_bb_test_cleaned.append(loss_faust_centered_bb_test_cleaned.avg)
        L2curve_faust_centered_bb_train.append(L2loss_faust_centered_bb_train.avg)
        L2curve_val_smlp_augmented.append(val_smpl_augmented.avg)

        # UPDATE CURVES
        L2curve_train_loss_L2_smpl.append(train_loss_L2_smpl.avg)
        test_loss_L2_smpl_curve.append(test_loss_L2_smpl.avg)

        vis.line(X=np.column_stack((np.arange(len(L2curve_train_loss_L2_smpl)), np.arange(len(L2curve_val_smlp_augmented)), np.arange(len(curve_faust_centered_bb_test_cleaned)), np.arange(len(L2curve_faust_centered_bb_train)), np.arange(len(test_loss_L2_smpl_curve)))),
                 Y=np.column_stack((np.array(L2curve_train_loss_L2_smpl), np.array(L2curve_val_smlp_augmented), np.array(curve_faust_centered_bb_test_cleaned), np.array(L2curve_faust_centered_bb_train), np.array(test_loss_L2_smpl_curve))),
                 win='loss',
                 opts=dict(title="loss", legend=["L2curve_train_loss_L2_smpl" + opt.env,"train_correspondances_curve" + opt.env,"val_curve" + opt.env,"val_correspondance_curve" + opt.env, "test_loss_L2_smpl_curve" + opt.env], markersize=2, ), )
        vis.line(X=np.column_stack((np.arange(len(L2curve_train_loss_L2_smpl)), np.arange(len(L2curve_val_smlp_augmented)), np.arange(len(curve_faust_centered_bb_test_cleaned)), np.arange(len(L2curve_faust_centered_bb_train)), np.arange(len(test_loss_L2_smpl_curve)))),
                 Y=np.log(np.column_stack((np.array(L2curve_train_loss_L2_smpl), np.array(L2curve_val_smlp_augmented), np.array(curve_faust_centered_bb_test_cleaned), np.array(L2curve_faust_centered_bb_train), np.array(test_loss_L2_smpl_curve)))),
                 win='log',
                 opts=dict(title="loss", legend=["L2curve_train_loss_L2_smpl" + opt.env,"train_correspondances_curve" + opt.env,"val_curve" + opt.env,"val_correspondance_curve" + opt.env, "test_loss_L2_smpl_curve" + opt.env], markersize=2, ), )

        # dump stats in log file
        log_table = {
            "val_smpl_augmented": val_smpl_augmented.avg,
            "train_loss_L2_smpl": train_loss_L2_smpl.avg,
            "loss_faust_centered_bb_test_cleaned": loss_faust_centered_bb_test_cleaned.avg,
            "L2loss_faust_centered_bb_train": L2loss_faust_centered_bb_train.avg,
            "test_loss_L2_smpl": test_loss_L2_smpl.avg,
            "epoch": epoch,
            "lr": lrate,
            "env": opt.env,
        }
        print(log_table)
        with open(logname, 'a') as f:  # open and append
            f.write('json_stats: ' + json.dumps(log_table) + '\n')
        #save latest network
        torch.save(network.state_dict(), '%s/network_last.pth' % (dir_name))
