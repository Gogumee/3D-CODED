from model import *

global network
global opt
global mesh_ref
global mesh_ref_LR
# load template at high and low resolution
mesh_ref = trimesh.load("./data/reg_color_ref_HR.ply", process=False)
mesh_ref_LR = trimesh.load("./data/reg_color_ref.ply", process=False)
red_LR = np.load("./data/red_LR.npy").astype("uint8")
green_LR = np.load("./data/green_LR.npy").astype("uint8")
blue_LR = np.load("./data/blue_LR.npy").astype("uint8")
red_HR = np.load("./data/red_HR.npy").astype("uint8")
green_HR = np.load("./data/green_HR.npy").astype("uint8")
blue_HR = np.load("./data/blue_HR.npy").astype("uint8")
