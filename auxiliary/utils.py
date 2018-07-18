import os
import random
import numpy as np
import sys
import pymesh

def test_orientation(input_mesh):
    # This fonction tests wether widest axis of the input mesh is the Z axis
    # input mesh
    # output : boolean or warning
    bbox = input_mesh.bbox
    if not np.argmax(np.abs(bbox[1] - bbox[0])) == 1:
        print("The widest axis is not the Y axis, you should make sure the mesh is aligned on the Y axis for the autoencoder to work (check out the example in /data)")
    return 

def clean(input_mesh):
    # This function remove faces, and vertex that doesn't belong to any face. Intended to be used before a feed forward pass in pointNet
    # Input : mesh
    # output : cleaned mesh
    pts = input_mesh.vertices
    faces = input_mesh.faces
    faces = faces.reshape(-1)
    unique_points_index = np.unique(faces)
    unique_points = pts[unique_points_index]
    mesh = pymesh.form_mesh(vertices=unique_points, faces=np.array([[0,0,0]]))
    return mesh

def scale(input_mesh, mesh_ref):
    # This function scales the input mesh to have the same volume as a reference mesh Intended to be used before a feed forward pass in pointNet
    # Input : file path
    # mesh_ref : reference mesh path
    # output : scaled mesh
    mesh = pymesh.load_mesh(input)
    area = np.power(mesh_ref.volume / input_mesh.volume, 1.0/3)
    mesh= pymesh.form_mesh( vertices =  input_mesh.vertices * area, faces= input_mesh.faces)
    return mesh

def rot(input_mesh,  theta = np.pi/2):
    # rotation around X axis of angle theta
    point = input_mesh.vertices
    rot_matrix = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    point_set = point.dot(np.transpose(rot_matrix, (1, 0)))
    #center the rotated mesh
    point_set = point_set - (input_mesh.bbox[0] + input_mesh.bbox[1]) / 2

    mesh = pymesh.form_mesh(vertices=point_set, faces=input_mesh.faces)
    return mesh



#initialize the weighs of the network for Convolutional layers and batchnorm layers
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def adjust_learning_rate(optimizer, epoch, phase):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if (epoch%phase==(phase-1)):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/10.


class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

#Example:
def get_colors(num_colors=10):
  colors = []
  for i in range(0,num_colors):
      colors.append(generate_new_color(colors,pastel_factor = 0.9))
  for i in range(0,num_colors):
      for j in range(0,3):
        colors[i][j] = int(colors[i][j]*256)
      colors[i].append(255)
  return colors




if __name__ == '__main__':

  #To make your color choice reproducible, uncomment the following line:
  #random.seed(10)

  colors = get_colors(10)
  print("Your colors:",colors)
