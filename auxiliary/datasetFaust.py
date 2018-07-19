from __future__ import print_function
import torch.utils.data as data
import os.path
import torch
import numpy as np
from utils import *
import pymesh

mypath = ''
class FAUST(data.Dataset):
    def __init__(self, train, rootpc = "/home/thibault/Downloads/MPI-FAUST/", npoints = 2500, correspondance=False):
        self.train = train
        self.rootpc = rootpc
        self.npoints = npoints
        self.datapath = []
        self.dataname = []
        self.datapathreg = []
        self.datapathregval = []
        self.datapathtxt = []
        self.datapathtxtval = []
        self.datanamereg = []
        self.datanameregval = []
        self.datanameregval = []
        self.correspondance = correspondance
        if not self.correspondance:
            if self.train:
                dir_ply  = os.path.join(self.rootpc, "training")
            else:
                dir_ply  = os.path.join(self.rootpc, "test")

            fns_ply = sorted(os.listdir(os.path.join(dir_ply, "scans_processed")))
            for fn in fns_ply:
                self.datapath.append(os.path.join(dir_ply, "scans_processed") + "/" + fn)
            for fn in fns_ply:
                self.dataname.append(fn)
            if self.train:
                fns_ply = sorted(os.listdir(os.path.join(dir_ply, "registrations")))
                for fn in fns_ply:
                    if fn.endswith(".ply"):
                        self.datapathreg.append(os.path.join(dir_ply, "registrations") + "/" + fn)
                        self.datapathtxt.append(os.path.join(dir_ply, "txt") + "/" + fn)
                for fn in fns_ply:
                    if fn.endswith(".ply"):
                        self.datanamereg.append(fn)
                fns_ply = sorted(os.listdir(os.path.join(os.path.join(self.rootpc, "val"), "registrations")))
                for fn in fns_ply:
                    if fn.endswith(".ply"):
                        self.datapathregval.append(os.path.join(os.path.join(self.rootpc, "val"), "registrations") + "/" + fn)
                        self.datapathtxtval.append(os.path.join(os.path.join(self.rootpc, "val"), "txt") + "/" + fn)
                for fn in fns_ply:
                    if fn.endswith(".ply"):
                        self.datanameregval.append(fn)
        else:
            if self.train:
                dir_ply  = os.path.join(self.rootpc, "training")
            else:
                dir_ply  = os.path.join(self.rootpc, "test")

            fns_ply = sorted(os.listdir(os.path.join(dir_ply, "scans_processed")))
            for fn in fns_ply:
                if (fn[0:6]=="tr_reg") and int(fn[7:10])>=80:
                    self.datapath.append(os.path.join(dir_ply, "scans_processed") + "/" + fn)
            for fn in fns_ply:
                if (fn[0:6]=="tr_reg") and int(fn[7:10])>=80:
                    self.dataname.append(fn)
            if self.train:
                fns_ply = sorted(os.listdir(os.path.join(dir_ply, "registrations")))
                for fn in fns_ply:
                    if fn.endswith(".ply"):
                        self.datapathreg.append(os.path.join(dir_ply, "registrations") + "/" + fn)
                        self.datapathtxt.append(os.path.join(dir_ply, "txt") + "/" + fn)
                for fn in fns_ply:
                    if fn.endswith(".ply"):
                        self.datanamereg.append(fn)
                fns_ply = sorted(os.listdir(os.path.join(os.path.join(self.rootpc, "val"), "registrations")))
                for fn in fns_ply:
                    if fn.endswith(".ply"):
                        self.datapathregval.append(os.path.join(os.path.join(self.rootpc, "val"), "registrations") + "/" + fn)
                        self.datapathtxtval.append(os.path.join(os.path.join(self.rootpc, "val"), "txt") + "/" + fn)
                for fn in fns_ply:
                    if fn.endswith(".ply"):
                        self.datanameregval.append(fn)


    def __getitem__(self, index):
        fn = self.datapath[index]
        with open(fn) as fp:
            for i, line in enumerate(fp):
                if i == 2:
                    try:
                        lenght = int(line.split()[2])
                    except ValueError:
                        lenght = -1
                    if lenght > 0:
                        break

                elif i == 3:
                    try:
                        print("trying line 3")
                        lenght = int(line.split()[2])
                    except ValueError:
                        print(fn)
                        print(line)
                        lenght = -1
                    break
        idx = np.random.randint(6890, size= self.npoints)
        for i in range(15):
            try:
                if self.dataname[index][3:6]=="reg":
                    point_set, idx = my_get_n_random_lines_reg(fn, n = self.npoints)
                else:
                    mystring = my_get_n_random_lines(fn, n = self.npoints)
                    point_set = np.loadtxt(mystring).astype(np.float32)
                break
            except ValueError as excep:
                print(fn)
                print(excep)
        point_set = torch.from_numpy(point_set)
        return point_set.contiguous(), self.dataname[index], idx

    def getTemplate(self):
        fn = "/home/thibault/Downloads/MPI-FAUST/training/scans_processed/tr_reg_000.ply"
        dataname = "tr_reg_000.ply"
        mesh = pymesh.load_mesh(fn)
        point_set = torch.from_numpy(mesh.vertices).float()
        return point_set.contiguous(), dataname, 0
        # return 0, self.dataname[index]

    def getCorrespondance(self, index, val=False):
        if self.train:
            if not val:
                fn = self.datapathreg[index]
                mesh = pymesh.load_mesh(fn)
                vertex = mesh.vertices
                with open(self.datapathtxt[index][:-11] + "scan_" + self.datapathtxt[index][-7:-3] + "txt", 'r') as f:
                    x = f.read()
                x = x.split('\n')
                center =  x[0].split(' ')
                vertex[: , 0] = vertex[: , 0] - float(center[0])
                vertex[: , 1] = vertex[: , 1] - float(center[1])
                vertex[: , 2] = vertex[: , 2] - float(center[2])
                vertex = vertex/float(x[1])
                vertex = vertex.astype(np.float32)
                vertex = torch.from_numpy(vertex).cuda().contiguous()
                center = np.array(center)

                # vertex = vertex - center
                radius = float(x[1])
                # vertex = vertex/radius

                return vertex, center, radius
            else:
                fn = self.datapathregval[index]
                mesh = pymesh.load_mesh(fn)
                vertex = mesh.vertices
                with open(self.datapathtxtval[index][:-11] + "scan_" + self.datapathtxtval[index][-7:-3] + "txt", 'r') as f:
                    x = f.read()
                x = x.split('\n')
                center =  x[0].split(' ')
                vertex[: , 0] = vertex[: , 0] - float(center[0])
                vertex[: , 1] = vertex[: , 1] - float(center[1])
                vertex[: , 2] = vertex[: , 2] - float(center[2])
                vertex = vertex/float(x[1])
                vertex = vertex.astype(np.float32)

                vertex = torch.from_numpy(vertex).cuda().contiguous()
                center = np.array(center)

                # vertex = vertex - center
                radius = float(x[1])
                # vertex = vertex/radius

                return vertex, center, radius
        else:
            fn = self.datapath[index]
            print(fn)
            mesh = pymesh.load_mesh(fn)
            vertex = mesh.vertices
            with open(self.datapathtxt[index][:-11] + "scan_" + self.datapathtxt[index][-7:-3] + "txt", 'r') as f:
                x = f.read()
            x = x.split('\n')
            center =  x[0].split(' ')
            vertex[: , 0] = vertex[: , 0] - float(center[0])
            vertex[: , 1] = vertex[: , 1] - float(center[1])
            vertex[: , 2] = vertex[: , 2] - float(center[2])
            vertex = vertex/float(x[1])
            vertex = vertex.astype(np.float32)
            vertex = torch.from_numpy(vertex).cuda().contiguous()
            center = np.array(center)

            # vertex = vertex - center
            radius = float(x[1])
            # vertex = vertex/radius

            return vertex, center, radius

    def getCorrespondance_with_mesh(self, index, val=False):
        if self.train:
            if not val:
                fn = self.datapathreg[index]
                mesh = pymesh.load_mesh(fn)
                vertex = mesh.vertices
                with open(self.datapathtxt[index][:-11] + "scan_" + self.datapathtxt[index][-7:-3] + "txt", 'r') as f:
                    x = f.read()
                x = x.split('\n')
                center =  x[0].split(' ')
                vertex[: , 0] = vertex[: , 0] - float(center[0])
                vertex[: , 1] = vertex[: , 1] - float(center[1])
                vertex[: , 2] = vertex[: , 2] - float(center[2])
                vertex = vertex/float(x[1])
                vertex = vertex.astype(np.float32)
                vertex = torch.from_numpy(vertex).cuda().contiguous()
                center = np.array(center)

                # vertex = vertex - center
                radius = float(x[1])
                # vertex = vertex/radius

                return vertex, center, radius, mesh
            else:
                fn = self.datapathregval[index]
                mesh = pymesh.load_mesh(fn)
                vertex = mesh.vertices
                with open(self.datapathtxtval[index][:-11] + "scan_" + self.datapathtxtval[index][-7:-3] + "txt", 'r') as f:
                    x = f.read()
                x = x.split('\n')
                center =  x[0].split(' ')
                vertex[: , 0] = vertex[: , 0] - float(center[0])
                vertex[: , 1] = vertex[: , 1] - float(center[1])
                vertex[: , 2] = vertex[: , 2] - float(center[2])
                vertex = vertex/float(x[1])
                vertex = vertex.astype(np.float32)

                vertex = torch.from_numpy(vertex).cuda().contiguous()
                center = np.array(center)

                # vertex = vertex - center
                radius = float(x[1])
                # vertex = vertex/radius

                return vertex, center, radius, mesh
        else:
            fn = self.datapath[index]
            print(fn)
            mesh = pymesh.load_mesh(fn)
            vertex = mesh.vertices
            with open(self.datapathtxt[index][:-11] + "scan_" + self.datapathtxt[index][-7:-3] + "txt", 'r') as f:
                x = f.read()
            x = x.split('\n')
            center =  x[0].split(' ')
            vertex[: , 0] = vertex[: , 0] - float(center[0])
            vertex[: , 1] = vertex[: , 1] - float(center[1])
            vertex[: , 2] = vertex[: , 2] - float(center[2])
            vertex = vertex/float(x[1])
            vertex = vertex.astype(np.float32)
            vertex = torch.from_numpy(vertex).cuda().contiguous()
            center = np.array(center)

            # vertex = vertex - center
            radius = float(x[1])
            # vertex = vertex/radius

            return vertex, center, radius, mesh


    def __len__(self):
        return len(self.datapath)



if __name__  == '__main__':
    d  =  FAUST(train=True, correspondance=True)
    a = len(d)
    print(a)
    a,b, idx = d[15]
    print(idx)
    print(b)
    print(a.size())
    d  =  FAUST(train=False)
    a =  len(d)
    print(a)
