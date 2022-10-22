#!/home/rtzhao/anaconda3/bin/python
#-*-coding:utf-8-*-
import os
import re
import copy
import numpy as np
import pandas as pd

class ExtractProcar(object):

    color_list = ['mediumblue','tomato', 'red','darkorange',  'palegreen', 'seagreen', 'darkturquoise', 'deepskyblue', 'dodgerblue', 'blueviolet', 'violet', 'deeppink', 'crimson']

    def __init__(self, path = './'):
        
        self._path = path
        self._ispin = int(os.popen("grep 'ISPIN' " + os.path.join(path,'OUTCAR'), 'r').read().split()[2])
        self._fermi = float(os.popen("grep 'E-fermi' " + os.path.join(path,'OUTCAR'), 'r').read().split()[2])
        
        with open(os.path.join(path,'PROCAR'),'r') as pro:
            pro.readline()
            kbions = pro.readline().split()
            self._kpoints = int(kbions[3])
            self._bands = int(kbions[7])
            self._ions = int(kbions[-1])
            pro.readline()
            self.__occ_state = pro.readlines()
        
        # energy s py pz px dxy dyz dz2-r2 dxz dx2-y2
        self.__label_col = np.array(['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2'])
        

    def _ext_atom_index(self):
        
        """
        Obtain the element-index dictionary: 
            self._atom_index, e.g.: {'Cs': [0, 2], 'In': [2, 3], 'Ag': [3, 4], 'Cl': [4, 10]} return atom index start from 0 !
        Obtain the element list: 
            self.__elements, e.g.: ['Cs', 'Cs', 'In', 'Ag', 'Cl', 'Cl', 'Cl', 'Cl', 'Cl', 'Cl']

        """
        with open(os.path.join(self._path,'POSCAR'),'r') as pos:
            pos.readline()
            scale = float(pos.readline())
            lattice = np.zeros((0,3), dtype=float)
            for i in range(3):
                lattice =  np.vstack((lattice, np.array(pos.readline().split(), dtype=float)))
            self._lattice = scale * lattice
            element = pos.readline().split()
            index = pos.readline().split()
            self._atom_index = {}
            count = 0
            self.__elements = []
            for i_index, i in enumerate(element):
                num = int(index[i_index])
                self._atom_index[i] = [count, count+num]
                count += num
                self.__elements += [i]*num
        
        return self
    
    @property
    def get_atom_index(self):

        self._ext_atom_index()
        return self._atom_index
    
    @property
    def get_cell(self):

        self._ext_atom_index()
        return self._lattice
    
    def __get_kpath(self):

        self._interpoint = int(os.popen("sed -n '2p' " + os.path.join(self._path,'KPOINTS')).read().split()[0])
        with open(os.path.join(self._path,'KPOINTS'),'r') as kpts:
            for i in range(4):
                kpts.readline()
            kpts = kpts.readlines()
        self._rec_vector = np.empty(shape = (0,3), dtype=float)
        self._kpt_label = []

        for i in kpts:
            self._rec_vector = np.append(self._rec_vector, np.array([i.split()[:3]], dtype=float), axis=0)
        
        for i in range(0, len(kpts)+1, 2):
            if i == 0:
                if kpts[i].split()[-1][1:] == 'G' or kpts[i].split()[-1][1:] == 'GAMMA' or kpts[i].split()[-1][1:] == 'gamma' or kpts[i-1].split()[-1][1:] == 'Gamma':  
                    self._kpt_label.append("$\Gamma$")
                else:
                    self._kpt_label.append(kpts[i].split()[-1][1:])
            else:
                if kpts[i-1].split()[-1][1:] == 'G' or kpts[i-1].split()[-1][1:] == 'GAMMA' or kpts[i-1].split()[-1][1:] == 'gamma' or kpts[i-1].split()[-1][1:] == 'Gamma':  
                    self._kpt_label.append("$\Gamma$")
                else:
                    self._kpt_label.append(kpts[i-1].split()[-1][1:])
        
        rec_line =  int(os.popen("cat -n " +  os.path.join(self._path,'OUTCAR') + " | grep 'reciprocal lattice vectors'", 'r').read().split()[0])
        bk = np.array(os.popen("sed -n " + str(rec_line+1) + "," + str(rec_line+3) + "p " + os.path.join(self._path,'OUTCAR')).read().split()).reshape(3, -1)
        self._rec_lattice = np.empty(shape = (0,3), dtype=float)
        for i in bk:
            self._rec_lattice = np.append(self._rec_lattice, np.array([i[3:]], dtype = float), axis = 0)
       
        delta = [np.linalg.norm(np.dot(self._rec_lattice, dk)) for dk in self._kpts[1:]-self._kpts[:-1] ]
        delta.insert(0,0)
        
        self._xkpt = np.cumsum(delta)
        
        # volume = self._lattice[0][0]*self._lattice[1][1]*self._lattice[2][2] + self._lattice[0][1]*self._lattice[1][2]*self._lattice[2][0] + self._lattice[0][2]*self._lattice[1][0]*self._lattice[2][1] - \
        #          self._lattice[0][0]*self._lattice[1][2]*self._lattice[2][1] - self._lattice[0][1]*self._lattice[1][0]*self._lattice[2][2] - self._lattice[0][2]*self._lattice[1][1]*self._lattice[2][0]
        # self._rec_lattice = np.empty(shape = (0,3), dtype=float)
        # for i in range(3):
        #     if i == 0:
        #         j = 1
        #         k = 2
        #     elif i == 1:
        #         j = 2
        #         k = 0
        #     elif i == 2:
        #         j = 0
        #         k = 1
        #     self._rec_lattice = np.append(self._rec_lattice, np.array([(2*np.pi*np.cross(self._lattice[j], self._lattice[k]))/volume]), axis = 0)

        return self

    def ext_procar_info(self):

        """
        Obtain the occupied state array:
            self._occ_state: shape (nspin, nkpoints, nbands, natoms+1(tot), orbitals+1(tot))
        Obtrain the energy band:
            self._energy: shape (nspin, nkpoints, nbands,) 

        """
        energy = np.empty(shape = (0,) ,dtype=float)
        self._kpts = np.empty(shape = (0,3), dtype = float)
        for i in range(len(self.__occ_state)-1, -1, -1):
            if self.__occ_state[i].split() == []:  # del blank line
                del self.__occ_state[i]
            elif self.__occ_state[i].split()[0] == 'ion':  # del ion and orbital label line
                del self.__occ_state[i]
            elif self.__occ_state[i].split()[0] == 'band':  # del band line 
                energy = np.append(energy, np.array([self.__occ_state[i].split()[4]], dtype=float)-self._fermi, axis = 0) 
                del self.__occ_state[i]
            elif self.__occ_state[i].split()[0] == 'k-point':
                self._kpts = np.append(self._kpts, np.array([self.__occ_state[i].split()[3:6]], dtype=float), axis = 0)
                del self.__occ_state[i]
            elif self.__occ_state[i].split()[0] == '#':
                del self.__occ_state[i]
        
        energy = energy[::-1]  # 由于是倒叙删除，所以为了保证索引一致，要将energy倒叙
        self._kpts = self._kpts[::-1] if self._ispin == 1 else self._kpts[::-1][:self._kpoints]
        occ_state = [i.split()[1:] for i in self.__occ_state]  # del 'tot'

        if self._ispin == 1:
            assert len(occ_state) == 1*self._kpoints*self._bands*(self._ions+1)  # +1 for tot
            self._occ_state = np.array(occ_state, dtype=float).reshape(1, self._kpoints, self._bands, self._ions+1, -1)
            self._energy = energy.reshape(1, self._kpoints, self._bands)
        elif self._ispin == 2:
            assert len(occ_state) == 2*self._kpoints*self._bands*(self._ions+1)
            self._occ_state = np.array(occ_state, dtype=float).reshape(2, self._kpoints, self._bands, self._ions+1, -1)
            self._energy = energy.reshape(2, self._kpoints, self._bands)
        # self._occ_state.shape -> ispin, k-points, bands, atoms, orbital 
        # self._energy -> ispin, kpoints, bands
     
        return self
    
    def process_proinfo(self, pro_dict, IFsum_element = True, IFsum_atom = True, IFsum_orbital = True):
        
        self._ext_atom_index()
        pro_data = {}
        if self._ispin == 1:
            for i in pro_dict:
                pro_data[i] = {}
                for j in pro_dict[i]:
                    pro_data[i][j] = {}
                    for k in pro_dict[i][j]:
                        if isinstance(j, int):
                            if k == 'tot':
                                pro_data[i][j][k] = self._occ_state[:,:,:,j-1,-1]  
                            elif k == 's':
                                pro_data[i][j][k] = self._occ_state[:,:,:,j-1,0]
                            elif k == 'p':
                                pro_data[i][j][k] = np.sum(self._occ_state[:,:,:,j-1,1:4], axis = 3)
                            elif k == 'd':
                                pro_data[i][j][k] = np.sum(self._occ_state[:,:,:,j-1,4:9], axis = 3)
                            elif k in self.__label_col:
                                pro_data[i][j][k] = self._occ_state[:,:,:,j-1, np.argwhere(self.__label_col==k)[0][0]]
                            elif '-' in k:
                                bk = []
                                for n in k.split('-'):
                                    if n == 'p':
                                        bk += [1,2,3]
                                    elif n == 'd':
                                        bk += [4, 5, 6, 7, 8]
                                    else:
                                        bk.append(np.argwhere(self.__label_col==n)[0][0])
                                pro_data[i][j][k] = np.sum(self._occ_state[:,:,:,j-1, bk], axis = 3)
                            
                        elif j == 'all':
                            start_i = self._atom_index[i][0]
                            end_i =  self._atom_index[i][-1]
                            if k == 'tot':
                                pro_data[i][j][k] = np.sum(self._occ_state[:,:,:,start_i:end_i, -1], axis = 3)
                            elif k == 's':
                                pro_data[i][j][k] = np.sum(self._occ_state[:,:,:,start_i:end_i, 0], axis = 3)
                            elif k == 'p':
                                pro_data[i][j][k] = np.sum(np.sum(self._occ_state[:,:,:,start_i:end_i, 1:4], axis = 4), axis = 3)
                            elif k == 'd':
                                pro_data[i][j][k] = np.sum(np.sum(self._occ_state[:,:,:,start_i:end_i, 4:9], axis = 4), axis = 3)
                            elif k in self.__label_col:
                                pro_data[i][j][k] = np.sum(self._occ_state[:,:,:,start_i:end_i, np.argwhere(self.__label_col==k)[0][0]], axis = 3)
                            elif '-' in k:
                                bk = []
                                for n in k.split('-'):
                                    if n == 'p':
                                        bk += [1,2,3]
                                    elif n == 'd':
                                        bk += [4, 5, 6, 7, 8]
                                    else:
                                        bk.append(np.argwhere(self.__label_col==n)[0][0])
                                pro_data[i][j][k] = np.sum(np.sum(self._occ_state[:,:,:,start_i:end_i, bk], axis = 4), axis = 3)
                                
                        elif ':' in j:
                            start_i = int(j.split(':')[0])-1
                            end_i = int(j.split(':')[-1])
                            if k == 'tot':
                                pro_data[i][j][k] = np.sum(self._occ_state[:,:,:,start_i:end_i, -1], axis = 3)
                            elif k == 's':
                                pro_data[i][j][k] = np.sum(self._occ_state[:,:,:,start_i:end_i,  0], axis = 3)
                            elif k == 'p':
                                pro_data[i][j][k] = np.sum(np.sum(self._occ_state[:,:,:,start_i:end_i,  1:4], axis = 4), axis = 3)
                            elif k == 'd':
                                pro_data[i][j][k] = np.sum(np.sum(self._occ_state[:,:,:,start_i:end_i,  4:9], axis = 4), axis = 3)
                            elif k in self.__label_col:
                                pro_data[i][j][k] = np.sum(self._occ_state[:,:,:,start_i:end_i, np.argwhere(self.__label_col==k)[0][0]], axis = 3)
                            elif '-' in k:
                                bk = []
                                for n in k.split('-'):
                                    if n == 'p':
                                        bk += [1,2,3]
                                    elif n == 'd':
                                        bk += [4, 5, 6, 7, 8]
                                    else:
                                        bk.append(np.argwhere(self.__label_col==n)[0][0])
                                pro_data[i][j][k] = np.sum(np.sum(self._occ_state[:,:,:,start_i:end_i, bk], axis = 4), axis = 3)

                        elif '-' in j:
                            ei = [int(n)-1 for n in j.split('-')]
                            if k == 'tot':
                                pro_data[i][j][k] = np.sum(self._occ_state[:,:,:,ei, -1], axis = 3)
                            elif k == 's':
                                pro_data[i][j][k] = np.sum(self._occ_state[:,:,:,ei,  0], axis = 3)
                            elif k == 'p':
                                pro_data[i][j][k] = np.sum(np.sum(self._occ_state[:,:,:,ei, 1:4], axis = 4), axis = 3)
                            elif k == 'd':
                                pro_data[i][j][k] = np.sum(np.sum(self._occ_state[:,:,:,ei,  4:9], axis = 4), axis = 3)
                            elif k in self.__label_col:
                                pro_data[i][j][k] = np.sum(self._occ_state[:,:,:,ei, np.argwhere(self.__label_col==k)[0][0]], axis = 3)
                            elif '-' in k:
                                bk = []
                                for n in k.split('-'):
                                    if n == 'p':
                                        bk += [1,2,3]
                                    elif n == 'd':
                                        bk += [4, 5, 6, 7, 8]
                                    else:
                                        bk.append(np.argwhere(self.__label_col==n)[0][0])
                                pro_data[i][j][k] = np.sum(np.sum(self._occ_state[:,:,:,ei, :], axis = 3)[:,:,:,bk], axis = 3)
        
        elif self._ispin == 2:
            for s in pro_dict:
                if s.lower() == 'up' or s.lower() == 'down':
                    pro_data[s] = {}
                else:
                    raise KeyError('Wrong Spin Keyword "{}"'.format(s))
                for i in pro_dict[s]:
                    pro_data[s][i] = {}
                    for j in pro_dict[s][i]:
                        pro_data[s][i][j] = {}
                        for k in pro_dict[s][i][j]:
                            if isinstance(j, int):
                                if k == 'tot':
                                    pro_data[s][i][j][k] = self._occ_state[:1,:,:,j-1,-1] if s.lower() == 'up' else self._occ_state[1:2,:,:,j-1,-1]
                                elif k == 's':
                                    pro_data[s][i][j][k] = self._occ_state[:1,:,:,j-1,0] if s.lower() == 'up' else self._occ_state[1:2,:,:,j-1,0]
                                elif k == 'p':
                                    pro_data[s][i][j][k] = np.sum(self._occ_state[:1,:,:,j-1,1:4], axis = 3) if s.lower() == 'up' else np.sum(self._occ_state[1:2,:,:,j-1,1:4], axis = 3)
                                elif k == 'd':
                                    pro_data[s][i][j][k] = np.sum(self._occ_state[:1,:,:,j-1,4:9], axis = 3) if s.lower() == 'up' else np.sum(self._occ_state[1:2,:,:,j-1,4:9], axis = 3)
                                elif k in self.__label_col:
                                    pro_data[s][i][j][k] = self._occ_state[:1,:,:,j-1, np.argwhere(self.__label_col==k)[0][0]] if s.lower() == 'up' else  self._occ_state[1:2,:,:,j-1, np.argwhere(self.__label_col==k)[0][0]]
                                elif '-' in k:
                                    bk = []
                                    for n in k.split('-'):
                                        if n == 'p':
                                            bk += [1,2,3]
                                        elif n == 'd':
                                            bk += [4, 5, 6, 7, 8]
                                        else:
                                            bk.append(np.argwhere(self.__label_col==n)[0][0])
                                    pro_data[s][i][j][k] = np.sum(self._occ_state[:1,:,:,j-1, bk], axis = 3) if s.lower() == 'up' else  np.sum(self._occ_state[1:2,:,:,j-1, bk], axis = 3)
                            
                            elif j == 'all':
                                start_i = self._atom_index[i][0]
                                end_i   = self._atom_index[i][-1]
                                if k == 'tot':
                                    pro_data[s][i][j][k] = np.sum(self._occ_state[:1,:,:,start_i:end_i, -1], axis = 3) if s.lower() == 'up' else np.sum(self._occ_state[1:2,:,:,start_i:end_i, -1], axis = 3)
                                elif k == 's':
                                    pro_data[s][i][j][k] = np.sum(self._occ_state[:1,:,:,start_i:end_i, 0], axis = 3) if s.lower() == 'up' else  np.sum(self._occ_state[1:2,:,:,start_i:end_i, 0], axis = 3)
                                elif k == 'p':
                                    pro_data[s][i][j][k] = np.sum(np.sum(self._occ_state[:1,:,:,start_i:end_i, 1:4], axis = 4), axis = 3) if s.lower() == 'up' else np.sum(np.sum(self._occ_state[1:2,:,:,start_i:end_i, 1:4], axis = 4), axis = 3)
                                elif k == 'd':
                                    pro_data[s][i][j][k] = np.sum(np.sum(self._occ_state[:1,:,:,start_i:end_i, 4:9], axis = 4), axis = 3) if s.lower() == 'up' else np.sum(np.sum(self._occ_state[1:2,:,:,start_i:end_i, 4:9], axis = 4), axis = 3)
                                elif k in self.__label_col:
                                    pro_data[s][i][j][k] = np.sum(self._occ_state[:1,:,:,start_i:end_i, np.argwhere(self.__label_col==k)[0][0]], axis = 3) if s.lower() == 'up' else np.sum(self._occ_state[1:2,:,:,start_i:end_i, np.argwhere(self.__label_col==k)[0][0]], axis = 3)
                                elif '-' in k:
                                    bk = []
                                    for n in k.split('-'):
                                        if n == 'p':
                                            bk += [1,2,3]
                                        elif n == 'd':
                                            bk += [4, 5, 6, 7, 8]
                                        else:
                                            bk.append(np.argwhere(self.__label_col==n)[0][0])
                                    pro_data[s][i][j][k] = np.sum(np.sum(self._occ_state[0:1,:,:,start_i:end_i, bk], axis = 4), axis = 3) if s.lower() == 'up' else np.sum(np.sum(self._occ_state[1:2,:,:,start_i:end_i, bk], axis = 4), axis = 3)
                                
                            elif ':' in j:
                                start_i = int(j.split(':')[0])-1
                                end_i = int(j.split(':')[-1])
                                if k == 'tot':
                                    pro_data[s][i][j][k] = np.sum(self._occ_state[:1,:,:,start_i:end_i, -1], axis = 3) if s.lower() == 'up' else np.sum(self._occ_state[1:2,:,:,start_i:end_i, -1], axis = 3)
                                elif k == 's':
                                    pro_data[s][i][j][k] = np.sum(self._occ_state[:1,:,:,start_i:end_i,  0], axis = 3) if s.lower() == 'up' else np.sum(self._occ_state[1:2,:,:,start_i:end_i,  0], axis = 3) 
                                elif k == 'p':
                                    pro_data[s][i][j][k] = np.sum(np.sum(self._occ_state[:1,:,:,start_i:end_i,  1:4], axis = 4), axis = 3) if s.lower() == 'up' else np.sum(np.sum(self._occ_state[1:2,:,:,start_i:end_i,  1:4], axis = 4), axis = 3)
                                elif k == 'd':
                                    pro_data[s][i][j][k] = np.sum(np.sum(self._occ_state[:1,:,:,start_i:end_i,  4:9], axis = 4), axis = 3) if s.lower() == 'up' else np.sum(np.sum(self._occ_state[1:2,:,:,start_i:end_i,  4:9], axis = 4), axis = 3)
                                elif k in self.__label_col:
                                    pro_data[s][i][j][k] = np.sum(self._occ_state[:1,:,:,start_i:end_i, np.argwhere(self.__label_col==k)[0][0]], axis = 3) if s.lower() == 'up' else np.sum(self._occ_state[1:2,:,:,start_i:end_i, np.argwhere(self.__label_col==k)[0][0]], axis = 3)
                                elif '-' in k:
                                    bk = []
                                    for n in k.split('-'):
                                        if n == 'p':
                                            bk += [1,2,3]
                                        elif n == 'd':
                                            bk += [4, 5, 6, 7, 8]
                                        else:
                                            bk.append(np.argwhere(self.__label_col==n)[0][0])
                                    pro_data[s][i][j][k] = np.sum(np.sum(self._occ_state[:1,:,:,start_i:end_i, bk], axis = 4), axis = 3) if s.lower() == 'up' else np.sum(np.sum(self._occ_state[1:2,:,:,start_i:end_i, bk], axis = 4), axis = 3)

                            elif '-' in j:
                                ei = [int(n)-1 for n in j.split('-')]
                                if k == 'tot':
                                    pro_data[s][i][j][k] = np.sum(self._occ_state[:1,:,:,ei, -1], axis = 3) if s.lower() == 'up' else np.sum(self._occ_state[1:2,:,:,ei, -1], axis = 3) 
                                elif k == 's':
                                    pro_data[s][i][j][k] = np.sum(self._occ_state[:1,:,:,ei,  0], axis = 3) if s.lower() == 'up' else np.sum(self._occ_state[1:2,:,:,ei,  0], axis = 3)
                                elif k == 'p':
                                    pro_data[s][i][j][k] = np.sum(np.sum(self._occ_state[:1,:,:,ei, 1:4], axis = 4), axis = 3) if s.lower() == 'up' else np.sum(np.sum(self._occ_state[1:2,:,:,ei, 1:4], axis = 4), axis = 3)
                                elif k == 'd':
                                    pro_data[s][i][j][k] = np.sum(np.sum(self._occ_state[:1,:,:,ei,  4:9], axis = 4), axis = 3) if s.lower() == 'up' else np.sum(np.sum(self._occ_state[1:2,:,:,ei,  4:9], axis = 4), axis = 3)
                                elif k in self.__label_col:
                                    pro_data[s][i][j][k] = np.sum(self._occ_state[:1,:,:,ei, np.argwhere(self.__label_col==k)[0][0]], axis = 3) if s.lower() == 'up' else np.sum(self._occ_state[1:2,:,:,ei, np.argwhere(self.__label_col==k)[0][0]], axis = 3)
                                elif '-' in k:
                                    bk = []
                                    for n in k.split('-'):
                                        if n == 'p':
                                            bk += [1,2,3]
                                        elif n == 'd':
                                            bk += [4, 5, 6, 7, 8]
                                        else:
                                            bk.append(np.argwhere(self.__label_col==n)[0][0])
                                    pro_data[s][i][j][k] = np.sum(np.sum(self._occ_state[:1,:,:,ei, :], axis = 3)[:,:,:,bk], axis = 3) if s.lower() == 'up' else np.sum(np.sum(self._occ_state[1:2,:,:,ei, :], axis = 3)[:,:,:,bk], axis = 3)
                                
        self.__get_kpath()
        self._pro_data = pro_data
        
        return self
    
    def plot_normal_band(self, eV_range = [-5, 5], filename = 'band.jpg'):
        
        self.__get_kpath()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 8))
        ax1 = plt.subplot(111)

        if self._ispin == 1:
            up_energy = self._energy[0].T
            for i in up_energy:
                ax1.plot(self._xkpt, i, '-', c = 'blue', linewidth=1.5, label = '$Non-spin$')
        elif self._ispin == 2:
            up_energy = self._energy[0].T
            down_energy = self._energy[1].T
            for i_index, i  in enumerate(up_energy):
                ax1.plot(self._xkpt, up_energy[i_index], '-', c = 'blue', linewidth=1.5, label= '$Spin \ up$')
                ax1.plot(self._xkpt, down_energy[i_index], '--', c = 'darkorchid', linewidth=1.5, alpha = 0.8, label = '$Spin \ down$')

        handles, labels = plt.gca().get_legend_handles_labels()
        bk = {}
        for i in range(len(labels)-1, -1 , -1):
            bk[labels[i]] = handles[i] 
        v = list(bk.values())
        v.reverse()
        k = list(bk.keys())
        k.reverse()
        legend_font = {'weight': 'normal', "size" : 15}  
        # if self._ispin == 2:
        legend_font = {'weight': 'normal', "size" : 15}  
        ax1.legend(v, k, loc = 'upper right', ncol = 1, fontsize = 'large', prop = legend_font)

        xticks = []
        for i in range(0, len(self._xkpt)+1, self._interpoint):
            if i > 0: i -= 1
            xticks.append(self._xkpt[i])
            ax1.axvline(x = self._xkpt[i], c='black', linewidth=1)
        ax1.axhline(0,  c = 'black', linewidth=1, linestyle='dashed')

        ax1.set_xticks(xticks)
        ax1.set_xticklabels(self._kpt_label)

        ax1.set_ylabel('$E - E_{f} \ [eV]$', fontsize = 25, fontweight = 'medium')
        # ax1.set_xlabel('', fontsize = 25, fontweight = 'medium')
        ax1.tick_params(axis = 'both', labelsize = 20, width = 2, length = 5, direction='in')
        ax1.set_xlim(np.min(self._xkpt), np.max(self._xkpt))
        ax1.set_ylim(eV_range[0], eV_range[1])
        ax1.spines['bottom'].set_linewidth(2)
        ax1.spines['left'].set_linewidth(2)
        ax1.spines['top'].set_linewidth(2)
        ax1.spines['right'].set_linewidth(2)
        plt.tight_layout()
        plt.savefig(filename, dpi = 600)
        
    
    def plot_fat_band_circles(self, eV_range = [-5,5], color = 'rainbow', size = 400, filename = 'fat-band.jpg'):

        import matplotlib.pyplot as plt
        
        if self._ispin == 1:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 8))
            ax1 = plt.subplot(111)
            energy = self._energy[0].T
            pro = np.empty(shape = (0, self._kpoints, self._bands,), dtype=float)
            for i in self._pro_data:
                for j in self._pro_data[i]:
                    for k in self._pro_data[i][j]:
                        pro = np.append(pro, np.array([self._pro_data[i][j][k][0]]), axis = 0)
            label = []
            for i in self._pro_data:
                for j in self._pro_data[i]:
                    for k in self._pro_data[i][j]:
                        if j == 'all' and k == 'tot':
                            label.append('$Element: ' + i + '$')
                        elif j != 'all' and k == 'tot':
                            label.append('$'+ i + '-' +  str(j) + '$')
                        elif j != 'all' and k != 'tot':
                            label.append('$'+i + '-' +  str(j) + '-' + k+'$')
                        elif j == 'all' and k != 'tot':
                            label.append('$Element: ' + i + '-' + k + '$')

            color_l = []
            r_l = []
            for i in range(len(label)):
                color_l.append(self.color_list[i])
                r, = ax1.plot([0,1],[0,1], c=self.color_list[i], label=label[i])
                legend_font = {'weight': 'normal', "size" : 15}  
                ax1.legend( loc = 'upper right', ncol = 1, fontsize = 'large', prop = legend_font,)
                r_l.append(r)
            for line in r_l:
                line.remove()

            for i in energy:
                ax1.plot(self._xkpt, i, '-', c = 'black', linewidth=1.5)

            for i in range(pro.shape[0]):
                for j in range(pro[i].T.shape[0]):
                    # ax1.scatter(self._xkpt, energy[j], marker = 'o',c = color_l[i], s=pro[i].T[j]*size, alpha = 0.8, linewidths=2)
                    ax1.scatter(self._xkpt, energy[j], marker = 'o', facecolors = 'none', edgecolors = color_l[i], s=pro[i].T[j]*size, alpha = 0.8, linewidths=2)
                    # ax1.scatter(self._xkpt , energy[j], marker = 'o', facecolors = 'none', edgecolors = cmap0(i), s=pro[i].T[j]*size, alpha = 0.8, linewidths=2)

            xticks = []
            for i in range(0, len(self._xkpt)+1, self._interpoint):
                if i > 0: i -= 1
                xticks.append(self._xkpt[i])
                ax1.axvline(x = self._xkpt[i], c='black', linewidth=1)
            ax1.axhline(0,  c = 'black', linewidth=1, linestyle='dashed')

            ax1.set_xticks(xticks)
            ax1.set_xticklabels(self._kpt_label)
        
            ax1.set_ylabel('$E - E_{f} \ [eV]$', fontsize = 25, fontweight = 'medium')
            # ax1.set_xlabel('', fontsize = 25, fontweight = 'medium')
            ax1.tick_params(axis = 'both', labelsize = 20, width = 2, length = 5, direction='in')
            ax1.set_xlim(np.min(self._xkpt), np.max(self._xkpt))
            ax1.set_ylim(eV_range[0], eV_range[1])
            ax1.spines['bottom'].set_linewidth(2)
            ax1.spines['left'].set_linewidth(2)
            ax1.spines['top'].set_linewidth(2)
            ax1.spines['right'].set_linewidth(2)

        elif self._ispin == 2:
            fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 8))
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)

            up_energy = self._energy[0].T
            down_energy = self._energy[1].T
            up_pro = np.empty(shape = (0, self._kpoints, self._bands,), dtype=float)
            down_pro = np.empty(shape = (0, self._kpoints, self._bands,), dtype=float)
            for s in self._pro_data:
                for i in self._pro_data[s]:
                    for j in self._pro_data[s][i]:
                        for k in self._pro_data[s][i][j]:
                            if s.lower() == 'up':
                                up_pro = np.append(up_pro, np.array([self._pro_data[s][i][j][k][0]]), axis = 0)
                            elif s.lower() == 'down':
                                down_pro = np.append(down_pro, np.array([self._pro_data[s][i][j][k][0]]), axis = 0)
        
            up_label = []
            down_label = []

            for s in self._pro_data:
                for i in self._pro_data[s]:
                    for j in self._pro_data[s][i]:
                        for k in self._pro_data[s][i][j]:
                            if j == 'all' and k == 'tot':
                                if s.lower() == 'up':
                                    up_label.append('$Element: ' + i + '$')
                                elif s.lower() == 'down':
                                    down_label.append('$Element: ' + i + '$')
                            elif j != 'all' and k == 'tot':
                                if s.lower() == 'up':
                                    up_label.append('$'+ i + '-' +  str(j) + '$')
                                elif s.lower() == 'down':
                                    down_label.append('$'+ i + '-' +  str(j) + '$')
                            elif j != 'all' and k != 'tot':
                                if s.lower() == 'up':
                                    up_label.append('$'+i + '-' +  str(j) + '-' + k+'$')
                                elif s.lower() == 'down':
                                    down_label.append('$'+i + '-' +  str(j) + '-' + k+'$')
                            elif j == 'all' and k != 'tot':
                                if s.lower() == 'up':
                                    up_label.append('$Element: ' + i + '-' + k + '$')
                                elif s.lower() == 'down':
                                    down_label.append('$Element: ' + i + '-' + k + '$')
        
            up_color_l = []
            up_r_l = []
            down_color_l = []
            down_r_l = []
            for i in range(len(up_label)):
                up_color_l.append(self.color_list[i])
                r, = ax1.plot([0,1],[0,1], c=self.color_list[i], label=up_label[i])
                legend_font = {'weight': 'normal', "size" : 15}  
                ax1.legend( loc = 'upper right', ncol = 1, fontsize = 'large', prop = legend_font,)
                up_r_l.append(r)
            for line in up_r_l:
                line.remove()

            for i in range(len(down_label)):
                down_color_l.append(self.color_list[i])
                r, = ax2.plot([0,1],[0,1], c=self.color_list[i], label=down_label[i])
                legend_font = {'weight': 'normal', "size" : 15}  
                ax2.legend( loc = 'upper right', ncol = 1, fontsize = 'large', prop = legend_font,)
                down_r_l.append(r)
            for line in down_r_l:
                line.remove()


            for i in up_energy:
                ax1.plot(self._xkpt, i, '-', c = 'blue', linewidth=1)
            for i in down_energy:
                ax2.plot(self._xkpt, i, '-', c = 'darkorchid', linewidth=1)

            # cmap0=plt.get_cmap('rainbow', len(up_label))

            for i in range(up_pro.shape[0]):
                for j in range(up_pro[i].T.shape[0]):
                    # ax1.scatter(self._xkpt, up_energy[j], marker = 'o',c = color_l[i], s=up_pro[i].T[j]*size, alpha = 0.8, linewidths=2)
                    ax1.scatter(self._xkpt, up_energy[j], marker = 'o', facecolors = 'none', edgecolors = up_color_l[i], s=up_pro[i].T[j]*size, alpha = 0.8, linewidths=2)
                    # ax1.scatter(self._xkpt , up_energy[j], marker = 'o', facecolors = 'none', edgecolors = cmap0(i), s=up_pro[i].T[j]*size, alpha = 0.8, linewidths=2)

            for i in range(down_pro.shape[0]):
                for j in range(down_pro[i].T.shape[0]):
                    # ax2.scatter(self._xkpt, up_energy[j], marker = 'o',c = color_l[i], s=down_pro[i].T[j]*size, alpha = 0.8, linewidths=2)
                    ax2.scatter(self._xkpt, down_energy[j], marker = 'o', facecolors = 'none', edgecolors = down_color_l[i], s=down_pro[i].T[j]*size, alpha = 0.8, linewidths=2)
                    # ax2.scatter(self._xkpt , up_energy[j], marker = 'o', facecolors = 'none', edgecolors = cmap0(i), s=down_pro[i].T[j]*size, alpha = 0.8, linewidths=2)

            xticks = []
            for i in range(0, len(self._xkpt)+1, self._interpoint):
                if i > 0: i -= 1
                xticks.append(self._xkpt[i])
                ax1.axvline(x = self._xkpt[i], c='black', linewidth=1)
                ax2.axvline(x = self._xkpt[i], c='black', linewidth=1)
            ax1.axhline(0,  c = 'black', linewidth=1, linestyle='dashed')
            ax2.axhline(0,  c = 'black', linewidth=1, linestyle='dashed')
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(self._kpt_label)
            ax2.set_xticks(xticks)
            ax2.set_xticklabels(self._kpt_label)

            ax1.text(self._xkpt[0] + 0.05*(self._xkpt[-1]-self._xkpt[0]), 0.9*eV_range[1], '$Spin \  up$', fontsize=18, style='oblique', weight = 'roman',
                     bbox = {
                             'facecolor' : 'white',
                             'edgecolor' : 'black',
                             'alpha' : 0.8,
                            'boxstyle' : 'round'
                            })
            ax2.text(self._xkpt[0] + 0.05*(self._xkpt[-1]-self._xkpt[0]), 0.9*eV_range[1], '$Spin \ down$', fontsize=18, style='oblique', weight = 'roman',
                     bbox = {
                             'facecolor' : 'white',
                             'edgecolor' : 'black',
                             'alpha' : 0.8,
                            'boxstyle' : 'round'
                            })
        
            ax1.set_ylabel('$E - E_{f} \ [eV]$', fontsize = 25, fontweight = 'medium')
            # ax1.set_xlabel('', fontsize = 25, fontweight = 'medium')
            ax1.tick_params(axis = 'both', labelsize = 20, width = 2, length = 5, direction='in')
            ax1.set_xlim(np.min(self._xkpt), np.max(self._xkpt))
            ax1.set_ylim(eV_range[0], eV_range[1])
            ax1.spines['bottom'].set_linewidth(2)
            ax1.spines['left'].set_linewidth(2)
            ax1.spines['top'].set_linewidth(2)
            ax1.spines['right'].set_linewidth(2)
            ax2.set_ylabel('$E - E_{f} \ [eV]$', fontsize = 25, fontweight = 'medium')
            # ax2.set_xlabel('', fontsize = 25, fontweight = 'medium')
            ax2.tick_params(axis = 'both', labelsize = 20, width = 2, length = 5, direction='in')
            ax2.set_xlim(np.min(self._xkpt), np.max(self._xkpt))
            ax2.set_ylim(eV_range[0], eV_range[1])
            ax2.spines['bottom'].set_linewidth(2)
            ax2.spines['left'].set_linewidth(2)
            ax2.spines['top'].set_linewidth(2)
            ax2.spines['right'].set_linewidth(2)
        plt.tight_layout()
        plt.savefig(filename, dpi = 600)
    
    def plot_fat_band_gradient(self, eV_range = [-5,15], color = 'rainbow',  filename = 'fat-band-gradient.jpg'):

        from matplotlib import cm
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.collections import LineCollection

        if self._ispin == 1:

            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 8))
            ax1 = plt.subplot(111)
            
            energy = self._energy[0].T
            pro = np.empty(shape = (0, self._kpoints, self._bands,), dtype=float)
            for i in self._pro_data:
                for j in self._pro_data[i]:
                    for k in self._pro_data[i][j]:
                        pro = np.append(pro, np.array([self._pro_data[i][j][k][0]]), axis = 0)
            
            label = []
            for i in self._pro_data:
                for j in self._pro_data[i]:
                    for k in self._pro_data[i][j]:
                        if j == 'all' and k == 'tot':
                            label.append('$Element: ' + i + '$')
                        elif j != 'all' and k == 'tot':
                            label.append('$'+ i + '-' +  str(j) + '$')
                        elif j != 'all' and k != 'tot':
                            label.append('$'+i + '-' +  str(j) + '-' + k+'$')
                        elif j == 'all' and k != 'tot':
                            label.append('$Element: ' + i + '-' + k + '$')

            if len(label) > 3:
                raise ValueError("Gradient color can only be applied to two or three component system !")

            tot_pro = np.sum(pro, axis = 0)
            ratio_pro = np.empty(shape = (0, self._kpoints, self._bands), dtype=float)
            for i in pro:
                ratio_pro = np.append(ratio_pro, np.array([np.true_divide(i, tot_pro)]), axis = 0)

            if len(label) == 2:
                a = np.sum([x for x in ratio_pro[0].reshape(-1) if np.isnan(x) == False])
                b = np.sum([x for x in ratio_pro[1].reshape(-1) if np.isnan(x) == False])
            if a >= b:
                order = 0
            else:
                order = 1
        
            bk_points = np.array([self._xkpt, energy[0]]).T.reshape(-1,1,2)
            bk_segments = np.concatenate([bk_points[:-1], bk_points[1:]], axis=1)
            lc = LineCollection(bk_segments, cmap=cm.get_cmap(color))
            line = ax1.add_collection(lc)
            cbar = fig.colorbar(line, ax = ax1)
            cbar.set_ticks(ticks = [0,1], )
            import copy
            label_bk = copy.deepcopy(label)  
            cbar.set_ticklabels([label[np.abs(1-order)], label[order]])
            # cbar.add_lines((0,1),('black','black'), linewidths = (1.5))
            lc.remove()

            if len(label) == 2:
                for i in range(len(ratio_pro[order].T)):
                    cmp_o = cm.get_cmap(color)(np.linspace(0.0, 1, 100))
                    points = np.array([self._xkpt, energy[i]]).T.reshape(-1,1,2)
                    segments = np.concatenate([points[:-1],points[1:]], axis=1)
                    newcmp = [tuple(list(cmp_o[int(j*100)-1])) if not np.isnan(j) else tuple([1.0, 1.0, 1.0, 0.5]) for j in ratio_pro[order].T[i]]
                    lc = LineCollection(segments, colors=newcmp, linewidths = 2)
                    ax1.add_collection(lc)

            xticks = []
            for i in range(0, len(self._xkpt)+1, self._interpoint):
                if i > 0: i -= 1
                xticks.append(self._xkpt[i])
                ax1.axvline(x = self._xkpt[i], c='black', linewidth=1)
            ax1.axhline(0,  c = 'black', linewidth=1, linestyle='dashed')

            ax1.set_xticks(xticks)
            ax1.set_xticklabels(self._kpt_label)

            ax1.set_ylabel('$E - E_{f} \ [eV]$', fontsize = 25, fontweight = 'medium')
            # ax1.set_xlabel('', fontsize = 25, fontweight = 'medium')
            ax1.tick_params(axis = 'both', labelsize = 20, width = 2, length = 5, direction='in')
            ax1.set_xlim(np.min(self._xkpt), np.max(self._xkpt))
            ax1.set_ylim(eV_range[0], eV_range[1])
            ax1.spines['bottom'].set_linewidth(2)
            ax1.spines['left'].set_linewidth(2)
            ax1.spines['top'].set_linewidth(2)
            ax1.spines['right'].set_linewidth(2)


        elif self._ispin == 2:

            fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 8))
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)

            up_energy = self._energy[0].T
            down_energy = self._energy[1].T
            up_pro = np.empty(shape = (0, self._kpoints, self._bands,), dtype=float)
            down_pro = np.empty(shape = (0, self._kpoints, self._bands,), dtype=float)
            for s in self._pro_data:
                for i in self._pro_data[s]:
                    for j in self._pro_data[s][i]:
                        for k in self._pro_data[s][i][j]:
                            if s.lower() == 'up':
                                up_pro = np.append(up_pro, np.array([self._pro_data[s][i][j][k][0]]), axis = 0)
                            elif s.lower() == 'down':
                                down_pro = np.append(down_pro, np.array([self._pro_data[s][i][j][k][0]]), axis = 0)
        
            up_label = []
            down_label = []
            for s in self._pro_data:
                for i in self._pro_data[s]:
                    for j in self._pro_data[s][i]:
                        for k in self._pro_data[s][i][j]:
                            if j == 'all' and k == 'tot':
                                if s.lower() == 'up':
                                    up_label.append('$Element: ' + i + '$')
                                elif s.lower() == 'down':
                                    down_label.append('$Element: ' + i + '$')
                            elif j != 'all' and k == 'tot':
                                if s.lower() == 'up':
                                    up_label.append('$'+ i + '-' +  str(j) + '$')
                                elif s.lower() == 'down':
                                    down_label.append('$'+ i + '-' +  str(j) + '$')
                            elif j != 'all' and k != 'tot':
                                if s.lower() == 'up':
                                    up_label.append('$'+i + '-' +  str(j) + '-' + k+'$')
                                elif s.lower() == 'down':
                                    down_label.append('$'+i + '-' +  str(j) + '-' + k+'$')
                            elif j == 'all' and k != 'tot':
                                if s.lower() == 'up':
                                    up_label.append('$Element: ' + i + '-' + k + '$')
                                elif s.lower() == 'down':
                                    down_label.append('$Element: ' + i + '-' + k + '$')
        
            if len(up_label) > 3:
                raise ValueError("Gradient color can only be applied to two or three component system !")

            if len(down_label) > 3:
                raise ValueError("Gradient color can only be applied to two or three component system !")

            up_tot_pro = np.sum(up_pro, axis = 0)
            up_ratio_pro = np.empty(shape = (0, self._kpoints, self._bands), dtype=float)
            for i in up_pro:
                up_ratio_pro = np.append(up_ratio_pro, np.array([np.true_divide(i, up_tot_pro)]), axis = 0)

            if len(up_label) == 2:
                a = np.sum([x for x in up_ratio_pro[0].reshape(-1) if np.isnan(x) == False])
                b = np.sum([x for x in up_ratio_pro[1].reshape(-1) if np.isnan(x) == False])
            if a >= b:
                order = 0
            else:
                order = 1

            bk_points = np.array([self._xkpt, up_energy[0]]).T.reshape(-1,1,2)
            bk_segments = np.concatenate([bk_points[:-1], bk_points[1:]], axis=1)
            lc = LineCollection(bk_segments, cmap=cm.get_cmap(color))
            line = ax1.add_collection(lc)
            cbar = fig.colorbar(line, ax = ax1)
            cbar.set_ticks(ticks = [0,1], )
            import copy
            up_label_bk = copy.deepcopy(up_label)  
            cbar.set_ticklabels([up_label[np.abs(1-order)], up_label[order]])
            # cbar.add_lines((0,1),('black','black'), linewidths = (1.5))
            lc.remove()

            if len(up_label) == 2:
                for i in range(len(up_ratio_pro[order].T)):
                    cmp_o = cm.get_cmap(color)(np.linspace(0.0, 1, 100))
                    points = np.array([self._xkpt, up_energy[i]]).T.reshape(-1,1,2)
                    segments = np.concatenate([points[:-1],points[1:]], axis=1)
                    newcmp = [tuple(list(cmp_o[int(j*100)-1])) if not np.isnan(j) else tuple([1.0, 1.0, 1.0, 0.5]) for j in up_ratio_pro[order].T[i]]
                    lc = LineCollection(segments, colors=newcmp, linewidths = 2)
                    ax1.add_collection(lc)
         
            down_tot_pro = np.sum(down_pro, axis = 0)
            down_ratio_pro = np.empty(shape = (0, self._kpoints, self._bands), dtype=float)
            for i in down_pro:
                down_ratio_pro = np.append(down_ratio_pro, np.array([np.true_divide(i, down_tot_pro)]), axis = 0)

            if len(down_label) == 2:
                a = np.sum([x for x in down_ratio_pro[0].reshape(-1) if np.isnan(x) == False])
                b = np.sum([x for x in down_ratio_pro[1].reshape(-1) if np.isnan(x) == False])
            if a >= b:
                order = 0
            else:
                order = 1

            bk_points = np.array([self._xkpt, down_energy[0]]).T.reshape(-1,1,2)
            bk_segments = np.concatenate([bk_points[:-1], bk_points[1:]], axis=1)
            lc = LineCollection(bk_segments, cmap=cm.get_cmap(color))
            line = ax2.add_collection(lc)
            cbar = fig.colorbar(line, ax = ax2)
            cbar.set_ticks(ticks = [0,1], )
            import copy
            down_label_bk = copy.deepcopy(down_label)  
            cbar.set_ticklabels([down_label[np.abs(1-order)], down_label[order]])
            # cbar.add_lines((0,1),('black','black'), linewidths = (1.5))
            lc.remove()

            if len(down_label) == 2:
                for i in range(len(down_ratio_pro[order].T)):
                    cmp_o = cm.get_cmap(color)(np.linspace(0.0, 1, 100))
                    points = np.array([self._xkpt, down_energy[i]]).T.reshape(-1,1,2)
                    segments = np.concatenate([points[:-1],points[1:]], axis=1)
                    newcmp = [tuple(list(cmp_o[int(j*100)-1])) if not np.isnan(j) else tuple([1.0, 1.0, 1.0, 0.5]) for j in down_ratio_pro[order].T[i]]
                    lc = LineCollection(segments, colors=newcmp, linewidths = 2)
                    ax2.add_collection(lc)

            xticks = []
            for i in range(0, len(self._xkpt)+1, self._interpoint):
                if i > 0: i -= 1
                xticks.append(self._xkpt[i])
                ax1.axvline(x = self._xkpt[i], c='black', linewidth=1)
                ax2.axvline(x = self._xkpt[i], c='black', linewidth=1)
            ax1.axhline(0,  c = 'black', linewidth=1, linestyle='dashed')
            ax2.axhline(0,  c = 'black', linewidth=1, linestyle='dashed')

            ax1.set_xticks(xticks)
            ax1.set_xticklabels(self._kpt_label)
            ax2.set_xticks(xticks)
            ax2.set_xticklabels(self._kpt_label)

            ax1.text(self._xkpt[0] + 0.05*(self._xkpt[-1]-self._xkpt[0]), 0.9*eV_range[1], '$Spin \  up$', fontsize=18, style='oblique', weight = 'roman',
                     bbox = {
                             'facecolor' : 'white',
                             'edgecolor' : 'black',
                             'alpha' : 0.8,
                            'boxstyle' : 'round'
                            })
            ax2.text(self._xkpt[0] + 0.05*(self._xkpt[-1]-self._xkpt[0]), 0.9*eV_range[1], '$Spin \ down$', fontsize=18, style='oblique', weight = 'roman',
                     bbox = {
                             'facecolor' : 'white',
                             'edgecolor' : 'black',
                             'alpha' : 0.8,
                            'boxstyle' : 'round'
                            })

            ax1.set_ylabel('$E - E_{f} \ [eV]$', fontsize = 25, fontweight = 'medium')
            # ax1.set_xlabel('', fontsize = 25, fontweight = 'medium')
            ax1.tick_params(axis = 'both', labelsize = 20, width = 2, length = 5, direction='in')
            ax1.set_xlim(np.min(self._xkpt), np.max(self._xkpt))
            ax1.set_ylim(eV_range[0], eV_range[1])
            ax1.spines['bottom'].set_linewidth(2)
            ax1.spines['left'].set_linewidth(2)
            ax1.spines['top'].set_linewidth(2)
            ax1.spines['right'].set_linewidth(2)

            ax2.set_ylabel('$E - E_{f} \ [eV]$', fontsize = 25, fontweight = 'medium')
            # ax1.set_xlabel('', fontsize = 25, fontweight = 'medium')
            ax2.tick_params(axis = 'both', labelsize = 20, width = 2, length = 5, direction='in')
            ax2.set_xlim(np.min(self._xkpt), np.max(self._xkpt))
            ax2.set_ylim(eV_range[0], eV_range[1])
            ax2.spines['bottom'].set_linewidth(2)
            ax2.spines['left'].set_linewidth(2)
            ax2.spines['top'].set_linewidth(2)
            ax2.spines['right'].set_linewidth(2)

        plt.tight_layout()
        plt.savefig(filename, dpi = 600)

        

if __name__ == '__main__':

    # 1. Non-Spin 
    # 1.1 Normal band
    ep = ExtractProcar('./example/CsAgInCl').ext_procar_info()
    ep.plot_normal_band(filename='./example/CsAgInCl/band.jpg')

    # 1.2 Fat-Band
    # 1.2.1 Circle 
    # Any number of components can be drawn 
    ep.process_proinfo({'Cs':{'1:2':['s']}, 
                        'In':{'all':['s-p-dx2', 'tot']},
                        'Ag':{4:['s-p-dx2','s', 'px', 'd']},  
                        'Cl':{'9-10':['p', 'd','px-pz-s'], 6:['tot'], 8:['tot']}}).plot_fat_band_circles(eV_range = [-2,8], size = 500, filename = './example/CsAgInCl/fat-band-circle.jpg')
    
    # 1.2.2 Gradient color
    # Currently only two components can be processed !!!
    ep.process_proinfo({"In":{'all':['p']},
                        'Cl':{'all':['p']},
                        }).plot_fat_band_gradient(color ='jet',eV_range = [-2,8],filename = './example/CsAgInCl/fat-band-gradient.jpg')


    # 2. Spin
    # 2.1 Normal band
    ep = ExtractProcar('./example/CsAgFeCl-Spin/AFM').ext_procar_info()
    ep.plot_normal_band(filename='./example/CsAgFeCl-Spin/AFM/band.jpg')

    ep = ExtractProcar('./example/CsAgFeCl-Spin/FM').ext_procar_info()
    ep.plot_normal_band(filename='./example/CsAgFeCl-Spin/FM/band.jpg')

    # 2.2 Fat-Band
    # 2.2.1 Circle
    # Any number of components can be drawn 
    ep = ExtractProcar('./example/CsAgFeCl-Spin/FM').ext_procar_info()
    ep.process_proinfo({'up':  {
                                'Ag':{'all':['s-p-dx2', 'tot']},
                                'Fe':{'13:17':['s', 'p', 'd']},  
                               },
                        'down':{
                                'Cl':{'17-20':['p', 'd','px-pz-s'], 28:['s','p','d']}},
                              }).plot_fat_band_circles(eV_range = [-5,5], size = 500, filename = './example/CsAgFeCl-Spin/FM/fat-band-circle.jpg')
    # 2.2.2 Gradient color
    # Currently only two components can be processed !!!
    ep = ExtractProcar('./example/CsAgFeCl-Spin/FM').ext_procar_info()
    ep.process_proinfo({'up':  {
                                'Ag':{'all':['s-p-dx2', 'tot']},
                                # 'Fe':{13:['s-p-dx2','s', 'px', 'd']},  
                               },
                        'down':{
                                'Fe':{'13:17':['p','d']}},
                              }).plot_fat_band_gradient(eV_range = [-5,5], color ='rainbow', filename = './example/CsAgFeCl-Spin/FM/fat-band-gradient.jpg')
