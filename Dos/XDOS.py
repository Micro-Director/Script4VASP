#!/home/rtzhao/anaconda3/bin/python
#-*-coding:utf-8-*-
from enum import Flag
import os
import re
import copy
import numpy as np
import pandas as pd

class XDos(object):

    def __init__(self, path = './'):

        self._path = path

        self._ispin = int(os.popen("grep 'ISPIN' " + os.path.join(path,'OUTCAR'), 'r').read().split()[2])
        with open(os.path.join(self._path, 'DOSCAR'), 'r') as dos:
            for i in range(5):
                dos.readline()
            info = dos.readline()
            
        self._eVmin = float(info.split()[0])
        self._eVmax = float(info.split()[1])
        self._Nedos = int(info.split()[2])
        self._fermi = float(info.split()[3])
    
    def __ext_atom_index(self):
        
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

    def ext_tdos(self,):
        
        self.__ext_atom_index()
        if self._ispin == 1:
            self._tdos = np.empty(shape=(0,3), dtype=float)
            with open(os.path.join(self._path, "DOSCAR"),'r') as td:
                for i in range(6): 
                    td.readline()
                for i in range(self._Nedos):
                    self._tdos = np.vstack((self._tdos, np.array(td.readline().split(), dtype=float)))
            self._tdos = self._tdos[:,:2] - np.array([self._fermi, 0]) 
            self.__label_col = ['energy', 'tot']
            self._energy = self._tdos[:,0]
        elif self._ispin == 2:
            self._tdos = np.empty(shape=(0,5), dtype=float)
            with open(os.path.join(self._path, "DOSCAR"),'r') as td:
                for i in range(6): 
                    td.readline()
                for i in range(self._Nedos):
                    self._tdos = np.vstack((self._tdos, np.array(td.readline().split(), dtype=float)))
            self._tdos = self._tdos[:,:3] - np.array([self._fermi, 0, 0]) 
            self.__label_col = ['energy', 'tot_up', 'tot_down']
            self._energy = self._tdos[:,0]

        return self

    def ext_ordos(self):

        """
        spd_dict = {
            's'   : [0],
            'p'   : [1, 2, 3],
            'd'   : [4, 5, 6, 7, 8],
            'f'   : [9, 10, 11, 12, 13, 14, 15],
            'py'  : [1],
            'pz'  : [2],
            'px'  : [3],
            'dxy' : [4],
            'dyz' : [5],
            'dz2' : [6],
            'dxz' : [7],
            'dx2' : [8],
    "fy(3x2-y2)"  : [9],
    "fxyz  "      : [10],
    "fyz2  "      : [11],
    "fz3   "      : [12],
    "fxz2  "      : [13],
    "fz(x2-y2)"   : [14],
    "fx(x2-3y2) " : [15],
    }
        """

        self.__ext_atom_index()
        _ordos_columns = len(os.popen("sed -n " + str(self._Nedos+8) + "p " + os.path.join(self._path, "DOSCAR"), 'r').read().split())
        self._pdos = np.empty(shape=(0, _ordos_columns), dtype=float)

        isr = 7 + 1*(self._Nedos + 1)
        for i in range(len(self.__elements)):
            # print (i, isr, isr+self._Nedos-1)
            bk = os.popen("sed -n " + str(isr) + ',' + str(isr+self._Nedos -1) + "p " + os.path.join(self._path, "DOSCAR"), 'r').read().split()
            isr += (self._Nedos+1)
            self._pdos = np.concatenate((self._pdos, 
                                         np.array(bk, dtype=float).reshape(self._Nedos,-1)), axis = 0)
        
        self._energy = self._pdos[:,:1].T - np.array([self._fermi,]) 
        self._energy = self._energy[0][:self._Nedos]
        self._pdos = self._pdos.reshape(len(self.__elements), self._Nedos, -1)  # (natoms, state, orbitals)
        
        if _ordos_columns == 4:
            # energy s p d
            self.__label_col = ['energy', 's', 'p', 'd']
        elif _ordos_columns == 10:
            # energy s py pz px dxy dyz dz2-r2 dxz dx2-y2
            self.__label_col = ['energy', 's', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2r2', 'dxz', 'dx2y2']
        elif _ordos_columns == 7:
            # energy s_up s_down p_up p_down d_up d_down
            self.__label_col = ['energy', 's up', 's down', 'p up', 'p down', 'd up', 'd down']
        elif _ordos_columns == 19:
            # energy s_up s_down py_up py_down pz_up pz_down px_up px_down  dxy_up dxy_down dyz_up dyz_down dz2-r2_up dz2-r2_down dxz_up dxz_down dx2-y2_up dx2-y2_down
            self.__label_col = ['energy', 's up', 's down', 'py up', 'py down', 'pz up', 'pz down', 'px up', 'px down', 'dxy up', 'dxy down', 'dyz up', 'dyz down', 'dz2r2 up', 'dz2r2 down','dxz up', 'dxz down','dx2y2 up', 'dx2y2 down']
        
        return self
    

    def __process_dos(self, dos_dict, IFave_atom):

        """
        Erange: [energy minimum, energy maximum]  

        {'Cl':{6:['s','p'], 7:['px','pz']}, 'Ag':{2:['p','d']}}
            
        {'Cl':{'all':['s', 'p', ]}}
        {'Cl':{'all':['tot']}}
        {'all':{'all':['tot']}}

        """
    
        dos_data = {}
        
        for i in dos_dict:
            for j in dos_dict[i]:
                for k in dos_dict[i][j]:

                    if len(self.__label_col) == 10:

                        if isinstance(j, int):
                            if k == 's':
                                label = i + '-' + str(j) + '-' + k
                                dos_data[label] = self._pdos[j-1,:,1]
                            elif k == 'p' :
                                label = i + '-' + str(j) + '-' + k
                                dos_data[label] = np.sum(self._pdos[j-1,:,2:5], axis = 1)
                            elif k == 'd' :
                                label = i + '-' + str(j) + '-' + k
                                dos_data[label] = np.sum(self._pdos[j-1,:,5:10], axis = 1)
                            elif k in self.__label_col:
                                label = i + '-' + str(j) + '-' + k
                                dos_data[label] = self._pdos[j-1,:,np.argwhere(np.array(self.__label_col)==k)[0][0]]
                            elif k == 'tot':
                                label = i + '-' + str(j)
                                dos_data[label] = np.sum(self._pdos[j-1,:,1:], axis = 1)
                            elif '-' in k:
                                bk = []
                                for n in k.split('-'):
                                    if 'p' in n:
                                        bk += [3,5,7]
                                    elif 'd' in n:
                                        bk += [9, 11, 13, 15, 17]
                                    else:
                                        bk.append(np.argwhere(self.__label_col == n))[0][0]
                                dos_data[i + '-' + str(j) + '-' + k] = np.sum(self._pdos[j-1,:, bk].T, axis = 1)
                        
                        elif j == 'all':
                            if i not in self.__elements:
                                raise ValueError("Please enter the correct element name, the {} is not in the element list ! ".format(i))  
                            elif k == 's':
                                label = i  + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,1], axis=0), self._atom_index[i][1]-self._atom_index[i][0]) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,1], axis=0) 
                            elif k == 'p' :
                                label = i  + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,2:5], axis=0), axis =1), self._atom_index[i][1]-self._atom_index[i][0]) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,2:5], axis=0), axis =1)
                            elif k == 'd' :
                                label = i + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,5:10], axis=0), axis=1), self._atom_index[i][1]-self._atom_index[i][0]) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,5:10], axis=0), axis=1)
                            elif k in self.__label_col:
                                label = i + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,np.argwhere(np.array(self.__label_col)==k)[0][0]], axis = 0), self._atom_index[i][1]-self._atom_index[i][0]) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,np.argwhere(np.array(self.__label_col)==k)[0][0]], axis = 0)
                            elif k == 'tot':
                                label = i
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,1:], axis=0), axis=1), self._atom_index[i][1]-self._atom_index[i][0])
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,1:], axis=0), axis=1)
                            elif '-' in k:
                                bk = []
                                for n in k.split('-'):
                                    if 'p' in n:
                                        bk += [3,5,7]
                                    elif 'd' in n:
                                        bk += [9, 11, 13, 15, 17]
                                    else:
                                        bk.append(np.argwhere(self.__label_col == n))[0][0]
                                dos_data[i + '-' + str(j) + '-' + k] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,bk], axis=0), axis=1)
                            
                            elif i not in self.__elements:

                                raise ValueError("Please enter the correct element name ! ")
                                
                        elif ':' in j:
                            es = int(j.split(':')[0])-1
                            ed = int(j.split(':')[1])-1
                            if k == 's':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[es:ed,:,1], axis=0), ed-es) 
                                else:
                                    dos_data[label] =np.sum(self._pdos[es:ed,:,1], axis=0)
                            elif k == 'p':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[es:ed,:,2:5], axis=0), axis =1), ed-es) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[es:ed,:,2:5], axis=0), axis =1)
                            elif k == 'd':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[es:ed,:,5:10], axis=0), axis =1), ed-es) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[es:ed,:,5:10], axis=0), axis =1)
                            elif k in self.__label_col:
                                label = i + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[es:ed,:,np.argwhere(np.array(self.__label_col)==k)[0][0]], axis = 0), ed-es) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[es:ed,:,np.argwhere(np.array(self.__label_col)==k)[0][0]], axis = 0)
                            elif k == 'tot':
                                label = i + '-' + j
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[es:ed,:,1:], axis=0), axis=1), ed-es)
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[es:ed,:,1:], axis=0), axis=1)
                            elif '-' in k:
                                bk = []
                                for n in k.split('-'):
                                    if 'p' in n:
                                        bk += [3,5,7]
                                    elif 'd' in n:
                                        bk += [9, 11, 13, 15, 17]
                                    else:
                                        bk.append(np.argwhere(self.__label_col == n))[0][0]
                                dos_data[i + '-' + str(j) + '-' + k] = np.sum(np.sum(self._pdos[es:ed,:,bk], axis=0), axis=1)

                        elif '-' in j:
                            ei = [int(n)-1 for n in j.split('-')]
                            if k == 's':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[ei,:,1], axis=0), len(ei)) 
                                else:
                                    dos_data[label] =np.sum(self._pdos[ei,:,1], axis=0)
                            elif k == 'p':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,2:5], axis=0), axis =1),  len(ei)) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,2:5], axis=0), axis =1)
                            elif k == 'd':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,5:10], axis=0), axis =1), len(ei)) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,5:10], axis=0), axis =1)
                            elif k in self.__label_col:
                                label = i + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[ei,:,np.argwhere(np.array(self.__label_col)==k)[0][0]], axis = 0), len(ei)) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[ei,:,np.argwhere(np.array(self.__label_col)==k)[0][0]], axis = 0)
                            elif k == 'tot':
                                label = i + '-' + j
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,1:], axis=0), axis=1), len(ei))
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,1:], axis=0), axis=1)
                            elif '-' in k:
                                bk = []
                                for n in k.split('-'):
                                    if 'p' in n:
                                        bk += [3,5,7]
                                    elif 'd' in n:
                                        bk += [9, 11, 13, 15, 17]
                                    else:
                                        bk.append(np.argwhere(self.__label_col == n))[0][0]
                                dos_data[i + '-' + str(j) + '-' + k] = np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,bk].T, axis = 0)
                        
                    elif len(self.__label_col) == 4:
                        if isinstance(j, int):
                            if k == 's':
                                label = i + '-' + str(j-1) + '-' + k
                                dos_data[label] = self._pdos[j-1,:,1]
                            elif k == 'p' :
                                label = i + '-' + str(j-1) + '-' + k
                                dos_data[label] = self._pdos[j-1,:,2]
                            elif k == 'd' :
                                label = i + '-' + str(j-1) + '-' + k
                                dos_data[label] = self._pdos[j-1,:,3]
                            elif k == 'tot':
                                label = i + '-' + k
                                dos_data[label] = np.sum(self._pdos[j-1,:,1:], axis = 1)
                            elif '-' in k:
                                bk = []
                                for n in k.split('-'):
                                    if 'p' in n:
                                        bk += [3,5,7]
                                    elif 'd' in n:
                                        bk += [9, 11, 13, 15, 17]
                                    else:
                                        bk.append(np.argwhere(self.__label_col == n))[0][0]
                                dos_data[i + '-' + str(j) + '-' + k] = np.sum(self._pdos[j-1,:, bk].T, axis = 1)

                        elif j == 'all':

                            if i not in self.__elements:
                                raise ValueError("Please enter the correct element name, the {} is not in the element list ! ".format(i))    

                            elif k == 's':
                                label = i + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,1], axis=0), self._atom_index[i][1]-self._atom_index[i][0]) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,1], axis=0)
                            elif k == 'p' :
                                label = i + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,2], axis=0), self._atom_index[i][1]-self._atom_index[i][0]) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,2], axis=0)
                            elif k == 'd' :
                                label = i + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,3], axis=0), self._atom_index[i][1]-self._atom_index[i][0]) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,3], axis=0)
                            elif k == 'tot':
                                label = i + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,1:], axis=0), axis = 1), self._atom_index[i][1]-self._atom_index[i][0])
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,1:], axis=0), axis = 1)             
                            elif '-' in k:
                                bk = []
                                for n in k.split('-'):
                                    if 'p' in n:
                                        bk += [3,5,7]
                                    elif 'd' in n:
                                        bk += [9, 11, 13, 15, 17]
                                    else:
                                        bk.append(np.argwhere(self.__label_col == n))[0][0]
                                dos_data[i + '-' + str(j) + '-' + k] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,bk], axis=0), axis=1)

                        elif ':' in j:
                            es = int(j.split(':')[0])-1
                            ed = int(j.split(':')[1])-1
                            if k == 's':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[es:ed,:,1], axis=0), ed-es) 
                                else:
                                    dos_data[label] =np.sum(self._pdos[es:ed,:,1], axis=0)
                            elif k == 'p':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[es:ed,:,2], axis=0), ed-es) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[es:ed,:, 2], axis=0)
                            elif k == 'd':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[es:ed,:,3], axis=0), ed-es) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[es:ed,:,3], axis=0)
                            elif k == 'tot':
                                label = i  + '-' + j
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[es:ed,:,1:], axis=0),axis = 1), ed-es) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[es:ed,:,1:], axis=0),axis = 1)
                            elif '-' in k:
                                bk = []
                                for n in k.split('-'):
                                    if 'p' in n:
                                        bk += [3,5,7]
                                    elif 'd' in n:
                                        bk += [9, 11, 13, 15, 17]
                                    else:
                                        bk.append(np.argwhere(self.__label_col == n))[0][0]
                                dos_data[i + '-' + str(j) + '-' + k] = np.sum(np.sum(self._pdos[es:ed,:,bk], axis=0), axis=1)

                        elif '-' in j:
                            ei = [int(n)-1 for n in j.split('-')]
                            if k == 's':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[ei,:,1], axis=0), len(ei)) 
                                else:
                                    dos_data[label] =np.sum(self._pdos[ei,:,1], axis=0)
                            elif k == 'p':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,2], axis=0), axis =1),  len(ei)) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,2], axis=0), axis =1)
                            elif k == 'd':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,3], axis=0), axis =1), len(ei)) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,3], axis=0), axis =1)
                            elif k == 'tot':
                                label = i + '-' + j
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,1:], axis=0), axis=1), len(ei))
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,1:], axis=0), axis=1)
                            elif '-' in k:
                                bk = []
                                for n in k.split('-'):
                                    if 'p' in n:
                                        bk += [3,5,7]
                                    elif 'd' in n:
                                        bk += [9, 11, 13, 15, 17]
                                    else:
                                        bk.append(np.argwhere(self.__label_col == n))[0][0]
                                dos_data[i + '-' + str(j) + '-' + k] = np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,bk].T, axis = 0)

                    elif len(self.__label_col) == 2:
                        # if i=='all' and j == 'all' and k == 'tot':
                        if IFave_atom:
                            dos_data['Total'] = np.true_divide(self._tdos[:,1], len(self.__elements))
                        else:
                            dos_data['Total'] = self._tdos[:,1]

                    elif len(self.__label_col) == 3:
                        # if i=='all' and j == 'all' and k == 'tot'
                        
                        if IFave_atom:
                            dos_data['Total-up'] = np.true_divide(self._tdos[:,1], len(self.__elements))

                            dos_data['Total-down'] = np.true_divide(self._tdos[:,2], len(self.__elements)) * -1
                        else:
                            dos_data['Total-up'] = self._tdos[:,1]
                            dos_data['Total-down'] = self._tdos[:,2] * -1

                    elif len(self.__label_col) == 19:

                        if isinstance(j, int):
                            if k == 's up':
                                label = i + '-' + str(j) + '-' + k
                                dos_data[label] = self._pdos[j-1,:,1]
                            elif k == 's down':
                                label = i + '-' + str(j) + '-' + k 
                                dos_data[label] = self._pdos[j-1,:,2] * -1
                            elif k == 'p up' :
                                label = i + '-' + str(j) + '-' + k
                                dos_data[label] = np.sum(self._pdos[j-1,:,[3,5,7]].T, axis = 1)
                            elif k == 'p down' :
                                label = i + '-' + str(j) + '-' + k
                                dos_data[label] = np.sum(self._pdos[j-1,:,[4,6,8]].T, axis = 1) * -1
                            elif k == 'd up':
                                label = i + '-' + str(j) + '-' + k
                                dos_data[label] = np.sum(self._pdos[j-1,:,[9,11,13,15,17]].T, axis = 1)
                            elif k == 'd down':
                                label = i + '-' + str(j) + '-' + k
                                dos_data[label] = np.sum(self._pdos[j-1,:,[10,12,14,16,18]].T, axis = 1) * -1
                            elif k in self.__label_col:
                                label = i + '-' + str(j) + '-' + k
                                if 'down' in k:
                                    dos_data[label] = self._pdos[j-1,:,np.argwhere(np.array(self.__label_col)==k)[0][0]] * -1
                                else:
                                    dos_data[label] = self._pdos[j-1,:,np.argwhere(np.array(self.__label_col)==k)[0][0]]
                            elif k == 'tot':
                                dos_data[i + '-' + str(j) + '-' +  'up'] = np.sum(self._pdos[j-1,:,[1,3,5,7,9,11,13,15,17]].T, axis = 1)
                                dos_data[i + '-' + str(j) + '-' +  'down'] = np.sum(self._pdos[j-1,:,[2,4,6,8,10,12,14,16,18]].T, axis = 1) * -1
                            elif '-' in k:
                                bk = []
                                if k.split(' ')[-1] == 'up':
                                    for n in k.split(' ')[0].split('-'):
                                        if 'p' in n:
                                            bk += [3,5,7]
                                        elif 'd' in n:
                                            bk += [9, 11, 13, 15, 17]
                                        else:
                                            bk.append(np.argwhere(self.__label_col == n +' up'))[0][0]
                                    dos_data[i + '-' + str(j) + '-' + k] = np.sum(self._pdos[j-1,:, bk].T, axis = 1)
                                elif k.split(' ')[-1] == 'down':
                                    for n in k.split(' ')[0].split('-'):
                                        if 'p' in n:
                                            bk += [4,6,8]
                                        elif 'd' in n:
                                            bk += [10, 12, 14, 16, 18]
                                        else:
                                            bk.append(np.argwhere(self.__label_col == n + ' down'))[0][0]
                                        dos_data[i + '-' + str(j) + '-' + k] = np.sum(self._pdos[j-1,:,bk].T, axis = 1) * -1

                        elif j == 'all':

                            if i not in self.__elements:
                                raise ValueError("Please enter the correct element name, the {} is not in the element list ! ".format(i))    
                            elif k == 's up':
                                label = i  + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,1], axis=0), self._atom_index[i][1]-self._atom_index[i][0]) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,1], axis=0) 
                            elif k == 's down':
                                label = i  + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,2], axis=0), self._atom_index[i][1]-self._atom_index[i][0]) * -1
                                else:
                                    dos_data[label] = np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,2], axis=0) * -1
                            elif k == 'p up' :
                                label = i  + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,[3,5,7]], axis=0), axis =1), self._atom_index[i][1]-self._atom_index[i][0]) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,[3,5,7]], axis=0), axis =1)
                            elif k == 'p down' :
                                label = i  + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,[4,6,8]], axis=0), axis =1), self._atom_index[i][1]-self._atom_index[i][0]) * -1
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,[4,6,8]], axis=0), axis =1) * -1
                            elif k == 'd up' :
                                label = i + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,[9,11,13,15,17]], axis=0), axis=1), self._atom_index[i][1]-self._atom_index[i][0]) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,[9,11,13,15,17]], axis=0), axis=1)
                            elif k == 'd down' :
                                label = i + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,[10,12,14,16,18]], axis=0), axis=1), self._atom_index[i][1]-self._atom_index[i][0]) * -1
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,[10,12,14,16,18]], axis=0), axis=1) * -1
                            elif k in self.__label_col and 'up' in k:
                                label = i + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,np.argwhere(np.array(self.__label_col)==k)[0][0]], axis = 0), self._atom_index[i][1]-self._atom_index[i][0]) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,np.argwhere(np.array(self.__label_col)==k)[0][0]], axis = 0)
                            elif k in self.__label_col and 'down' in k:
                                label = i + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,np.argwhere(np.array(self.__label_col)==k)[0][0]], axis = 0), self._atom_index[i][1]-self._atom_index[i][0]) * -1
                                else:
                                    dos_data[label] = np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,np.argwhere(np.array(self.__label_col)==k)[0][0]], axis = 0) * -1
                            elif k == 'tot up':
                                label = i + ' up'
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,[1,3,5,7,9,11,13,15,17]], axis=0), axis=1), self._atom_index[i][1]-self._atom_index[i][0])
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,[1,3,5,7,9,11,13,15,17]], axis=0), axis=1)
                            elif k == 'tot down':
                                label = i + ' down'
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,[2,4,6,8,10,12,14,16,18]], axis=0), axis=1), self._atom_index[i][1]-self._atom_index[i][0]) * -1
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,[2,4,6,8,10,12,14,16,18]], axis=0), axis=1) * -1
                            elif '-' in k:
                                bk = []
                                if k.split(' ')[-1] == 'up':
                                    for n in k.split(' ')[0].split('-'):
                                        if 'p' in n:
                                            bk += [3,5,7]
                                        elif 'd' in n:
                                            bk += [9, 11, 13, 15, 17]
                                        else:
                                            bk.append(np.argwhere(self.__label_col == n +' up'))[0][0]
                                    if IFave_atom:
                                        dos_data[i + '-' + k] = np.true_divide(np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,bk], axis=0), axis=1), self._atom_index[i][1]-self._atom_index[i][0]) 
                                    else:
                                        dos_data[i + '-' + k] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,bk], axis=0), axis=1)
                                elif k.split(' ')[-1] == 'down':
                                    for n in k.split(' ')[0].split('-'):
                                        if 'p' in n:
                                            bk += [4,6,8]
                                        elif 'd' in n:
                                            bk += [10, 12, 14, 16, 18]
                                        else:
                                            bk.append(np.argwhere(self.__label_col == n + ' down'))[0][0]
                                    if IFave_atom:
                                        dos_data[i + '-' + k] = np.true_divide(np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,bk], axis=0), axis=1), self._atom_index[i][1]-self._atom_index[i][0]) * -1
                                    else:
                                        dos_data[i + '-' + k] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,bk], axis=0), axis=1) * -1

                        elif ':' in j:
                            es = int(j.split(':')[0]) - 1
                            ed = int(j.split(':')[1]) - 1
                            if k == 's up':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[es:ed,:,1], axis=0), ed-es) 
                                else:
                                    dos_data[label] =np.sum(self._pdos[es:ed,:,1], axis=0)
                            elif k == 's down':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[es:ed,:,2], axis=0), ed-es) * -1
                                else:
                                    dos_data[label] =np.sum(self._pdos[es:ed,:,2], axis=0) * -1
                            elif k == 'p up':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[es:ed,:,[3,5,7]], axis=0), axis =1), ed-es) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[es:ed,:,[3,5,7]], axis=0), axis =1)
                            elif k == 'p down':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[es:ed,:,[4,6,8]], axis=0), axis =1), ed-es) * -1
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[es:ed,:,[4,6,8]], axis=0), axis =1) * -1
                            elif k == 'd up':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[es:ed,:,[9,11,13,15,17]], axis=0), axis =1), ed-es) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[es:ed,:,[9,11,13,15,17]], axis=0), axis =1)
                            elif k == 'd down':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[es:ed,:,[10,12,14,16,18]], axis=0), axis =1), ed-es) * -1
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[es:ed,:,[10,12,14,16,18]], axis=0), axis =1) * -1
                            elif k in self.__label_col and 'up' in k:
                                label = i + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[es:ed,:,np.argwhere(np.array(self.__label_col)==k)[0][0]], axis = 0), ed-es) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[es:ed,:,np.argwhere(np.array(self.__label_col)==k)[0][0]], axis = 0)
                            elif k in self.__label_col and 'down' in k:
                                label = i + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[es:ed,:,np.argwhere(np.array(self.__label_col)==k)[0][0]], axis = 0), ed-es) * -1
                                else:
                                    dos_data[label] = np.sum(self._pdos[es:ed,:,np.argwhere(np.array(self.__label_col)==k)[0][0]], axis = 0) * -1
                            elif k == 'tot up':
                                label = i + '-' + j + ' up'
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[es:ed,:,[1,3,5,7,9,11,13,15,17]], axis=0), axis=1), ed-es)
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[es:ed,:,[1,3,5,7,9,11,13,15,17]], axis=0), axis=1)
                            elif k == 'tot down':
                                label = i + '-' + j + ' down'
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[es:ed,:,[2,4,6,8,10,12,14,16,18]], axis=0), axis=1), ed-es) * -1
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[es:ed,:,[2,4,6,8,10,12,14,16,18]], axis=0), axis=1) * -1
                            elif '-' in k:
                                bk = []
                                if k.split(' ')[-1] == 'up':
                                    for n in k.split(' ')[0].split('-'):
                                        if 'p' in n:
                                            bk += [3,5,7]
                                        elif 'd' in n:
                                            bk += [9, 11, 13, 15, 17]
                                        else:
                                            bk.append(np.argwhere(self.__label_col == n +' up'))[0][0]
                                    if IFave_atom:
                                        dos_data[i + '-' + j + '-' + k] = np.true_divide(np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,bk], axis=0), axis=1), self._atom_index[i][1]-self._atom_index[i][0]) 
                                    else:
                                        dos_data[i + '-' + j + '-' + k] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,bk], axis=0), axis=1)
                                elif k.split(' ')[-1] == 'down':
                                    for n in k.split(' ')[0].split('-'):
                                        if 'p' in n:
                                            bk += [4,6,8]
                                        elif 'd' in n:
                                            bk += [10, 12, 14, 16, 18]
                                        else:
                                            bk.append(np.argwhere(self.__label_col == n + ' down'))[0][0]
                                    if IFave_atom:
                                        dos_data[i + '-' + j + '-' + k] = np.true_divide(np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,bk], axis=0), axis=1), self._atom_index[i][1]-self._atom_index[i][0]) * -1
                                    else:
                                        dos_data[i + '-' + j + '-' + k] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,bk], axis=0), axis=1) * -1

                        elif '-' in j:
                            ei = [int(n)-1 for n in j.split('-')]
                            if k == 's up':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[1]].T, axis = 0), len(ei)) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[1]].T, axis = 0)
                            elif k == 's down':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[2]].T, axis = 0), len(ei))  * -1
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[2]].T, axis = 0) * -1
                            elif k == 'p up':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[3,5,7]].T, axis = 0), len(ei)) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[3,5,7]].T, axis = 0)
                            elif k == 'p down':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[4,6,8]].T, axis = 0), len(ei))  * -1
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[4,6,8]].T, axis = 0) * -1
                            elif k == 'd up':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[9,11,13,15,17]].T, axis = 0), len(ei)) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[9,11,13,15,17]].T, axis = 0)
                            elif k == 'd down':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[10,12,14,16,18]].T, axis = 0), len(ei)) * -1
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[10,12,14,16,18]].T, axis = 0) * -1
                            elif k in self.__label_col and 'up' in k:
                                label = i + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[ei,:,:], axis =0)[:,np.argwhere(np.array(self.__label_col)==k)[0][0]], len(ei)) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[ei,:,:], axis =0)[:,np.argwhere(np.array(self.__label_col)==k)[0][0]]
                            elif k in self.__label_col and 'down' in k:
                                label = i + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[ei,:,:], axis =0)[:,np.argwhere(np.array(self.__label_col)==k)[0][0]], len(ei)) * -1
                                else:
                                    dos_data[label] = np.sum(self._pdos[ei,:,:], axis =0)[:,np.argwhere(np.array(self.__label_col)==k)[0][0]] * -1
                            elif k == 'tot up':
                                label = i + '-' + j + ' up'
                                
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[1,3,5,7,9,11,13,15,17]].T, axis = 0), len(ei)) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[1,3,5,7,9,11,13,15,17]].T, axis = 0)
                            elif k == 'tot down':
                                label = i + '-' + j + ' down'
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[2,4,6,8,10,12,14,16,18]].T, axis = 0), len(ei))  * -1
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[2,4,6,8,10,12,14,16,18]].T, axis = 0) * -1
                            elif '-' in k:
                                bk = []
                                if k.split(' ')[-1] == 'up':
                                    for n in k.split(' ')[0].split('-'):
                                        if 'p' in n:
                                            bk += [3,5,7]
                                        elif 'd' in n:
                                            bk += [9, 11, 13, 15, 17]
                                        else:
                                            # print(n)
                                            bk.append(np.argwhere(self.__label_col == n +' up'))[0][0]
                                    if IFave_atom:                        
                                        dos_data[i + '-' + j + '-' + k] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,bk].T, axis = 0), len(ei)) 
                                    else:
                                        dos_data[i + '-' + j + '-' + k] = np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,bk].T, axis = 0)
                                elif k.split(' ')[-1] == 'down':
                                    for n in k.split(' ')[0].split('-'):
                                        if 'p' in n:
                                            bk += [4,6,8]
                                        elif 'd' in n:
                                            bk += [10, 12, 14, 16, 18]
                                        else:
                                            bk.append(np.argwhere(self.__label_col == n + ' down'))[0][0]
                                    if IFave_atom:
                                        dos_data[i + '-' + j + '-' + k] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,bk].T, axis = 0), len(ei)) * -1
                                    else:
                                        dos_data[i + '-' + j + '-' + k] = np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,bk].T, axis = 0) * -1
                        
                    elif len(self.__label_col) == 7:
                        if isinstance(j, int):
                            if k == 's up':
                                label = i + '-' + str(j) + '-' + k
                                dos_data[label] = self._pdos[j-1,:,1]
                            elif k == 's down':
                                label = i + '-' + str(j) + '-' + k 
                                dos_data[label] = self._pdos[j-1,:,2] * -1
                            elif k == 'p up' :
                                label = i + '-' + str(j) + '-' + k
                                dos_data[label] = self._pdos[j-1,:,3]
                            elif k == 'p down' :
                                label = i + '-' + str(j) + '-' + k
                                dos_data[label] = self._pdos[j-1,:,4] * -1
                            elif k == 'd up':
                                label = i + '-' + str(j) + '-' + k
                                dos_data[label] = self._pdos[j-1,:,5]
                            elif k == 'd down':
                                label = i + '-' + str(j) + '-' + k
                                dos_data[label] = self._pdos[j-1,:,6] * -1
                            elif k == 'tot':
                                dos_data[i + '-' + str(j) + '-' +  'up'] = np.sum(self._pdos[j-1,:,[1,3,5,]].T, axis = 1)
                                dos_data[i + '-' + str(j) + '-' +  'down'] = np.sum(self._pdos[j-1,:,[2,4,6,]].T, axis = 1) * -1
                            elif '-' in k:
                                bk = []
                                if k.split(' ')[-1] == 'up':
                                    for n in k.split(' ')[0].split('-'):
                                        # if 'p' in n:
                                        #     bk += [3]
                                        # elif 'd' in n:
                                        #     bk += [5]
                                        # else:
                                        bk.append(np.argwhere(self.__label_col == n +' up'))[0][0]
                                    dos_data[i + '-' + str(j) + '-' + k] = np.sum(self._pdos[j-1,:, bk].T, axis = 1)
                                elif k.split(' ')[-1] == 'down':
                                    for n in k.split(' ')[0].split('-'):
                                        # if 'p' in n:
                                        #     bk += [4,]
                                        # elif 'd' in n:
                                        #     bk += [6,]
                                        # else:
                                        bk.append(np.argwhere(self.__label_col == n + ' down'))[0][0]
                                        dos_data[i + '-' + str(j) + '-' + k] = np.sum(self._pdos[j-1,:,bk].T, axis = 1) * -1

                        elif j == 'all':

                            if i not in self.__elements:
                                raise ValueError("Please enter the correct element name, the {} is not in the element list ! ".format(i))    

                            elif k == 's up':
                                label = i + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,1], axis=0), self._atom_index[i][1]-self._atom_index[i][0]) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,1], axis=0)
                            elif k == 's down':
                                label = i + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,2], axis=0), self._atom_index[i][1]-self._atom_index[i][0]) * -1
                                else:
                                    dos_data[label] = np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,2], axis=0) * -1
                            elif k == 'p up' :
                                label = i + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,3], axis=0), self._atom_index[i][1]-self._atom_index[i][0]) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,3], axis=0)
                            elif k == 'p down' :
                                label = i + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,4], axis=0), self._atom_index[i][1]-self._atom_index[i][0]) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,4], axis=0)
                            elif k == 'd up' :
                                label = i + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,5], axis=0), self._atom_index[i][1]-self._atom_index[i][0]) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,5], axis=0)
                            elif k == 'd down' :
                                label = i + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,6], axis=0), self._atom_index[i][1]-self._atom_index[i][0]) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,6], axis=0)
                            elif k == 'tot up':
                                label = i + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,[1,3,5]], axis=0), axis = 1), self._atom_index[i][1]-self._atom_index[i][0])
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,[1,3,5]], axis=0), axis = 1)    
                            elif k == 'tot down':
                                label = i + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,[2,4,6]], axis=0), axis = 1), self._atom_index[i][1]-self._atom_index[i][0])
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,[2,4,6]], axis=0), axis = 1)    

                            elif '-' in k:
                                bk = []
                                if k.split(' ')[-1] == 'up':
                                    for n in k.split(' ')[0].split('-'):
                                #     if 'p' in n:
                                #         bk += [3,5,7]
                                #     elif 'd' in n:
                                #         bk += [9, 11, 13, 15, 17]
                                #     else:
                                        bk.append(np.argwhere(self.__label_col == n))[0][0]
                                    if IFave_atom:
                                        dos_data[i + '-' + str(j) + '-' + k] = np.true_divide(np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,bk], axis=0), axis=1), self._atom_index[i][1]-self._atom_index[i][0])
                                    else:
                                        dos_data[i + '-' + str(j) + '-' + k] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,bk], axis=0), axis=1)
                                elif k.split(' ')[-1] == 'down':
                                    for n in k.split(' ')[0].split('-'):
                                #     if 'p' in n:
                                #         bk += [3,5,7]
                                #     elif 'd' in n:
                                #         bk += [9, 11, 13, 15, 17]
                                #     else:
                                        bk.append(np.argwhere(self.__label_col == n))[0][0]
                                    if IFave_atom:
                                        dos_data[i + '-' + str(j) + '-' + k] = np.true_divide(np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,bk], axis=0), axis=1), self._atom_index[i][1]-self._atom_index[i][0])
                                    else:
                                        dos_data[i + '-' + str(j) + '-' + k] = np.sum(np.sum(self._pdos[self._atom_index[i][0]:self._atom_index[i][1],:,bk], axis=0), axis=1)

                        elif ':' in j:
                            es = int(j.split(':')[0])-1
                            ed = int(j.split(':')[1])-1
                            if k == 's up':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[es:ed,:,1], axis=0), ed-es) 
                                else:
                                    dos_data[label] =np.sum(self._pdos[es:ed,:,1], axis=0)
                            elif k == 's down':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[es:ed,:,2], axis=0), ed-es) * -1
                                else:
                                    dos_data[label] =np.sum(self._pdos[es:ed,:,2], axis=0) * -1
                            elif k == 'p up':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[es:ed,:,3], axis=0), ed-es) 
                                else:
                                    dos_data[label] =np.sum(self._pdos[es:ed,:,3], axis=0)
                            elif k == 'p down':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[es:ed,:,4], axis=0), ed-es) * -1
                                else:
                                    dos_data[label] =np.sum(self._pdos[es:ed,:,4], axis=0) * -1
                            elif k == 'd up':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[es:ed,:,5], axis=0), ed-es) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[es:ed,:,5], axis=0)
                            elif k == 'd down':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[es:ed,:,6], axis=0), ed-es) * -1
                                else:
                                    dos_data[label] = np.sum(self._pdos[es:ed,:,6], axis=0) * -1
                            elif k == 'tot up':
                                label = i  + '-' + j
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[es:ed,:,[1,3,5]], axis=0),axis = 1), ed-es) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[es:ed,:,[1,3,5]], axis=0),axis = 1)
                            elif k == 'tot down':
                                label = i  + '-' + j
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[es:ed,:,[2,4,6]], axis=0),axis = 1), ed-es) * -1
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[es:ed,:,[2,4,6]], axis=0),axis = 1) * -1
                            
                            elif '-' in k:
                                bk = []
                                if k.split(' ')[-1] == 'up':
                                    for n in k.split(' ')[0].split('-'):
                                #     if 'p' in n:
                                #         bk += [3,5,7]
                                #     elif 'd' in n:
                                #         bk += [9, 11, 13, 15, 17]
                                #     else:
                                        bk.append(np.argwhere(self.__label_col == n))[0][0]
                                    if IFave_atom:
                                        dos_data[i + '-' + str(j) + '-' + k] = np.true_divide(np.sum(np.sum(self._pdos[es:ed,:,bk], axis=0), axis=1), ed-es)
                                    else:
                                        dos_data[i + '-' + str(j) + '-' + k] = np.sum(np.sum(self._pdos[es:ed,:,bk], axis=0), axis=1)
                                elif k.split(' ')[-1] == 'down':
                                    for n in k.split(' ')[0].split('-'):
                                #     if 'p' in n:
                                #         bk += [3,5,7]
                                #     elif 'd' in n:
                                #         bk += [9, 11, 13, 15, 17]
                                #     else:
                                        bk.append(np.argwhere(self.__label_col == n))[0][0]
                                    if IFave_atom:
                                        dos_data[i + '-' + str(j) + '-' + k] = np.true_divide(np.sum(np.sum(self._pdos[es:ed,:,bk], axis=0), axis=1), ed-es) * -1
                                    else:
                                        dos_data[i + '-' + str(j) + '-' + k] = np.sum(np.sum(self._pdos[es:ed,:,bk], axis=0), axis=1) * -1

                        elif '-' in j:
                            ei = [int(n)-1 for n in j.split('-')]
                            if k == 's up':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[ei,:,1], axis=0), len(ei)) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[ei,:,1], axis=0)
                            elif k == 's down':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(self._pdos[ei,:,2], axis=0), len(ei)) 
                                else:
                                    dos_data[label] = np.sum(self._pdos[ei,:,2], axis=0)
                            elif k == 'p up':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,3], axis=0), axis =1),  len(ei)) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,3], axis=0), axis =1)
                            elif k == 'p down':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,4], axis=0), axis =1),  len(ei)) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,4], axis=0), axis =1)
                            elif k == 'd up':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,5], axis=0), axis =1), len(ei)) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,5], axis=0), axis =1)
                            elif k == 'd down':
                                label = i  + '-' + j + '-' + k
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,6], axis=0), axis =1), len(ei)) 
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,6], axis=0), axis =1)
                            elif k == 'tot up':
                                label = i + '-' + j
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[1,3,5]].T, axis = 0), len(ei))
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[1,3,5,]].T, axis = 0)
                            elif k == 'tot down':
                                label = i + '-' + j
                                if IFave_atom:
                                    dos_data[label] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[2,4,6]].T, axis = 0), len(ei))
                                else:
                                    dos_data[label] = np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,[2,4,6]].T, axis = 0)
                            elif '-' in k:
                                bk = []
                                if k.split(' ')[-1] == 'up':
                                    for n in k.split(' ')[0].split('-'):
                                        # if 'p' in n:
                                        #     bk += [3,5,7]
                                        # elif 'd' in n:
                                        #     bk += [9, 11, 13, 15, 17]
                                        # else:
                                        bk.append(np.argwhere(self.__label_col == n +' up'))[0][0]
                                    if IFave_atom:                        
                                        dos_data[i + '-' + j + '-' + k] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,bk].T, axis = 0), len(ei)) 
                                    else:
                                        dos_data[i + '-' + j + '-' + k] = np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,bk].T, axis = 0)
                                elif k.split(' ')[-1] == 'down':
                                    for n in k.split(' ')[0].split('-'):
                                        # if 'p' in n:
                                        #     bk += [4,6,8]
                                        # elif 'd' in n:
                                        #     bk += [10, 12, 14, 16, 18]
                                        # else:
                                        bk.append(np.argwhere(self.__label_col == n + ' down'))[0][0]
                                    if IFave_atom:
                                        dos_data[i + '-' + j + '-' + k] = np.true_divide(np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,bk].T, axis = 0), len(ei)) * -1
                                    else:
                                        dos_data[i + '-' + j + '-' + k] = np.sum(np.sum(self._pdos[ei,:,:], axis =0)[:,bk].T, axis = 0) * -1

        self._output_dos = dos_data
        
        return self

    def output_xdos(self, dos_dict, IFave_atom):

        self.__process_dos(dos_dict, IFave_atom)
        return pd.DataFrame({'energy' : self._energy , **self._output_dos})

    def plot_xdos(self, DOS_range,eV_range, dos_dict, IFave_atom, plot_name):

        self.__process_dos( dos_dict, IFave_atom)
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch
        from matplotlib.ticker import AutoMinorLocator
        from matplotlib.patches import Polygon
        import matplotlib.colors as mcolors
        
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 7))
        ax1 = plt.subplot(111)
        
        for i in self._output_dos:
            line, = ax1.plot(self._energy, self._output_dos[i], '-', linewidth=2, label = i)
            z = np.empty((100, 1, 4), dtype=float)
            rgb = mcolors.colorConverter.to_rgb(line.get_color())
            z[:, :, :3] = rgb
            z[:, :, -1] = np.linspace(0.1, 1, 100).reshape(-1, 1)
            if self._output_dos[i].max() > 0:
                xmin, xmax, ymin, ymax = self._energy.min(), self._energy.max(), self._output_dos[i].min(), self._output_dos[i].max()
                im = ax1.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                                origin='lower',)
                xy = np.column_stack([self._energy, self._output_dos[i]])
                xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
                clip_path = Polygon(xy, lw=0.0, facecolor='none',
                        edgecolor='none', closed=True)
            elif self._output_dos[i].min() < 0:
                xmin, xmax, ymax, ymin = self._energy.min(), self._energy.max(), self._output_dos[i].max(), self._output_dos[i].min()
                im = ax1.imshow(z, aspect='auto', extent=[xmin, xmax, ymax, ymin],
                                origin='lower',)
                xy = np.column_stack([self._energy, self._output_dos[i]])
                xy = np.vstack([[xmin, ymax], xy, [xmax, ymax], [xmin, ymax]])
                clip_path = Polygon(xy, lw=0.0, facecolor='none',
                                    edgecolor='none', closed=True)
            ax.add_patch(clip_path)
            im.set_clip_path(clip_path)

        ax1.axvline(x=0, linestyle="--",color='k')
        legend_font = {'weight': 'normal', "size" : 15}
        ax1.legend(loc = 'upper right', ncol = 2, fontsize = 'large', prop = legend_font)
        ax1.set_xlim(eV_range[0], eV_range[1])
        ax1.set_ylim(DOS_range[0], DOS_range[1])
        # ax1.set_xticks([-4,-2,0,2,4,6,])
        # ax1.set_yticks([-2,0,2,4,6,])
        ax1.tick_params(axis = 'both', labelsize = 25, width = 2, length = 8, direction='in')
        ax1.set_xlabel('$E - E_{f} \ (eV)$', fontsize = 30, fontweight = 'medium')
        ax1.set_ylabel('$DOS \ (states/eV)$', fontsize = 30, fontweight = 'medium')
        ax1.spines['bottom'].set_linewidth(3)
        ax1.spines['left'].set_linewidth(3)
        ax1.spines['top'].set_linewidth(3)
        ax1.spines['right'].set_linewidth(3)
        plt.tight_layout()
        plt.savefig(plot_name, dpi = 100)


if __name__ == '__main__':


    """
    
    orbital: ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2']
    
    """

    # Type one : Total DOS
    # 1. Total dos Fig
    XDos('./example/s1-scf/').ext_tdos().plot_xdos(DOS_range = [0,1000], 
                                                   eV_range  = [-5,5], 
                                                   dos_dict = {'all':{'all':['tot']}},  #  Can not modified                                      
                                                   IFave_atom =False,
                                                   plot_name='./example/s1-scf/TDOS.jpg')
    # 2. Total dos data file
    XDos('./example/s1-scf/').ext_tdos().output_xdos(dos_dict = {'all':{'all':['tot']}},  #  Can not modified                                      
                                                     IFave_atom =False).to_csv('./example/s1-scf/TDOS.csv')

    # Type two : Partial or Local DOS
    # 1. pdos Fig
    XDos('./example/s1-scf/').ext_ordos().plot_xdos(DOS_range = [0,200], 
                                                   eV_range  = [-5,5], 
                                                   dos_dict = {'Sb':{'all':['s', 'p', 'd']},
                                                               'Te':{'all':['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2']}},  #  Can not modified                                      
                                                   IFave_atom =False,
                                                   plot_name='./example/s1-scf/PDOS.jpg')
    
    # 2. Ldos Fig
    # 2.1 Multiple atoms
    XDos('./example/s1-scf/').ext_ordos().plot_xdos(DOS_range = [0,2], 
                                                   eV_range  = [-5,5], 
                                                   dos_dict = {'Sb':{'271-281-291-301':['s', 'p', 'd']},
                                                               'Te':{'1:5':['s', 'p', 'd']}},                                    
                                                   IFave_atom =False,
                                                   plot_name='./example/s1-scf/LmDOS.jpg')

    # 2.2 Single atom
    XDos('./example/s1-scf/').ext_ordos().plot_xdos(DOS_range = [0,2], 
                                                   eV_range  = [-5,5], 
                                                   dos_dict = {'Sb':{271:['s', 'p', 'd'], 272:['s', 'p', 'd']}, 
                                                               'Te':{1:['s', 'p', 'd'], 2:['s', 'p', 'd']}},                                    
                                                   IFave_atom =False,
                                                   plot_name='./example/s1-scf/LsDOS.jpg')
    
    # 3. Ldos File
    # 3.1 Multiple atoms
    XDos('./example/s1-scf/').ext_ordos().output_xdos(dos_dict = {'Sb':{'271-281-291-301':['s', 'p', 'd']},
                                                                  'Te':{'1:5':['s', 'p', 'd']}},                                    
                                                      IFave_atom =False).to_csv('./example/s1-scf/LmDOS.csv')

    # 3.2 Single atom
    XDos('./example/s1-scf/').ext_ordos().output_xdos(dos_dict = {'Sb':{271:['s', 'p', 'd'], 272:['s', 'p', 'd']}, 
                                                                  'Te':{1:['s', 'p', 'd'], 2:['s', 'p', 'd']}},                                    
                                                      IFave_atom =False).to_csv('./example/s1-scf/LsDOS.csv')  

    # Type three : Spin
    # 1. Spin Fig 
    # 1.1 Total spin dos 
    XDos('./example/dos-AFM/').ext_tdos().plot_xdos(DOS_range = [-100, 100], 
                                                    eV_range  = [-5,10], 
                                                    dos_dict = {'all':{'all':['tot up', 'tot down']}},                                   
                                                    IFave_atom = False,
                                                    plot_name='./example/dos-AFM/TDOS.jpg')

    # 1.2  LDOS or PDOS dos
    # 1.2.1 Multiple atoms
    XDos('./example/dos-AFM/').ext_ordos().plot_xdos(DOS_range = [-100, 100], 
                                                     eV_range  = [-5,10], 
                                                     dos_dict = {'Cs':{'1:9':['s up', 's down','p up', 'p down', 'd up', 'd down']},
                                                                 'Ag':{'9-10-11-12':['s up', 's down','p up', 'p down', 'd up', 'd down']},
                                                                 'Fe':{'all':['py up', 'py down', 'pz up', 'pz down', 'px up', 'px down']},
                                                                 'Cl':{'all':['tot up', 'tot down']}},                                   
                                                     IFave_atom = False,
                                                     plot_name='./example/dos-AFM/LmDOS.jpg')

    # 1.2.2 Single atoms
    XDos('./example/dos-AFM/').ext_ordos().plot_xdos(DOS_range = [-5, 5], 
                                                     eV_range  = [-5,10], 
                                                     dos_dict = {'Cs':{1:['s up', 's down','p up', 'p down', 'd up', 'd down'], 2:['s up', 's down','p up', 'p down', 'd up', 'd down']},
                                                                 'Ag':{9:['s up', 's down','p up', 'p down', 'd up', 'd down']},
                                                                 'Fe':{13:['py up', 'py down', 'pz up', 'pz down', 'px up', 'px down']},
                                                                 'Cl':{40:['tot up', 'tot down']}},                                   
                                                     IFave_atom = False,
                                                     plot_name='./example/dos-AFM/LsDOS.jpg')

    # 1.2.2 Label atoms
    XDos('./example/dos-AFM/').ext_ordos().plot_xdos(DOS_range = [-100, 100], 
                                                     eV_range  = [-5,10], 
                                                     dos_dict = {
                                                                 'A site':{'1:9':['s up', 's down','p up', 'p down', 'd up', 'd down'],},
                                                                 'B site':{'9:17':['s up', 's down','p up', 'p down', 'd up', 'd down']},
                                                                 'X site':{'17:41':['py up', 'py down', 'pz up', 'pz down', 'px up', 'px down']},
                                                                },                                   
                                                     IFave_atom = False,
                                                     plot_name='./example/dos-AFM/LlDOS.jpg')

    # 3. Spin File
    # 3.1 Multiple atoms
    XDos('./example/dos-AFM/').ext_ordos().output_xdos(dos_dict = {'Cs':{'1:9':['s up', 's down','p up', 'p down', 'd up', 'd down']},
                                                                   'Ag':{'9-10-11-12':['s up', 's down','p up', 'p down', 'd up', 'd down']},
                                                                   'Fe':{'all':['py up', 'py down', 'pz up', 'pz down', 'px up', 'px down']},
                                                                   'Cl':{'all':['tot up', 'tot down']}},                                     
                                                       IFave_atom =False).to_csv('./example/dos-AFM/LmDOS.csv')

    # 3.2 Single atoms
    XDos('./example/dos-AFM/').ext_ordos().output_xdos(dos_dict = {'Cs':{1:['s up', 's down','p up', 'p down', 'd up', 'd down'], 2:['s up', 's down','p up', 'p down', 'd up', 'd down']},
                                                                   'Ag':{9:['s up', 's down','p up', 'p down', 'd up', 'd down']},
                                                                   'Fe':{13:['py up', 'py down', 'pz up', 'pz down', 'px up', 'px down']},
                                                                   'Cl':{40:['tot up', 'tot down']}},                                    
                                                       IFave_atom =False).to_csv('./example/dos-AFM/LsDOS.csv') 
    # 3.3 Label atoms
    XDos('./example/dos-AFM/').ext_ordos().output_xdos(dos_dict = {'A site':{'1:9':['s up', 's down','p up', 'p down', 'd up', 'd down']},
                                                                   'B site':{'9:17':['s up', 's down','p up', 'p down', 'd up', 'd down']},
                                                                   'X site':{'17:41':['py up', 'py down', 'pz up', 'pz down', 'px up', 'px down']}},                                  
                                                       IFave_atom =False).to_csv('./example/dos-AFM/LlDOS.csv') 

