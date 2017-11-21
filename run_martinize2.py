# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:48:46 2017

@author: Peter Kroon
"""
#import matplotlib.pyplot as plt
#plt.close('all')
from martinize2 import *

import os.path as osp

import numpy as np

#PATH = '../molecules/cycliclipopeptide_2.pdb'
PATH = '../molecules/cyclicpeptide_2.pdb'
#PATH = '../molecules/glkfk.pdb'
#PATH = '../molecules/6-macro-8_cartwheel.gro'
#PATH = '../molecules/6-macro-16.gro'
#PATH = '../molecules/6-macro-16-rtc-eq-nodisre.pdb'
#PATH = '../molecules/3-macro-1.gro'
#
#write_pdb(CG_graph, "6-macro-16-rtc-eq-nodisre-CG.pdb", conect=True)

system = System()

ext = osp.splitext(PATH)[-1]

if ext.casefold() == '.pdb':
    PDBInput().run_system(system, PATH)
elif ext.casefold() == '.gro':
    GROInput().run_system(system, PATH)
else:
    raise RuntimeError
MakeBonds().run_system(system)
RepairGraph().run_system(system)
DoMapping().run_system(system)
ApplyBlocks().run_system(system)
DoLinks().run_system(system)

print(system)


for mol in system.molecules:
    to_remove = set()
    for idx in mol:
        if 'position' not in mol.nodes[idx]:
            node = mol.nodes[idx]
            print(node['resname'], node['resid'], node['atomname'])
            to_remove.add(idx)
#            mol.nodes[idx]['position'] = np.array([np.nan, np.nan, np.nan])
    mol.remove_nodes_from(to_remove)
            
    draw(mol, node_size=30, node_color=tuple(np.random.rand(3)), with_label=True)

show()
