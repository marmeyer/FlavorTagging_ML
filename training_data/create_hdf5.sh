#!/bin/bash

rm input/b.root
rm input/c.root
rm input/uds.root

rm output/data_b.h5
rm output/data_c.h5
rm output/data_uds.h5

hadd input/b.root input/*_b.root
hadd input/c.root input/*_c.root
hadd input/uds.root input/*_uds.root

python create_hdf5.py --rootfile input/b.root   --tree ntp --branches trk1d0sig trk2d0sig trk1z0sig trk2z0sig trk1pt trk2pt jprobr jprobr5sigma jprobz jprobz5sigma d0bprob d0cprob d0qprob z0bprob z0cprob z0qprob nmuon nelectron trkmass 1vtxprob vtxlen1 vtxlen2 vtxlen12 vtxlen12all_jete vtxlen2_jete vtxlen12_jete vtxlen1_jete vtxsig1 vtxsig2 vtxsig12  vtxsig1_jete vtxsig2_jete vtxsig12_jete vtxdirang1 vtxdirang2 vtxdirang1_jete vtxdirang2_jete vtxdirang12_jete  vtxmult1 vtxmult2 vtxmult vtxmom1 vtxmom2 vtxmom1_jete vtxmom2_jete vtxmom_jete vtxmass1 vtxmass2 vtxmass vtxmasspc  vtxmassall vtxprob  nvtx nvtxall trk1pt_jete trk2pt_jete jprobr25sigma jprobz25sigma jprobr2 jprobz2 d0bprob2 d0cprob2 d0qprob2 z0bprob2 z0cprob2 z0qprob2 --output output/data_b.h5 --target 0
python create_hdf5.py --rootfile input/c.root   --tree ntp --branches trk1d0sig trk2d0sig trk1z0sig trk2z0sig trk1pt trk2pt jprobr jprobr5sigma jprobz jprobz5sigma d0bprob d0cprob d0qprob z0bprob z0cprob z0qprob nmuon nelectron trkmass 1vtxprob vtxlen1 vtxlen2 vtxlen12 vtxlen12all_jete vtxlen2_jete vtxlen12_jete vtxlen1_jete vtxsig1 vtxsig2 vtxsig12  vtxsig1_jete vtxsig2_jete vtxsig12_jete vtxdirang1 vtxdirang2 vtxdirang1_jete vtxdirang2_jete vtxdirang12_jete  vtxmult1 vtxmult2 vtxmult vtxmom1 vtxmom2 vtxmom1_jete vtxmom2_jete vtxmom_jete vtxmass1 vtxmass2 vtxmass vtxmasspc  vtxmassall vtxprob  nvtx nvtxall trk1pt_jete trk2pt_jete jprobr25sigma jprobz25sigma jprobr2 jprobz2 d0bprob2 d0cprob2 d0qprob2 z0bprob2 z0cprob2 z0qprob2 --output output/data_c.h5 --target 1
python create_hdf5.py --rootfile input/uds.root --tree ntp --branches trk1d0sig trk2d0sig trk1z0sig trk2z0sig trk1pt trk2pt jprobr jprobr5sigma jprobz jprobz5sigma d0bprob d0cprob d0qprob z0bprob z0cprob z0qprob nmuon nelectron trkmass 1vtxprob vtxlen1 vtxlen2 vtxlen12 vtxlen12all_jete vtxlen2_jete vtxlen12_jete vtxlen1_jete vtxsig1 vtxsig2 vtxsig12  vtxsig1_jete vtxsig2_jete vtxsig12_jete vtxdirang1 vtxdirang2 vtxdirang1_jete vtxdirang2_jete vtxdirang12_jete  vtxmult1 vtxmult2 vtxmult vtxmom1 vtxmom2 vtxmom1_jete vtxmom2_jete vtxmom_jete vtxmass1 vtxmass2 vtxmass vtxmasspc  vtxmassall vtxprob  nvtx nvtxall trk1pt_jete trk2pt_jete jprobr25sigma jprobz25sigma jprobr2 jprobz2 d0bprob2 d0cprob2 d0qprob2 z0bprob2 z0cprob2 z0qprob2 --output output/data_uds.h5 --target 2