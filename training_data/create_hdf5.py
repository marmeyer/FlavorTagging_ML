import uproot
import argparse
import numpy as np
import h5py

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootfile', type=str, required=True, help='input root file')
    parser.add_argument('--tree', type=str, required=True, help='tree name')
    parser.add_argument('--branches', nargs="+", required=True, help='list of branches to be added to hdf5 file')
    parser.add_argument('--output', type=str, required=True, help='name of output hdf5 file')
    parser.add_argument('--target', type=int, required=True, help='target class')
    args = parser.parse_args()
    
    root_file = str(args.rootfile)
    root_tree = str(args.tree)
    root_branches=list(args.branches)
    output_file=str(args.output)
    target_class=int(args.target)
    
    file = uproot.open(root_file)
    tree = file[root_tree]
    
    hf = h5py.File(output_file, 'w')
    
    for branch in root_branches:
        hf.create_dataset(branch, data=tree.arrays()[branch])
        
    branch_target_class = np.full(len(tree.arrays()[root_branches[0]]), target_class, dtype=int)
    hf.create_dataset('target', data=branch_target_class)
    
    hf.close()