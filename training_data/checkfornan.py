import argparse
import h5py
import torch
import numpy as np
import math

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5file', type=str, required=True, help='input hdf5 file')
    parser.add_argument('--features', nargs="+", required=True, help='list of features to be checked')
    parser.add_argument('--output', type=str, required=True, help='name of output hdf5 file')
    args = parser.parse_args()
    
    input=str(args.h5file)
    features_to_check=list(args.features)
    output_file=str(args.output)
    
    data = h5py.File(input, 'r')
    
    list_features = [] 
    for feature in features_to_check:
        list_features.append(data[feature][:])
    list_features.append(data["target"][:])
    data_all=tuple(zip(*list_features))
    res = [t for t in data_all if not any(np.isnan(t))]    
    
    if len(data_all)!= len(res):
        print('Found NaNs! ', len(data_all)-len(res), ' events removed')
    
    hf = h5py.File(output_file, 'w')
    for feature in features_to_check:
        result=np.array(res)
        hf.create_dataset(feature, data=result[:,features_to_check.index(feature)])
    hf.create_dataset("target", data=result[:,len(features_to_check)])