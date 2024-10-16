import os
import torch
from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.train import *
def load_tensor_batch(batch, feature_type, feature_directory):
    batchsize = len(batch[0])
    list_of_lists = [[] for _ in range(batchsize)]
    
    # Iterate through each batch of filenames and values
    for item in range(len(batch)):
        filenames, values = batch[item]
        
        # Iterate through the filenames and load tensors
        for file_index, file_name in enumerate(filenames):
            file_path = os.path.join(feature_directory, file_name)
            
            # Load the tensor (either heavy or light) and move to CUDA if available
            feature = torch.load(file_path)[feature_type][0]
            feature = feature.cuda() if torch.cuda.is_available() else feature
            
            # Append the feature to the corresponding index list
            list_of_lists[file_index].append(feature)
    
    # Stack the tensors for each file across the batch dimension
    tensor_list = [torch.stack(listitem) for listitem in list_of_lists]
    return torch.stack(tensor_list)
def load_data_batch(batch, data_type, feature_directory):
    batchsize = len(batch[0])
    list_of_lists = [[] for _ in range(batchsize)]
    
    # Iterate through each batch of filenames and values
    for item in range(len(batch)):
        filenames, values = batch[item]
        
        # Iterate through the filenames and load tensors
        for file_index, file_name in enumerate(filenames):
            file_path = os.path.join(feature_directory, file_name)
            
            # Load the tensor (either heavy or light) and move to CUDA if available
            data = torch.load(file_path)[data_type]
            data = recursive_to(data,'cuda')
            
            # Append the feature to the corresponding index list
            list_of_lists[file_index].append(data)


    return list_of_lists
def retrieval_feature_with_id(topnheavy_batch, topnlight_batch, feature_directory):
    # Process heavy batch
    heavy_tensor = load_tensor_batch(topnheavy_batch, 'heavy_feature', feature_directory)
    
    # Process light batch
    light_tensor = load_tensor_batch(topnlight_batch, 'light_feature', feature_directory)
    
    return heavy_tensor, light_tensor
def retrieval_data_with_id(topnheavy_batch, topnlight_batch, feature_directory):
    # Process heavy batch
    heavy_data = load_data_batch(topnheavy_batch, 'heavy_data', feature_directory)
    
    # Process light batch
    light_data = load_data_batch(topnlight_batch, 'light_data', feature_directory)
    
    return heavy_data, light_data