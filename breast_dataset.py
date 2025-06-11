####### breast datasets (CBGRA and TCGA datasets)


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import h5py

class RadPathDataset(Dataset):
    def __init__(self, 
                 root_data,
                 outcome_csv_path,
                 split='train', 
                 split_type='institution', 
                 random_seed=42, 
                 max_patches=2000):
        """
        Dataset for combined radiological and pathological data.
        Radiology features (from CSV) and WSI embeddings (from h5 file) 
        are assumed to be pre-normalized to [0, 1].
        
        Args:
            root_data: Root directory (e.g., .../pathology_HIPT) that contains a "WSI_norm" folder.
            outcome_csv_path: Path to outcome CSV file with radiological features.
            split: 'train' or 'test'.
            split_type: 'institution' or 'random'.
            random_seed: For reproducibility.
            max_patches: Maximum number of patches to sample per patient.
        """
        self.root_data = root_data
        self.outcome_csv_path = outcome_csv_path
        self.split = split
        self.split_type = split_type
        self.random_seed = random_seed
        self.max_patches = max_patches
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Load outcome data from CSV and process patient splits using available WSI files
        self.load_data()
        self.process_data()
        
    def load_data(self):
        """Load outcome and radiological feature data from CSV."""
        try:
            self.data = pd.read_csv(self.outcome_csv_path)
            self.data.columns = self.data.columns.str.strip()
            
            # Ensure a patient identifier column exists.
            if 'PatientID' not in self.data.columns:
                if 'PatientCode' in self.data.columns:
                    self.data.rename(columns={'PatientCode': 'PatientID'}, inplace=True)
                else:
                    raise ValueError("Could not find PatientID column in data")
            
            # Define radiological feature columns (assumed to start with "phase")
            self.rad_feature_cols = [col for col in self.data.columns if col.startswith('phase')]
            if len(self.rad_feature_cols) == 0:
                raise ValueError("No radiological feature columns found (columns starting with 'phase')")
                
            print(f"Found {len(self.rad_feature_cols)} radiological feature columns")
            
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")
    
    def process_data(self):
        """Process outcome data and build a mapping of patients to their WSI h5 file.
           The WSI files are assumed to be stored under:
             root_data/WSI_norm/TCGA and root_data/WSI_norm/CBCGA
        """
        self.patient_mapping = {}
        
        # List of institutions (as subdirectories under "WSI_norm")
        for inst in ["TCGA", "CBCGA"]:
            inst_path = os.path.join(self.root_data, "WSI_norm", inst)
            if os.path.exists(inst_path):
                for file in os.listdir(inst_path):
                    if file.lower().endswith('.h5'):
                        # Patient ID is the file name without extension.
                        patient_id = os.path.splitext(file)[0]
                        self.patient_mapping[patient_id] = {
                            'path': os.path.join(inst_path, file),
                            'institution': inst
                        }
                        
        print(f"Found WSI data for {len(self.patient_mapping)} patients")
        if len(self.patient_mapping) == 0:
            raise ValueError("No WSI data found in the specified root_data/WSI_norm folder.")
        
        # Initialize lists for outcome data.
        all_patient_ids = []
        all_patient_files = []
        all_times = []
        all_events = []
        all_institutions = []
        
        # Collect patients present in the CSV that also have matching WSI data.
        for idx, row in self.data.iterrows():
            patient_id = row['PatientID']
            if patient_id in self.patient_mapping:
                all_patient_ids.append(patient_id)
                all_patient_files.append(self.patient_mapping[patient_id]['path'])
                all_events.append(row.get('RFS_status', 0))
                all_times.append(row.get('RFS_time', 0.0))
                institution = 1 if self.patient_mapping[patient_id]['institution'] == 'TCGA' else 0
                all_institutions.append(institution)
        
        print(f"Found {len(all_patient_ids)} patients in CSV with matching WSI data")
        if len(all_patient_ids) == 0:
            raise ValueError("No patients in the CSV have matching WSI data.")
        
        # Split the data based on institution or randomly.
        if self.split_type == 'institution':
            cbcga_indices = [i for i, inst in enumerate(all_institutions) if inst == 0]
            tcga_indices = [i for i, inst in enumerate(all_institutions) if inst == 1]
            print(f"Institution split: CBCGA={len(cbcga_indices)}, TCGA={len(tcga_indices)}")
            if len(cbcga_indices) == 0 or len(tcga_indices) == 0:
                print("Warning: One institution has no data. Using random split instead.")
                all_indices = list(range(len(all_patient_ids)))
                train_indices, test_indices = train_test_split(
                    all_indices, test_size=0.2, random_state=self.random_seed,
                    stratify=all_events if len(set(all_events)) > 1 else None
                )
            else:
                train_indices = cbcga_indices
                test_indices = tcga_indices
        else:
            all_indices = list(range(len(all_patient_ids)))
            train_indices, test_indices = train_test_split(
                all_indices, test_size=0.2, random_state=self.random_seed,
                stratify=all_events if len(set(all_events)) > 1 else None
            )
        
        indices = train_indices if self.split == 'train' else test_indices
        
        self.patient_ids = [all_patient_ids[i] for i in indices]
        self.patient_files = [all_patient_files[i] for i in indices]
        self.times = np.array([all_times[i] for i in indices], dtype=np.float32)
        self.events = np.array([all_events[i] for i in indices], dtype=np.float32)
        self.institutions = [all_institutions[i] for i in indices]
        
        self.patient_data = {
            pid: {
                'file': pfile,
                'time': time,
                'event': event,
                'institution': inst
            }
            for pid, pfile, time, event, inst in zip(
                self.patient_ids, self.patient_files, self.times, self.events, self.institutions
            )
        }
        
        print(f"{self.split} set: Found {len(self.patient_ids)} patients")
        print(f"  - TCGA patients: {sum(1 for i in self.institutions if i == 1)}")
        print(f"  - CBCGA patients: {sum(1 for i in self.institutions if i == 0)}")

    def load_pathology_embedding(self, file_path):
        """
        Load and sample pathology embeddings for a patient from a single HDF5 file.
        Assumes the file contains a dataset named 'features' with shape (n_patches, feature_dim).
        Returns:
            A tensor of fixed size (max_patches, feature_dim) and a corresponding mask.
        """
        try:
            with h5py.File(file_path, 'r') as hf:
                if 'features' not in hf:
                    raise ValueError(f"'features' dataset not found in {file_path}")
                data = hf['features'][:]  # load as numpy array
            
            if len(data.shape) != 2:
                raise ValueError(f"Unexpected shape in {file_path}: {data.shape}")
            
            combined_tensor = torch.from_numpy(data).float()
            num_patches = combined_tensor.shape[0]
            feature_dim = combined_tensor.shape[1]
            
            fixed_tensor = torch.zeros((self.max_patches, feature_dim))
            mask = torch.ones(self.max_patches)
            
            if num_patches >= self.max_patches:
                indices = torch.randperm(num_patches)[:self.max_patches]
                fixed_tensor = combined_tensor[indices]
                mask.fill_(0)
            else:
                fixed_tensor[:num_patches] = combined_tensor
                mask[:num_patches] = 0
            
            return fixed_tensor, mask
            
        except Exception as e:
            print(f"Error loading pathology embedding from {file_path}: {e}")
            dummy_embedding = torch.zeros((self.max_patches, 194))
            dummy_mask = torch.ones(self.max_patches)
            return dummy_embedding, dummy_mask
            
    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        """
        Retrieve a single patient's data.
        Returns a dictionary with:
            - 'x_rad': radiological features (from CSV; pre-normalized).
            - 'x_wsi': pathology features (from the HDF5 file).
            - 'y': discretized label (placeholder).
            - 'time': survival time.
            - 'event': survival event status.
            - 'patient_id': patient identifier.
            - 'mask': mask for pathology features.
            - 'institution': institution code (0 for CBCGA, 1 for TCGA).
        """
        try:
            patient_id = self.patient_ids[index]
            patient_info = self.patient_data[patient_id]
            file_path = patient_info['file']
            
            patient_row = self.data[self.data['PatientID'] == patient_id]
            if patient_row.empty:
                raise ValueError(f"Patient {patient_id} not found in CSV data")
            patient_row = patient_row.iloc[0]
            
            rad_features = patient_row[self.rad_feature_cols].values.astype(np.float32)
            pathology, mask = self.load_pathology_embedding(file_path)
            
            time = patient_info['time']
            event = patient_info['event']
            disc_label = 0  # Placeholder
            
            return {
                'x_rad': rad_features,
                'x_wsi': pathology,
                'y': disc_label,
                'time': time,
                'event': event,
                'patient_id': patient_id,
                'mask': mask,
                'institution': patient_info['institution']
            }
            
        except Exception as e:
            print(f"Error getting item at index {index}, patient {patient_id if 'patient_id' in locals() else 'unknown'}: {e}")
            feature_dim = len(self.rad_feature_cols)
            dummy_rad = np.zeros(feature_dim, dtype=np.float32)
            dummy_wsi = torch.zeros((1, 194))
            dummy_mask = torch.zeros(1)
            
            return {
                'x_rad': dummy_rad,
                'x_wsi': dummy_wsi,
                'y': 0,
                'time': 0.0,
                'event': 0,
                'patient_id': "ERROR",
                'mask': dummy_mask,
                'institution': 0
            }
        
def survival_collate_fn(batch):
    """
    Custom collate function for survival analysis batches.
    Returns a dictionary with collated tensor data.
    """
    rad_features = torch.tensor(np.stack([item['x_rad'] for item in batch]))
    wsi_features = torch.stack([item['x_wsi'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    disc_labels = torch.tensor([item['y'] for item in batch]).unsqueeze(1)
    times = torch.tensor([item['time'] for item in batch]).unsqueeze(1)
    events = torch.tensor([item['event'] for item in batch]).unsqueeze(1)
    patient_ids = [item['patient_id'] for item in batch]
    institutions = torch.tensor([item['institution'] for item in batch])
    
    return {
        'x_rad': rad_features,
        'x_wsi': wsi_features,
        'y': disc_labels,
        'time': times,
        'event': events,
        'patient_id': patient_ids,
        'mask': masks,
        'institution': institutions
    }

############# Example usage (for testing purposes)
if __name__ == "__main__":
    root_data = "/home/tanmoy/Documents/Bolin/Radpath/SMURF/SMuRF_MultiModal_OPSCC/Instutional/data_breast/forTanmoy/pathology_HIPT/"
    outcome_csv = "/home/tanmoy/Documents/Bolin/Radpath/SMURF/SMuRF_MultiModal_OPSCC/Instutional/data_breast/forTanmoy/rad_global_scaled_final.csv"
    print("Creating datasets...")
    train_dataset = RadPathDataset(
        root_data=root_data,
        outcome_csv_path=outcome_csv,
        split='train',
        split_type= 'institution',    ###'random',
        max_patches=2000
    )
    test_dataset = RadPathDataset(
        root_data=root_data,
        outcome_csv_path=outcome_csv,
        split='test',
        split_type= 'institution', ########'random',
        max_patches=2000
    )
    batch_size = 20
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=survival_collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=survival_collate_fn,
        num_workers=0
    )
    for i, batch in enumerate(test_loader):
        print(f"Batch {i+1}:")
        print(f"  Radiological features shape: {batch['x_rad'].shape}")
        print(f"  WSI features shape: {batch['x_wsi'].shape}")
        print(f"  Mask shape: {batch['mask'].shape}")
        print(f"  Sample patient IDs: {batch['patient_id']}")
        print(f"  Sample patient IDs: {batch['time']}")
        print(f"  Sample patient IDs: {batch['event']}")
        break





###############################################################################################################################################
#####################################################################################################################
#################################################################################################################################


# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import pickle
# import random
# from sklearn.preprocessing import StandardScaler    

# class RadPathDataset(Dataset):
#     def __init__(self, 
#                  root_data,
#                  outcome_csv_path,
#                  split='train', 
#                  split_type='institution', 
#                  random_seed=42, 
#                  max_patches=2000):
#         """
#         Dataset for combined radiological and pathological data.
        
#         Args:
#             root_data: Root directory of pathology data
#             outcome_csv_path: Path to outcome CSV file with radiological features
#             split: 'train' or 'test'
#             split_type: 'institution' or 'random'
#             random_seed: For reproducibility
#             max_patches: Maximum number of patches to sample per patient
#         """
#         self.root_data = root_data
#         self.outcome_csv_path = outcome_csv_path
#         self.split = split
#         self.split_type = split_type
#         self.random_seed = random_seed
#         self.max_patches = max_patches
        
#         # Set random seeds for reproducibility
#         np.random.seed(random_seed)
#         random.seed(random_seed)
#         torch.manual_seed(random_seed)
        
#         # Load data
#         self.load_data()
#         self.process_data()
        
#     def load_data(self):
#         """Load outcome and radiological feature data"""
#         try:
#             # Load outcome and radiological features from the same file
#             self.data = pd.read_csv(self.outcome_csv_path)

#             self.data.columns = self.data.columns.str.strip()

#             ###print("Loaded columns names in the datasets", self.data.columns)
            
#             # Check if PatientID column exists
#             if 'PatientID' not in self.data.columns:
#                 # Try alternate column names
#                 if 'PatientCode' in self.data.columns:
#                     self.data.rename(columns={'PatientCode': 'PatientID'}, inplace=True)
#                 else:
#                     raise ValueError("Could not find PatientID column in data")
            
#             # Define radiological feature columns - assuming they start with "phase"
#             self.rad_feature_cols = [col for col in self.data.columns if col.startswith('phase')]
            
#             if len(self.rad_feature_cols) == 0:
#                 raise ValueError("No radiological feature columns found (columns starting with 'phase')")
                
#             print(f"Found {len(self.rad_feature_cols)} radiological feature columns")
            
#         except Exception as e:
#             raise RuntimeError(f"Error loading data: {str(e)}")

#     def process_data(self):
#         """Process and split data"""
#         # Map folder names for both TCGA and CBCGA
#         self.patient_mapping = {}
        
#         # Process TCGA folders
#         tcga_path = os.path.join(self.root_data, "TCGA", "embeddings")
#         if os.path.exists(tcga_path):
#             for folder_name in os.listdir(tcga_path):
#                 folder_path = os.path.join(tcga_path, folder_name)
#                 if os.path.isdir(folder_path) and any(f.endswith('.pkl') for f in os.listdir(folder_path)):
#                     self.patient_mapping[folder_name] = {
#                         'path': folder_path,
#                         'institution': 'TCGA'
#                     }
        
#         # Process CBCGA folders
#         cbcga_path = os.path.join(self.root_data, "CBCGA", "embeddings")
#         if os.path.exists(cbcga_path):
#             for folder_name in os.listdir(cbcga_path):
#                 folder_path = os.path.join(cbcga_path, folder_name)
#                 if os.path.isdir(folder_path) and any(f.endswith('.pkl') for f in os.listdir(folder_path)):
#                     self.patient_mapping[folder_name] = {
#                         'path': folder_path,
#                         'institution': 'CBCGA'
#                     }
                    
#         print(f"Found WSI data for {len(self.patient_mapping)} patients")
        
#         # Initialize lists
#         all_patient_ids = []
#         all_patient_dirs = []
#         all_times = []
#         all_events = []
#         all_institutions = []
        
#         # Process each patient in the CSV
#         for idx, row in self.data.iterrows():
#             patient_id = row['PatientID']
            
#             # Check if this patient has WSI data
#             if patient_id in self.patient_mapping:
#                 all_patient_ids.append(patient_id)
#                 all_patient_dirs.append(self.patient_mapping[patient_id]['path'])
                
#                 # Get survival event status
#                 if 'RFS_status' in row:
#                     all_events.append(row['RFS_status'])
#                 else:
#                     all_events.append(0)  # Default if missing
                    
#                 # Get survival time
#                 if 'RFS_time' in row:
#                     all_times.append(row['RFS_time'])
#                 else:
#                     all_times.append(0.0)  # Default if missing
                
#                 # Mark institution (0 for CBCGA, 1 for TCGA)
#                 institution = 1 if self.patient_mapping[patient_id]['institution'] == 'TCGA' else 0
#                 all_institutions.append(institution)
        
#         print(f"Found {len(all_patient_ids)} patients in CSV with matching WSI data")

#         ######print(f"Found {len(all_times)} patients in CSV with matching events data", all_times)
        
#         # Perform split based on institution or random
#         if self.split_type == 'institution':
#             cbcga_indices = [i for i, inst in enumerate(all_institutions) if inst == 0]
#             tcga_indices = [i for i, inst in enumerate(all_institutions) if inst == 1]
            
#             print(f"Institution split: CBCGA={len(cbcga_indices)}, TCGA={len(tcga_indices)}")
#             ####all_indices = list(range(len(all_patient_ids)))

#             ########print("find the total number indices in the two datasets", len(all_indices))

#             ########### this is the instituation 
            
#             if len(cbcga_indices) == 0 or len(tcga_indices) == 0:
#                 print("Warning: One institution has no data. Using random split instead.")
#                 # Fall back to random split
#                 all_indices = list(range(len(all_patient_ids)))
#                 ####### understanding the total indices available in the dataset    
                
#                 train_indices, test_indices = train_test_split(
#                     all_indices, test_size=0.2, random_state=self.random_seed,
#                     stratify=all_events if len(set(all_events)) > 1 else None
#                 )
#             else:
#                 # Use CBCGA for train, TCGA for test
#                 train_indices = cbcga_indices  # CBCGA
#                 test_indices = tcga_indices    # TCGA
#         else:
#             # Random split
#             all_indices = list(range(len(all_patient_ids)))
#             ####print("find the total number indices in the two datasets", len(all_indices))
#             train_indices, test_indices = train_test_split(
#                 all_indices, test_size=0.2, random_state=self.random_seed,
#                 stratify=all_events if len(set(all_events)) > 1 else None
#             )
        
#         # Select indices based on split
#         indices = train_indices if self.split == 'train' else test_indices
        
#         # Filter data
#         self.patient_ids = [all_patient_ids[i] for i in indices]
#         self.patient_dirs = [all_patient_dirs[i] for i in indices]
#         self.times = np.array([all_times[i] for i in indices], dtype=np.float32)
#         self.events = np.array([all_events[i] for i in indices], dtype=np.float32)
#         self.institutions = [all_institutions[i] for i in indices]
        
#         # Create a mapping for faster lookup
#         self.patient_data = {
#             pid: {
#                 'dir': pdir,
#                 'time': time,
#                 'event': event,
#                 'institution': inst
#             }
#             for pid, pdir, time, event, inst in zip(
#                 self.patient_ids, self.patient_dirs, self.times, self.events, self.institutions
#             )
#         }
        
#         print(f"{self.split} set: Found {len(self.patient_ids)} patients")
#         print(f"  - TCGA patients: {sum(1 for i in self.institutions if i == 1)}")
#         print(f"  - CBCGA patients: {sum(1 for i in self.institutions if i == 0)}")

#     def load_pathology_embedding(self, path):
#         """
#         Load and sample pathology embeddings for a patient
        
#         Args:
#             path: Directory containing .pkl files for the patient
            
#         Returns:
#             Combined embeddings tensor and mask
#         """
#         try:
#             # Get all pickle files
#             pkl_files = [f for f in os.listdir(path) if f.endswith('.pkl')]
#             if not pkl_files:
#                 raise ValueError(f"No pkl files found in {path}")
            
#             # Store combined features from all files
#             all_features = []
#             total_patches = 0
#             patient_id = os.path.basename(path)
            
#             # Process each pickle file
#             for pkl_file in pkl_files:
#                 file_path = os.path.join(path, pkl_file)
#                 try:
#                     with open(file_path, 'rb') as f:
#                         embedding = pickle.load(f)
                        
#                         # Convert to numpy array if needed
#                         if not isinstance(embedding, np.ndarray):
#                             embedding = np.array(embedding)
                        
#                         # Ensure it's the right shape (n_patches, 194)
#                         if embedding.shape[1] == 194:
#                             # Correct shape, add to our collection
#                             all_features.append(embedding)
#                             total_patches += embedding.shape[0]
# #                         
#                         else:
#                             print(f"Warning: Unexpected shape in {file_path}: {embedding.shape}")
#                             continue
                            
#                 except Exception as e:
#                     print(f"Warning: Error loading {file_path}: {e}")
#                     continue
            
#             if len(all_features) == 0:
#                 raise ValueError(f"No valid embeddings loaded from {path}")
            
#             # Concatenate all features from all files
#             if len(all_features) > 1:
#                 combined_features = np.concatenate(all_features, axis=0)
#             else:
#                 combined_features = all_features[0]
            
#             # Convert to tensor
#             combined_tensor = torch.from_numpy(combined_features).float()
            
                     
#             # Number of available patches
#             num_patches = combined_tensor.shape[0]
            
#             # Create fixed-size tensor - we'll fill this with sampled/padded data
#             fixed_tensor = torch.zeros((self.max_patches, 194))
            
#             # Create mask (0 for real data, 1 for padding)
#             mask = torch.ones(self.max_patches)
            
#             if num_patches >= self.max_patches:
#                 # If we have more patches than max_patches, randomly sample
#                 indices = torch.randperm(num_patches)[:self.max_patches]
#                 fixed_tensor = combined_tensor[indices]
#                 # All patches are real (no padding)
#                 mask.fill_(0)
#             else:
#                 # If we have fewer patches than max_patches, copy and pad with zeros
#                 fixed_tensor[:num_patches] = combined_tensor
#                 # Mark real patches in mask (first num_patches are real, rest are padding)
#                 mask[:num_patches] = 0
            
#             ##print(f"Patient {patient_id}: Combined {len(all_features)} files, {total_patches} patches -> fixed size {self.max_patches}")
#             return fixed_tensor, mask
            
#         except Exception as e:
#             print(f"Error loading pathology embedding from {path}: {e}")
#             # Return a dummy tensor as fallback (zero tensor of correct size)
#             dummy_embedding = torch.zeros((self.max_patches, 194))
#             dummy_mask = torch.ones(self.max_patches)  # All masked as padding
#             return dummy_embedding, dummy_mask
            
            
#     def __len__(self):
#         return len(self.patient_ids)

#     def __getitem__(self, index):
#         """
#         Get a single patient's data
        
#         Returns:
#             Dictionary with all modalities and labels
#         """
#         try:
#             patient_id = self.patient_ids[index]
#             patient_info = self.patient_data[patient_id]
#             patient_dir = patient_info['dir']
            
#             # Get patient row from data
#             patient_row = self.data[self.data['PatientID'] == patient_id]
#             if len(patient_row) == 0:
#                 raise ValueError(f"Patient {patient_id} not found in CSV data")
#             patient_row = patient_row.iloc[0]
            
#             # Extract radiological features
#             rad_features = patient_row[self.rad_feature_cols].values.astype(np.float32)
            
#             # Load pathology embeddings
#             pathology, mask = self.load_pathology_embedding(patient_dir)
            
#             # Get survival information
#             time = patient_info['time']
#             event = patient_info['event']
            
#             # Create discretized label (placeholder - would be replaced with proper binning)
#             # For survival data, this would typically be based on quantiles of survival time
#             disc_label = 0   #### keep this one for better undderstanding of the classification score
            
#             return {
#                 'x_rad': rad_features,
#                 'x_wsi': pathology,
#                 'y': disc_label,
#                 'time': time,
#                 'event': event,
#                 'patient_id': patient_id,
#                 'mask': mask,
#                 'institution': patient_info['institution']
#             }
            
#         except Exception as e:
#             print(f"Error getting item at index {index}, patient {patient_id if 'patient_id' in locals() else 'unknown'}: {e}")
#             # Return dummy data as fallback
#             feature_dim = len(self.rad_feature_cols)
#             dummy_rad = np.zeros(feature_dim, dtype=np.float32)
#             dummy_wsi = torch.zeros((1, 194))
#             dummy_mask = torch.zeros(1)
            
#             return {
#                 'x_rad': dummy_rad,
#                 'x_wsi': dummy_wsi,
#                 'y': 0,
#                 'time': 0.0,
#                 'event': 0,
#                 'patient_id': "ERROR",
#                 'mask': dummy_mask,
#                 'institution': 0
#             }
        

        
# ##### survival data collate_fn of new approach

# def survival_collate_fn(batch):
#     """
#     Custom collate function for survival analysis batches
    
#     Args:
#         batch: List of dictionaries from __getitem__
        
#     Returns:
#         Collated batch with consistent tensor dimensions
#     """
#     # Extract all components
#     rad_features = torch.tensor(np.stack([item['x_rad'] for item in batch]))
    
#     # Stack WSI features - all should be the same size already
#     wsi_features = torch.stack([item['x_wsi'] for item in batch])
    
#     # Stack masks
#     masks = torch.stack([item['mask'] for item in batch])
    
#     # Collect other data
#     disc_labels = torch.tensor([item['y'] for item in batch]).unsqueeze(1)
#     times = torch.tensor([item['time'] for item in batch]).unsqueeze(1)
#     events = torch.tensor([item['event'] for item in batch]).unsqueeze(1)
#     patient_ids = [item['patient_id'] for item in batch]
#     institutions = torch.tensor([item['institution'] for item in batch])
    
#     # Structure for PIBD model compatibility
#     return {
#         'x_rad': rad_features,
#         'x_wsi': wsi_features,
#         'y': disc_labels,
#         'time': times,
#         'event': events,
#         'patient_id': patient_ids,
#         'mask': masks,
#         'institution': institutions
#     }


# if __name__ == "__main__":
#     # Example usage
#     root_data = "/home/tanmoy/Documents/Bolin/Radpath/SMURF/SMuRF_MultiModal_OPSCC/data_breast/forTanmoy/pathology_HIPT"
#     outcome_csv = "/home/tanmoy/Documents/Bolin/Radpath/SMURF/SMuRF_MultiModal_OPSCC/data_breast/forTanmoy/merged_outcome_final.csv"
    
#     print("Creating datasets...")
    
#     # Create train dataset
#     train_dataset = RadPathDataset(
#         root_data=root_data,
#         outcome_csv_path=outcome_csv,
#         split='train',
#         split_type='random',  # 'institution' or 'random'
#         max_patches=2000
#     )
    
#     # Create test dataset
#     test_dataset = RadPathDataset(
#         root_data=root_data,
#         outcome_csv_path=outcome_csv,
#         split='test',
#         split_type='random',  # 'institution' or 'random'
#         max_patches=2000
#     )
    
#     # Create dataloaders
#     batch_size = 4  # Small batch size for testing
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=survival_collate_fn,
#         num_workers=0  # Set to 0 for easier debugging
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=survival_collate_fn,
#         num_workers=0  # Set to 0 for easier debugging
#     )

#     for i, batch in enumerate(test_loader):
#         print(f"Batch {i+1}:")
#         print(f"  Radiological features shape: {batch['x_rad'].shape}")
#         print(f"  WSI features shape: {batch['x_wsi'].shape}")
#         print(f"  Mask shape: {batch['mask'].shape}")
#         print(f"  Sample patient IDs: {batch['patient_id'][:3]}")
#         print(f"  Sample patient IDs: {batch['time']}")
#         print(f"  Sample patient IDs: {batch['event']}")
#         break
        



















































##################################################################################################################################################
###################################################################################################
####################################################################################

# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import glob
# import os
# from math import ceil, floor
# from medpy.io import load, header
# from models import Model
# import utils
# import pandas as pd
# import matplotlib.pyplot as plt

# def custom_collate(data):
#     rad_feat,  pathology, y, time, event, ID = zip(*data)
#     max_sizes = (max([path.shape[0] for path in pathology]), max([path.shape[1] for path in pathology]))
#     pathology = list(pathology)
#     ID = list(ID)  # Convert ID back to a list if needed
#     for i in range(len(pathology)):
#         pathology[i] = torch.moveaxis(pathology[i], -1,0)
#         pad_2d = max_sizes[1] - pathology[i].shape[2]
#         pad_3d = max_sizes[0] - pathology[i].shape[1]
#         padding = (floor(pad_2d/2), ceil(pad_2d/2), floor(pad_2d/2), ceil(pad_2d/2), floor(pad_3d/2), ceil(pad_3d/2))
#         m = torch.nn.ConstantPad3d(padding, 0)
#         pathology[i] = m(pathology[i])
#         pathology[i] = torch.permute(pathology[i], (1,0,2,3)).float()
#     return torch.stack(rad_feat), ID



# class RadPathDataset(Dataset):
#     def __init__(
#         self, df, root_data, index=None, dim=[128, 128, 3], ring=15
#     ):   #### dim=[48, 48, 3]
#         self.df = df
#         if index is not None:
#             df = df.iloc[index]
#         self.transforms = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.RandomHorizontalFlip(0.5),
#             transforms.RandomVerticalFlip(0.5)])
#         self.y = np.array(df["grade"]).astype(np.float32)
#         self.time = np.array(df["DFS"]).astype(np.float32)
#         self.event = np.array(df["DFS_censor"]).astype(np.float32)
#         self.ID = np.array(df["radiology_folder_name"])

#         self.dim = dim
#         self.ring = ring
#         self.root_data = root_data

#         # Define feature columns
#         self.feature_columns = [
#             'original_firstorder_Minimum',
#             'original_glrlm_RunVariance',
#             'wavelet_LLH_firstorder_Minimum',
#             'wavelet_LHL_glcm_MCC',
#             'wavelet_LHH_firstorder_Energy',
#             'wavelet_LHH_firstorder_Range',
#             'wavelet_LHH_firstorder_TotalEnergy',
#             'wavelet_LHH_glcm_Imc2',
#             'wavelet_HHL_glcm_MCC',
#             'wavelet_HHH_firstorder_InterquartileRange',
#             'wavelet_HHH_firstorder_Maximum'
#         ]

        

#     def __len__(self):
#         return len(self.y)

#     def get_radiology(self, index):

        
        

#         # Get radiological features
#         rad_feat1_data = self.rad_feat1[self.rad_feat1['PatientID'] == index][self.feature_columns].iloc[0].values
#         rad_feat2_data = self.rad_feat2[self.rad_feat2['PatientID'] == index][self.feature_columns].iloc[0].values
#         rad_feat3_data = self.rad_feat3[self.rad_feat3['PatientID'] == index][self.feature_columns].iloc[0].values
            
#             # Combine features
#         rad_features = np.concatenate([
#                 rad_feat1_data.astype(np.float32),
#                 rad_feat2_data.astype(np.float32),
#                 rad_feat3_data.astype(np.float32)
#             ])

        
#         return rad_features

#     def __getitem__(self, index):
#         # print(index)
#         # print(self.df["radiology_folder_name"][index])
#         ct_feat = self.get_radiology(index)
            
#         pathology_file = os.path.join(self.root_data, "pathology", self.df["PatientCode"][index], "embeddings.npy")
#         pathology = np.load(pathology_file)
#         pathology = torch.from_numpy(pathology)
        

#         return ct_feat, pathology, self.y[index], self.time[index], self.event[index], self.ID[index] 



#########################################################################################################################################################################################################################################################

# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import os
# from math import ceil, floor
# import pandas as pd
# import pickle as pkl

# def custom_collate(data):
#     """Custom collate function for the dataloader"""
#     rad_feat, pathology, time, event, ID = zip(*data)
    
#     # Convert radiological features to tensor
#     rad_feat = torch.tensor(rad_feat).float()
    
#     # Handle pathology data
#     max_sizes = (
#         max([path.shape[0] for path in pathology]),
#         max([path.shape[1] for path in pathology])
#     )
    
#     pathology = list(pathology)
#     ID = list(ID)
    
#     for i in range(len(pathology)):
#         pathology[i] = torch.moveaxis(pathology[i], -1, 0)
#         pad_2d = max_sizes[1] - pathology[i].shape[2]
#         pad_3d = max_sizes[0] - pathology[i].shape[1]
#         padding = (
#             floor(pad_2d/2), ceil(pad_2d/2),
#             floor(pad_2d/2), ceil(pad_2d/2),
#             floor(pad_3d/2), ceil(pad_3d/2)
#         )
#         m = torch.nn.ConstantPad3d(padding, 0)
#         pathology[i] = m(pathology[i])
#         pathology[i] = torch.permute(pathology[i], (1,0,2,3)).float()
    
#     pathology = torch.stack(pathology)
    
#     return (
#         rad_feat,
#         pathology,
#         torch.tensor(time).float(),
#         torch.tensor(event).float(),
#         ID
#     )

# class RadPathDataset(Dataset):
#     def __init__(self, df, root_data, index=None, dim=[128, 128, 3], ring=15):
#         """
#         Args:
#             df: Pandas DataFrame containing merged data
#             root_data: Root directory for pathology data
#             index: Optional indices for subset selection
#             dim: Dimensions for pathology data
#             ring: Ring parameter
#         """
#         self.df = df
#         if index is not None:
#             self.df = self.df.iloc[index]
            
#         self.root_data = root_data
#         self.dim = dim
#         self.ring = ring
        
#         # Get labels and time data
#         ###self.y = np.array(self.df["OS_status"]).astype(np.float32)
#         self.time = np.array(self.df["RFS time (month)"]).astype(np.float32)
#         self.event = np.array(self.df["RFS status"]).astype(np.float32)
#         self.ID = np.array(self.df["PatientID"])
        
#         # Get feature column names for each phase
#         self.feature_base_names = [
#             'original_firstorder_Minimum',
#             'original_glrlm_RunVariance',
#             'wavelet_LLH_firstorder_Minimum',
#             'wavelet_LHL_glcm_MCC',
#             'wavelet_LHH_firstorder_Energy',
#             'wavelet_LHH_firstorder_Range',
#             'wavelet_LHH_firstorder_TotalEnergy',
#             'wavelet_LHH_glcm_Imc2',
#             'wavelet_HHL_glcm_MCC',
#             'wavelet_HHH_firstorder_InterquartileRange',
#             'wavelet_HHH_firstorder_Maximum'
#         ]
        
#         # Create phase-specific feature column names
#         self.phase0_cols = [f'phase0_{col}' for col in self.feature_base_names]
#         self.phase1_cols = [f'phase1_{col}' for col in self.feature_base_names]
#         self.phase2_cols = [f'phase2_{col}' for col in self.feature_base_names]
        
#     def __len__(self):
#         return len(self.df)
    
#     def get_radiology(self, index):
#         """Get radiological features for a patient"""
#         # Get features from each phase
#         phase0_features = self.df.iloc[index][self.phase0_cols].values
#         phase1_features = self.df.iloc[index][self.phase1_cols].values
#         phase2_features = self.df.iloc[index][self.phase2_cols].values
        
#         # Combine features
#         rad_features = np.concatenate([
#             phase0_features.astype(np.float32),
#             phase1_features.astype(np.float32),
#             phase2_features.astype(np.float32)
#         ])
        
#         return rad_features
    
#     def __getitem__(self, index):
#         """Get item from dataset"""
#         try:
#             # Get radiological features
#             ct_feat = self.get_radiology(index)
            
#             # Get pathology data
#             patient_id = self.df.iloc[index]['PatientID']
            
#             # Determine pathology path based on institution
#             if '-' in patient_id:  # TCGA
#                 pathology_path = os.path.join(self.root_data, "TCGA", "embeddings", patient_id)
#             else:  # CBCGA
#                 pathology_path = os.path.join(self.root_data, "CBCGA", "embeddings", patient_id)
            
#             # Get all pkl files for this patient
#             pkl_files = [f for f in os.listdir(pathology_path) if f.endswith('.pkl')]
#             if not pkl_files:
#                 raise ValueError(f"No pkl files found for patient {patient_id}")
            
#             # Load and combine all pkl files
#             embeddings = []
#             for pkl_file in pkl_files:
#                 file_path = os.path.join(pathology_path, pkl_file)
#                 with open(file_path, 'rb') as f:
#                     embedding = pkl.load(f)
#                     if not isinstance(embedding, torch.Tensor):
#                         embedding = torch.tensor(embedding)
#                     embeddings.append(embedding.float())
            
#             # Combine embeddings if multiple files exist
#             if len(embeddings) > 1:
#                 pathology = torch.cat(embeddings, dim=0)
#             else:
#                 pathology = embeddings[0]
            
#             return (
#                 ct_feat,
#                 pathology,
#                 self.time[index],
#                 self.event[index],
#                 self.ID[index]
#             )
            
#         except Exception as e:
#             print(f"Error loading data for index {index}, patient {self.ID[index]}: {str(e)}")
#             raise

# # Example usage:
# if __name__ == "__main__":
#     # Load merged data
#     merged_df = pd.read_csv("/home/tanmoy/Documents/Project_code/Bolin/SMURF/SMuRF_MultiModal_OPSCC/data_breast/forTanmoy/merged_outcome_final.csv")
    
#     # Create dataset
#     dataset = RadPathDataset(
#         df=merged_df,
#         root_data="/home/tanmoy/Documents/Project_code/Bolin/SMURF/SMuRF_MultiModal_OPSCC/data_breast/forTanmoy/pathology_HIPT/",
#         dim=[128, 128, 3]
#     )
    
#     # Create dataloader
#     dataloader = DataLoader(
#         dataset,
#         batch_size=32,
#         shuffle=True,
#         collate_fn=custom_collate,
#         num_workers=4
#     )
    
#     # Test loading a batch
#     for batch_idx, (rad_feats, pathology,  time, event, ids) in enumerate(dataloader):
#         print(f"\nBatch {batch_idx + 1}:")
#         print(f"Radiological features shape: {rad_feats.shape}")
#         print(f"Pathology features shape: {pathology.shape}")
        
#         print(f"Time shape: {time.shape}")
#         print(f"Event shape: {event.shape}")
#         print(f"Number of IDs: {len(ids)}")
#         break  # Just test first batch


# ######################################################################################################################################################################################################################################################################################################
# ########## uniform 2D structure here 

# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# import os
# import pickle as pkl
# import pandas as pd
# from typing import List, Tuple, Optional

# def custom_collate(batch) -> Tuple:
#     """
#     Custom collate function that handles variable-sized pathology embeddings.
#     Pads sequences to the maximum length in the batch.
#     """
#     rad_feat, path_feat, time, event, ID = zip(*batch)
    
#     # Convert radiological features to tensor
#     rad_feat = torch.stack([torch.tensor(x) for x in rad_feat]).float()
    
#     # Get max sequence length in this batch
#     max_seq_len = max(x.shape[0] for x in path_feat)
    
#     # Pad pathology features
#     padded_path = []
#     attention_masks = []
    
#     for feat in path_feat:
#         # Create attention mask
#         mask = torch.zeros(max_seq_len)
#         mask[:feat.shape[0]] = 1
#         attention_masks.append(mask)
        
#         # Pad if necessary
#         if feat.shape[0] < max_seq_len:
#             padding = torch.zeros((max_seq_len - feat.shape[0], feat.shape[1]))
#             feat = torch.cat([feat, padding], dim=0)
#         padded_path.append(feat)
    
#     # Stack into tensors
#     path_feat = torch.stack(padded_path).float()
#     attention_masks = torch.stack(attention_masks).bool()
    
#     return (
#         rad_feat,
#         path_feat,
#         attention_masks,
#         torch.tensor(time).float(),
#         torch.tensor(event).float(),
#         ID
#     )

# class RadPathDataset(Dataset):
#     def __init__(
#         self, 
#         df: pd.DataFrame,
#         root_data: str,
#         index: Optional[List[int]] = None,
#         max_seq_length: Optional[int] = None
#     ):
#         """
#         Enhanced RadPath dataset that handles variable-sized pathology embeddings.
        
#         Args:
#             df: Pandas DataFrame containing merged data
#             root_data: Root directory for pathology data
#             index: Optional indices for subset selection
#             max_seq_length: Optional maximum sequence length for pathology features
#         """
#         self.df = df.iloc[index] if index is not None else df
#         self.root_data = root_data
#         self.max_seq_length = max_seq_length
        
#         # Get clinical data
#         self.time = np.array(self.df["RFS time (month)"]).astype(np.float32)
#         self.event = np.array(self.df["RFS status"]).astype(np.float32)
#         self.ID = np.array(self.df["PatientID"])
        
#         # Radiological feature names
#         self.feature_base_names = [
#             'original_firstorder_Minimum',
#             'original_glrlm_RunVariance',
#             'wavelet_LLH_firstorder_Minimum',
#             'wavelet_LHL_glcm_MCC',
#             'wavelet_LHH_firstorder_Energy',
#             'wavelet_LHH_firstorder_Range',
#             'wavelet_LHH_firstorder_TotalEnergy',
#             'wavelet_LHH_glcm_Imc2',
#             'wavelet_HHL_glcm_MCC',
#             'wavelet_HHH_firstorder_InterquartileRange',
#             'wavelet_HHH_firstorder_Maximum'
#         ]
        
#         # Create phase-specific feature column names
#         self.phase0_cols = [f'phase0_{col}' for col in self.feature_base_names]
#         self.phase1_cols = [f'phase1_{col}' for col in self.feature_base_names]
#         self.phase2_cols = [f'phase2_{col}' for col in self.feature_base_names]
        
#         # Pre-compute pathology paths for efficiency
#         self.pathology_paths = self._get_pathology_paths()
    
#     def _get_pathology_paths(self) -> dict:
#         """Pre-compute pathology file paths for all patients"""
#         paths = {}
#         for patient_id in self.ID:
#             # Determine institution
#             institution = "TCGA" if '-' in patient_id else "CBCGA"
#             base_path = os.path.join(self.root_data, institution, "embeddings", patient_id)
            
#             if os.path.exists(base_path):
#                 pkl_files = [f for f in os.listdir(base_path) if f.endswith('.pkl')]
#                 paths[patient_id] = [os.path.join(base_path, f) for f in pkl_files]
            
#         return paths
    
#     def get_radiology(self, index: int) -> np.ndarray:
#         """Get radiological features for a patient"""
#         # Get features from each phase
#         phase0_features = self.df.iloc[index][self.phase0_cols].values
#         phase1_features = self.df.iloc[index][self.phase1_cols].values
#         phase2_features = self.df.iloc[index][self.phase2_cols].values
        
#         # Combine features
#         rad_features = np.concatenate([
#             phase0_features.astype(np.float32),
#             phase1_features.astype(np.float32),
#             phase2_features.astype(np.float32)
#         ])
        
#         return rad_features
    
#     def get_pathology(self, patient_id: str) -> torch.Tensor:
#         """
#         Load and combine pathology embeddings for a patient.
#         Returns padded sequence if max_seq_length is set.
#         """
#         embeddings = []
        
#         # Load all pkl files for this patient
#         for file_path in self.pathology_paths.get(patient_id, []):
#             with open(file_path, 'rb') as f:
#                 embedding = pkl.load(f)
#                 if not isinstance(embedding, torch.Tensor):
#                     embedding = torch.tensor(embedding)
#                 embeddings.append(embedding.float())
        
#         if not embeddings:
#             raise ValueError(f"No embeddings found for patient {patient_id}")
        
#         # Combine all embeddings
#         combined = torch.cat(embeddings, dim=0)
        
#         # Apply sequence length limit if specified
#         if self.max_seq_length is not None and combined.shape[0] > self.max_seq_length:
#             combined = combined[:self.max_seq_length]
            
#         return combined
    
#     def __len__(self) -> int:
#         return len(self.df)
    
#     def __getitem__(self, index: int) -> Tuple:
#         """Get item from dataset"""
#         try:
#             # Get radiological features
#             ct_feat = self.get_radiology(index)
            
#             # Get pathology features
#             patient_id = self.ID[index]
#             path_feat = self.get_pathology(patient_id)
            
#             return (
#                 ct_feat,
#                 path_feat,
#                 self.time[index],
#                 self.event[index],
#                 patient_id
#             )
            
#         except Exception as e:
#             print(f"Error loading data for index {index}, patient {self.ID[index]}: {str(e)}")
#             raise

# # Example usage:
# if __name__ == "__main__":
#     # Load merged data
#     merged_df = pd.read_csv("path/to/merged_outcome_final.csv")
    
#     # Create dataset with maximum sequence length of 512
#     dataset = RadPathDataset(
#         df=merged_df,
#         root_data="path/to/pathology_data",
#         max_seq_length=512  # Optional: limit sequence length
#     )
    
#     # Create dataloader
#     dataloader = DataLoader(
#         dataset,
#         batch_size=32,
#         shuffle=True,
#         collate_fn=custom_collate,
#         num_workers=4
#     )
    
#     # Test loading a batch
#     for batch_idx, (rad_feats, path_feats, attention_masks, time, event, ids) in enumerate(dataloader):
#         print(f"\nBatch {batch_idx + 1}:")
#         print(f"Radiological features shape: {rad_feats.shape}")
#         print(f"Pathology features shape: {path_feats.shape}")
#         print(f"Attention masks shape: {attention_masks.shape}")
#         print(f"Time shape: {time.shape}")
#         print(f"Event shape: {event.shape}")
#         print(f"Number of IDs: {len(ids)}")
#         break