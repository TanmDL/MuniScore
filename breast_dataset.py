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

















