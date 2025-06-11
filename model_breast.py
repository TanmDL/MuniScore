
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from lifelines.utils import concordance_index


class WSIEncoder(nn.Module):
    """
    Encodes WSI patch features into a fixed-size representation
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, dropout=0.1):
        super(WSIEncoder, self).__init__()
        
        # Feature transformation for patches
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Multi-head attention for patch interaction
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm and MLP
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Aggregation token (learnable)
        self.agg_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.agg_token, std=0.02)
        
    def forward(self, x, mask=None):
        """
        Forward pass through WSI encoder
        
        Args:
            x: Input tensor of shape [batch, patches, input_dim]
            mask: Mask tensor of shape [batch, patches], where 1 indicates padding
            
        Returns:
            Tensor of shape [batch, output_dim]
        """
        batch_size, num_patches, _ = x.shape
        
        # Apply feature transformation
        x = self.feature_transform(x)  # [batch, patches, hidden_dim]
        
        # Expand aggregation token for each sample in batch
        agg_tokens = self.agg_token.expand(batch_size, -1, -1)  # [batch, 1, hidden_dim]
        
        # Concatenate aggregation token with patch features
        x_with_agg = torch.cat([agg_tokens, x], dim=1)  # [batch, 1+patches, hidden_dim]
        
        # Create attention mask if needed
        key_padding_mask = None
        if mask is not None:
            # Create mask for aggregation token (not masked) and patches
            agg_mask = torch.zeros(batch_size, 1, device=mask.device)
            extended_mask = torch.cat([agg_mask, mask], dim=1)  # [batch, 1+patches]
            
            # Convert to key_padding_mask format for MultiheadAttention
            # (batch_size, seq_len) where True values are masked
            key_padding_mask = extended_mask.bool()
        
        # Apply self-attention
        x_norm = self.layer_norm1(x_with_agg)
        x_attn, _ = self.self_attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # Add residual connection
        x_with_agg = x_with_agg + x_attn
        
        # Apply MLP with layer norm and residual
        x_norm = self.layer_norm2(x_with_agg)
        x_mlp = self.mlp(x_norm)
        x_with_agg = x_with_agg + x_mlp
        
        # Extract the aggregation token as the representation of the WSI
        wsi_representation = x_with_agg[:, 0]  # [batch, hidden_dim]
        
        # Project to output dimension
        wsi_output = self.output_proj(wsi_representation)  # [batch, output_dim]
        
        return wsi_output


class SurvivalTaskModel(nn.Module):
    """
    Task-specific model for survival prediction
    """
    def __init__(self, input_dim, hidden_units, dropout=0.25):
        super(SurvivalTaskModel, self).__init__()
        
        # Hidden layers
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_units:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer for hazard prediction
        self.hazard_layer = nn.Linear(hidden_units[-1], 1)
        
    def forward(self, x):
        """
        Forward pass through survival task model
        
        Args:
            x: Input tensor of shape [batch, input_dim]
            
        Returns:
            Hazard scores tensor of shape [batch, 1]
        """
        features = self.hidden_layers(x)
        hazard = self.hazard_layer(features)
        return hazard


class RadPathAttention(nn.Module):
    """
    Self-attention mechanism for fusing modalities
    """
    def __init__(self, rad_dim, wsi_dim, hidden_dim):
        super(RadPathAttention, self).__init__()
        self.rad_transform = nn.Linear(rad_dim, hidden_dim)
        self.wsi_transform = nn.Linear(wsi_dim, hidden_dim)
        self.q_transform = nn.Linear(hidden_dim, hidden_dim)
        self.k_transform = nn.Linear(hidden_dim, hidden_dim)
        self.v_transform = nn.Linear(hidden_dim, hidden_dim)
        self.scaling = hidden_dim ** -0.5
        self.out_transform = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, rad_features, wsi_features):
        """
        Forward pass through attention mechanism
        
        Args:
            rad_features: Radiological features tensor of shape [batch, rad_dim]
            wsi_features: WSI features tensor of shape [batch, wsi_dim]
            
        Returns:
            Tuple of attended features (rad_attended, wsi_attended)
        """
        # Transform features to common space
        rad_hidden = self.rad_transform(rad_features)
        wsi_hidden = self.wsi_transform(wsi_features)
        
        # Stack features from both modalities
        combined = torch.stack([rad_hidden, wsi_hidden], dim=1)
        
        # Compute QKV
        q = self.q_transform(combined)
        k = self.k_transform(combined)
        v = self.v_transform(combined)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attn_weights, v)
        attended = self.out_transform(attended)
        
        # Return attended features for both modalities
        return attended[:, 0], attended[:, 1]  # rad_attended, wsi_attended


class CoxLoss(nn.Module):
    """
    Cox proportional hazards loss for survival analysis
    """
    def forward(self, hazards, times, events):
        """
        Compute Cox proportional hazards loss
        
        Args:
            hazards: Hazard scores tensor of shape [batch, 1]
            times: Survival times tensor of shape [batch]
            events: Event indicators tensor of shape [batch]
            
        Returns:
            Cox partial likelihood loss
        """
        # Get batch size
        current_batch_len = len(times)
        
        # Create risk matrix (R_mat): R_mat[i,j] = 1 if times[j] >= times[i]
        # This identifies which patients were at risk at each event time
        R_mat = times.reshape((1, current_batch_len)) >= times.reshape((current_batch_len, 1))
        
        # Reshape hazards to a flat vector
        theta = hazards.reshape(-1)
        
        # Compute exponential of hazards
        exp_theta = torch.exp(theta)
        
        # Calculate negative log partial likelihood
        # For each patient i with an event (events[i] = 1):
        # log(exp(theta_i) / sum(exp(theta_j) for all j at risk at time i))
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * events)
        
        return loss_cox


class RadPathFusionModule(pl.LightningModule):
    """
    PyTorch Lightning module for RadPath fusion model with global c-index and CSV export
    """
    
    def __init__(self, 
                 rad_input_dim: int,
                 wsi_input_dim: int = 194,
                 hidden_dim: int = 256,
                 fusion_dim: int = 128,
                 fusion_type: str = "attention",
                 task_hidden_units: List[int] = [64, 32],
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 use_test_as_val: bool = False,
                 output_dir: Optional[str] = None):
        super(RadPathFusionModule, self).__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Initialize network components
        self._init_network(rad_input_dim, wsi_input_dim, hidden_dim, fusion_dim, 
                          fusion_type, task_hidden_units, dropout)
        
        # Loss function
        self.cox_loss = CoxLoss()
        
        # Directory for saving prediction results
        self.output_dir = output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # For accumulating predictions
        self.all_val_hazards = []
        self.all_val_times = []
        self.all_val_events = []
        self.all_val_patient_ids = []
        
        self.all_test_hazards = []
        self.all_test_times = []
        self.all_test_events = []
        self.all_test_patient_ids = []
        
        self.all_train_hazards = []
        self.all_train_times = []
        self.all_train_events = []
        self.all_train_patient_ids = []
        
        # Configuration
        self.fusion_type = fusion_type
        self.use_test_as_val = use_test_as_val
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Track best validation c-index
        self.best_val_c_index = 0.0
        
    def _init_network(self, rad_input_dim, wsi_input_dim, hidden_dim, fusion_dim, 
                     fusion_type, task_hidden_units, dropout):
        """
        Initialize network components
        """
        # Radiological feature encoder
        self.rad_encoder = nn.Sequential(
            nn.Linear(rad_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim)
        )
        
        # WSI feature encoder
        self.wsi_encoder = WSIEncoder(
            input_dim=wsi_input_dim,
            hidden_dim=hidden_dim,
            output_dim=fusion_dim,
            dropout=dropout
        )
        
        # Fusion mechanism
        self.fusion_type = fusion_type
        
        if fusion_type == "attention":
            self.attention = RadPathAttention(
                rad_dim=fusion_dim,
                wsi_dim=fusion_dim,
                hidden_dim=fusion_dim
            )
            self.task_input_dim = fusion_dim * 2
            
        elif fusion_type == "kronecker":
            self.task_input_dim = (fusion_dim + 1) * (fusion_dim + 1)
            
        elif fusion_type == "concatenation":
            self.task_input_dim = fusion_dim * 2
            
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        
        # Task-specific model
        self.task_model = SurvivalTaskModel(
            input_dim=self.task_input_dim,
            hidden_units=task_hidden_units,
            dropout=dropout
        )
    
    def forward(self, batch):
        """
        Forward pass through the model
        
        Args:
            batch: Dictionary containing input data
            
        Returns:
            Hazard scores tensor
        """
        # Extract features
        rad_features = batch['x_rad']
        wsi_features = batch['x_wsi']
        mask = batch['mask']
        
        # Encode features
        rad_encoded = self.rad_encoder(rad_features)
        wsi_encoded = self.wsi_encoder(wsi_features, mask)
        
        # Apply fusion strategy
        if self.fusion_type == "attention":
            rad_attended, wsi_attended = self.attention(rad_encoded, wsi_encoded)
            fused_features = torch.cat([rad_attended, wsi_attended], dim=1)
            
        elif self.fusion_type == "kronecker":
            # Add bias term
            rad_bias = torch.cat([rad_encoded, torch.ones(rad_encoded.size(0), 1, device=rad_encoded.device)], dim=1)
            wsi_bias = torch.cat([wsi_encoded, torch.ones(wsi_encoded.size(0), 1, device=wsi_encoded.device)], dim=1)
            
            # Compute Kronecker product
            fused_features = torch.bmm(
                rad_bias.unsqueeze(2), 
                wsi_bias.unsqueeze(1)
            ).flatten(start_dim=1)
            
        elif self.fusion_type == "concatenation":
            # Simple concatenation
            fused_features = torch.cat([rad_encoded, wsi_encoded], dim=1)
        
        # Apply task-specific model
        hazard = self.task_model(fused_features)
        
        return hazard
    
    def _step(self, batch, batch_type):
        """
        Generic step function for training, validation, and testing
        
        Args:
            batch: Dictionary containing input data
            batch_type: String indicating step type ('train', 'val', or 'test')
            
        Returns:
            Dictionary containing loss and predictions
        """
        # Forward pass
        hazards = self(batch)
        times = batch['time']
        events = batch['event']
        
        # Get patient IDs if available
        patient_ids = batch.get('patient_id', [])
        
        # Compute loss
        loss = self.cox_loss(hazards, times, events)
        
        # Store data for global metrics calculation
        if batch_type == "val":
            self.all_val_hazards.append(hazards.detach())
            self.all_val_times.append(times.detach())
            self.all_val_events.append(events.detach())
            if len(patient_ids) > 0:
                self.all_val_patient_ids.extend(patient_ids)
                
        elif batch_type == "test":
            self.all_test_hazards.append(hazards.detach())
            self.all_test_times.append(times.detach())
            self.all_test_events.append(events.detach())
            if len(patient_ids) > 0:
                self.all_test_patient_ids.extend(patient_ids)
                
        elif batch_type == "train":
            self.all_train_hazards.append(hazards.detach())
            self.all_train_times.append(times.detach())
            self.all_train_events.append(events.detach())
            if len(patient_ids) > 0:
                self.all_train_patient_ids.extend(patient_ids)
        
        # Log batch-level metrics
        self.log(f'{batch_type}_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {
            'loss': loss,
            'hazards': hazards.detach(),
            'times': times.detach(),
            'events': events.detach()
        }
    
    def training_step(self, batch, batch_idx):
        """
        Training step
        
        Args:
            batch: Dictionary containing input data
            batch_idx: Index of the current batch
            
        Returns:
            Loss value
        """
        outputs = self._step(batch, "train")
        return outputs['loss']
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step
        
        Args:
            batch: Dictionary containing input data
            batch_idx: Index of the current batch
            
        Returns:
            Loss value
        """
        outputs = self._step(batch, "val")
        return outputs['loss']
    
    def test_step(self, batch, batch_idx):
        """
        Test step
        
        Args:
            batch: Dictionary containing input data
            batch_idx: Index of the current batch
            
        Returns:
            Loss value
        """
        outputs = self._step(batch, "test")
        
        # If using test as validation, also track as validation data
        if self.use_test_as_val:
            self.all_val_hazards.append(outputs['hazards'])
            self.all_val_times.append(outputs['times'])
            self.all_val_events.append(outputs['events'])
            if 'patient_id' in batch:
                self.all_val_patient_ids.extend(batch['patient_id'])
        
        return outputs['loss']
    
    def _compute_global_c_index(self, hazards, times, events):
        """
        Compute c-index on the entire dataset at once
        
        Args:
            hazards: List of hazard scores tensors
            times: List of survival times tensors
            events: List of event indicators tensors
            
        Returns:
            Concordance index (c-index)
        """
        if not hazards:
            return 0.0
            
        # Concatenate all batches
        hazards_all = torch.cat(hazards, dim=0).cpu().numpy().flatten()
        times_all = torch.cat(times, dim=0).cpu().numpy().flatten()
        events_all = torch.cat(events, dim=0).cpu().numpy().flatten()
        
        # Compute c-index (negative hazards = better prognosis)
        c_index = concordance_index(times_all, -hazards_all, events_all)
        
        return c_index
    
    def on_validation_epoch_end(self):
        """
        Process the entire validation set at once
        """
        if len(self.all_val_hazards) > 0:
            # Compute global c-index
            val_c_index = self._compute_global_c_index(self.all_val_hazards, self.all_val_times, self.all_val_events)
            
            # Log the global c-index
            self.log('val_c_index', val_c_index, prog_bar=True)
            
            # Check if this is the best model so far
            if val_c_index > self.best_val_c_index:
                self.best_val_c_index = val_c_index
                
                # Export predictions for best model
                if self.output_dir is not None and self.global_rank == 0:  # Only save on main process
                    self._save_predictions_to_csv("validation")
            
            # Clear accumulated data
            self.all_val_hazards = []
            self.all_val_times = []
            self.all_val_events = []
            self.all_val_patient_ids = []
    
    def on_test_epoch_end(self):
        """
        Process the entire test set at once
        """
        if len(self.all_test_hazards) > 0:
            # Compute global c-index
            test_c_index = self._compute_global_c_index(self.all_test_hazards, self.all_test_times, self.all_test_events)
            
            # Log the global c-index
            self.log('test_c_index', test_c_index)
            
            # Export predictions
            if self.output_dir is not None and self.global_rank == 0:  # Only save on main process
                self._save_predictions_to_csv("test")
            
            # Clear accumulated data
            self.all_test_hazards = []
            self.all_test_times = []
            self.all_test_events = []
            self.all_test_patient_ids = []
    
    def on_train_epoch_end(self):
        """
        Process the entire training set at once
        """
        if len(self.all_train_hazards) > 0:
            # Compute global c-index
            train_c_index = self._compute_global_c_index(self.all_train_hazards, self.all_train_times, self.all_train_events)
            
            # Log the global c-index
            self.log('train_c_index', train_c_index, prog_bar=True)
            
            # Export predictions for the final epoch
            if self.current_epoch == self.trainer.max_epochs - 1 and self.output_dir is not None and self.global_rank == 0:
                self._save_predictions_to_csv("train")
            
            # Clear accumulated data
            self.all_train_hazards = []
            self.all_train_times = []
            self.all_train_events = []
            self.all_train_patient_ids = []
    
    def _save_predictions_to_csv(self, dataset_type):
        """
        Save predictions to CSV file
        
        Args:
            dataset_type: String indicating the dataset type ('train', 'validation', or 'test')
        """
        import pandas as pd
        
        # Determine which data to use based on dataset type
        if dataset_type == "train":
            hazards = self.all_train_hazards
            times = self.all_train_times
            events = self.all_train_events
            patient_ids = self.all_train_patient_ids
        elif dataset_type == "validation":
            hazards = self.all_val_hazards
            times = self.all_val_times
            events = self.all_val_events
            patient_ids = self.all_val_patient_ids
        elif dataset_type == "test":
            hazards = self.all_test_hazards
            times = self.all_test_times
            events = self.all_test_events
            patient_ids = self.all_test_patient_ids
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Concatenate all batches
        hazards_all = torch.cat(hazards, dim=0).cpu().numpy().flatten()
        times_all = torch.cat(times, dim=0).cpu().numpy().flatten()
        events_all = torch.cat(events, dim=0).cpu().numpy().flatten()
        
        # Create DataFrame
        data = {
            'hazard_score': hazards_all,
            'time': times_all,
            'event': events_all
        }
        
        # Add patient IDs if available
        if len(patient_ids) > 0:
            data['patient_id'] = patient_ids
            
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fusion_type = self.fusion_type
        
        # For validation from best model, add indication
        filename_suffix = ""
        if dataset_type == "validation" and abs(self._compute_global_c_index(hazards, times, events) - self.best_val_c_index) < 1e-5:
            filename_suffix = "_best_model"
        
        filename = f"{self.output_dir}/{dataset_type}_{fusion_type}_predictions{filename_suffix}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        print(f"Saved {dataset_type} predictions to {filename}")
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers
        
        Returns:
            Optimizer or dictionary with optimizer and scheduler
        """
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Add scheduler when we have validation data (either explicit or using test)
        has_val_data = (
            hasattr(self.trainer, 'datamodule') and 
            getattr(self.trainer.datamodule, 'val_dataloader', None) is not None
        ) or self.use_test_as_val
        
        if has_val_data:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            # If not using validation, return just the optimizer
            return optimizer
        
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        """
        This method is called by trainer.predict().
        Return hazard, time, event, patient_id in a dictionary
        so your main script can collect them easily.
        """
        # Forward pass for hazards
        hazards = self(batch)  # [batch_size, 1]

        # Gather times, events, optional IDs
        times  = batch["time"]         # [batch_size]
        events = batch["event"]        # [batch_size]
        pids   = batch.get("patient_id", [""] * len(times))

        return {
            "hazard": hazards,     # shape [B, 1]
            "time": times,         # shape [B]
            "event": events,       # shape [B]
            "patient_id": pids     # shape [B]
        }
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model-specific arguments to the parser
        
        Args:
            parent_parser: Parent argument parser
            
        Returns:
            Updated parser with model-specific arguments
        """
        parser = parent_parser.add_argument_group("RadPathFusionModule")
        parser.add_argument('--hidden_dim', type=int, default=256)
        parser.add_argument('--fusion_dim', type=int, default=128)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--fusion_type', type=str, default='attention', 
                        choices=['attention', 'kronecker', 'concatenation'])
        return parent_parser
##############################################################################################
#####################################################################################
########################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pytorch_lightning as pl
# from typing import List
# #####from sklearn.metrics import concordance_index
# from lifelines.utils import concordance_index  # Corrected import here

# class WSIEncoder(nn.Module):
#     """
#     Encodes WSI patch features into a fixed-size representation
#     """
#     def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, dropout=0.1):
#         super(WSIEncoder, self).__init__()
        
#         # Feature transformation for patches
#         self.feature_transform = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout)
#         )
        
#         # Multi-head attention for patch interaction
#         self.self_attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True
#         )
        
#         # Layer norm and MLP
#         self.layer_norm1 = nn.LayerNorm(hidden_dim)
#         self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim * 2, hidden_dim)
#         )
        
#         # Output projection
#         self.output_proj = nn.Linear(hidden_dim, output_dim)
        
#         # Aggregation token (learnable)
#         self.agg_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
#         nn.init.normal_(self.agg_token, std=0.02)
        
#     def forward(self, x, mask=None):
#         # x: [batch, patches, input_dim]
#         # mask: [batch, patches] - 0 for real data, 1 for padding
        
#         batch_size, num_patches, _ = x.shape
        
#         # Apply feature transformation
#         x = self.feature_transform(x)  # [batch, patches, hidden_dim]
        
#         # Expand aggregation token for each sample in batch
#         agg_tokens = self.agg_token.expand(batch_size, -1, -1)  # [batch, 1, hidden_dim]
        
#         # Concatenate aggregation token with patch features
#         x_with_agg = torch.cat([agg_tokens, x], dim=1)  # [batch, 1+patches, hidden_dim]
        
#         # Create attention mask if needed
#         key_padding_mask = None
#         if mask is not None:
#             # Create mask for aggregation token (not masked) and patches
#             agg_mask = torch.zeros(batch_size, 1, device=mask.device)
#             extended_mask = torch.cat([agg_mask, mask], dim=1)  # [batch, 1+patches]
            
#             # Convert to key_padding_mask format for MultiheadAttention
#             # (batch_size, seq_len) where True values are masked
#             key_padding_mask = extended_mask.bool()
        
#         # Apply self-attention
#         x_norm = self.layer_norm1(x_with_agg)
#         x_attn, _ = self.self_attention(
#             query=x_norm,
#             key=x_norm,
#             value=x_norm,
#             key_padding_mask=key_padding_mask,
#             need_weights=False
#         )
        
#         # Add residual connection
#         x_with_agg = x_with_agg + x_attn
        
#         # Apply MLP with layer norm and residual
#         x_norm = self.layer_norm2(x_with_agg)
#         x_mlp = self.mlp(x_norm)
#         x_with_agg = x_with_agg + x_mlp
        
#         # Extract the aggregation token as the representation of the WSI
#         wsi_representation = x_with_agg[:, 0]  # [batch, hidden_dim]
        
#         # Project to output dimension
#         wsi_output = self.output_proj(wsi_representation)  # [batch, output_dim]
        
#         return wsi_output


# class SurvivalTaskModel(nn.Module):
#     """
#     Task-specific model for survival prediction
#     """
#     def __init__(self, input_dim, hidden_units, dropout=0.25):
#         super(SurvivalTaskModel, self).__init__()
        
#         # Hidden layers
#         layers = []
#         current_dim = input_dim
        
#         for hidden_dim in hidden_units:
#             layers.extend([
#                 nn.Linear(current_dim, hidden_dim),
#                 nn.LeakyReLU(0.1, inplace=True),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.Dropout(dropout)
#             ])
#             current_dim = hidden_dim
        
#         self.hidden_layers = nn.Sequential(*layers)
        
#         # Output layer for hazard prediction
#         self.hazard_layer = nn.Linear(hidden_units[-1], 1)
        
#     def forward(self, x):
#         features = self.hidden_layers(x)
#         hazard = self.hazard_layer(features)
#         return hazard


# class RadPathAttention(nn.Module):
#     """Self-attention mechanism for fusing modalities"""
#     def __init__(self, rad_dim, wsi_dim, hidden_dim):
#         super(RadPathAttention, self).__init__()
#         self.rad_transform = nn.Linear(rad_dim, hidden_dim)
#         self.wsi_transform = nn.Linear(wsi_dim, hidden_dim)
#         self.q_transform = nn.Linear(hidden_dim, hidden_dim)
#         self.k_transform = nn.Linear(hidden_dim, hidden_dim)
#         self.v_transform = nn.Linear(hidden_dim, hidden_dim)
#         self.scaling = hidden_dim ** -0.5
#         self.out_transform = nn.Linear(hidden_dim, hidden_dim)
        
#     def forward(self, rad_features, wsi_features):
#         # Transform features to common space
#         rad_hidden = self.rad_transform(rad_features)
#         wsi_hidden = self.wsi_transform(wsi_features)
        
#         # Stack features from both modalities
#         # rad_hidden: [batch, hidden_dim]
#         # wsi_hidden: [batch, hidden_dim]
#         # combined: [batch, 2, hidden_dim]
#         combined = torch.stack([rad_hidden, wsi_hidden], dim=1)
        
#         # Compute QKV
#         q = self.q_transform(combined)
#         k = self.k_transform(combined)
#         v = self.v_transform(combined)
        
#         # Compute attention scores
#         scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
#         attn_weights = F.softmax(scores, dim=-1)
        
#         # Apply attention
#         attended = torch.matmul(attn_weights, v)
#         attended = self.out_transform(attended)
        
#         # Return attended features for both modalities
#         return attended[:, 0], attended[:, 1]  # rad_attended, wsi_attended


# class CoxLoss(nn.Module):
#     """Cox proportional hazards loss for survival analysis using matrix formulation"""
#     def forward(self, hazards, times, events):
#         # Get batch size
#         current_batch_len = len(times)
        
#         # Create risk matrix (R_mat): R_mat[i,j] = 1 if times[j] >= times[i]
#         # This identifies which patients were at risk at each event time
#         R_mat = times.reshape((1, current_batch_len)) >= times.reshape((current_batch_len, 1))
        
#         # Reshape hazards to a flat vector
#         theta = hazards.reshape(-1)
        
#         # Compute exponential of hazards
#         exp_theta = torch.exp(theta)
        
#         # Calculate negative log partial likelihood
#         # For each patient i with an event (events[i] = 1):
#         # log(exp(theta_i) / sum(exp(theta_j) for all j at risk at time i))
#         loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * events)
        
#         return loss_cox


# class RadPathFusionModule(pl.LightningModule):
#     """PyTorch Lightning module for RadPath fusion model (without validation set)"""
    
#     def __init__(self, 
#                  rad_input_dim: int,
#                  wsi_input_dim: int = 194,
#                  hidden_dim: int = 256,
#                  fusion_dim: int = 128,
#                  fusion_type: str = "attention",
#                  task_hidden_units: List[int] = [64, 32],
#                  dropout: float = 0.2,
#                  learning_rate: float = 0.001,
#                  use_test_as_val: bool = True):  # New parameter to use test set for validation
#         super(RadPathFusionModule, self).__init__()
        
#         # Save hyperparameters
#         self.save_hyperparameters()
        
#         # Radiological feature encoder
#         self.rad_encoder = nn.Sequential(
#             nn.Linear(rad_input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, fusion_dim)
#         )
        
#         # WSI feature encoder
#         self.wsi_encoder = WSIEncoder(
#             input_dim=wsi_input_dim,
#             hidden_dim=hidden_dim,
#             output_dim=fusion_dim,
#             dropout=dropout
#         )
        
#         # Fusion mechanism
#         self.fusion_type = fusion_type
        
#         if fusion_type == "attention":
#             self.attention = RadPathAttention(
#                 rad_dim=fusion_dim,
#                 wsi_dim=fusion_dim,
#                 hidden_dim=fusion_dim
#             )
#             self.task_input_dim = fusion_dim * 2  # Concatenate attended features
            
#         elif fusion_type == "kronecker":
#             # Add +1 for the bias term in kronecker product
#             self.task_input_dim = (fusion_dim + 1) * (fusion_dim + 1)
            
#         elif fusion_type == "concatenation":
#             self.task_input_dim = fusion_dim * 2
            
#         else:
#             raise ValueError(f"Unsupported fusion type: {fusion_type}")
        
#         # Task-specific model
#         self.task_model = SurvivalTaskModel(
#             input_dim=self.task_input_dim,
#             hidden_units=task_hidden_units,
#             dropout=dropout
#         )
        
#         # Loss function
#         self.cox_loss = CoxLoss()
        
#         # For tracking metrics
#         self.training_step_outputs = []
#         self.test_step_outputs = []
        
#         # Use test set as validation if specified
#         self.use_test_as_val = use_test_as_val
        
#     def forward(self, batch):
#         # Extract features
#         rad_features = batch['x_rad']  # [batch, rad_dim]
#         wsi_features = batch['x_wsi']  # [batch, patches, wsi_dim]
#         mask = batch['mask']  # [batch, patches]
        
#         # Encode features
#         rad_encoded = self.rad_encoder(rad_features)  # [batch, fusion_dim]
#         wsi_encoded = self.wsi_encoder(wsi_features, mask)  # [batch, fusion_dim]
        
#         # Apply fusion strategy
#         if self.fusion_type == "attention":
#             rad_attended, wsi_attended = self.attention(rad_encoded, wsi_encoded)
#             fused_features = torch.cat([rad_attended, wsi_attended], dim=1)
            
#         elif self.fusion_type == "kronecker":
#             # Add bias term
#             rad_bias = torch.cat([rad_encoded, torch.ones(rad_encoded.size(0), 1, device=rad_encoded.device)], dim=1)
#             wsi_bias = torch.cat([wsi_encoded, torch.ones(wsi_encoded.size(0), 1, device=wsi_encoded.device)], dim=1)
            
#             # Compute Kronecker product
#             fused_features = torch.bmm(
#                 rad_bias.unsqueeze(2),  # [batch, fusion_dim+1, 1]
#                 wsi_bias.unsqueeze(1)   # [batch, 1, fusion_dim+1]
#             ).flatten(start_dim=1)  # [batch, (fusion_dim+1)*(fusion_dim+1)]
            
#         elif self.fusion_type == "concatenation":
#             # Simple concatenation
#             fused_features = torch.cat([rad_encoded, wsi_encoded], dim=1)
        
#         # Apply task-specific model
#         hazard = self.task_model(fused_features)
        
#         return hazard
    
#     def _compute_metrics(self, hazards, times, events):
#         # Convert to numpy for c-index calculation, detaching from computation graph first
#         hazards_np = hazards.detach().cpu().numpy().flatten()
#         times_np = times.detach().cpu().numpy().flatten()
#         events_np = events.detach().cpu().numpy().flatten()
        
#         # Compute c-index (using negative hazards as prediction, higher means better prognosis)
#         c_index = concordance_index(times_np, -hazards_np, events_np)
        
#         return {
#             'c_index': torch.tensor(c_index, device=hazards.device)
#         }
    
#     def training_step(self, batch, batch_idx):
#         hazards = self(batch)
#         times = batch['time']
#         events = batch['event']
        
#         # Compute loss
#         loss = self.cox_loss(hazards, times, events)
        
#         # Compute metrics
#         metrics = self._compute_metrics(hazards, times, events)
#         c_index = metrics['c_index']
        
#         # Log metrics
#         self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
#         self.log('train_c_index', c_index, on_step=False, on_epoch=True, prog_bar=True)
        
#         # Store outputs for epoch end
#         self.training_step_outputs.append({
#             'loss': loss,
#             'hazards': hazards.detach(),
#             'times': times.detach(),
#             'events': events.detach()
#         })
        
#         return loss
    
#     def test_step(self, batch, batch_idx):
#         hazards = self(batch)
#         times = batch['time']
#         events = batch['event']
        
#         # Compute loss
#         loss = self.cox_loss(hazards, times, events)
        
#         # Compute metrics
#         metrics = self._compute_metrics(hazards, times, events)
#         c_index = metrics['c_index']
        
#         # Log metrics
#         self.log('test_loss', loss, on_step=False, on_epoch=True)
#         self.log('test_c_index', c_index, on_step=False, on_epoch=True)
        
#         # If using test as validation, also log with val_ prefix for scheduler
#         if self.use_test_as_val:
#             self.log('val_loss', loss, on_step=False, on_epoch=True)
#             self.log('val_c_index', c_index, on_step=False, on_epoch=True)
        
#         # Store outputs for epoch end
#         self.test_step_outputs.append({
#             'loss': loss,
#             'hazards': hazards.detach(),
#             'times': times.detach(),
#             'events': events.detach()
#         })
        
#         return loss
    
#     # Define validation_step to be the same as test_step when needed
#     def validation_step(self, batch, batch_idx):
#         return self.test_step(batch, batch_idx)
    
#     def on_training_epoch_end(self):
#         # Clear stored outputs
#         self.training_step_outputs.clear()
    
#     def on_test_epoch_end(self):
#         # Clear stored outputs
#         self.test_step_outputs.clear()
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
#         # If using test set as validation, we can use a scheduler
#         if self.use_test_as_val:
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer, mode='min', factor=0.5, patience=5, verbose=True
#             )
            
#             return {
#                 'optimizer': optimizer,
#                 'lr_scheduler': {
#                     'scheduler': scheduler,
#                     'monitor': 'val_loss',  # Monitor val_loss (which is actually test_loss)
#                     'interval': 'epoch',
#                     'frequency': 1
#                 }
#             }
#         else:
#             # If not using validation, return just the optimizer
#             return optimizer
        
#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         parser = parent_parser.add_argument_group("RadPathFusionModule")
#         parser.add_argument('--hidden_dim', type=int, default=256)
#         parser.add_argument('--fusion_dim', type=int, default=128)
#         parser.add_argument('--dropout', type=float, default=0.2)
#         parser.add_argument('--learning_rate', type=float, default=0.001)
#         parser.add_argument('--fusion_type', type=str, default='attention', 
#                         choices=['attention', 'kronecker', 'concatenation'])
#         parser.add_argument('--use_test_as_val', type=bool, default=True,
#                         help='Use test set for validation during training')
#         return parent_parser



########### model checking here
# model = RadPathFusionModule(
#     rad_input_dim=33,
#     wsi_input_dim=194,
#     hidden_dim=256,
#     fusion_dim=128,
#     fusion_type='attention',
#     learning_rate=0.001
# )

###print("model", model)
























#######################################################################################################################
################################################################################################################
#########################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from torch.utils.data import DataLoader
# import math
# from typing import List, Dict, Union, Tuple, Optional
# from pytorch_lightning.loggers import TensorBoardLogger
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import concordance_index
# import os
# from argparse import ArgumentParser


# class WSIEncoder(nn.Module):
#     """
#     Encodes WSI patch features into a fixed-size representation
#     """
#     def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, dropout=0.1):
#         super(WSIEncoder, self).__init__()
        
#         # Feature transformation for patches
#         self.feature_transform = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout)
#         )
        
#         # Multi-head attention for patch interaction
#         self.self_attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True
#         )
        
#         # Layer norm and MLP
#         self.layer_norm1 = nn.LayerNorm(hidden_dim)
#         self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim * 2, hidden_dim)
#         )
        
#         # Output projection
#         self.output_proj = nn.Linear(hidden_dim, output_dim)
        
#         # Aggregation token (learnable)
#         self.agg_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
#         nn.init.normal_(self.agg_token, std=0.02)
        
#     def forward(self, x, mask=None):
#         # x: [batch, patches, input_dim]
#         # mask: [batch, patches] - 0 for real data, 1 for padding
        
#         batch_size, num_patches, _ = x.shape
        
#         # Apply feature transformation
#         x = self.feature_transform(x)  # [batch, patches, hidden_dim]
        
#         # Expand aggregation token for each sample in batch
#         agg_tokens = self.agg_token.expand(batch_size, -1, -1)  # [batch, 1, hidden_dim]
        
#         # Concatenate aggregation token with patch features
#         x_with_agg = torch.cat([agg_tokens, x], dim=1)  # [batch, 1+patches, hidden_dim]
        
#         # Create attention mask if needed
#         attn_mask = None
#         if mask is not None:
#             # Create mask for aggregation token (not masked) and patches
#             agg_mask = torch.zeros(batch_size, 1, device=mask.device)
#             extended_mask = torch.cat([agg_mask, mask], dim=1)  # [batch, 1+patches]
            
#             # Convert to attention mask format (boolean mask where True values are masked)
#             attn_mask = extended_mask.bool().unsqueeze(1).expand(-1, extended_mask.size(1), -1)
        
#         # Apply self-attention
#         x_norm = self.layer_norm1(x_with_agg)
#         x_attn, _ = self.self_attention(
#             query=x_norm,
#             key=x_norm,
#             value=x_norm,
#             key_padding_mask=attn_mask if attn_mask is not None else None,
#             need_weights=False
#         )
        
#         # Add residual connection
#         x_with_agg = x_with_agg + x_attn
        
#         # Apply MLP with layer norm and residual
#         x_norm = self.layer_norm2(x_with_agg)
#         x_mlp = self.mlp(x_norm)
#         x_with_agg = x_with_agg + x_mlp
        
#         # Extract the aggregation token as the representation of the WSI
#         wsi_representation = x_with_agg[:, 0]  # [batch, hidden_dim]
        
#         # Project to output dimension
#         wsi_output = self.output_proj(wsi_representation)  # [batch, output_dim]
        
#         return wsi_output


# class SurvivalTaskModel(nn.Module):
#     """
#     Task-specific model for survival prediction
#     """
#     def __init__(self, input_dim, hidden_units, dropout=0.25):
#         super(SurvivalTaskModel, self).__init__()
        
#         # Hidden layers
#         layers = []
#         current_dim = input_dim
        
#         for hidden_dim in hidden_units:
#             layers.extend([
#                 nn.Linear(current_dim, hidden_dim),
#                 nn.LeakyReLU(0.1, inplace=True),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.Dropout(dropout)
#             ])
#             current_dim = hidden_dim
        
#         self.hidden_layers = nn.Sequential(*layers)
        
#         # Output layer for hazard prediction
#         self.hazard_layer = nn.Linear(hidden_units[-1], 1)
        
#     def forward(self, x):
#         features = self.hidden_layers(x)
#         hazard = self.hazard_layer(features)
#         return hazard


# class RadPathAttention(nn.Module):
#     """Self-attention mechanism for fusing modalities"""
#     def __init__(self, rad_dim, wsi_dim, hidden_dim):
#         super(RadPathAttention, self).__init__()
#         self.rad_transform = nn.Linear(rad_dim, hidden_dim)
#         self.wsi_transform = nn.Linear(wsi_dim, hidden_dim)
#         self.q_transform = nn.Linear(hidden_dim, hidden_dim)
#         self.k_transform = nn.Linear(hidden_dim, hidden_dim)
#         self.v_transform = nn.Linear(hidden_dim, hidden_dim)
#         self.scaling = hidden_dim ** -0.5
#         self.out_transform = nn.Linear(hidden_dim, hidden_dim)
        
#     def forward(self, rad_features, wsi_features):
#         # Transform features to common space
#         rad_hidden = self.rad_transform(rad_features)
#         wsi_hidden = self.wsi_transform(wsi_features)
        
#         # Stack features from both modalities
#         # rad_hidden: [batch, hidden_dim]
#         # wsi_hidden: [batch, hidden_dim]
#         # combined: [batch, 2, hidden_dim]
#         combined = torch.stack([rad_hidden, wsi_hidden], dim=1)
        
#         # Compute QKV
#         q = self.q_transform(combined)
#         k = self.k_transform(combined)
#         v = self.v_transform(combined)
        
#         # Compute attention scores
#         scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
#         attn_weights = F.softmax(scores, dim=-1)
        
#         # Apply attention
#         attended = torch.matmul(attn_weights, v)
#         attended = self.out_transform(attended)
        
#         # Return attended features for both modalities
#         return attended[:, 0], attended[:, 1]  # rad_attended, wsi_attended


# class CoxLoss(nn.Module):
#     """Cox proportional hazards loss for survival analysis"""
#     def forward(self, hazards, times, events):
#         # Sort in descending order
#         _, indices = torch.sort(times, descending=True)
#         hazards = hazards[indices]
#         events = events[indices]
        
#         # Compute log partial likelihood
#         hazards_exp = torch.exp(hazards)
#         log_risk = torch.cumsum(hazards_exp, dim=0)
#         log_risk = torch.log(log_risk)
        
#         uncensored_likelihood = hazards - log_risk
#         censored_likelihood = uncensored_likelihood * events
        
#         # Negative log likelihood
#         neg_likelihood = -torch.sum(censored_likelihood) / torch.sum(events)
#         return neg_likelihood


# class RadPathFusionModule(pl.LightningModule):
#     """PyTorch Lightning module for RadPath fusion model"""
    
#     def __init__(self, 
#                  rad_input_dim: int,
#                  wsi_input_dim: int = 194,
#                  hidden_dim: int = 256,
#                  fusion_dim: int = 128,
#                  fusion_type: str = "attention",
#                  task_hidden_units: List[int] = [64, 32],
#                  dropout: float = 0.2,
#                  learning_rate: float = 0.001):
#         super(RadPathFusionModule, self).__init__()
        
#         # Save hyperparameters
#         self.save_hyperparameters()
        
#         # Radiological feature encoder
#         self.rad_encoder = nn.Sequential(
#             nn.Linear(rad_input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, fusion_dim)
#         )
        
#         # WSI feature encoder
#         self.wsi_encoder = WSIEncoder(
#             input_dim=wsi_input_dim,
#             hidden_dim=hidden_dim,
#             output_dim=fusion_dim,
#             dropout=dropout
#         )
        
#         # Fusion mechanism
#         self.fusion_type = fusion_type
        
#         if fusion_type == "attention":
#             self.attention = RadPathAttention(
#                 rad_dim=fusion_dim,
#                 wsi_dim=fusion_dim,
#                 hidden_dim=fusion_dim
#             )
#             self.task_input_dim = fusion_dim * 2  # Concatenate attended features
            
#         elif fusion_type == "kronecker":
#             # Add +1 for the bias term in kronecker product
#             self.task_input_dim = (fusion_dim + 1) * (fusion_dim + 1)
            
#         elif fusion_type == "concatenation":
#             self.task_input_dim = fusion_dim * 2
            
#         else:
#             raise ValueError(f"Unsupported fusion type: {fusion_type}")
        
#         # Task-specific model
#         self.task_model = SurvivalTaskModel(
#             input_dim=self.task_input_dim,
#             hidden_units=task_hidden_units,
#             dropout=dropout
#         )
        
#         # Loss function
#         self.cox_loss = CoxLoss()
        
#         # For tracking metrics
#         self.training_step_outputs = []
#         self.validation_step_outputs = []
#         self.test_step_outputs = []
        
#     def forward(self, batch):
#         # Extract features
#         rad_features = batch['x_rad']  # [batch, rad_dim]
#         wsi_features = batch['x_wsi']  # [batch, patches, wsi_dim]
#         mask = batch['mask']  # [batch, patches]
        
#         # Encode features
#         rad_encoded = self.rad_encoder(rad_features)  # [batch, fusion_dim]
#         wsi_encoded = self.wsi_encoder(wsi_features, mask)  # [batch, fusion_dim]
        
#         # Apply fusion strategy
#         if self.fusion_type == "attention":
#             rad_attended, wsi_attended = self.attention(rad_encoded, wsi_encoded)
#             fused_features = torch.cat([rad_attended, wsi_attended], dim=1)
            
#         elif self.fusion_type == "kronecker":
#             # Add bias term
#             rad_bias = torch.cat([rad_encoded, torch.ones(rad_encoded.size(0), 1, device=rad_encoded.device)], dim=1)
#             wsi_bias = torch.cat([wsi_encoded, torch.ones(wsi_encoded.size(0), 1, device=wsi_encoded.device)], dim=1)
            
#             # Compute Kronecker product
#             fused_features = torch.bmm(
#                 rad_bias.unsqueeze(2),  # [batch, fusion_dim+1, 1]
#                 wsi_bias.unsqueeze(1)   # [batch, 1, fusion_dim+1]
#             ).flatten(start_dim=1)  # [batch, (fusion_dim+1)*(fusion_dim+1)]
            
#         elif self.fusion_type == "concatenation":
#             # Simple concatenation
#             fused_features = torch.cat([rad_encoded, wsi_encoded], dim=1)
        
#         # Apply task-specific model
#         hazard = self.task_model(fused_features)
        
#         return hazard
    
#     def _compute_metrics(self, hazards, times, events):
#         # Convert to numpy for c-index calculation
#         hazards_np = hazards.cpu().numpy().flatten()
#         times_np = times.cpu().numpy().flatten()
#         events_np = events.cpu().numpy().flatten()
        
#         # Compute c-index (using negative hazards as prediction, higher means better prognosis)
#         c_index = concordance_index(times_np, -hazards_np, events_np)
        
#         return {
#             'c_index': torch.tensor(c_index, device=hazards.device)
#         }
    
#     def training_step(self, batch, batch_idx):
#         hazards = self(batch)
#         times = batch['time']
#         events = batch['event']
        
#         # Compute loss
#         loss = self.cox_loss(hazards, times, events)
        
#         # Compute metrics
#         metrics = self._compute_metrics(hazards, times, events)
#         c_index = metrics['c_index']
        
#         # Log metrics
#         self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
#         self.log('train_c_index', c_index, on_step=False, on_epoch=True, prog_bar=True)
        
#         # Store outputs for epoch end
#         self.training_step_outputs.append({
#             'loss': loss,
#             'hazards': hazards.detach(),
#             'times': times.detach(),
#             'events': events.detach()
#         })
        
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         hazards = self(batch)
#         times = batch['time']
#         events = batch['event']
        
#         # Compute loss
#         loss = self.cox_loss(hazards, times, events)
        
#         # Compute metrics
#         metrics = self._compute_metrics(hazards, times, events)
#         c_index = metrics['c_index']
        
#         # Log metrics
#         self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
#         self.log('val_c_index', c_index, on_step=False, on_epoch=True, prog_bar=True)
        
#         # Store outputs for epoch end
#         self.validation_step_outputs.append({
#             'loss': loss,
#             'hazards': hazards.detach(),
#             'times': times.detach(),
#             'events': events.detach()
#         })
        
#         return loss
    
#     def test_step(self, batch, batch_idx):
#         hazards = self(batch)
#         times = batch['time']
#         events = batch['event']
        
#         # Compute loss
#         loss = self.cox_loss(hazards, times, events)
        
#         # Compute metrics
#         metrics = self._compute_metrics(hazards, times, events)
#         c_index = metrics['c_index']
        
#         # Log metrics
#         self.log('test_loss', loss, on_step=False, on_epoch=True)
#         self.log('test_c_index', c_index, on_step=False, on_epoch=True)
        
#         # Store outputs for epoch end
#         self.test_step_outputs.append({
#             'loss': loss,
#             'hazards': hazards.detach(),
#             'times': times.detach(),
#             'events': events.detach()
#         })
        
#         return loss
    
#     def on_training_epoch_end(self):
#         # Clear stored outputs
#         self.training_step_outputs.clear()
    
#     def on_validation_epoch_end(self):
#         # Clear stored outputs
#         self.validation_step_outputs.clear()
    
#     def on_test_epoch_end(self):
#         # Clear stored outputs
#         self.test_step_outputs.clear()
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='min', factor=0.5, patience=5, verbose=True
#         )
        
#         return {
#             'optimizer': optimizer,
#             'lr_scheduler': {
#                 'scheduler': scheduler,
#                 'monitor': 'val_loss',
#                 'interval': 'epoch',
#                 'frequency': 1
#             }
#         }


# # Function to train and evaluate the model
# def train_rad_path_model(train_loader, val_loader, test_loader, 
#                          rad_input_dim, 
#                          fusion_type="attention",
#                          max_epochs=50,
#                          learning_rate=0.001):
    
#     # Create model
#     model = RadPathFusionModule(
#         rad_input_dim=rad_input_dim,
#         fusion_type=fusion_type,
#         learning_rate=learning_rate
#     )
    
#     # Define callbacks
#     checkpoint_callback = ModelCheckpoint(
#         dirpath='checkpoints/',
#         filename=f'radpath-{fusion_type}-{{epoch:02d}}-{{val_loss:.2f}}-{{val_c_index:.3f}}',
#         monitor='val_c_index',
#         mode='max',
#         save_top_k=3
#     )
    
#     early_stop_callback = EarlyStopping(
#         monitor='val_loss',
#         patience=10,
#         mode='min'
#     )
    
#     # Define logger
#     logger = TensorBoardLogger('logs/', name=f'radpath-{fusion_type}')
    
#     # Create trainer
#     trainer = pl.Trainer(
#         max_epochs=max_epochs,
#         callbacks=[checkpoint_callback, early_stop_callback],
#         logger=logger,
#         log_every_n_steps=10,
#         deterministic=True
#     )
    
#     # Train the model
#     trainer.fit(model, train_loader, val_loader)
    
#     # Test the model
#     trainer.test(model, test_loader)
    
#     return model, trainer


# # Main function for command-line usage
# def main():
#     parser = ArgumentParser()
    
#     # Dataset parameters
#     parser.add_argument('--root_data', type=str, required=True)
#     parser.add_argument('--outcome_csv', type=str, required=True)
#     parser.add_argument('--split_type', type=str, default='institution', choices=['institution', 'random'])
#     parser.add_argument('--max_patches', type=int, default=2000)
    
#     # Model parameters
#     parser.add_argument('--fusion_type', type=str, default='attention', 
#                         choices=['attention', 'kronecker', 'concatenation'])
#     parser.add_argument('--hidden_dim', type=int, default=256)
#     parser.add_argument('--fusion_dim', type=int, default=128)
#     parser.add_argument('--dropout', type=float, default=0.2)
    
#     # Training parameters
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--max_epochs', type=int, default=50)
#     parser.add_argument('--learning_rate', type=float, default=0.001)
#     parser.add_argument('--seed', type=int, default=42)
    
#     # Add model specific args
#     parser = RadPathFusionModule.add_model_specific_args(parser)
    
#     args = parser.parse_args()
    
#     # Set random seeds
#     pl.seed_everything(args.seed)
    
#     # Import here to avoid circular imports
#     from radpath_dataset import RadPathDataset, survival_collate_fn
    
#     # Create dataset 
#     train_dataset = RadPathDataset(
#         root_data=args.root_data,
#         outcome_csv_path=args.outcome_csv,
#         split='train',
#         split_type=args.split_type,
#         max_patches=args.max_patches,
#         normalize=True
#     )
    
#     # Get normalization stats
#     norm_stats_path = os.path.join(os.path.dirname(args.outcome_csv), "normalization_stats.json")
    
#     val_dataset = RadPathDataset(
#         root_data=args.root_data,
#         outcome_csv_path=args.outcome_csv,
#         split='test',  # Use part of test set as validation
#         split_type=args.split_type,
#         max_patches=args.max_patches,
#         normalize=True,
#         norm_stats_path=norm_stats_path
#     )
    
#     # Create dataloaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         collate_fn=survival_collate_fn,
#         num_workers=4
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         collate_fn=survival_collate_fn,
#         num_workers=4
#     )
    
#     # Get radiological feature dimension
#     rad_input_dim = len(train_dataset.rad_feature_cols)
    
#     # Train model
#     model, trainer = train_rad_path_model(
#         train_loader=train_loader,
#         val_loader=val_loader,
#         test_loader=val_loader,  # Use validation set as test set for simplicity
#         rad_input_dim=rad_input_dim,
#         fusion_type=args.fusion_type,
#         max_epochs=args.max_epochs,
#         learning_rate=args.learning_rate
#     )
    
#     print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")
    

# if __name__ == "__main__":
#     main()