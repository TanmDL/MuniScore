import os
import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

from model_breast import RadPathFusionModule  # ensure you have predict_step
from breast_dataset import RadPathDataset, survival_collate_fn

def set_seed(seed):
    """
    Set seeds for all potential sources of randomness.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    
    # For PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pl.seed_everything(seed)


# class LossHistoryCallback(Callback):
#     """
#     Callback to store training and validation losses after every epoch.
#     Lets us plot train vs val loss curves at the end.
#     """
#     def __init__(self):
#         super().__init__()
#         self.train_losses = []
#         self.val_losses = []

#     def on_train_epoch_end(self, trainer, pl_module):
#         # The "train_loss" is logged each epoch; retrieve it
#         train_loss = trainer.callback_metrics.get("train_loss", None)
#         if train_loss is not None:
#             self.train_losses.append(float(train_loss))

#     def on_validation_epoch_end(self, trainer, pl_module):
#         # The "val_loss" is logged each epoch; retrieve it
#         val_loss = trainer.callback_metrics.get("val_loss", None)
#         if val_loss is not None:
#             self.val_losses.append(float(val_loss))

class LossHistoryCallback(Callback):
    """
    Callback to store training and validation losses after every epoch.
    Lets us plot train vs val loss curves at the end.
    """
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.val_c_indices = []  # Add this to track validation c-indices

    def on_train_epoch_end(self, trainer, pl_module):
        # The "train_loss" is logged each epoch; retrieve it
        train_loss = trainer.callback_metrics.get("train_loss", None)
        if train_loss is not None:
            self.train_losses.append(float(train_loss))

    def on_validation_epoch_end(self, trainer, pl_module):
        # The "val_loss" is logged each epoch; retrieve it
        val_loss = trainer.callback_metrics.get("val_loss", None)
        if val_loss is not None:
            self.val_losses.append(float(val_loss))
        
        # Also track validation c-index
        val_c_index = trainer.callback_metrics.get("val_c_index", None)
        if val_c_index is not None:
            self.val_c_indices.append(float(val_c_index))


def robust_flatten(x):
    """
    Convert x to a flat 1D NumPy array, handling Tensors or lists of Tensors.
    """
    if x is None:
        return np.array([], dtype=float)

    # If x is a list/tuple, possibly of Tensors
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return np.array([], dtype=float)
        # If all items are Tensors, concatenate them
        if all(isinstance(elem, torch.Tensor) for elem in x):
            x = torch.cat(x, dim=0)  # cat along dim=0 => shape [B,...]
        else:
            # Otherwise just turn it into an array
            arr = np.array(x)
            return arr.ravel()

    # If x is still a Tensor, convert to NumPy
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    # Now x should be a NumPy array
    x = np.squeeze(x)  # remove any size=1 dims
    x = x.reshape(-1)  # ensure 1D
    return x


def main():
    """
    Main function for training and evaluating the RadPath fusion model
    with logging, predictions, and combined KM plots.
    
    IMPORTANT: This version uses the training-set median hazard 
    for the test set's single-dataset KM curve.
    """
    parser = argparse.ArgumentParser()
    
    # Dataset parameters
    parser.add_argument('--root_data', type=str, required=True, help="Root directory containing pathology data")
    parser.add_argument('--outcome_csv', type=str, required=True, help="Path to the outcome CSV file")
    parser.add_argument('--split_type', type=str, default='institution', choices=['institution', 'random'],
                        help="Split type: 'institution' or 'random'")
    parser.add_argument('--max_patches', type=int, default=2000, help="Maximum number of patches per WSI")
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true', help="Use deterministic algorithms")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs to use")
    parser.add_argument('--precision', type=str, default="32-true", 
                        choices=["32-true", "16-mixed", "bf16-mixed"], 
                        help="Precision for training")
    parser.add_argument('--val_size', type=float, default=0.2,
                        help="Fraction of training set used as validation (only for random split)")
    parser.add_argument('--use_test_as_val', action='store_true', 
                        help="Use test set as validation (only for institution split)")
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results', help="Directory to save results")
    parser.add_argument('--save_predictions', action='store_true', help="Save predictions to CSV files")
    parser.add_argument('--log_dir', type=str, default='logs', help="Directory to save logs")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory to save checkpoints")
    
    # Visualization parameters
    parser.add_argument('--create_plots', action='store_true', help="Create plots from the results")
    parser.add_argument('--plot_dir', type=str, default='plots', help="Directory to save plots")
    
    # Model parameters
    parser.add_argument('--fusion_type', type=str, default='attention', 
                    choices=['attention', 'kronecker', 'concatenation'],
                    help="Type of fusion mechanism to use")
    parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden dimension size")
    parser.add_argument('--fusion_dim', type=int, default=128, help="Fusion dimension size")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate")
    parser.add_argument('--learning_rate', type=float, default=0.0002, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay (L2 regularization)")
    
    args = parser.parse_args()
    
    # Set random seeds
    print(f"Seed set to {args.seed}")
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    if args.create_plots:
        os.makedirs(args.plot_dir, exist_ok=True)
    
    # Create unique experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"radpath-{args.fusion_type}-{timestamp}"
    
    # Create the training dataset
    print("Creating training dataset...")
    train_dataset = RadPathDataset(
        root_data=args.root_data,
        outcome_csv_path=args.outcome_csv,
        split='train',
        split_type=args.split_type,
        random_seed=args.seed,
        max_patches=args.max_patches
    )
    
    # Create the test dataset
    print("Creating test dataset...")
    test_dataset = RadPathDataset(
        root_data=args.root_data,
        outcome_csv_path=args.outcome_csv,
        split='test',
        split_type=args.split_type,
        random_seed=args.seed,
        max_patches=args.max_patches
    )
    
    # Handle validation set
    val_loader = None
    use_test_as_val = False
    
    if args.split_type == 'random':
        # Randomly split train -> train+val
        print(f"Creating validation set from training data ({args.val_size*100:.0f}%)...")
        train_size = int((1 - args.val_size) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_subset, val_subset = random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        print(f"Dataset sizes - Train: {len(train_subset)}, Validation: {len(val_subset)}, Test: {len(test_dataset)}")
        
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=survival_collate_fn,
            num_workers=args.num_workers
        )
        train_data = train_subset
        
    else:  # institution-based
        if args.use_test_as_val:
            print("Using institution-based split with test set as validation")
            use_test_as_val = True
            val_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=survival_collate_fn,
                num_workers=args.num_workers
            )
        else:
            print("Using institution-based split without validation")
        
        print(f"Dataset sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        train_data = train_dataset
    
    # Create training DataLoader
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=survival_collate_fn,
        num_workers=args.num_workers
    )
    
    # Create test DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=survival_collate_fn,
        num_workers=args.num_workers
    )
    
    # Radiological feature dimension
    rad_input_dim = len(train_dataset.rad_feature_cols)
    print(f"Radiological feature dimension: {rad_input_dim}")
    print(f"Using fusion type: {args.fusion_type}")
    
    # Possibly speed up matmul
    torch.set_float32_matmul_precision('high')
    
    # Create the model
    model = RadPathFusionModule(
        rad_input_dim=rad_input_dim,
        wsi_input_dim=194,
        fusion_type=args.fusion_type,
        hidden_dim=args.hidden_dim,
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_test_as_val=use_test_as_val,
        output_dir=args.output_dir if args.save_predictions else None
    )
    
    # Callbacks
    callbacks = []
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    loss_callback = LossHistoryCallback()
    callbacks.append(loss_callback)

    

    # Checkpoint callback / Early stopping
    # We choose to monitor 'val_c_index' with mode='max'
    if val_loader is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.checkpoint_dir, experiment_name),
            filename="{epoch:02d}-{val_loss:.2f}-{val_c_index:.3f}",
            monitor='val_c_index',
            mode='max',
            save_top_k=1,
            save_last=True
        )
        early_stop_callback = EarlyStopping(
            monitor='val_c_index',
            patience=args.max_epochs,
            mode='max',
            verbose=True
        )
        callbacks.extend([checkpoint_callback, early_stop_callback])
    else:
        # No validation => monitor train_c_index
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.checkpoint_dir, experiment_name),
            filename="{epoch:02d}-{train_loss:.2f}-{train_c_index:.3f}",
            monitor='train_c_index',
            mode='max',
            save_top_k=1,
            save_last=True
        )
        callbacks.append(checkpoint_callback)
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=experiment_name,
        default_hp_metric=False
    )
    logger.log_hyperparams({
        'fusion_type': args.fusion_type,
        'hidden_dim': args.hidden_dim,
        'fusion_dim': args.fusion_dim,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'split_type': args.split_type,
        'use_test_as_val': use_test_as_val,
        'max_patches': args.max_patches,
        'rad_input_dim': rad_input_dim
    })
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        deterministic=args.deterministic,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.num_gpus if torch.cuda.is_available() else None,
        precision=args.precision,
        strategy="ddp" if args.num_gpus > 1 else "auto",
        enable_progress_bar=True,
        num_sanity_val_steps=0
    )
    
    # Train
    print(f"\nStarting model training for {experiment_name}...")
    if val_loader is not None:
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader)
    
    # Test
    print("\nEvaluating model on test set...")
    test_results = trainer.test(model, test_loader)
    print("\nTest Results:")
    for k, v in test_results[0].items():
        print(f"{k}: {v:.4f}")
    
    # Save test results
    test_results_df = pd.DataFrame([test_results[0]])
    test_results_df['fusion_type'] = args.fusion_type
    test_results_df['timestamp'] = timestamp
    test_results_path = os.path.join(args.output_dir, f"test_metrics_{experiment_name}.csv")
    test_results_df.to_csv(test_results_path, index=False)
    print(f"Test metrics saved to {test_results_path}")
    
    # Best model path
    best_model_path = None
    if hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path:
        best_model_path = checkpoint_callback.best_model_path
        print(f"\nBest model saved at: {best_model_path}")
    
    # Export training set predictions
    if args.save_predictions:
        print("\nExporting predictions for the entire training set...")
        if best_model_path is not None:
            best_model = RadPathFusionModule.load_from_checkpoint(best_model_path)
            export_predictions_for_train(best_model, trainer, train_loader, args.output_dir, experiment_name)
        else:
            export_predictions_for_train(model, trainer, train_loader, args.output_dir, experiment_name)
    
    # (Optional) re-run test with best model for final predictions export
    if args.save_predictions and best_model_path is not None:
        print("\nLoading best model for final test predictions export...")
        best_model = RadPathFusionModule.load_from_checkpoint(best_model_path)
        trainer.test(best_model, test_loader)
    
    # Single-dataset plots
    if args.create_plots and args.save_predictions:
        print("\nCreating single-dataset visualization plots (train/test separately)...")
        create_visualization_plots(args, experiment_name, use_train_cutoff=True)
    
    # Combined KM plot
    if args.create_plots and args.save_predictions:
        print("\nCreating combined KM plot with both training and test sets...")
        train_pred_file = find_prediction_csv(args.output_dir, 'train', args.fusion_type)
        test_pred_file  = find_prediction_csv(args.output_dir, 'test',  args.fusion_type)
        
        if train_pred_file and test_pred_file:
            combined_km_plot_path = os.path.join(args.plot_dir, f"KM_combined_{experiment_name}.png")
            create_combined_km_plot(
                os.path.join(args.output_dir, train_pred_file),
                os.path.join(args.output_dir, test_pred_file),
                combined_km_plot_path
            )
        else:
            print("Could not find train/test prediction CSV files to make a combined KM plot.")
    
    # Plot train vs val loss
    if args.create_plots:
        print("\nPlotting training vs. validation loss curves...")
        plot_loss_curves(loss_callback.train_losses, loss_callback.val_losses, args.plot_dir, experiment_name)
    
    print(f"\nExperiment {experiment_name} completed successfully!")
    return model, trainer, checkpoint_callback, experiment_name


def export_predictions_for_train(model, trainer, train_loader, output_dir, experiment_name):
    """
    Use trainer.predict(...) on the training set, flatten shapes, then write CSV.
    """
    all_preds = trainer.predict(model, train_loader)
    
    hazards_all = []
    times_all = []
    events_all = []
    pid_all = []
    
    for batch_dict in all_preds:
        h = robust_flatten(batch_dict.get('hazard', None))
        t = robust_flatten(batch_dict.get('time',   None))
        e = robust_flatten(batch_dict.get('event',  None))
        p = robust_flatten(batch_dict.get('patient_id', None))
        
        hazards_all.append(h)
        times_all.append(t)
        events_all.append(e)
        pid_all.append(p)
    
    hazards_all = np.concatenate(hazards_all, axis=0)
    times_all   = np.concatenate(times_all,   axis=0)
    events_all  = np.concatenate(events_all,  axis=0)
    pid_all     = np.concatenate(pid_all,     axis=0)
    
    df = pd.DataFrame({
        'hazard_score': hazards_all,
        'time': times_all,
        'event': events_all
    })
    df['patient_id'] = pid_all
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"train_{model.fusion_type}_predictions_{timestamp}.csv"
    out_path = os.path.join(output_dir, csv_name)
    df.to_csv(out_path, index=False)
    print(f"Saved training predictions to {out_path}")


def find_prediction_csv(output_dir, dataset_type, fusion_type):
    """
    Look in output_dir for a file with e.g. "train_attention_predictions" or
    "test_attention_predictions" in its name.
    """
    for f in os.listdir(output_dir):
        if f.endswith('.csv') and dataset_type in f and fusion_type in f and 'predictions' in f:
            return f
    return None


def create_combined_km_plot(train_pred_csv, test_pred_csv, save_path):
    """
    Loads train/test prediction CSVs, merges them, splits by high/low risk
    using a global median hazard, then plots 4 curves. Also fits a Cox model
    and does a log-rank test, printing c-index, hazard ratio, and p-values.
    
    This still uses the combined median. If you want the train median 
    for the combined approach, you'd need to load the train CSV first, 
    compute train_median, then apply that to both train & test.
    """
    train_df = pd.read_csv(train_pred_csv)
    test_df  = pd.read_csv(test_pred_csv)
    
    train_df['dataset'] = 'train'
    test_df['dataset']  = 'test'
    
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # Define risk groups by global median hazard
    global_median = combined_df['hazard_score'].median()
    combined_df['risk_group'] = (combined_df['hazard_score'] > global_median).astype(int)
    
    # Compute c-index (negative hazard => better outcome)
    c_index = concordance_index(
        combined_df['time'],
        -combined_df['hazard_score'],
        combined_df['event']
    )
    
    # Fit Cox model
    cph = CoxPHFitter()
    cph_df = combined_df[['time','event','hazard_score']].copy()
    cph.fit(cph_df, duration_col='time', event_col='event', show_progress=False)
    hr   = np.exp(cph.params_['hazard_score'])
    p_val= cph.summary.loc['hazard_score','p']
    
    # Log-rank test
    low_group  = combined_df[combined_df['risk_group'] == 0]
    high_group = combined_df[combined_df['risk_group'] == 1]
    results = logrank_test(
        low_group['time'],
        high_group['time'],
        event_observed_A=low_group['event'],
        event_observed_B=high_group['event']
    )
    p_val_logrank = results.p_value
    
    # Plot 4 curves
    plt.figure(figsize=(8, 6))
    kmf = KaplanMeierFitter()
    
    for (ds, rg), subset_df in combined_df.groupby(['dataset','risk_group']):
        label = f"{ds.title()} - {'LowRisk' if rg==0 else 'HighRisk'}"
        kmf.fit(subset_df['time'], event_observed=subset_df['event'], label=label)
        kmf.plot(ci_show=False)
    
    plt.title(f"Combined KM (Train + Test)\nC-index={c_index:.3f}, HR={hr:.3f}, p={p_val:.3e}")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.grid(alpha=0.3)
    
    plt.text(
        0.05, 0.05,
        f"Log-rank p={p_val_logrank:.3e}",
        transform=plt.gca().transAxes
    )
    
    plt.legend(loc="best")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined KM plot saved to {save_path}")


def create_visualization_plots(args, experiment_name, use_train_cutoff=False):
    """
    Create standard single-dataset visualization plots for train/test
    from any CSVs named e.g. "train_attention_predictions..." or
    "test_attention_predictions...".

    If use_train_cutoff=True, then for the test set we 
    find the train median hazard and apply it to the test set 
    instead of computing the test set's median hazard.
    """
    try:
        files = [f for f in os.listdir(args.output_dir) if f.endswith('.csv') and 'predictions' in f]
        if not files:
            print("No prediction CSV files found for single-dataset visualization.")
            return
        
        train_df = None
        test_df  = None
        
        # Find train predictions
        train_pred_file = None
        for f in files:
            if 'train' in f and args.fusion_type in f:
                train_pred_file = f
                break
        
        # Load train data if found
        train_median = None
        if train_pred_file:
            train_path = os.path.join(args.output_dir, train_pred_file)
            train_df   = pd.read_csv(train_path)
            
            # If you want to do single-dataset train KM
            plot_single_dataset_km(train_df, "train", experiment_name, args, custom_cutoff=None)
            
            # We'll store the median hazard from train 
            train_median = train_df['hazard_score'].median()
        
        # Find test predictions
        test_pred_file = None
        for f in files:
            if 'test' in f and args.fusion_type in f:
                test_pred_file = f
                break
        
        if test_pred_file:
            test_path = os.path.join(args.output_dir, test_pred_file)
            test_df   = pd.read_csv(test_path)
            
            # If we want to use the train median for test (and we have train_median)
            # otherwise, fallback to None
            if use_train_cutoff and (train_median is not None):
                plot_single_dataset_km(test_df, "test", experiment_name, args, custom_cutoff=train_median)
            else:
                # Use the test's own median hazard
                plot_single_dataset_km(test_df, "test", experiment_name, args, custom_cutoff=None)
        
        if not train_pred_file:
            print("No train predictions found for single-dataset plots.")
        if not test_pred_file:
            print("No test predictions found for single-dataset plots.")
    
    except Exception as e:
        print(f"Error creating visualization plots: {e}")


def plot_single_dataset_km(df, dataset_type, experiment_name, args, custom_cutoff=None):
    """
    Plots KM curves for a single dataset (train or test).
    If custom_cutoff is not None, we use that hazard threshold.
    Otherwise, we use the dataset's own median hazard.
    Also prints c-index, HR, p-value, log-rank p-value.
    """
    if 'hazard_score' not in df.columns:
        print(f"No hazard_score in {dataset_type} DataFrame. Skipping KM.")
        return
    
    # If we have a custom cutoff (from train), apply it to the dataset
    if custom_cutoff is not None:
        df['risk_group'] = df['hazard_score'] > custom_cutoff
        print(f"{dataset_type.capitalize()} set risk split using TRAIN median hazard={custom_cutoff:.4f}")
    else:
        # Use this dataset's own median
        local_median = df['hazard_score'].median()
        df['risk_group'] = df['hazard_score'] > local_median
        print(f"{dataset_type.capitalize()} set risk split using its own median hazard={local_median:.4f}")
    
    # c-index: negative hazards => better outcome
    c_index = concordance_index(df['time'], -df['hazard_score'], df['event'])
    
    # Fit Cox model on entire dataset
    cph_df = df[['time', 'event', 'hazard_score']].copy()
    cph = CoxPHFitter()
    cph.fit(cph_df, duration_col='time', event_col='event', show_progress=False)
    hr = np.exp(cph.params_['hazard_score'])
    p_val = cph.summary.loc['hazard_score', 'p']
    
    # Log-rank test for the two risk groups
    low_group  = df[df['risk_group'] == False]
    high_group = df[df['risk_group'] == True]
    results = logrank_test(
        low_group['time'], high_group['time'],
        event_observed_A=low_group['event'],
        event_observed_B=high_group['event']
    )
    p_val_logrank = results.p_value
    
    # Plot KM
    plt.figure()
    kmf = KaplanMeierFitter()
    
    if len(high_group) > 0:
        kmf.fit(high_group['time'], event_observed=high_group['event'], label='High Risk')
        kmf.plot(ci_show=False)
    if len(low_group) > 0:
        kmf.fit(low_group['time'], event_observed=low_group['event'], label='Low Risk')
        kmf.plot(ci_show=False)
    
    plt.title(
        f"{dataset_type.capitalize()} KM Curves ({args.fusion_type.capitalize()})\n"
        f"c-index={c_index:.3f}, HR={hr:.3f}, p={p_val:.3e}"
    )
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.grid(alpha=0.3)
    
    plt.text(
        0.05, 0.05,
        f"Log-rank p={p_val_logrank:.3e}",
        transform=plt.gca().transAxes
    )
    
    km_plot_path = os.path.join(args.plot_dir, f"km_{dataset_type}_{experiment_name}.png")
    plt.savefig(km_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{dataset_type.capitalize()} KM with c-index/HR/p-values => {km_plot_path}")


def plot_loss_curves(train_losses, val_losses, plot_dir, experiment_name):
    """
    Plot the train vs val/test loss curves if val_losses is not empty.
    Otherwise, just plot train loss alone.
    """
    plt.figure()
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    
    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, marker='s', label='Val/Test Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Loss Curves - {experiment_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    
    loss_plot_path = os.path.join(plot_dir, f"loss_curves_{experiment_name}.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curves plot saved to {loss_plot_path}")


if __name__ == "__main__":
    main()



































# import os
# import argparse
# import torch
# import pytorch_lightning as pl
# from torch.utils.data import DataLoader, random_split
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
# from pytorch_lightning.loggers import TensorBoardLogger
# from datetime import datetime
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import random

# from lifelines import KaplanMeierFitter, CoxPHFitter
# from lifelines.statistics import logrank_test
# from lifelines.utils import concordance_index

# from model_breast import RadPathFusionModule  # Make sure you've added predict_step
# from breast_dataset import RadPathDataset, survival_collate_fn


# def set_seed(seed):
#     """
#     Set seeds for all potential sources of randomness.
#     """
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # For multi-GPU
#     np.random.seed(seed)
#     random.seed(seed)
    
#     # For PyTorch
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     pl.seed_everything(seed)


# class LossHistoryCallback(Callback):
#     """
#     Callback to store training and validation losses after every epoch.
#     Lets us plot train vs val loss curves at the end.
#     """
#     def __init__(self):
#         super().__init__()
#         self.train_losses = []
#         self.val_losses = []

#     def on_train_epoch_end(self, trainer, pl_module):
#         # The "train_loss" is logged each epoch; retrieve it
#         train_loss = trainer.callback_metrics.get("train_loss", None)
#         if train_loss is not None:
#             self.train_losses.append(float(train_loss))

#     def on_validation_epoch_end(self, trainer, pl_module):
#         # The "val_loss" is logged each epoch; retrieve it
#         val_loss = trainer.callback_metrics.get("val_loss", None)
#         if val_loss is not None:
#             self.val_losses.append(float(val_loss))


# def robust_flatten(x):
#     """
#     Convert x to a flat 1D NumPy array, handling Tensors or lists of Tensors.
#     """
#     if x is None:
#         return np.array([], dtype=float)

#     # If x is a list/tuple, possibly of Tensors
#     if isinstance(x, (list, tuple)):
#         if len(x) == 0:
#             return np.array([], dtype=float)
#         # If all items are Tensors, concatenate them
#         if all(isinstance(elem, torch.Tensor) for elem in x):
#             x = torch.cat(x, dim=0)  # cat along dim=0 => shape [B,...]
#         else:
#             # Otherwise just turn it into an array
#             arr = np.array(x)
#             return arr.ravel()

#     # If x is still a Tensor, convert to NumPy
#     if isinstance(x, torch.Tensor):
#         x = x.detach().cpu().numpy()

#     # Now x should be a NumPy array
#     x = np.squeeze(x)  # remove any size=1 dims
#     x = x.reshape(-1)  # ensure 1D
#     return x


# def main():
#     """
#     Main function for training and evaluating the RadPath fusion model
#     with logging, predictions, and combined KM plots.
#     """
#     parser = argparse.ArgumentParser()
    
#     # Dataset parameters
#     parser.add_argument('--root_data', type=str, required=True, help="Root directory containing pathology data")
#     parser.add_argument('--outcome_csv', type=str, required=True, help="Path to the outcome CSV file")
#     parser.add_argument('--split_type', type=str, default='institution', choices=['institution', 'random'],
#                         help="Split type: 'institution' or 'random'")
#     parser.add_argument('--max_patches', type=int, default=3000, help="Maximum number of patches per WSI")
    
#     # Training parameters
#     parser.add_argument('--batch_size', type=int, default=16)
#     parser.add_argument('--max_epochs', type=int, default=50)
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--deterministic', action='store_true', help="Use deterministic algorithms")
#     parser.add_argument('--num_workers', type=int, default=4)
#     parser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs to use")
#     parser.add_argument('--precision', type=str, default="32-true", 
#                         choices=["32-true", "16-mixed", "bf16-mixed"], 
#                         help="Precision for training")
#     parser.add_argument('--val_size', type=float, default=0.2,
#                         help="Fraction of training set used as validation (only for random split)")
#     parser.add_argument('--use_test_as_val', action='store_true', 
#                         help="Use test set as validation (only for institution split)")
    
#     # Output parameters
#     parser.add_argument('--output_dir', type=str, default='results', help="Directory to save results")
#     parser.add_argument('--save_predictions', action='store_true', help="Save predictions to CSV files")
#     parser.add_argument('--log_dir', type=str, default='logs', help="Directory to save logs")
#     parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory to save checkpoints")
    
#     # Visualization parameters
#     parser.add_argument('--create_plots', action='store_true', help="Create plots from the results")
#     parser.add_argument('--plot_dir', type=str, default='plots', help="Directory to save plots")
    
#     # Model parameters
#     parser.add_argument('--fusion_type', type=str, default='attention', 
#                     choices=['attention', 'kronecker', 'concatenation'],
#                     help="Type of fusion mechanism to use")
#     parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden dimension size")
#     parser.add_argument('--fusion_dim', type=int, default=128, help="Fusion dimension size")
#     parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate")
#     parser.add_argument('--learning_rate', type=float, default=0.0002, help="Learning rate")
#     parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay (L2 regularization)")
    
#     args = parser.parse_args()
    
#     # Set random seeds
#     print(f"Seed set to {args.seed}")
#     set_seed(args.seed)
    
#     # Create output directories
#     os.makedirs(args.output_dir, exist_ok=True)
#     os.makedirs(args.checkpoint_dir, exist_ok=True)
#     os.makedirs(args.log_dir, exist_ok=True)
#     if args.create_plots:
#         os.makedirs(args.plot_dir, exist_ok=True)
    
#     # Create unique experiment name with timestamp
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     experiment_name = f"radpath-{args.fusion_type}-{timestamp}"
    
#     # Create the training dataset
#     print("Creating training dataset...")
#     train_dataset = RadPathDataset(
#         root_data=args.root_data,
#         outcome_csv_path=args.outcome_csv,
#         split='train',
#         split_type=args.split_type,
#         random_seed=args.seed,
#         max_patches=args.max_patches
#     )
    
#     # Create the test dataset
#     print("Creating test dataset...")
#     test_dataset = RadPathDataset(
#         root_data=args.root_data,
#         outcome_csv_path=args.outcome_csv,
#         split='test',
#         split_type=args.split_type,
#         random_seed=args.seed,
#         max_patches=args.max_patches
#     )
    
#     # Handle validation set
#     val_loader = None
#     use_test_as_val = False
    
#     if args.split_type == 'random':
#         # Randomly split train -> train+val
#         print(f"Creating validation set from training data ({args.val_size*100:.0f}%)...")
#         train_size = int((1 - args.val_size) * len(train_dataset))
#         val_size = len(train_dataset) - train_size
        
#         train_subset, val_subset = random_split(
#             train_dataset, 
#             [train_size, val_size],
#             generator=torch.Generator().manual_seed(args.seed)
#         )
#         print(f"Dataset sizes - Train: {len(train_subset)}, Validation: {len(val_subset)}, Test: {len(test_dataset)}")
        
#         val_loader = DataLoader(
#             val_subset,
#             batch_size=args.batch_size,
#             shuffle=False,
#             collate_fn=survival_collate_fn,
#             num_workers=args.num_workers
#         )
#         train_data = train_subset
        
#     else:  # institution-based
#         if args.use_test_as_val:
#             print("Using institution-based split with test set as validation")
#             use_test_as_val = True
#             val_loader = DataLoader(
#                 test_dataset,
#                 batch_size=args.batch_size,
#                 shuffle=False,
#                 collate_fn=survival_collate_fn,
#                 num_workers=args.num_workers
#             )
#         else:
#             print("Using institution-based split without validation")
        
#         print(f"Dataset sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
#         train_data = train_dataset
    
#     # Create training DataLoader
#     train_loader = DataLoader(
#         train_data,
#         batch_size=args.batch_size,
#         shuffle=True,
#         collate_fn=survival_collate_fn,
#         num_workers=args.num_workers
#     )
    
#     # Create test DataLoader
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         collate_fn=survival_collate_fn,
#         num_workers=args.num_workers
#     )
    
#     # Radiological feature dimension
#     rad_input_dim = len(train_dataset.rad_feature_cols)
#     print(f"Radiological feature dimension: {rad_input_dim}")
#     print(f"Using fusion type: {args.fusion_type}")
    
#     # Possibly speed up matmul
#     torch.set_float32_matmul_precision('high')
    
#     # Create the model
#     model = RadPathFusionModule(
#         rad_input_dim=rad_input_dim,
#         wsi_input_dim=194,
#         fusion_type=args.fusion_type,
#         hidden_dim=args.hidden_dim,
#         fusion_dim=args.fusion_dim,
#         dropout=args.dropout,
#         learning_rate=args.learning_rate,
#         weight_decay=args.weight_decay,
#         use_test_as_val=use_test_as_val,
#         output_dir=args.output_dir if args.save_predictions else None
#     )
    
#     # Callbacks
#     callbacks = []
#     lr_monitor = LearningRateMonitor(logging_interval='epoch')
#     callbacks.append(lr_monitor)
    
#     loss_callback = LossHistoryCallback()
#     callbacks.append(loss_callback)

#     # Checkpoint callback / Early stopping
#     if val_loader is not None:
#         checkpoint_callback = ModelCheckpoint(
#             dirpath=os.path.join(args.checkpoint_dir, experiment_name),
#             filename="{epoch:02d}-{val_loss:.2f}-{val_c_index:.3f}",
#             monitor='val_c_index',
#             mode='max',
#             save_top_k=3,
#             save_last=True
#         )
#         early_stop_callback = EarlyStopping(
#             monitor='val_loss',
#             patience=args.max_epochs,
#             mode='min',
#             verbose=True
#         )
#         callbacks.extend([checkpoint_callback, early_stop_callback])
#     else:
#         # No validation => monitor train_c_index
#         checkpoint_callback = ModelCheckpoint(
#             dirpath=os.path.join(args.checkpoint_dir, experiment_name),
#             filename="{epoch:02d}-{train_loss:.2f}-{train_c_index:.3f}",
#             monitor='train_c_index',
#             mode='max',
#             save_top_k=2,
#             save_last=True
#         )
#         callbacks.append(checkpoint_callback)
    
#     # Logger
#     logger = TensorBoardLogger(
#         save_dir=args.log_dir,
#         name=experiment_name,
#         default_hp_metric=False
#     )
#     logger.log_hyperparams({
#         'fusion_type': args.fusion_type,
#         'hidden_dim': args.hidden_dim,
#         'fusion_dim': args.fusion_dim,
#         'dropout': args.dropout,
#         'learning_rate': args.learning_rate,
#         'weight_decay': args.weight_decay,
#         'batch_size': args.batch_size,
#         'split_type': args.split_type,
#         'use_test_as_val': use_test_as_val,
#         'max_patches': args.max_patches,
#         'rad_input_dim': rad_input_dim
#     })
    
#     # Trainer
#     trainer = pl.Trainer(
#         max_epochs=args.max_epochs,
#         callbacks=callbacks,
#         logger=logger,
#         log_every_n_steps=10,
#         deterministic=args.deterministic,
#         accelerator="gpu" if torch.cuda.is_available() else "cpu",
#         devices=args.num_gpus if torch.cuda.is_available() else None,
#         precision=args.precision,
#         strategy="ddp" if args.num_gpus > 1 else "auto",
#         enable_progress_bar=True,
#         num_sanity_val_steps=0
#     )
    
#     # Train
#     print(f"\nStarting model training for {experiment_name}...")
#     if val_loader is not None:
#         trainer.fit(model, train_loader, val_loader)
#     else:
#         trainer.fit(model, train_loader)
    
#     # Test
#     print("\nEvaluating model on test set...")
#     test_results = trainer.test(model, test_loader)
#     print("\nTest Results:")
#     for k, v in test_results[0].items():
#         print(f"{k}: {v:.4f}")
    
#     # Save test results
#     test_results_df = pd.DataFrame([test_results[0]])
#     test_results_df['fusion_type'] = args.fusion_type
#     test_results_df['timestamp'] = timestamp
#     test_results_path = os.path.join(args.output_dir, f"test_metrics_{experiment_name}.csv")
#     test_results_df.to_csv(test_results_path, index=False)
#     print(f"Test metrics saved to {test_results_path}")
    
#     # Best model path
#     best_model_path = None
#     if hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path:
#         best_model_path = checkpoint_callback.best_model_path
#         print(f"\nBest model saved at: {best_model_path}")
    
#     # Export training set predictions
#     if args.save_predictions:
#         print("\nExporting predictions for the entire training set...")
#         if best_model_path is not None:
#             best_model = RadPathFusionModule.load_from_checkpoint(best_model_path)
#             export_predictions_for_train(best_model, trainer, train_loader, args.output_dir, experiment_name)
#         else:
#             export_predictions_for_train(model, trainer, train_loader, args.output_dir, experiment_name)
    
#     # (Optional) re-run test with best model for final predictions export
#     if args.save_predictions and best_model_path is not None:
#         print("\nLoading best model for final test predictions export...")
#         best_model = RadPathFusionModule.load_from_checkpoint(best_model_path)
#         trainer.test(best_model, test_loader)
    
#     # Single-dataset plots
#     if args.create_plots and args.save_predictions:
#         print("\nCreating single-dataset visualization plots (train/test separately)...")
#         create_visualization_plots(args, experiment_name)
    
#     # Combined KM plot
#     if args.create_plots and args.save_predictions:
#         print("\nCreating combined KM plot with both training and test sets...")
#         train_pred_file = find_prediction_csv(args.output_dir, 'train', args.fusion_type)
#         test_pred_file  = find_prediction_csv(args.output_dir, 'test',  args.fusion_type)
        
#         if train_pred_file and test_pred_file:
#             combined_km_plot_path = os.path.join(args.plot_dir, f"KM_combined_{experiment_name}.png")
#             create_combined_km_plot(
#                 os.path.join(args.output_dir, train_pred_file),
#                 os.path.join(args.output_dir, test_pred_file),
#                 combined_km_plot_path
#             )
#         else:
#             print("Could not find train/test prediction CSV files to make a combined KM plot.")
    
#     # Plot train vs val loss
#     if args.create_plots:
#         print("\nPlotting training vs. validation loss curves...")
#         plot_loss_curves(loss_callback.train_losses, loss_callback.val_losses, args.plot_dir, experiment_name)
    
#     print(f"\nExperiment {experiment_name} completed successfully!")
#     return model, trainer, checkpoint_callback, experiment_name


# def export_predictions_for_train(model, trainer, train_loader, output_dir, experiment_name):
#     """
#     Use trainer.predict(...) on the training set, flatten shapes, then write CSV.
#     """
#     all_preds = trainer.predict(model, train_loader)
    
#     hazards_all = []
#     times_all = []
#     events_all = []
#     pid_all = []
    
#     for batch_dict in all_preds:
#         h = robust_flatten(batch_dict.get('hazard', None))
#         t = robust_flatten(batch_dict.get('time',   None))
#         e = robust_flatten(batch_dict.get('event',  None))
#         p = robust_flatten(batch_dict.get('patient_id', None))
        
#         hazards_all.append(h)
#         times_all.append(t)
#         events_all.append(e)
#         pid_all.append(p)
    
#     hazards_all = np.concatenate(hazards_all, axis=0)
#     times_all   = np.concatenate(times_all,   axis=0)
#     events_all  = np.concatenate(events_all,  axis=0)
#     pid_all     = np.concatenate(pid_all,     axis=0)
    
#     df = pd.DataFrame({
#         'hazard_score': hazards_all,
#         'time': times_all,
#         'event': events_all
#     })
#     df['patient_id'] = pid_all
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     csv_name = f"train_{model.fusion_type}_predictions_{timestamp}.csv"
#     out_path = os.path.join(output_dir, csv_name)
#     df.to_csv(out_path, index=False)
#     print(f"Saved training predictions to {out_path}")


# def find_prediction_csv(output_dir, dataset_type, fusion_type):
#     """
#     Look in output_dir for a file with e.g. "train_attention_predictions" or
#     "test_attention_predictions" in its name.
#     """
#     for f in os.listdir(output_dir):
#         if f.endswith('.csv') and dataset_type in f and fusion_type in f and 'predictions' in f:
#             return f
#     return None


# def create_combined_km_plot(train_pred_csv, test_pred_csv, save_path):
#     """
#     Loads train/test prediction CSVs, merges them, splits by high/low risk
#     using a global median hazard, then plots 4 curves. Also fits a Cox model
#     and does a log-rank test, printing c-index, hazard ratio, and p-values.
#     """
#     train_df = pd.read_csv(train_pred_csv)
#     test_df  = pd.read_csv(test_pred_csv)
    
#     train_df['dataset'] = 'train'
#     test_df['dataset']  = 'test'
    
#     combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
#     # Define risk groups by global median hazard
#     global_median = combined_df['hazard_score'].median()
#     combined_df['risk_group'] = (combined_df['hazard_score'] > global_median).astype(int)
    
#     # Compute c-index (negative hazard => better outcome)
#     c_index = concordance_index(
#         combined_df['time'],
#         -combined_df['hazard_score'],
#         combined_df['event']
#     )
    
#     # Fit Cox model
#     cph = CoxPHFitter()
#     cph_df = combined_df[['time','event','hazard_score']].copy()
#     # We do NOT rename 'time' to 'duration'; we pass 'time' directly:
#     cph.fit(cph_df, duration_col='time', event_col='event', show_progress=False)
#     hr   = np.exp(cph.params_['hazard_score'])
#     p_val= cph.summary.loc['hazard_score','p']
    
#     # Log-rank test between high vs low
#     low_group  = combined_df[combined_df['risk_group'] == 0]
#     high_group = combined_df[combined_df['risk_group'] == 1]
#     results = logrank_test(
#         low_group['time'],
#         high_group['time'],
#         event_observed_A=low_group['event'],
#         event_observed_B=high_group['event']
#     )
#     p_val_logrank = results.p_value
    
#     # Plot 4 curves: (Train, Low), (Train, High), (Test, Low), (Test, High)
#     plt.figure(figsize=(8, 6))
#     kmf = KaplanMeierFitter()
    
#     for (ds, rg), subset_df in combined_df.groupby(['dataset','risk_group']):
#         label = f"{ds.title()} - {'LowRisk' if rg==0 else 'HighRisk'}"
#         kmf.fit(subset_df['time'], event_observed=subset_df['event'], label=label)
#         kmf.plot(ci_show=False)
    
#     plt.title(f"Combined KM (Train + Test)\nC-index={c_index:.3f}, HR={hr:.3f}, p={p_val:.3e}")
#     plt.xlabel("Time")
#     plt.ylabel("Survival Probability")
#     plt.grid(alpha=0.3)
    
#     plt.text(
#         0.05, 0.05,
#         f"Log-rank p={p_val_logrank:.3e}",
#         transform=plt.gca().transAxes
#     )
    
#     plt.legend(loc="best")
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Combined KM plot saved to {save_path}")


# def create_visualization_plots(args, experiment_name):
#     """
#     Create standard single-dataset visualization plots for train/test
#     from any CSVs named e.g. "train_attention_predictions..." or
#     "test_attention_predictions...".
#     """
#     try:
#         files = [f for f in os.listdir(args.output_dir) if f.endswith('.csv') and 'predictions' in f]
#         if not files:
#             print("No prediction CSV files found for single-dataset visualization.")
#             return
        
#         # Look for test predictions
#         test_pred_file = None
#         for f in files:
#             if 'test' in f and args.fusion_type in f:
#                 test_pred_file = f
#                 break
        
#         if test_pred_file:
#             test_path = os.path.join(args.output_dir, test_pred_file)
#             df_test   = pd.read_csv(test_path)
#             plot_single_dataset_km(df_test, "test", experiment_name, args)
#             plot_hazard_distribution(df_test, "test", experiment_name, args)
#             plot_event_comparison(df_test, "test", experiment_name, args)
#         else:
#             print("No test predictions found for single-dataset plots.")
        
#         # Look for train predictions
#         train_pred_file = None
#         for f in files:
#             if 'train' in f and args.fusion_type in f:
#                 train_pred_file = f
#                 break
        
#         if train_pred_file:
#             train_path = os.path.join(args.output_dir, train_pred_file)
#             df_train   = pd.read_csv(train_path)
#             plot_single_dataset_km(df_train, "train", experiment_name, args)
#             plot_hazard_distribution(df_train, "train", experiment_name, args)
#             plot_event_comparison(df_train, "train", experiment_name, args)
#         else:
#             print("No train predictions found for single-dataset plots.")
    
#     except Exception as e:
#         print(f"Error creating visualization plots: {e}")


# def plot_single_dataset_km(df, dataset_type, experiment_name, args):
#     if 'hazard_score' not in df.columns:
#         print(f"No hazard_score in {dataset_type} DataFrame. Skipping KM.")
#         return
    
#     # Split into two risk groups by median hazard
#     df['risk_group'] = df['hazard_score'] > df['hazard_score'].median()
    
#     # ----------------------------
#     # (A) Compute c-index, HR, p-value for entire dataset
#     # ----------------------------
#     # c-index: negative hazards => better outcome
#     c_index = concordance_index(df['time'], -df['hazard_score'], df['event'])
    
#     # Fit a Cox model
#     cph_df = df[['time', 'event', 'hazard_score']].copy()
#     cph = CoxPHFitter()
#     cph.fit(cph_df, duration_col='time', event_col='event', show_progress=False)
#     hr = np.exp(cph.params_['hazard_score'])
#     p_val = cph.summary.loc['hazard_score', 'p']
    
#     # Also do a log-rank test for the two risk groups
#     low_group  = df[df['risk_group'] == False]
#     high_group = df[df['risk_group'] == True]
#     results = logrank_test(
#         low_group['time'], high_group['time'],
#         event_observed_A=low_group['event'],
#         event_observed_B=high_group['event']
#     )
#     p_val_logrank = results.p_value
    
#     # ----------------------------
#     # (B) Plot KM curves for High vs. Low risk
#     # ----------------------------
#     plt.figure()
#     kmf = KaplanMeierFitter()
    
#     if len(high_group) > 0:
#         kmf.fit(high_group['time'], event_observed=high_group['event'], label='High Risk')
#         kmf.plot(ci_show=False)
#     if len(low_group) > 0:
#         kmf.fit(low_group['time'], event_observed=low_group['event'], label='Low Risk')
#         kmf.plot(ci_show=False)
    
#     # Put the c-index, HR, and p in the title, plus log-rank test p as text
#     plt.title(
#         f"{dataset_type.capitalize()} KM Curves ({args.fusion_type.capitalize()})\n"
#         f"c-index={c_index:.3f}, HR={hr:.3f}, p={p_val:.3e}"
#     )
#     plt.xlabel('Time')
#     plt.ylabel('Survival Probability')
#     plt.grid(alpha=0.3)
    
#     # Show log-rank p-value in corner
#     plt.text(
#         0.05, 0.05,
#         f"Log-rank p={p_val_logrank:.3e}",
#         transform=plt.gca().transAxes
#     )
    
#     # Save figure
#     km_plot_path = os.path.join(args.plot_dir, f"km_{dataset_type}_{experiment_name}.png")
#     plt.savefig(km_plot_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"{dataset_type.capitalize()} KM with c-index/HR/p-values => {km_plot_path}")


# def plot_hazard_distribution(df, dataset_type, experiment_name, args):
#     """
#     Plot histogram of hazard scores for a single dataset (train/test).
#     """
#     if 'hazard_score' not in df.columns:
#         print(f"No hazard_score in {dataset_type} DataFrame. Skipping hazard distribution.")
#         return
    
#     plt.figure()
#     plt.hist(df['hazard_score'], bins=20, alpha=0.7)
#     plt.axvline(df['hazard_score'].median(), color='red', linestyle='--', label='Median')
#     plt.title(f"{dataset_type.capitalize()} Hazard Score Distribution\n{args.fusion_type.capitalize()} Fusion")
#     plt.xlabel('Hazard Score')
#     plt.ylabel('Count')
#     plt.grid(alpha=0.3)
#     plt.legend()
    
#     hist_path = os.path.join(args.plot_dir, f"hazard_distribution_{dataset_type}_{experiment_name}.png")
#     plt.savefig(hist_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"{dataset_type.capitalize()} hazard distribution plot saved to {hist_path}")


# def plot_event_comparison(df, dataset_type, experiment_name, args):
#     """
#     Boxplot comparing hazard scores for event vs no-event in a single dataset.
#     """
#     if 'hazard_score' not in df.columns:
#         print(f"No hazard_score in {dataset_type} DataFrame. Skipping event comparison.")
#         return
    
#     plt.figure()
#     event_scores    = df[df['event'] == 1]['hazard_score']
#     nonevent_scores = df[df['event'] == 0]['hazard_score']
#     plt.boxplot([event_scores, nonevent_scores], labels=['Event', 'No Event'])
#     plt.title(f"{dataset_type.capitalize()} Hazard Scores by Event Status\n{args.fusion_type.capitalize()} Fusion")
#     plt.ylabel('Hazard Score')
#     plt.grid(alpha=0.3)
    
#     event_plot_path = os.path.join(args.plot_dir, f"event_comparison_{dataset_type}_{experiment_name}.png")
#     plt.savefig(event_plot_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"{dataset_type.capitalize()} event comparison plot saved to {event_plot_path}")


# def plot_loss_curves(train_losses, val_losses, plot_dir, experiment_name):
#     """
#     Plot the train vs val/test loss curves if val_losses is not empty.
#     Otherwise, just plot train loss alone.
#     """
#     plt.figure()
#     epochs = range(1, len(train_losses) + 1)
#     plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    
#     if val_losses:
#         val_epochs = range(1, len(val_losses) + 1)
#         plt.plot(val_epochs, val_losses, marker='s', label='Val/Test Loss')
    
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title(f"Loss Curves - {experiment_name}")
#     plt.legend()
#     plt.grid(alpha=0.3)
    
#     loss_plot_path = os.path.join(plot_dir, f"loss_curves_{experiment_name}.png")
#     plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Loss curves plot saved to {loss_plot_path}")


# if __name__ == "__main__":
#     main()

