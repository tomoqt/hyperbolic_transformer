#!/usr/bin/env python
"""
Gradio app for real-time visualization of GPT model loop representations.
Shows PCA trajectories evolving as the model processes through loops.
"""

import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pickle
import os
import re
import time
from typing import List, Tuple, Iterator, Optional
import matplotlib.patches as mpatches
from threading import Lock

# Import model components
from model import GPTConfig, GPT

# Color palettes for visualization
PALETTES = [plt.cm.viridis, plt.cm.plasma, plt.cm.inferno, plt.cm.magma, 
            plt.cm.cividis, plt.cm.coolwarm, plt.cm.spring, plt.cm.autumn, 
            plt.cm.winter, plt.cm.summer]

class RealTimeTrajectoryVisualizer:
    def __init__(self):
        self.model = None
        self.tokenizer_encode = None
        self.tokenizer_decode_single = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pca_model = None
        self.trajectory_data = []
        self.current_loop_count = 0
        self.token_strings = []
        self.plot_lock = Lock()
        
    def load_model(self, checkpoint_path: str, max_loops_override: Optional[int] = None):
        """Load model from checkpoint"""
        try:
            print(f"🔄 Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print("✅ Checkpoint loaded from disk")
            
            if 'model_args' not in checkpoint:
                raise ValueError("'model_args' not found in checkpoint")
            
            print("🔄 Configuring model...")
            gpt_model_config = checkpoint['model_args']
            gpt_model_config['loops_representation'] = True
            gpt_model_config['automatic_loop_exit'] = False
            
            if max_loops_override is not None:
                gpt_model_config['max_loops'] = max_loops_override
            
            if 'effective_n_layer' not in gpt_model_config:
                gpt_model_config['effective_n_layer'] = None
                
            if 'loop_groups' not in gpt_model_config:
                gpt_model_config['loop_groups'] = []
            
            print("🔄 Creating model instance...")
            gptconf = GPTConfig(**gpt_model_config)
            self.model = GPT(gptconf)
            print("✅ Model instance created")
            
            print("🔄 Loading state dict...")
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
            self.model.load_state_dict(state_dict)
            print("✅ State dict loaded")
            
            print(f"🔄 Moving model to device: {self.device}")
            self.model.eval()
            # Move to device more carefully
            if self.device.type == 'cuda':
                # Check if CUDA is actually available
                if torch.cuda.is_available():
                    self.model = self.model.to(self.device)
                    print(f"✅ Model moved to {self.device}")
                else:
                    print("⚠️ CUDA requested but not available, using CPU")
                    self.device = torch.device('cpu')
                    self.model = self.model.to(self.device)
            else:
                self.model = self.model.to(self.device)
                print(f"✅ Model moved to {self.device}")
            
            result = f"✅ Model loaded successfully!\nDevice: {self.device}\nVocab: {self.model.config.vocab_size}, Block size: {self.model.config.block_size}, Layers: {self.model.config.n_layer}, Max loops: {self.model.config.max_loops}"
            if hasattr(self.model.config, 'loop_groups') and self.model.config.loop_groups:
                result += f"\nLoop groups: {self.model.config.loop_groups}"
            
            print("✅ Model loading completed successfully!")
            return result
            
        except Exception as e:
            error_msg = f"❌ Error loading model: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg
    
    def load_tokenizer(self, meta_path: str):
        """Load tokenizer from meta.pkl"""
        try:
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            
            if 'encode' in meta and callable(meta['encode']) and \
               'decode' in meta and callable(meta['decode']):
                self.tokenizer_encode = meta['encode']
                self.tokenizer_decode_single = lambda token_id: meta['decode']([token_id])
            else:
                if 'stoi' not in meta or 'itos' not in meta:
                    raise ValueError("Meta file missing required tokenizer components")
                
                stoi, itos = meta['stoi'], meta['itos']
                self.tokenizer_encode = lambda s: [stoi[c] for c in s if c in stoi]
                self.tokenizer_decode_single = lambda token_id: itos.get(token_id, '?')
            
            return "✅ Tokenizer loaded successfully!"
            
        except Exception as e:
            return f"❌ Error loading tokenizer: {str(e)}"
    
    def reset_trajectory_data(self):
        """Reset trajectory data for new run"""
        self.trajectory_data = []
        self.current_loop_count = 0
        self.pca_model = None
        
    def process_loop_representations_incremental(self, prompt: str, max_loops: int = 50, n_components: int = 2, step_size: int = 5):
        """Generator that shows each generated token's trajectory evolution in real-time"""
        if not self.model or not self.tokenizer_encode:
            yield "❌ Model or tokenizer not loaded", None, None
            return
        
        # Reset for new run
        self.reset_trajectory_data()
        
        # Tokenize prompt
        input_ids = self.tokenizer_encode(prompt)
        if not input_ids:
            yield "❌ Could not tokenize prompt", None, None
            return
        
        if len(input_ids) > self.model.config.block_size:
            input_ids = input_ids[:self.model.config.block_size]
        
        prompt_tokens = [self.tokenizer_decode_single(id_) for id_ in input_ids]
        yield f"🔄 Starting generation from prompt: {prompt_tokens}", None, None
        
        # Generate tokens one by one and show their loop trajectories
        current_sequence = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        num_tokens_to_generate = 5  # Generate 5 new tokens
        
        with torch.no_grad():
            try:
                for token_step in range(num_tokens_to_generate):
                    print(f"🔄 Generating token {token_step + 1}/{num_tokens_to_generate}...")
                    
                    # Generate one new token and get its loop representations
                    new_sequence, loop_representations_raw = self.model.generate(
                        current_sequence, max_new_tokens=1, return_first_step_loop_reps=True)
                    
                    if not loop_representations_raw:
                        yield f"❌ No loop representations for token {token_step + 1}", None, None
                        continue
                    
                    # Get the newly generated token
                    new_token_id = new_sequence[0, -1].item()
                    new_token_str = self.tokenizer_decode_single(new_token_id)
                    current_sequence = new_sequence  # Update sequence for next generation
                    
                    print(f"✅ Generated token: '{new_token_str}' (ID: {new_token_id})")
                    
                    # Convert loop representations to CPU
                    all_loop_reps = [loop_repr.squeeze(0).cpu() for loop_repr in loop_representations_raw]
                    total_loops = len(all_loop_reps)
                    print(f"🔍 DEBUG: Got {total_loops} loop representations")
                    
                    if total_loops > 0:
                        print(f"🔍 DEBUG: First rep shape: {all_loop_reps[0].shape}")
                    
                    # We want to track how the model's internal representations evolve as it processes
                    # the sequence to predict the next token. Let's look at the last position that gets
                    # the most "processing" - this is typically the last token in the sequence
                    if token_step == 0:
                        # For the first generated token, look at the last prompt token
                        focus_position = len(input_ids) - 1
                    else:
                        # For subsequent tokens, look at the most recently generated token
                        focus_position = len(input_ids) + token_step - 1
                    
                    print(f"🔍 DEBUG: Focusing on token at position: {focus_position} (token_step: {token_step})")
                    
                    # Show trajectory evolution at specific milestones: 1, 5, 10, 15, etc.
                    milestones = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
                    milestones = [m for m in milestones if m <= total_loops]
                    
                    if not milestones:
                        milestones = [min(total_loops, 1)]
                    
                    print(f"🔍 DEBUG: Milestones to process: {milestones}")
                    
                    for milestone in milestones:
                        print(f"🔍 DEBUG: Processing milestone {milestone}/{total_loops}")
                        
                        # Build trajectory data up to milestone
                        milestone_reps = all_loop_reps[:milestone]
                        
                        # Extract trajectory for the focus token
                        focus_token_trajectory = []
                        for i, rep in enumerate(milestone_reps):
                            print(f"🔍 DEBUG: Rep {i} shape: {rep.shape}, looking at pos: {focus_position}")
                            if focus_position < rep.shape[0]:  # Make sure token exists
                                focus_token_trajectory.append(rep[focus_position, :])
                                print(f"🔍 DEBUG: Added token rep with shape: {rep[focus_position, :].shape}")
                            else:
                                print(f"🔍 DEBUG: Token position {focus_position} >= sequence length {rep.shape[0]}")
                        
                        print(f"🔍 DEBUG: Collected {len(focus_token_trajectory)} trajectory points")
                        
                        if len(focus_token_trajectory) >= 2:
                            try:
                                # Convert to tensor for PCA
                                trajectory_tensor = torch.stack(focus_token_trajectory)
                                print(f"🔍 DEBUG: Trajectory tensor shape: {trajectory_tensor.shape}")
                                
                                # Update visualization data - trajectory_tensor has shape (milestone, hidden_dim)
                                # We need to convert this to the expected format: list of (seq_len, hidden_dim) tensors
                                self.trajectory_data = []
                                for loop_idx in range(milestone):
                                    # Each entry is one loop iteration with shape (1, hidden_dim) for our single token
                                    loop_data = trajectory_tensor[loop_idx].unsqueeze(0)  # Shape: (1, hidden_dim)
                                    self.trajectory_data.append(loop_data)
                                
                                self.current_loop_count = milestone
                                # Show which token we're actually tracking
                                if token_step == 0:
                                    focus_token_str = prompt_tokens[-1] if prompt_tokens else "?"
                                    display_str = f"Last prompt token: '{focus_token_str}'"
                                else:
                                    focus_token_str = new_token_str
                                    display_str = f"Generated token: '{focus_token_str}'"
                                self.token_strings = [display_str]
                                
                                print("🔍 DEBUG: Updating PCA...")
                                # Update PCA
                                self.update_pca(n_components)
                                
                                print("🔍 DEBUG: Creating plots...")
                                # Create plots focused on this single token with zoom on latest 5 loops
                                fig_2d, fig_3d = self.create_single_token_focused_plots(
                                    n_components, new_token_str, token_step + 1, milestone, total_loops)
                                
                                print(f"🔍 DEBUG: Plots created: 2D={fig_2d is not None}, 3D={fig_3d is not None}")
                                
                                status = f"🔄 Token {token_step + 1}/{num_tokens_to_generate}: '{new_token_str}' at loop {milestone}/{total_loops}"
                                yield status, fig_2d, fig_3d
                                
                                # Small delay between milestones
                                time.sleep(0.4)
                            except Exception as e:
                                error_msg = f"❌ Error processing milestone {milestone}: {str(e)}"
                                print(error_msg)
                                import traceback
                                traceback.print_exc()
                                yield error_msg, None, None
                        else:
                            msg = f"🔄 Token {token_step + 1}: '{new_token_str}' loop {milestone} (only {len(focus_token_trajectory)} points, need ≥2 for PCA)"
                            print(f"🔍 DEBUG: {msg}")
                            yield msg, None, None
                        
            except Exception as e:
                error_msg = f"❌ Error during generation: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                yield error_msg, None, None
                return
    
    def update_pca(self, n_components: int):
        """Update PCA model with current trajectory data"""
        print(f"🔍 DEBUG PCA: trajectory_data length: {len(self.trajectory_data)}")
        if len(self.trajectory_data) < 1:
            print("🔍 DEBUG PCA: No trajectory data")
            return
        
        # Stack all representations
        all_reps = torch.stack(self.trajectory_data).numpy()  # (num_loops, seq_len, hidden_dim)
        num_loops, seq_len, hidden_dim = all_reps.shape
        print(f"🔍 DEBUG PCA: all_reps shape: {all_reps.shape}")
        
        # Reshape for PCA
        data_for_pca = all_reps.reshape(-1, hidden_dim)
        print(f"🔍 DEBUG PCA: data_for_pca shape: {data_for_pca.shape}")
        
        # Check if we have enough data for PCA
        if data_for_pca.shape[0] < 2:
            print(f"🔍 DEBUG PCA: Not enough data points ({data_for_pca.shape[0]} < 2)")
            return
        
        # Fit PCA
        actual_n_components = min(n_components, data_for_pca.shape[0], data_for_pca.shape[1])
        print(f"🔍 DEBUG PCA: actual_n_components: {actual_n_components}")
        self.pca_model = PCA(n_components=actual_n_components)
        transformed_flat = self.pca_model.fit_transform(data_for_pca)
        print(f"🔍 DEBUG PCA: transformed_flat shape: {transformed_flat.shape}")
        
        # Reshape back
        self.transformed_data = transformed_flat.reshape(num_loops, seq_len, actual_n_components)
        print(f"🔍 DEBUG PCA: transformed_data shape: {self.transformed_data.shape}")
        print(f"🔍 DEBUG PCA: explained variance ratio: {self.pca_model.explained_variance_ratio_}")
        print("🔍 DEBUG PCA: PCA update completed successfully")
    
    def create_current_plots(self, n_components: int):
        """Create current 2D and 3D plots"""
        if self.pca_model is None:
            return None, None
        
        with self.plot_lock:
            fig_2d = self.plot_2d_trajectories() if n_components >= 2 else None
            fig_3d = self.plot_3d_trajectories() if n_components >= 3 else None
        
        return fig_2d, fig_3d
    
    def create_individual_token_plots(self, n_components: int):
        """Create plots focused on individual token trajectories"""
        if self.pca_model is None:
            return None, None
        
        with self.plot_lock:
            fig_2d = self.plot_individual_token_trajectories_2d() if n_components >= 2 else None
            fig_3d = self.plot_individual_token_trajectories_3d() if n_components >= 3 else None
        
        return fig_2d, fig_3d
    
    def create_single_token_focused_plots(self, n_components: int, token_str: str, token_num: int, current_loops: int, total_loops: int):
        """Create plots focused on a single token's trajectory with zoom on latest loops"""
        if self.pca_model is None:
            return None, None
        
        with self.plot_lock:
            fig_2d = self.plot_single_token_focused_2d(token_str, token_num, current_loops, total_loops) if n_components >= 2 else None
            fig_3d = self.plot_single_token_focused_3d(token_str, token_num, current_loops, total_loops) if n_components >= 3 else None
        
        return fig_2d, fig_3d
    
    def plot_2d_trajectories(self):
        """Create 2D trajectory plot"""
        if self.transformed_data.shape[2] < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 9))
        num_loops, seq_len, _ = self.transformed_data.shape
        
        # Get model configuration for loop groups
        num_loop_groups = 1
        total_model_loops = self.model.config.max_loops
        if hasattr(self.model.config, 'loop_groups') and self.model.config.loop_groups:
            num_loop_groups = len(self.model.config.loop_groups)
        
        # Color map for tokens
        token_cmap = plt.cm.get_cmap('tab10', seq_len if seq_len <= 10 else 20)
        
        for token_idx in range(seq_len):
            trajectory = self.transformed_data[:, token_idx, :2]
            token_label = self.token_strings[token_idx] if token_idx < len(self.token_strings) else f"Token {token_idx}"
            
            # Plot trajectory line
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   color=token_cmap(token_idx % token_cmap.N), 
                   alpha=0.4, linewidth=1.5, zorder=1)
            
            # Plot points with loop group coloring
            for loop_k in range(num_loops):
                point_coords = trajectory[loop_k, :]
                
                # Color by loop group
                group_idx = loop_k % num_loop_groups
                cmap_for_point = PALETTES[group_idx % len(PALETTES)]
                occurrence = loop_k // num_loop_groups
                max_occ_idx = (total_model_loops - 1) // num_loop_groups if total_model_loops > 0 else 0
                
                norm_val = occurrence / max(1, max_occ_idx) if max_occ_idx > 0 else 0.0
                point_color = cmap_for_point(norm_val)
                
                ax.scatter(point_coords[0], point_coords[1], 
                          color=point_color, s=25, alpha=0.8, zorder=3,
                          edgecolors='white', linewidth=0.5)
            
            # Mark start and current end
            if num_loops > 0:
                ax.scatter(trajectory[0, 0], trajectory[0, 1], 
                          color=token_cmap(token_idx % token_cmap.N), 
                          marker='X', s=100, ec='black', zorder=5, 
                          label=f"'{token_label}' start" if token_idx == 0 else "")
                if num_loops > 1:
                    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                              color=token_cmap(token_idx % token_cmap.N), 
                              marker='*', s=100, ec='black', zorder=5,
                              label=f"'{token_label}' current" if token_idx == 0 else "")
        
        # Add loop group information to title instead of legend for simplicity
        group_info = ""
        if num_loop_groups > 1:
            group_info = f", Loop groups: {num_loop_groups}"
        
        ax.set_title(f'Real-time 2D PCA Trajectories (Loop {self.current_loop_count})\n'
                    f'Model: {total_model_loops} max loops{group_info}, '
                    f'Explained variance: {np.sum(self.pca_model.explained_variance_ratio_[:2]):.3f}')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        
        # Main legend for tokens
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='black', linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_3d_trajectories(self):
        """Create 3D trajectory plot"""
        if self.transformed_data.shape[2] < 3:
            return None
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        num_loops, seq_len, _ = self.transformed_data.shape
        
        # Get model configuration for loop groups
        num_loop_groups = 1
        total_model_loops = self.model.config.max_loops
        if hasattr(self.model.config, 'loop_groups') and self.model.config.loop_groups:
            num_loop_groups = len(self.model.config.loop_groups)
        
        # Color map for tokens
        token_cmap = plt.cm.get_cmap('tab10', seq_len if seq_len <= 10 else 20)
        
        for token_idx in range(seq_len):
            trajectory = self.transformed_data[:, token_idx, :3]
            token_label = self.token_strings[token_idx] if token_idx < len(self.token_strings) else f"Token {token_idx}"
            
            # Plot trajectory line
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                   color=token_cmap(token_idx % token_cmap.N), 
                   alpha=0.4, linewidth=1.5, zorder=1)
            
            # Plot points with loop group coloring
            for loop_k in range(num_loops):
                point_coords = trajectory[loop_k, :]
                
                # Color by loop group
                group_idx = loop_k % num_loop_groups
                cmap_for_point = PALETTES[group_idx % len(PALETTES)]
                occurrence = loop_k // num_loop_groups
                max_occ_idx = (total_model_loops - 1) // num_loop_groups if total_model_loops > 0 else 0
                
                norm_val = occurrence / max(1, max_occ_idx) if max_occ_idx > 0 else 0.0
                point_color = cmap_for_point(norm_val)
                
                ax.scatter(point_coords[0], point_coords[1], point_coords[2], 
                          color=point_color, s=20, alpha=0.8, depthshade=True,
                          edgecolors='white', linewidth=0.3)
            
            # Mark start and current end
            if num_loops > 0:
                ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                          color=token_cmap(token_idx % token_cmap.N), 
                          marker='X', s=80, ec='black', depthshade=True,
                          label=f"'{token_label}' start" if token_idx == 0 else "")
                if num_loops > 1:
                    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                              color=token_cmap(token_idx % token_cmap.N), 
                              marker='*', s=80, ec='black', depthshade=True,
                              label=f"'{token_label}' current" if token_idx == 0 else "")
        
        ax.set_title(f'Real-time 3D PCA Trajectories (Loop {self.current_loop_count})\n'
                    f'Model: {total_model_loops} max loops, Groups: {num_loop_groups}, '
                    f'Explained variance: {np.sum(self.pca_model.explained_variance_ratio_[:3]):.3f}')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        plt.tight_layout()
        return fig
    
    def plot_individual_token_trajectories_2d(self):
        """Create 2D plot showing individual token trajectories evolving in real-time"""
        if self.transformed_data.shape[2] < 2:
            return None
        
        num_loops, seq_len, _ = self.transformed_data.shape
        
        # Create subplots for each token
        cols = min(3, seq_len)  # Max 3 columns
        rows = (seq_len + cols - 1) // cols  # Ceiling division
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if seq_len == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Get model configuration for loop groups
        num_loop_groups = 1
        total_model_loops = self.model.config.max_loops
        if hasattr(self.model.config, 'loop_groups') and self.model.config.loop_groups:
            num_loop_groups = len(self.model.config.loop_groups)
        
        for token_idx in range(seq_len):
            ax = axes[token_idx]
            trajectory = self.transformed_data[:, token_idx, :2]
            token_label = self.token_strings[token_idx] if token_idx < len(self.token_strings) else f"Token {token_idx}"
            
            # Plot trajectory line
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   color='blue', alpha=0.6, linewidth=2, zorder=1)
            
            # Plot points colored by loop progression
            for loop_k in range(num_loops):
                point_coords = trajectory[loop_k, :]
                
                # Color by loop progression (darker = later loops)
                color_intensity = loop_k / max(1, num_loops - 1)
                point_color = plt.cm.viridis(color_intensity)
                
                ax.scatter(point_coords[0], point_coords[1], 
                          color=point_color, s=30, alpha=0.8, zorder=3,
                          edgecolors='white', linewidth=0.5)
                
                # Add loop number annotation for every few points
                if loop_k % max(1, num_loops // 5) == 0 or loop_k == num_loops - 1:
                    ax.annotate(f'{loop_k+1}', (point_coords[0], point_coords[1]), 
                               xytext=(3, 3), textcoords='offset points', 
                               fontsize=8, alpha=0.7)
            
            # Mark start and current end
            if num_loops > 0:
                ax.scatter(trajectory[0, 0], trajectory[0, 1], 
                          color='green', marker='o', s=80, ec='black', zorder=5, 
                          label='Start')
                if num_loops > 1:
                    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                              color='red', marker='*', s=100, ec='black', zorder=5,
                              label='Current')
            
            ax.set_title(f"Token '{token_label}' (pos {token_idx})\nLoop {self.current_loop_count}", fontsize=10)
            ax.set_xlabel('PC1', fontsize=8)
            ax.set_ylabel('PC2', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            if token_idx == 0:  # Only show legend on first subplot
                ax.legend(fontsize=8)
        
        # Hide unused subplots
        for i in range(seq_len, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(f'Individual Token Trajectories - Loop {self.current_loop_count}\n'
                    f'PCA Explained Variance: {np.sum(self.pca_model.explained_variance_ratio_[:2]):.3f}', 
                    fontsize=12)
        plt.tight_layout()
        return fig
    
    def plot_individual_token_trajectories_3d(self):
        """Create 3D plot showing individual token trajectories evolving in real-time"""
        if self.transformed_data.shape[2] < 3:
            return None
        
        num_loops, seq_len, _ = self.transformed_data.shape
        
        # Create subplots for each token
        cols = min(2, seq_len)  # Max 2 columns for 3D plots
        rows = (seq_len + cols - 1) // cols
        
        fig = plt.figure(figsize=(8*cols, 6*rows))
        
        for token_idx in range(seq_len):
            ax = fig.add_subplot(rows, cols, token_idx + 1, projection='3d')
            trajectory = self.transformed_data[:, token_idx, :3]
            token_label = self.token_strings[token_idx] if token_idx < len(self.token_strings) else f"Token {token_idx}"
            
            # Plot trajectory line
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                   color='blue', alpha=0.6, linewidth=2, zorder=1)
            
            # Plot points colored by loop progression
            for loop_k in range(num_loops):
                point_coords = trajectory[loop_k, :]
                
                # Color by loop progression
                color_intensity = loop_k / max(1, num_loops - 1)
                point_color = plt.cm.viridis(color_intensity)
                
                ax.scatter(point_coords[0], point_coords[1], point_coords[2],
                          color=point_color, s=25, alpha=0.8, depthshade=True)
            
            # Mark start and current end
            if num_loops > 0:
                ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                          color='green', marker='o', s=60, ec='black', depthshade=True,
                          label='Start')
                if num_loops > 1:
                    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                              color='red', marker='*', s=80, ec='black', depthshade=True,
                              label='Current')
            
            ax.set_title(f"Token '{token_label}' (pos {token_idx})\nLoop {self.current_loop_count}", fontsize=10)
            ax.set_xlabel('PC1', fontsize=8)
            ax.set_ylabel('PC2', fontsize=8)
            ax.set_zlabel('PC3', fontsize=8)
            
            if token_idx == 0:  # Only show legend on first subplot
                ax.legend(fontsize=8)
        
        fig.suptitle(f'Individual Token 3D Trajectories - Loop {self.current_loop_count}\n'
                    f'PCA Explained Variance: {np.sum(self.pca_model.explained_variance_ratio_[:3]):.3f}', 
                    fontsize=12)
        plt.tight_layout()
        return fig
    
    def plot_single_token_focused_2d(self, token_str: str, token_num: int, current_loops: int, total_loops: int):
        """Create 2D plot focused on a single token with latest 5 loops zoomed"""
        print(f"🔍 DEBUG PLOT 2D: transformed_data shape: {self.transformed_data.shape if hasattr(self, 'transformed_data') else 'None'}")
        if not hasattr(self, 'transformed_data') or self.transformed_data.shape[2] < 2 or self.transformed_data.shape[1] == 0:
            print("🔍 DEBUG PLOT 2D: Insufficient data for 2D plot")
            return None
        
        # We have data for a single token
        trajectory = self.transformed_data[:, 0, :2]  # Shape: (loops, 2)
        num_loops = trajectory.shape[0]
        
        fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Main plot - full trajectory
        ax_main.plot(trajectory[:, 0], trajectory[:, 1], 
                    color='blue', alpha=0.7, linewidth=2.5, zorder=1)
        
        # Plot points colored by loop progression
        for loop_k in range(num_loops):
            point_coords = trajectory[loop_k, :]
            color_intensity = loop_k / max(1, num_loops - 1)
            point_color = plt.cm.viridis(color_intensity)
            
            ax_main.scatter(point_coords[0], point_coords[1], 
                          color=point_color, s=40, alpha=0.8, zorder=3,
                          edgecolors='white', linewidth=0.8)
            
            # Annotate every few points
            if loop_k % max(1, num_loops // 8) == 0 or loop_k == num_loops - 1:
                ax_main.annotate(f'{loop_k+1}', (point_coords[0], point_coords[1]), 
                               xytext=(4, 4), textcoords='offset points', 
                               fontsize=9, alpha=0.8, weight='bold')
        
        # Mark start and end
        if num_loops > 0:
            ax_main.scatter(trajectory[0, 0], trajectory[0, 1], 
                          color='green', marker='o', s=100, ec='black', zorder=5, 
                          label='Start')
            if num_loops > 1:
                ax_main.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                              color='red', marker='*', s=120, ec='black', zorder=5,
                              label='Current')
        
        ax_main.set_title(f"Full Trajectory: Token #{token_num} '{token_str}'\nLoops 1-{current_loops} of {total_loops}", fontsize=12)
        ax_main.set_xlabel('Principal Component 1', fontsize=10)
        ax_main.set_ylabel('Principal Component 2', fontsize=10)
        ax_main.grid(True, alpha=0.3)
        ax_main.legend()
        
        # Zoom plot - latest 5 loops
        zoom_start = max(0, num_loops - 5)
        if num_loops > 1 and zoom_start < num_loops:
            zoom_trajectory = trajectory[zoom_start:, :]
            zoom_loops = num_loops - zoom_start
            
            ax_zoom.plot(zoom_trajectory[:, 0], zoom_trajectory[:, 1], 
                        color='red', alpha=0.8, linewidth=3, zorder=1)
            
            for i in range(zoom_loops):
                loop_k = zoom_start + i
                point_coords = zoom_trajectory[i, :]
                color_intensity = loop_k / max(1, num_loops - 1)
                point_color = plt.cm.viridis(color_intensity)
                
                ax_zoom.scatter(point_coords[0], point_coords[1], 
                              color=point_color, s=60, alpha=0.9, zorder=3,
                              edgecolors='white', linewidth=1)
                
                # Annotate all points in zoom
                ax_zoom.annotate(f'{loop_k+1}', (point_coords[0], point_coords[1]), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=10, alpha=0.9, weight='bold')
            
            # Mark end in zoom
            ax_zoom.scatter(zoom_trajectory[-1, 0], zoom_trajectory[-1, 1], 
                          color='red', marker='*', s=140, ec='black', zorder=5)
            
            ax_zoom.set_title(f"Zoomed: Latest {zoom_loops} Loops\n(Loops {zoom_start+1}-{current_loops})", fontsize=12)
        else:
            ax_zoom.text(0.5, 0.5, 'Need more loops\nfor zoom view', 
                        ha='center', va='center', transform=ax_zoom.transAxes, fontsize=12)
            ax_zoom.set_title("Zoom View (Need ≥2 loops)", fontsize=12)
        
        ax_zoom.set_xlabel('Principal Component 1', fontsize=10)
        ax_zoom.set_ylabel('Principal Component 2', fontsize=10)
        ax_zoom.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_single_token_focused_3d(self, token_str: str, token_num: int, current_loops: int, total_loops: int):
        """Create 3D plot focused on a single token with latest 5 loops zoomed"""
        print(f"🔍 DEBUG PLOT 3D: transformed_data shape: {self.transformed_data.shape if hasattr(self, 'transformed_data') else 'None'}")
        if not hasattr(self, 'transformed_data') or self.transformed_data.shape[2] < 3 or self.transformed_data.shape[1] == 0:
            print("🔍 DEBUG PLOT 3D: Insufficient data for 3D plot")
            return None
        
        trajectory = self.transformed_data[:, 0, :3]  # Shape: (loops, 3)
        num_loops = trajectory.shape[0]
        
        fig = plt.figure(figsize=(16, 7))
        
        # Main 3D plot
        ax_main = fig.add_subplot(121, projection='3d')
        ax_main.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                    color='blue', alpha=0.7, linewidth=2.5, zorder=1)
        
        # Plot points
        for loop_k in range(num_loops):
            point_coords = trajectory[loop_k, :]
            color_intensity = loop_k / max(1, num_loops - 1)
            point_color = plt.cm.viridis(color_intensity)
            
            ax_main.scatter(point_coords[0], point_coords[1], point_coords[2],
                          color=point_color, s=35, alpha=0.8, depthshade=True)
        
        # Mark start and end
        if num_loops > 0:
            ax_main.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                          color='green', marker='o', s=80, ec='black', depthshade=True,
                          label='Start')
            if num_loops > 1:
                ax_main.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                              color='red', marker='*', s=100, ec='black', depthshade=True,
                              label='Current')
        
        ax_main.set_title(f"Full 3D: Token #{token_num} '{token_str}'\nLoops 1-{current_loops} of {total_loops}", fontsize=11)
        ax_main.set_xlabel('PC1', fontsize=9)
        ax_main.set_ylabel('PC2', fontsize=9)
        ax_main.set_zlabel('PC3', fontsize=9)
        ax_main.legend()
        
        # Zoom 3D plot
        ax_zoom = fig.add_subplot(122, projection='3d')
        zoom_start = max(0, num_loops - 5)
        if num_loops > 1 and zoom_start < num_loops:
            zoom_trajectory = trajectory[zoom_start:, :]
            zoom_loops = num_loops - zoom_start
            
            ax_zoom.plot(zoom_trajectory[:, 0], zoom_trajectory[:, 1], zoom_trajectory[:, 2],
                        color='red', alpha=0.8, linewidth=3, zorder=1)
            
            for i in range(zoom_loops):
                loop_k = zoom_start + i
                point_coords = zoom_trajectory[i, :]
                color_intensity = loop_k / max(1, num_loops - 1)
                point_color = plt.cm.viridis(color_intensity)
                
                ax_zoom.scatter(point_coords[0], point_coords[1], point_coords[2],
                              color=point_color, s=50, alpha=0.9, depthshade=True)
            
            # Mark end in zoom
            ax_zoom.scatter(zoom_trajectory[-1, 0], zoom_trajectory[-1, 1], zoom_trajectory[-1, 2],
                          color='red', marker='*', s=120, ec='black', depthshade=True)
            
            ax_zoom.set_title(f"Zoomed: Latest {zoom_loops} Loops\n(Loops {zoom_start+1}-{current_loops})", fontsize=11)
        else:
            ax_zoom.text(0.5, 0.5, 0.5, 'Need more loops\nfor zoom view', 
                        ha='center', va='center', fontsize=12)
            ax_zoom.set_title("Zoom View (Need ≥2 loops)", fontsize=11)
        
        ax_zoom.set_xlabel('PC1', fontsize=9)
        ax_zoom.set_ylabel('PC2', fontsize=9)
        ax_zoom.set_zlabel('PC3', fontsize=9)
        
        plt.tight_layout()
        return fig

# Global visualizer instance
visualizer = RealTimeTrajectoryVisualizer()

def load_model_interface(checkpoint_path, max_loops_override):
    """Interface function for loading model"""
    max_loops = int(max_loops_override) if max_loops_override else None
    return visualizer.load_model(checkpoint_path, max_loops)

def load_tokenizer_interface(meta_path):
    """Interface function for loading tokenizer"""
    return visualizer.load_tokenizer(meta_path)

def generate_trajectories_interface(prompt, max_loops, n_components, step_size):
    """Interface function for generating trajectories without progress bars"""
    max_loops = int(max_loops)
    n_components = int(n_components)
    step_size = int(step_size)
    
    last_status = ""
    last_fig_2d = None
    last_fig_3d = None
    
    for status, fig_2d, fig_3d in visualizer.process_loop_representations_incremental(prompt, max_loops, n_components, step_size):
        # Close previous figures to avoid memory issues
        if last_fig_2d is not None and fig_2d is not None:
            plt.close(last_fig_2d)
        if last_fig_3d is not None and fig_3d is not None:
            plt.close(last_fig_3d)
        
        last_status = status
        if fig_2d is not None:
            last_fig_2d = fig_2d
        if fig_3d is not None:
            last_fig_3d = fig_3d
        
        # Yield intermediate results for real-time updates
        yield last_status, last_fig_2d, last_fig_3d
    
    # Final result
    yield last_status, last_fig_2d, last_fig_3d

# Create Gradio interface
def create_gradio_app():
    with gr.Blocks(title="Real-time Loop Trajectory Visualizer", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🔄 Real-time GPT Loop Trajectory Visualizer
        
        This app shows how each **newly generated token's** representation evolves through model loops in real-time.
        Watch individual tokens move through latent space at key loop milestones (1, 5, 10, 15...)!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Model Setup")
                
                checkpoint_path = gr.Textbox(
                    label="Model Checkpoint Path",
                    placeholder="/path/to/model.pt",
                    value="out_looped/ckpt.pt"
                )
                
                max_loops_override = gr.Number(
                    label="Max Loops Override (optional)",
                    value=None,
                    info="Leave empty to use model default"
                )
                
                load_model_btn = gr.Button("🔄 Load Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", interactive=False)
                
                meta_path = gr.Textbox(
                    label="Tokenizer Meta Path",
                    value="data/fineweb/meta.pkl"
                )
                
                load_tokenizer_btn = gr.Button("🔄 Load Tokenizer")
                tokenizer_status = gr.Textbox(label="Tokenizer Status", interactive=False)
                
                gr.Markdown("### ⚙️ Generation Settings")
                
                prompt = gr.Textbox(
                    label="Input Prompt",
                    value="Hello world, this is a test.",
                    lines=3
                )
                
                with gr.Row():
                    max_loops = gr.Slider(
                        label="Max Loops",
                        minimum=5,
                        maximum=100,
                        value=30,
                        step=1
                    )
                    
                    n_components = gr.Slider(
                        label="PCA Components",
                        minimum=2,
                        maximum=3,
                        value=3,
                        step=1
                    )
                
                with gr.Row():
                    step_size = gr.Slider(
                        label="Loop Step Size",
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        info="Show trajectories in incremental steps of this size"
                    )
                
                generate_btn = gr.Button("🚀 Start Real-time Generation", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Real-time Visualization")
                
                status_output = gr.Textbox(label="Generation Status", interactive=False)
                
                with gr.Tab("2D Token Trajectory"):
                    plot_2d = gr.Plot(label="2D PCA Token Trajectory")
                
                with gr.Tab("3D Token Trajectory"):
                    plot_3d = gr.Plot(label="3D PCA Token Trajectory")
        
        gr.Markdown("""
        ### 📖 How to Use:
        1. **Load Model**: Specify path to your trained GPT model checkpoint
        2. **Load Tokenizer**: Point to the meta.pkl file for tokenization
        3. **Set Parameters**: Choose your prompt and visualization settings
        4. **Generate**: Watch each generated token's trajectory evolve through loop milestones
        
        ### 🎨 Visualization Features:
        - **Per-Token Focus**: Each generated token shown individually
        - **Milestone Progression**: View trajectories at loops 1, 5, 10, 15, etc.
        - **Dual View**: Full trajectory + zoomed view of latest 5 loops
        - **Real-time Evolution**: See how representations move through latent space
        - **Loop Annotations**: Loop numbers marked on trajectory points
        """)
        
        # Event handlers
        load_model_btn.click(
            load_model_interface,
            inputs=[checkpoint_path, max_loops_override],
            outputs=[model_status]
        )
        
        load_tokenizer_btn.click(
            load_tokenizer_interface,
            inputs=[meta_path],
            outputs=[tokenizer_status]
        )
        
        generate_btn.click(
            generate_trajectories_interface,
            inputs=[prompt, max_loops, n_components, step_size],
            outputs=[status_output, plot_2d, plot_3d]
        )
    
    return app

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    ) 