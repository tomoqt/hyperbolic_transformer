import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.svd_entropy import calculate_model_svd_entropy, calculate_svd_entropy
import seaborn as sns

# Import necessary classes from both model files
from model import GPTConfig as HyperbolicGPTConfig, GPT as HyperbolicGPT
import sys
import os

# Add the Shakespeare-char directory to system path if needed
# Assuming it's in the parent directory
shakespeare_dir = os.path.abspath(".")
if shakespeare_dir not in sys.path:
    sys.path.append(shakespeare_dir)

# Import the baseline model
from model_baseline import GPTConfig as BaselineGPTConfig, GPT as BaselineGPT

def load_model(model_dir, is_hyperbolic=True):
    """
    Load a model from checkpoint file
    
    Args:
        model_dir: Directory containing the checkpoint
        is_hyperbolic: Whether the model is hyperbolic (True) or baseline (False)
    """
    ckpt_path = os.path.join(model_dir, 'ckpt.pt')
    print(f"Loading checkpoint from {ckpt_path}")
    
    # Load with appropriate device mapping
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Select appropriate model class based on type
    if is_hyperbolic:
        gptconf = HyperbolicGPTConfig(**checkpoint['model_args'])
        model = HyperbolicGPT(gptconf)
    else:
        # For baseline model, we need to filter out hyperbolic-specific config args
        model_args = checkpoint['model_args'].copy()
        # Remove hyperbolic-specific parameters if they exist
        hyperbolic_params = ['curvature_mode', 'curvature', 'curvature_initialization', 'map_back_after_attention']
        for param in hyperbolic_params:
            if param in model_args:
                del model_args[param]
        
        gptconf = BaselineGPTConfig(**model_args)
        model = BaselineGPT(gptconf)
    
    # Handle potential unwanted prefix in state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    # Try loading the state dict, handling potential mismatches
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning: Error loading state dict: {e}")
        print("Attempting to load with strict=False")
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()  # Set to evaluation mode
    return model

def analyze_layer_entropy(model, model_name):
    """
    Analyze SVD entropy for different layer types in the model
    Returns a dictionary with layer types and their entropy values
    """
    layer_entropies = {}
    for name, param in model.named_parameters():
        if param.dim() > 1:  # Only analyze matrices
            entropy = calculate_svd_entropy(param)
            # Extract layer type from parameter name
            if 'wte' in name or 'wpe' in name:
                layer_type = 'embedding'
            elif 'attn' in name:
                if 'c_attn' in name:
                    layer_type = 'attention_qkv'
                elif 'c_proj' in name:
                    layer_type = 'attention_proj'
                else:
                    layer_type = 'attention_other'
            elif 'mlp' in name:
                if 'c_fc' in name:
                    layer_type = 'mlp_fc'
                elif 'c_proj' in name:
                    layer_type = 'mlp_proj'
                else:
                    layer_type = 'mlp_other'
            else:
                layer_type = 'other'
                
            if layer_type not in layer_entropies:
                layer_entropies[layer_type] = []
            layer_entropies[layer_type].append((name, entropy.item()))
    
    return layer_entropies

def create_bar_plot(data, keys, title, ylabel, filename, output_dir, is_log=False, bar_colors=None):
    """Helper function to create bar plots with consistent styling in both linear and log scales"""
    plt.figure(figsize=(10, 6))
    
    # Ensure we have bar colors
    if bar_colors is None:
        bar_colors = ['blue', 'orange']
    
    # Ensure data is non-zero for log scale
    if is_log:
        # Add a small epsilon to zero values to avoid log(0)
        plot_data = np.array([max(val, 1e-10) for val in data])
    else:
        plot_data = data
        
    bars = plt.bar(keys, plot_data, color=bar_colors[:len(keys)])
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        # Format based on scale
        if is_log:
            value_text = f'{height:.2e}'
        else:
            value_text = f'{height:.4f}'
        
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                value_text, ha='center', va='bottom', fontsize=9)
    
    # Set y-axis to log scale if requested
    if is_log:
        plt.yscale('log')
        title = f"{title} (Log Scale)"
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    # Save figure
    scale_suffix = "_log" if is_log else ""
    plt.savefig(os.path.join(output_dir, f'{filename}{scale_suffix}.png'), dpi=300)
    plt.close()

def create_grouped_bar_plot(data_list, model_names, categories, title, ylabel, filename, output_dir, is_log=False):
    """Helper function to create grouped bar plots with consistent styling"""
    plt.figure(figsize=(max(12, len(categories)), 7))
    
    x = np.arange(len(categories))
    width = 0.35
    
    for i, (model_name, data) in enumerate(zip(model_names, data_list)):
        # For log scale, ensure non-zero values
        if is_log:
            plot_data = [max(val, 1e-10) for val in data]
        else:
            plot_data = data
            
        offset = width * (i - 0.5)
        plt.bar(x + offset, plot_data, width, label=model_name)
    
    # Set y-axis to log scale if requested
    if is_log:
        plt.yscale('log')
        title = f"{title} (Log Scale)"
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    scale_suffix = "_log" if is_log else ""
    plt.savefig(os.path.join(output_dir, f'{filename}{scale_suffix}.png'), dpi=300)
    plt.close()

def create_violin_plot(data_list, model_names, title, ylabel, filename, output_dir, is_log=False):
    """Helper function to create violin plots with consistent styling"""
    plt.figure(figsize=(10, 6))
    
    # For log scale, ensure all values are positive
    if is_log:
        plot_data = [[max(val, 1e-10) for val in data] for data in data_list]
        plt.yscale('log')
        title = f"{title} (Log Scale)"
    else:
        plot_data = data_list
    
    # Create violin plot
    plt.violinplot(plot_data, showmeans=True, showmedians=True)
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(1, len(model_names) + 1), model_names)
    plt.tight_layout()
    
    # Save figure
    scale_suffix = "_log" if is_log else ""
    plt.savefig(os.path.join(output_dir, f'{filename}{scale_suffix}.png'), dpi=300)
    plt.close()

def visualize_svd_entropy():
    # Create output directory if it doesn't exist
    output_dir = "svd entropy visualizations"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to directory: {output_dir}")
    
    # Model paths and their types
    model_configs = {
        'Standard': {'path': 'out-shakespeare-char', 'is_hyperbolic': False},
        'Hyperbolic': {'path': 'Hyperbolic-out-shakespeare-char', 'is_hyperbolic': True}
    }
    
    models = {}
    avg_entropies = {}
    layer_entropies = {}
    
    # Load models and calculate entropies
    for model_name, config in model_configs.items():
        print(f"Loading {model_name} model from {config['path']}...")
        models[model_name] = load_model(config['path'], is_hyperbolic=config['is_hyperbolic'])
        
        # Calculate average SVD entropy
        avg_entropies[model_name] = calculate_model_svd_entropy(models[model_name], average=True).item()
        
        # Calculate layer-wise entropies
        layer_entropies[model_name] = analyze_layer_entropy(models[model_name], model_name)
    
    # Set up the visualization style
    sns.set(style="whitegrid")
    model_names = list(model_configs.keys())
    
    # 1. Overall average SVD entropy comparison - Linear and Log Scale
    create_bar_plot(
        data=list(avg_entropies.values()),
        keys=list(avg_entropies.keys()),
        title='Average SVD Entropy Comparison',
        ylabel='SVD Entropy',
        filename='average_entropy_comparison',
        output_dir=output_dir,
        is_log=False
    )
    
    # Create log-scale version
    create_bar_plot(
        data=list(avg_entropies.values()),
        keys=list(avg_entropies.keys()),
        title='Average SVD Entropy Comparison',
        ylabel='SVD Entropy (log scale)',
        filename='average_entropy_comparison',
        output_dir=output_dir,
        is_log=True
    )
    
    # 2. Layer type comparison - Linear and Log Scale
    # Prepare data for grouped bar chart
    layer_types = set()
    for model_name in layer_entropies:
        layer_types.update(layer_entropies[model_name].keys())
    
    layer_types = sorted(layer_types)
    
    # Calculate mean entropy for each layer type and model
    layer_type_means = []
    for model_name in model_names:
        means = []
        for layer_type in layer_types:
            if layer_type in layer_entropies[model_name]:
                values = [v for _, v in layer_entropies[model_name][layer_type]]
                means.append(np.mean(values))
            else:
                means.append(0)
        layer_type_means.append(means)
    
    # Create linear and log scale plots
    create_grouped_bar_plot(
        data_list=layer_type_means,
        model_names=model_names,
        categories=layer_types,
        title='Average SVD Entropy by Layer Type',
        ylabel='SVD Entropy',
        filename='layer_type_entropy',
        output_dir=output_dir,
        is_log=False
    )
    
    create_grouped_bar_plot(
        data_list=layer_type_means,
        model_names=model_names,
        categories=layer_types,
        title='Average SVD Entropy by Layer Type',
        ylabel='SVD Entropy (log scale)',
        filename='layer_type_entropy',
        output_dir=output_dir,
        is_log=True
    )
    
    # 3. SVD Entropy distribution - Linear and Log Scale
    # Collect all entropy values for both models
    all_entropies = []
    for model_name in model_names:
        model_values = []
        for layer_type in layer_entropies[model_name]:
            model_values.extend([v for _, v in layer_entropies[model_name][layer_type]])
        all_entropies.append(model_values)
    
    # Create violin plots
    create_violin_plot(
        data_list=all_entropies,
        model_names=model_names,
        title='Distribution of SVD Entropy Values',
        ylabel='SVD Entropy',
        filename='entropy_distribution',
        output_dir=output_dir,
        is_log=False
    )
    
    create_violin_plot(
        data_list=all_entropies,
        model_names=model_names,
        title='Distribution of SVD Entropy Values',
        ylabel='SVD Entropy (log scale)',
        filename='entropy_distribution',
        output_dir=output_dir,
        is_log=True
    )
    
    # 4. Create per-layer comparison for each model - Linear and Log Scale
    for model_name in model_names:
        # Extract per-layer entropy values
        layer_data = {}
        for param_name, entropy_value in [(name, val) for layer_type in layer_entropies[model_name] 
                                        for name, val in layer_entropies[model_name][layer_type]]:
            # Extract layer number if possible
            if '.h.' in param_name:
                layer_num = int(param_name.split('.h.')[1].split('.')[0])
                if layer_num not in layer_data:
                    layer_data[layer_num] = []
                layer_data[layer_num].append(entropy_value)
        
        if layer_data:  # Only create plot if we have layer data
            # Calculate average entropy per layer
            layers = sorted(layer_data.keys())
            layer_means = [np.mean(layer_data[layer]) for layer in layers]
            
            # Linear scale
            plt.figure(figsize=(14, 8))
            plt.bar(layers, layer_means, alpha=0.7)
            plt.title(f'{model_name} Model: Average SVD Entropy by Layer')
            plt.xlabel('Layer Number')
            plt.ylabel('Average SVD Entropy')
            plt.xticks(layers)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name.lower()}_layer_entropy.png'), dpi=300)
            plt.close()
            
            # Log scale
            plt.figure(figsize=(14, 8))
            plt.bar(layers, layer_means, alpha=0.7)
            plt.title(f'{model_name} Model: Average SVD Entropy by Layer (Log Scale)')
            plt.xlabel('Layer Number')
            plt.ylabel('Average SVD Entropy (log scale)')
            plt.xticks(layers)
            plt.yscale('log')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name.lower()}_layer_entropy_log.png'), dpi=300)
            plt.close()
    
    # 5. Direct comparison of layer-wise entropy between models
    # Only proceed if both models have layer data
    if 'Standard' in models and 'Hyperbolic' in models:
        # Extract layers from both models
        std_layers = []
        std_means = []
        hyp_layers = []
        hyp_means = []
        
        # Process standard model
        for param_name, entropy_value in [(name, val) for layer_type in layer_entropies['Standard'] 
                                       for name, val in layer_entropies['Standard'][layer_type]]:
            if '.h.' in param_name:
                layer_num = int(param_name.split('.h.')[1].split('.')[0])
                if layer_num not in std_layers:
                    std_layers.append(layer_num)
                    std_values = [v for n, v in [(name, val) for layer_type in layer_entropies['Standard'] 
                                              for name, val in layer_entropies['Standard'][layer_type]]
                               if f'.h.{layer_num}.' in n]
                    std_means.append(np.mean(std_values))
        
        # Process hyperbolic model
        for param_name, entropy_value in [(name, val) for layer_type in layer_entropies['Hyperbolic'] 
                                       for name, val in layer_entropies['Hyperbolic'][layer_type]]:
            if '.h.' in param_name:
                layer_num = int(param_name.split('.h.')[1].split('.')[0])
                if layer_num not in hyp_layers:
                    hyp_layers.append(layer_num)
                    hyp_values = [v for n, v in [(name, val) for layer_type in layer_entropies['Hyperbolic'] 
                                              for name, val in layer_entropies['Hyperbolic'][layer_type]]
                               if f'.h.{layer_num}.' in n]
                    hyp_means.append(np.mean(hyp_values))
        
        # Create side-by-side layer comparison
        if std_layers and hyp_layers:
            # Linear scale
            plt.figure(figsize=(14, 8))
            
            x = np.arange(len(std_layers))
            width = 0.35
            
            plt.bar(x - width/2, std_means, width, label='Standard', alpha=0.7, color='blue')
            plt.bar(x + width/2, hyp_means, width, label='Hyperbolic', alpha=0.7, color='orange')
            
            plt.title('Layer-wise SVD Entropy Comparison Between Models')
            plt.xlabel('Layer Number')
            plt.ylabel('Average SVD Entropy')
            plt.xticks(x, std_layers)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'layer_comparison.png'), dpi=300)
            plt.close()
            
            # Log scale
            plt.figure(figsize=(14, 8))
            
            plt.bar(x - width/2, std_means, width, label='Standard', alpha=0.7, color='blue')
            plt.bar(x + width/2, hyp_means, width, label='Hyperbolic', alpha=0.7, color='orange')
            
            plt.title('Layer-wise SVD Entropy Comparison Between Models (Log Scale)')
            plt.xlabel('Layer Number')
            plt.ylabel('Average SVD Entropy (log scale)')
            plt.xticks(x, std_layers)
            plt.yscale('log')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'layer_comparison_log.png'), dpi=300)
            plt.close()
            
            # 6. Calculate and plot entropy ratio (Hyperbolic/Standard)
            if len(std_means) == len(hyp_means):
                # Calculate ratios, handling division by zero
                entropy_ratios = []
                for std, hyp in zip(std_means, hyp_means):
                    if std > 0:
                        entropy_ratios.append(hyp / std)
                    else:
                        entropy_ratios.append(1.0)  # Default to 1 if standard is zero
                
                # Plot entropy ratios
                plt.figure(figsize=(14, 8))
                plt.bar(x, entropy_ratios, alpha=0.7, color='green')
                plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
                
                plt.title('Entropy Ratio (Hyperbolic/Standard) by Layer')
                plt.xlabel('Layer Number')
                plt.ylabel('Entropy Ratio')
                plt.xticks(x, std_layers)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'entropy_ratio.png'), dpi=300)
                plt.close()
    
    # Print summary statistics
    summary_file = os.path.join(output_dir, 'summary_statistics.txt')
    with open(summary_file, 'w') as f:
        f.write("Summary Statistics:\n")
        for model_name in model_names:
            f.write(f"\n{model_name} Model:\n")
            f.write(f"  Overall Average SVD Entropy: {avg_entropies[model_name]:.4f} (log: {np.log10(avg_entropies[model_name]):.4f})\n")
            f.write("  Layer Type Average SVD Entropy:\n")
            for layer_type in sorted(layer_entropies[model_name].keys()):
                values = [v for _, v in layer_entropies[model_name][layer_type]]
                mean_value = np.mean(values)
                f.write(f"    {layer_type}: {mean_value:.4f} (log: {np.log10(max(mean_value, 1e-10)):.4f}) from {len(values)} parameters\n")
            
            # Add per-layer statistics if available
            if model_name in models and hasattr(models[model_name].config, 'n_layer'):
                f.write("\n  Layer-wise Average SVD Entropy:\n")
                for layer_num in range(models[model_name].config.n_layer):
                    layer_params = [v for name, v in [(name, val) for layer_type in layer_entropies[model_name] 
                                                    for name, val in layer_entropies[model_name][layer_type]]
                                  if f'.h.{layer_num}.' in name]
                    if layer_params:
                        mean_value = np.mean(layer_params)
                        f.write(f"    Layer {layer_num}: {mean_value:.4f} (log: {np.log10(max(mean_value, 1e-10)):.4f})\n")
        
        # Add comparison statistics
        if 'Standard' in avg_entropies and 'Hyperbolic' in avg_entropies:
            f.write("\nComparison Statistics:\n")
            ratio = avg_entropies['Hyperbolic'] / avg_entropies['Standard']
            log_diff = np.log10(max(avg_entropies['Hyperbolic'], 1e-10)) - np.log10(max(avg_entropies['Standard'], 1e-10))
            f.write(f"  Overall Entropy Ratio (Hyperbolic/Standard): {ratio:.4f}\n")
            f.write(f"  Log10 Difference (Hyperbolic - Standard): {log_diff:.4f}\n")
            f.write(f"  Percent Difference: {(ratio - 1) * 100:.2f}%\n")
            
        # Add a separate section for log-scale summary
        f.write("\n\nLOG SCALE SUMMARY:\n")
        f.write("==================\n")
        for model_name in model_names:
            f.write(f"\n{model_name} Model (Log10 Values):\n")
            f.write(f"  Overall Average SVD Entropy (log10): {np.log10(max(avg_entropies[model_name], 1e-10)):.4f}\n")
            
            # Group by order of magnitude
            log_values = {layer_type: np.log10(max(np.mean([v for _, v in values]), 1e-10)) 
                          for layer_type, values in layer_entropies[model_name].items()}
            
            # Sort by log value (ascending order)
            sorted_log_values = sorted(log_values.items(), key=lambda x: x[1])
            
            f.write("  Layer Types by Order of Magnitude (ascending):\n")
            for layer_type, log_val in sorted_log_values:
                values = [v for _, v in layer_entropies[model_name][layer_type]]
                f.write(f"    {layer_type}: {log_val:.4f} (from {len(values)} parameters)\n")
    
    # Also print to console
    print("\nSummary Statistics:")
    for model_name in model_names:
        print(f"\n{model_name} Model:")
        print(f"  Overall Average SVD Entropy: {avg_entropies[model_name]:.4f} (log: {np.log10(avg_entropies[model_name]):.4f})")
        print("  Layer Type Average SVD Entropy:")
        for layer_type in sorted(layer_entropies[model_name].keys()):
            values = [v for _, v in layer_entropies[model_name][layer_type]]
            mean_value = np.mean(values)
            print(f"    {layer_type}: {mean_value:.4f} (log: {np.log10(max(mean_value, 1e-10)):.4f}) from {len(values)} parameters")
    
    # Print comparison statistics to console
    if 'Standard' in avg_entropies and 'Hyperbolic' in avg_entropies:
        print("\nComparison Statistics:")
        ratio = avg_entropies['Hyperbolic'] / avg_entropies['Standard']
        log_diff = np.log10(max(avg_entropies['Hyperbolic'], 1e-10)) - np.log10(max(avg_entropies['Standard'], 1e-10))
        print(f"  Overall Entropy Ratio (Hyperbolic/Standard): {ratio:.4f}")
        print(f"  Log10 Difference (Hyperbolic - Standard): {log_diff:.4f}")
        print(f"  Percent Difference: {(ratio - 1) * 100:.2f}%")
        
    # Print log scale summary to console
    print("\n\nLOG SCALE SUMMARY:")
    print("==================")
    for model_name in model_names:
        print(f"\n{model_name} Model (Log10 Values):")
        print(f"  Overall Average SVD Entropy (log10): {np.log10(max(avg_entropies[model_name], 1e-10)):.4f}")
        
        # Group by order of magnitude
        log_values = {layer_type: np.log10(max(np.mean([v for _, v in values]), 1e-10)) 
                      for layer_type, values in layer_entropies[model_name].items()}
        
        # Sort by log value (ascending order)
        sorted_log_values = sorted(log_values.items(), key=lambda x: x[1])
        
        print("  Layer Types by Order of Magnitude (ascending):")
        for layer_type, log_val in sorted_log_values:
            values = [v for _, v in layer_entropies[model_name][layer_type]]
            print(f"    {layer_type}: {log_val:.4f} (from {len(values)} parameters)")
    
    # 7. Create comprehensive overview plots (regular and log scale)
    print("\nCreating comprehensive overview plots...")
    
    # Regular scale overview
    plt.figure(figsize=(22, 16))
    plt.suptitle('SVD Entropy Analysis Overview (Regular Scale)', fontsize=16)
    
    # 1. Overall comparison
    plt.subplot(2, 3, 1)
    bars = plt.bar(avg_entropies.keys(), avg_entropies.values(), color=['blue', 'orange'])
    plt.title('Average SVD Entropy Comparison')
    plt.ylabel('SVD Entropy')
    # Add the values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Layer type comparison
    plt.subplot(2, 3, 2)
    x = np.arange(len(layer_types))
    width = 0.35
    
    for i, model_name in enumerate(model_names):
        means = layer_type_means[i]
        offset = width * (i - 0.5)
        plt.bar(x + offset, means, width, label=model_name)
    
    plt.title('Average SVD Entropy by Layer Type')
    plt.ylabel('SVD Entropy')
    plt.xticks(x, layer_types, rotation=45, ha='right')
    plt.legend()
    
    # 3. Distribution
    plt.subplot(2, 3, 3)
    plt.violinplot(all_entropies, showmeans=True, showmedians=True)
    plt.title('Distribution of SVD Entropy Values')
    plt.ylabel('SVD Entropy')
    plt.xticks(np.arange(1, len(model_names) + 1), model_names)
    
    # 4. Layer comparison
    if 'Standard' in models and 'Hyperbolic' in models and std_layers and hyp_layers:
        plt.subplot(2, 3, 4)
        x = np.arange(len(std_layers))
        width = 0.35
        
        plt.bar(x - width/2, std_means, width, label='Standard', alpha=0.7, color='blue')
        plt.bar(x + width/2, hyp_means, width, label='Hyperbolic', alpha=0.7, color='orange')
        
        plt.title('Layer-wise SVD Entropy Comparison')
        plt.xlabel('Layer Number')
        plt.ylabel('Average SVD Entropy')
        plt.xticks(x, std_layers)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 5. Entropy ratio
    if 'Standard' in models and 'Hyperbolic' in models and std_layers and hyp_layers and len(std_means) == len(hyp_means):
        plt.subplot(2, 3, 5)
        plt.bar(x, entropy_ratios, alpha=0.7, color='green')
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        
        plt.title('Entropy Ratio (Hyperbolic/Standard)')
        plt.xlabel('Layer Number')
        plt.ylabel('Entropy Ratio')
        plt.xticks(x, std_layers)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 6. Summary text
    plt.subplot(2, 3, 6)
    plt.axis('off')
    summary_text = "Summary Statistics:\n"
    for model_name in model_names:
        summary_text += f"\n{model_name} Model:\n"
        summary_text += f"  Overall Average SVD Entropy: {avg_entropies[model_name]:.4f}\n"
    
    if 'Standard' in avg_entropies and 'Hyperbolic' in avg_entropies:
        summary_text += "\nComparison:\n"
        ratio = avg_entropies['Hyperbolic'] / avg_entropies['Standard']
        summary_text += f"  Entropy Ratio (H/S): {ratio:.4f}\n"
        summary_text += f"  Percent Difference: {(ratio - 1) * 100:.2f}%\n"
    
    plt.text(0, 0.5, summary_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig(os.path.join(output_dir, 'overview_regular.png'), dpi=300)
    plt.close()
    
    # Log scale overview
    plt.figure(figsize=(22, 16))
    plt.suptitle('SVD Entropy Analysis Overview (Log Scale)', fontsize=16)
    
    # 1. Overall comparison
    plt.subplot(2, 3, 1)
    plot_data = [max(val, 1e-10) for val in avg_entropies.values()]
    bars = plt.bar(avg_entropies.keys(), plot_data, color=['blue', 'orange'])
    plt.yscale('log')
    plt.title('Average SVD Entropy Comparison')
    plt.ylabel('SVD Entropy (log scale)')
    # Add the values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                f'{height:.2e}', ha='center', va='bottom', fontsize=9)
    
    # 2. Layer type comparison
    plt.subplot(2, 3, 2)
    x = np.arange(len(layer_types))
    width = 0.35
    
    for i, model_name in enumerate(model_names):
        means = [max(val, 1e-10) for val in layer_type_means[i]]
        offset = width * (i - 0.5)
        plt.bar(x + offset, means, width, label=model_name)
    
    plt.yscale('log')
    plt.title('Average SVD Entropy by Layer Type')
    plt.ylabel('SVD Entropy (log scale)')
    plt.xticks(x, layer_types, rotation=45, ha='right')
    plt.legend()
    
    # 3. Distribution
    plt.subplot(2, 3, 3)
    log_plot_data = [[max(val, 1e-10) for val in data] for data in all_entropies]
    plt.violinplot(log_plot_data, showmeans=True, showmedians=True)
    plt.yscale('log')
    plt.title('Distribution of SVD Entropy Values')
    plt.ylabel('SVD Entropy (log scale)')
    plt.xticks(np.arange(1, len(model_names) + 1), model_names)
    
    # 4. Layer comparison
    if 'Standard' in models and 'Hyperbolic' in models and std_layers and hyp_layers:
        plt.subplot(2, 3, 4)
        x = np.arange(len(std_layers))
        width = 0.35
        
        plt.bar(x - width/2, [max(val, 1e-10) for val in std_means], width, label='Standard', alpha=0.7, color='blue')
        plt.bar(x + width/2, [max(val, 1e-10) for val in hyp_means], width, label='Hyperbolic', alpha=0.7, color='orange')
        
        plt.yscale('log')
        plt.title('Layer-wise SVD Entropy Comparison')
        plt.xlabel('Layer Number')
        plt.ylabel('Average SVD Entropy (log scale)')
        plt.xticks(x, std_layers)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 5. Entropy ratio
    if 'Standard' in models and 'Hyperbolic' in models and std_layers and hyp_layers and len(std_means) == len(hyp_means):
        plt.subplot(2, 3, 5)
        plt.bar(x, entropy_ratios, alpha=0.7, color='green')
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        
        plt.title('Entropy Ratio (Hyperbolic/Standard)')
        plt.xlabel('Layer Number')
        plt.ylabel('Entropy Ratio')
        plt.xticks(x, std_layers)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 6. Summary text
    plt.subplot(2, 3, 6)
    plt.axis('off')
    summary_text = "Log Scale Summary Statistics:\n"
    for model_name in model_names:
        log_val = np.log10(max(avg_entropies[model_name], 1e-10))
        summary_text += f"\n{model_name} Model:\n"
        summary_text += f"  Log10 Average SVD Entropy: {log_val:.4f}\n"
    
    if 'Standard' in avg_entropies and 'Hyperbolic' in avg_entropies:
        log_diff = np.log10(max(avg_entropies['Hyperbolic'], 1e-10)) - np.log10(max(avg_entropies['Standard'], 1e-10))
        summary_text += "\nComparison:\n"
        summary_text += f"  Log10 Difference (H-S): {log_diff:.4f}\n"
    
    # Add layer type order by magnitude
    if len(sorted_log_values) > 0:  # Use the last model's sorted values
        summary_text += "\nLayer Types by Order of Magnitude:\n"
        for i, (layer_type, log_val) in enumerate(sorted_log_values):
            if i < 3:  # Show top 3 to keep it concise
                summary_text += f"  {layer_type}: {log_val:.4f}\n"
    
    plt.text(0, 0.5, summary_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig(os.path.join(output_dir, 'overview_log.png'), dpi=300)
    plt.close()
    
    print(f"\nAll visualizations and statistics saved to '{output_dir}' directory")

if __name__ == "__main__":
    visualize_svd_entropy()
