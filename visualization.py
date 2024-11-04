from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns



sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16})



def RealTime_visualization(losses, epoch):
    clear_output(wait=True)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=list(range(len(losses))), y=losses, marker='o', color="dodgerblue")
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Average Loss', fontsize=14, fontweight='bold')
    plt.title(f'Training Loss - Epoch {epoch + 1}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_loss_curves(results):
    plt.figure(figsize=(14, 10))
    for name, losses, _, _, _, data_type in results:
        sns.lineplot(x=range(len(losses)), y=losses, label=f"{name} ({data_type})", marker='o')
    plt.xlabel('Epochs', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Loss Curves for Different Optimizers', fontsize=18, fontweight='bold')
    plt.legend(title="Optimizers", loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_accuracy_bars(results):
    plt.figure(figsize=(12, 8))
    labels = [f"{name} ({data_type})" for name, _, acc, _, _, data_type in results]
    accuracies = [acc for _, _, acc, _, _, _ in results]
    ax = sns.barplot(x=labels, y=accuracies, palette="Blues_d")
    plt.xlabel('Optimizers', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('Accuracy Comparison Across Optimizers', fontsize=18, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    for i, acc in enumerate(accuracies):
        offset = max(0.01 * max(accuracies), 0.1) 
        ax.text(i, acc + offset, f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, color='black')
        
    plt.tight_layout()
    plt.show()


def plot_time_bars(results):
    plt.figure(figsize=(12, 8))
    labels = [f"{name} ({data_type})" for name, _, _, time, _, data_type in results]
    times = [time for _, _, _, time, _, _ in results]
    ax = sns.barplot(x=labels, y=times, palette="Greens_d")
    plt.xlabel('Optimizers', fontsize=14, fontweight='bold')
    plt.ylabel('Training Time (seconds)', fontsize=14, fontweight='bold')
    plt.title('Training Time Comparison Across Optimizers', fontsize=18, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    for i, time in enumerate(times):
        offset = max(0.01 * max(times), 0.5) 
        ax.text(i, time + offset, f'{time:.2f}s', ha='center', va='bottom', fontsize=12, color='black')
        
    plt.tight_layout()
    plt.show()


def plot_memory_bars(results):
    plt.figure(figsize=(12, 8))
    labels = [f"{name} ({data_type})" for name, _, _, _, memory, data_type in results]
    memory_usage = [memory for _, _, _, _, memory, _ in results]
    ax = sns.barplot(x=labels, y=memory_usage, palette="Oranges_d")
    plt.xlabel('Optimizers', fontsize=14, fontweight='bold')
    plt.ylabel('Memory Usage (MB)', fontsize=14, fontweight='bold')
    plt.title('Memory Usage Comparison Across Optimizers', fontsize=18, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    for i, memory in enumerate(memory_usage):
        offset = max(0.01 * max(memory_usage), 0.01)
        ax.text(i, memory + offset, f'{memory:.2f} MB', ha='center', va='bottom', fontsize=12, color='black')
        
    plt.tight_layout()
    plt.show()
