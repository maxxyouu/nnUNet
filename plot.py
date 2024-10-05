import re
import matplotlib.pyplot as plt

# Function to extract pseudo dice and epochs from a log file
def extract_pseudo_dice_and_epochs_precisely(log_data):
    dice_pattern = r"Yayy! New best EMA pseudo Dice:\s*(\d+\.\d+)"
    epoch_pattern = r"Epoch\s+(\d+)"
    pseudo_dice_pattern = r"Pseudo dice \[(\d+\.\d+)\]"
    
    # Find all matches for pseudo Dice values
    pseudo_dice_scores = re.findall(dice_pattern, log_data)
    pseudo_dice_scores = [float(score) for score in pseudo_dice_scores]
    
    # Find all epoch matches
    epoch_matches = list(re.finditer(epoch_pattern, log_data))
    
    # Epoch numbers only when there's a new best pseudo Dice
    relevant_epochs = []
    current_epoch_index = 0
    
    for dice_match in re.finditer(dice_pattern, log_data):
        # Find the closest preceding epoch
        while current_epoch_index < len(epoch_matches) and epoch_matches[current_epoch_index].end() < dice_match.start():
            current_epoch_index += 1
        
        # Add the last valid epoch number
        relevant_epochs.append(int(epoch_matches[current_epoch_index - 1].group(1)))
    
    # Extract all pseudo dice values and return the max value
    pseudo_dice_scores_all = re.findall(pseudo_dice_pattern, log_data)
    pseudo_dice_scores_all = [float(score) for score in pseudo_dice_scores_all]
    max_pseudo_dice = max(pseudo_dice_scores_all) if pseudo_dice_scores_all else None
    
    return pseudo_dice_scores, relevant_epochs, max_pseudo_dice

# Function to extract data from N log files
def extract_from_multiple_logs(log_file_paths):
    return [extract_pseudo_dice_and_epochs_precisely(log_data) for log_data in log_file_paths]

# Function to plot pseudo dice scores vs. epochs with custom labels for each line
def plot_pseudo_dice_vs_epoch_with_labels(all_pseudo_dice_and_epochs, custom_labels=None, title="Best EMA Pseudo Dice Over Epochs", 
                                          y_label="EMA Pseudo Dice", margin=(0.05, 0.1)):
    plt.figure(figsize=(10, 6))

    # Loop through the pseudo dice scores, epochs, and use custom label if provided
    for i, (pseudo_dice_scores, epochs, _) in enumerate(all_pseudo_dice_and_epochs):
        label = custom_labels[i] if custom_labels and i < len(custom_labels) else f'Log File {i+1}'
        plt.plot(epochs, pseudo_dice_scores, marker='o', linestyle='-', label=label)
    
    # Apply custom margins
    plt.margins(x=margin[0], y=margin[1])

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

# Function to handle any arbitrary number of log files and custom labels for each line
def process_and_plot_log_files_with_labels(log_file_paths, custom_labels=None, title="Best EMA Pseudo Dice Over Epochs", 
                                           y_label="EMA Pseudo Dice", margin=(0.05, 0.1)):
    # Read the contents of each log file
    log_data_list = [open(log_file, 'r').read() for log_file in log_file_paths]
    
    # Extract data from the log files
    all_pseudo_dice_and_epochs_precise = extract_from_multiple_logs(log_data_list)

    # Plot the data with custom labels for each line
    plot_pseudo_dice_vs_epoch_with_labels(all_pseudo_dice_and_epochs_precise, custom_labels=custom_labels, title=title, y_label=y_label, margin=margin)

    # Return the max pseudo dice scores from each file
    return [result[2] for result in all_pseudo_dice_and_epochs_precise]

# Example usage with N log files and custom labels
log_file_paths = [
    # "C:\\Users\\MaxYo\\OneDrive\\Desktop\\nnunet_parent\\nnUNet_results\\Dataset011_GE\\nnUNetTrainer_100epochsSaveEvery5Epochs__nnUNetResEncUNetMPlans__3d_fullres\\fold_0\\training_log_2024_9_19_12_45_38.txt",  # Replace with actual log file paths
    "C:\\Users\\MaxYo\\OneDrive\\Desktop\\nnunet_parent\\nnUNet_results\\Dataset011_GE\\nnUNetTrainer_100epochsSaveEvery5EpochsWithAdditionalIntensityAugmentation__nnUNetResEncUNetMPlans__3d_fullres\\fold_0\\training_log_2024_9_30_22_19_42.txt",
]

# Custom labels for each line plot
custom_labels = [
    # "vanilla nnUNet",
    "vanilla nnUNet + IA",
    # "nnUNet + IA only"
    # "nnUNet + no DA"
]

# Process and plot data from N log files with custom labels for each line
max_pseudo_dice_scores_combined = process_and_plot_log_files_with_labels(log_file_paths, custom_labels=custom_labels, 
                                                                        title="Exponential Moving Average (EMA) Dice Score Plot", 
                                                                        y_label="EMA Dice Score", 
                                                                        margin=(0.1, 0.15))

# Output the max Pseudo Dice scores
print(max_pseudo_dice_scores_combined)
