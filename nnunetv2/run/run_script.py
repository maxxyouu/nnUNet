import subprocess

# Define the script to run
script_path = "C:/Users/MaxYo/OneDrive/Desktop/nnunet_parent/nnUNet/nnunetv2/run/run_training.py"


# python run_training.py 11 3d_fullres 0 -tr nnUNetTrainer_100epochsSaveEvery5EpochsWithAdditionalIntensityAugmentation -p nnUNetResEncUNetMPlans
# Positional argument (input file)
input_file = "input.txt"

# Flag argument --verbose and optional flag argument -o for output
args = ['11', '3d_fullres', '0', "-tr", "nnUNetTrainer_100epochsSaveEvery5EpochsWithAdditionalIntensityAugmentation", '-p' "nnUNetResEncUNetMPlans"]

# Run the script with arguments
result = subprocess.run(["python", script_path] + args, capture_output=True, text=True)

# Output the result
print("Output:\n", result.stdout)
print("Error (if any):\n", result.stderr)
print("Return code:", result.returncode)
