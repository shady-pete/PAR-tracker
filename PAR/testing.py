import os


'''
    This file test all the models in a folder using our custom testing dataset.
'''

FOLDER_PATH = "models_v2/validation"    # Path to the folder containing the models

# Get all file in the folder (NOTE: THIS CODE DOES NOT FILTER THE FILES SO THE FILE IN THE FOLDER MUST BE .pth FILE)
files = [f for f in os.listdir(FOLDER_PATH) if os.path.isfile(os.path.join(FOLDER_PATH, f))]

# Sort the files by name so that each row in the metrics output file corresponds to successive models
files_sorted = sorted(files, key=lambda x: int(x.split('_')[-1].replace('.pth','')))

for file_name in files_sorted:
    file_path = os.path.join(FOLDER_PATH, file_name)
    print(f"Processing file: {file_path}")
    # Run the testing script test.py with the correct model
    os.system(f"python test.py --load_model {file_path}")
