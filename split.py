import splitfolders
# Path to the original dataset
original_dataset_path = "C:\\Users\\manis\\Downloads\plantvillage dataset"

# Path to the output directory where the split dataset will be stored
output_dataset_path = "Generated_dataset"

# Split the dataset into training, validation, and test sets
splitfolders.ratio(original_dataset_path, output=output_dataset_path, seed=1337, ratio=(0.6, 0.2, 0.2))
