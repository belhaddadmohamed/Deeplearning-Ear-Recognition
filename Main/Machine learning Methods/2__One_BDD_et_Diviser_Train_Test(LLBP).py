import pandas as pd
import os
import glob

# Directory where all class folders are stored
root_directory = "data\extract_features"  # Update with your directory path

# List of all folder names (each folder represents a class)
folder_names = [f.name for f in os.scandir(root_directory) if f.is_dir()]

# List to hold all DataFrames
dataframes = []

# Iterate through each folder to get CSV files and their label from the folder name
for folder_name in folder_names:
    folder_path = os.path.join(root_directory, folder_name) # Person_path
    # Get all CSV files in this folder
    csv_files = glob.glob(os.path.join(folder_path, "LLBP", "*.csv"))
    print('person: '+str(folder_name))
    
    for csv_file in csv_files:
        # Read the CSV file, without headers (assuming no headers in original CSV)
        df = pd.read_csv(csv_file, header=None)  # No header

        # Flatten the DataFrame to 1D
        flattened_array = df.values.flatten()  # Convert DataFrame to a 1D array

        # Convert the flattened array back to a DataFrame with a single row
        flattened_df = pd.DataFrame([flattened_array])

        # Save to a new CSV file (with or without header as desired)
        # flattened_df.to_csv("flattened_output.csv", index=False, header=False)  # No header, single row
        
        # Add a new column for the label based on folder name
        flattened_df["label"] = folder_name
        
        # Append this DataFrame to the list
        dataframes.append(flattened_df)

# Concatenate all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)
print(combined_df.shape)
print(f"Image files successfully flattened")

# Save the combined DataFrame to a new CSV file without a header
# output_csv_path = "data/combined_data_LLBP.csv"  # Output CSV path
# combined_df.to_csv(output_csv_path, index=False)  # Write without headers
# df = combined_df
# df = pd.read_csv(output_csv_path)



# =================================================================================================
                    # DIVISER TRAIN/TEST
# =================================================================================================


# Load the entire CSV
# csv_path = "data\combined_data_LLBP.csv" 
# data = pd.read_csv(csv_path)  
print("Divsier Train/Test...")
data = combined_df

# Number of rows to slice for training and testing
train_size = 25
test_size = 3
iteration_size = train_size + test_size

# Total number of iterations
total_rows = data.shape[0]
iterations = total_rows // iteration_size  # Full iterations available in the dataset

# List to store the results of training and testing
train_dfs = []
test_dfs = []

# Iterate through the data to create training and testing sets
for i in range(iterations):
    print('iteration: ', i)
    start = i * iteration_size
    end = start + iteration_size
    
    # Slice out the train and test sets for this iteration
    train_df = data.iloc[start:start + train_size]
    test_df = data.iloc[start + train_size:start + iteration_size]
    
    # Store the results
    train_dfs.append(train_df)
    test_dfs.append(test_df)

# Concatenate all training sets and all testing sets
final_train_df = pd.concat(train_dfs, ignore_index=True)
final_test_df = pd.concat(test_dfs, ignore_index=True)

# Save the final training and testing datasets
final_train_df.to_csv("data/train_test/LLBP/train_LLBP.csv", index=False) 
final_test_df.to_csv("data/train_test/LLBP/test_LLBP.csv", index=False)   


print("Training and testing datasets have been created and saved")
print("Training Shape:" + str(final_train_df.shape))
print("Testing Shape:" + str(final_test_df.shape))
