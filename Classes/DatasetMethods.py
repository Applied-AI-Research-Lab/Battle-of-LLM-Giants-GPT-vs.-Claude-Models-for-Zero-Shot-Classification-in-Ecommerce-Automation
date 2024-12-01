import os
import re
import pandas as pd
import csv
from sklearn.model_selection import train_test_split


class DatasetMethods:
    def __init__(self, dataset_path=''):
        self.dataset_path = dataset_path
        self.pre_path = '../Datasets/'

    def just_read_csv(self):
        return pd.read_csv(self.pre_path + self.dataset_path)

    """
    If the 'id' column is missing, create a new column named 'id' starting from 1
    """

    def add_id_column(self):
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Add a new column 'id' with sequential IDs starting from 1
        df.insert(0, 'id', range(1, len(df) + 1))

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_before_id.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame back to a new CSV file
        df.to_csv(self.pre_path + self.dataset_path, index=False)

    """
    Merge column1 and column2 into a new column named merged_column, while keeping the remaining columns unchanged.
    """

    def merge_columns(self, column1, column2, merged_column):
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Combine the two columns into a new merged_column
        df[merged_column] = df[column1] + ' ' + df[column2]

        # Drop the original column1 and column2, but keep the rest of the columns
        df = df.drop([column1, column2], axis=1)

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_before_merge_columns.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame back to a new CSV file
        df.to_csv(self.pre_path + self.dataset_path, index=False)

    """
    Create a new dataset with rows where the text in the column has less than max_len characters.
    """

    def remove_max_len_rows(self, column_name, max_len):
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Filter rows where the length of the text in column_name is less than max_len
        new_df = df[df[column_name].str.len() < max_len]

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_before_max_len.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame back to a new CSV file
        new_df.to_csv(self.pre_path + self.dataset_path, index=False)

        return {"status": True, "data": 'Rows where the text in ' + column_name + ' has fewer than ' + str(
            max_len) + ' characters have been removed'}

    """
    Make changes to a specific column
    """

    def make_changes_to_column(self, column):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Extract 'class' from column
        df[column] = df[column].apply(lambda x: eval(x)['class'])

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_before_changes_column.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame back to a new CSV file
        df.to_csv(self.pre_path + self.dataset_path, index=False)

    """
    Create a new column changing the categorical label to numerical
    """

    def create_numeric_column_from_categorical(self, column, new_column):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Define a mapping for sentiment values
        mapping = {
            'positive': 1,
            'negative': 0,
            'neutral': 2
        }

        # Create a new column 'sentiment_binary' based on the mapping
        df[new_column] = df[column].map(mapping)

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_before_numerical_column.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame back to a new CSV file
        df.to_csv(self.pre_path + self.dataset_path, index=False)

    """
    Create a new column changing the numerical label to categorical
    """

    def create_categorical_column_from_numeric(self, column, new_column):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Define a mapping for sentiment values
        mapping = {
            1: 'positive',
            0: 'negative',
            2: 'neutral'
        }

        # Create a new column 'sentiment_binary' based on the mapping
        df[new_column] = df[column].map(mapping)

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_before_categorical_column.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame back to a new CSV file
        df.to_csv(self.pre_path + self.dataset_path, index=False)

    """
    The method takes an array of column names (column_names) as input and removes empty rows.
    If the array is empty, each and every column is checked.
    Caution! The original dataset will be renamed to _original1,
         while the most current dataset will take the name of the original dataset
    """

    def remove_rows_with_empty_fields(self, column_names):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # If column_names is empty, check for empty fields in all columns
        if not column_names:
            # Check for empty fields in all columns and remove corresponding rows
            df = df.dropna(how='any')
        else:
            # Check for empty fields in specified columns and remove corresponding rows
            df = df.dropna(subset=column_names, how='any')

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original1.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame to a new CSV file
        df.to_csv(self.pre_path + self.dataset_path, index=False)

        return {"status": True, "data": 'Empty rows removed'}

    """    
    The method takes an array of column names (columns_to_remove) as input and removes them entirely.
    Caution! The original dataset will be renamed to _original2,
         while the most current dataset will take the name of the original dataset
    """

    def remove_columns_and_save(self, columns_to_remove):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Automatically remove 'Unnamed: 0' column if present
        if 'Unnamed: 0' in df.columns:
            columns_to_remove.append('Unnamed: 0')

        # Remove the specified columns
        df = df.drop(columns=columns_to_remove, errors='ignore')

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original2' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original2.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame to a new CSV file
        df.to_csv(self.pre_path + self.dataset_path, index=False)

        return {"status": True, "data": 'The specified columns have been removed'}

    """
    Display the unique labels in a specific column (column_name)
    """

    def display_unique_values(self, column_name):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Get the unique values and their counts
        unique_values_counts = df[column_name].value_counts()

        print(f"Unique values in column '{column_name}' ({len(unique_values_counts)}):")
        for value, count in unique_values_counts.items():
            print(f"Label: {value}: Count: {count}")

    """
    Returns the unique labels in a specific column (column_name) as an array.
    """
    def display_unique_values_array(self, column_name):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Get the unique values
        unique_values = df[column_name].unique().tolist() # .tolist() convert a NumPy array to a Python list

        return unique_values

    """
    The method removes rows containing a specific value (value_to_remove) in a given column (column_name)
    Caution! The original dataset will be renamed to _original4,
         while the most current dataset will take the name of the original dataset
    """

    def remove_rows_by_value(self, column_name, value_to_remove):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Use boolean indexing to filter out rows with the specified value
        filtered_dataframe = df[df[column_name] != value_to_remove]

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original4.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame to a new CSV file
        filtered_dataframe.to_csv(self.pre_path + self.dataset_path, index=False)

        return {"status": True, "data": f"Fields having value '{value_to_remove}' removed"}

    """
    This method cleans and standardizes the values of each row in all columns
    Caution! The original dataset will be renamed to _original3,
         while the most current dataset will take the name of the original dataset
    """

    def standardize_text(self, text):
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and punctuation
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Remove extra whitespaces and newline characters
        text = re.sub(r'\s+', ' ', text)

        # Remove newline characters specifically
        text = text.replace('\n', ' ').strip()

        return text

    def standardize_and_write_csv(self):
        # Rename the original dataset
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]
        original_file_path = f'../Datasets/{file_name_without_extension}_original3.csv'
        os.rename(f'../Datasets/{self.dataset_path}', original_file_path)

        # Write the standardized data to a new CSV file, excluding empty rows
        # Open the original file for reading and the new file for writing
        with open(original_file_path, 'r', encoding='utf-8') as infile, \
                open(f'../Datasets/{self.dataset_path}', 'w', newline='', encoding='utf-8') as outfile:

            csv_reader = csv.reader(infile)
            csv_writer = csv.writer(outfile)

            # Read and write the header
            header = next(csv_reader)
            csv_writer.writerow(header)

            # Process and write the remaining rows
            for row in csv_reader:
                standardized_row = [self.standardize_text(field) for field in row]

                # Check if the row is not empty after standardization
                if any(standardized_row):
                    csv_writer.writerow(standardized_row)

        return {"status": True, "data": "Standardization completed"}

    """
    This method creates a subset (total_rows) of the original dataset,
    ensuring the appropriate distribution of the (stratified_column) values
    Caution! The original dataset will be renamed to _original5,
         while the most current dataset will take the name of the original dataset
    """

    def create_stratified_subset(self, total_rows, stratified_column):
        # Load the dataset
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Check the unique values in the stratified column
        unique_values = df[stratified_column].unique()

        # Create an empty DataFrame to store the subset
        subset_df = pd.DataFrame()

        # Define the number of rows you want for each value in the stratified column
        rows_per_value = total_rows // len(unique_values)

        # Loop through each unique value and sample rows
        for value in unique_values:
            value_subset = df[df[stratified_column] == value].sample(rows_per_value, random_state=42)
            subset_df = pd.concat([subset_df, value_subset])

        # If the total number of rows is less than the specified total, sample the remaining rows from the entire dataset
        remaining_rows = total_rows - len(subset_df)
        remaining_subset = df.sample(remaining_rows, random_state=42)
        subset_df = pd.concat([subset_df, remaining_subset])

        # Optionally, you can shuffle the final subset
        subset_df = subset_df.sample(frac=1, random_state=42)

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original5.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame to a new CSV file
        subset_df.to_csv(self.pre_path + self.dataset_path, index=False)

        return {"status": True, "data": "Subset created"}

    """
    Split the dataset into train, validation, and test sets.
    By providing the stratify_column argument, the stratify function ensures that
    the distribution of labels or classes is maintained in both sets.
    """

    def split_dataset(self, stratify_column=''):
        train_file_path = 'train_set.csv'
        valid_file_path = 'validation_set.csv'
        test_file_path = 'test_set.csv'

        df = pd.read_csv(self.pre_path + self.dataset_path, on_bad_lines='skip')  # Read the cleaned dataset CSV file

        # Split the dataset into train, validation, and test sets while stratifying by the stratify_column
        if stratify_column:  # If stratify_column is provided, then stratify
            train_valid, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[stratify_column])
            train, valid = train_test_split(train_valid, test_size=0.2, random_state=42,
                                            stratify=train_valid[stratify_column])
        else:  # Split the dataset without stratifying
            train_valid, test = train_test_split(df, test_size=0.2, random_state=42)
            train, valid = train_test_split(train_valid, test_size=0.2, random_state=42)

        # Save the split datasets to separate CSV files
        train.to_csv(self.pre_path + train_file_path, index=False)
        valid.to_csv(self.pre_path + valid_file_path, index=False)
        test.to_csv(self.pre_path + test_file_path, index=False)

        return {"status": True, "data": "Splitting succeed"}

    """
    Remove phrases from a specific column
    By providing the phrase and the column a new clean dataset will be created
    """

    def remove_phrase_from_column(self, phrase, column):
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Function to remove phrase from column
        def remove_subject(text):
            return text.replace(phrase, "")

        # Apply the function to the column
        df[column] = df[column].apply(remove_subject)

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original7.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame to a new CSV file
        df.to_csv(self.pre_path + self.dataset_path, index=False)

        return {"status": True, "data": "Phrases removed"}

    """
    Rename a specific column
    """

    def rename_column_in_csv(self, old_column_name, new_column_name):
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Check if the old column name exists in the DataFrame
        if old_column_name not in df.columns:
            print(f"Column '{old_column_name}' does not exist in the CSV file.")
            return

        # Rename the specific column
        df.rename(columns={old_column_name: new_column_name}, inplace=True)

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original10.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the DataFrame back to the CSV file
        df.to_csv(self.pre_path + self.dataset_path, index=False)

    """
    Find min, max, avg for a specific column
    """

    def find_max_min_avg_length(self, column_name):
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Replace NaN values with empty strings
        df[column_name] = df[column_name].fillna('')

        # Calculate the length of each text in the specified column
        df['T_Length'] = df[column_name].apply(len)

        # Find the row with the maximum and minimum length
        max_row = df.loc[df['T_Length'].idxmax()]
        min_row = df.loc[df['T_Length'].idxmin()]

        # Get the text lengths for max and min
        max_text_length = len(max_row[column_name])
        min_text_length = len(min_row[column_name])

        # Calculate the average text length
        avg_text_length = df['T_Length'].mean()

        return {
            'max_length': max_text_length,
            'min_length': min_text_length,
            'avg_length': avg_text_length
        }

    """
    Check if all entries in the specified column of the DataFrame match the expected type and return the IDs of rows that do not match.
    """

    def check_column_types(self, id_column: str, column_name: str, column_type: str) -> list:
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Check if the specified columns exist
        if id_column not in df.columns:
            raise ValueError(f"ID column '{id_column}' does not exist in the DataFrame.")
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

        # Define a mapping of column types to Python types
        type_map = {
            'string': str,
            'float': float,
            'int': int
        }

        # Check if the specified column type is valid
        if column_type not in type_map:
            raise ValueError(f"Invalid column_type '{column_type}'. Choose from {list(type_map.keys())}.")

        # Get the expected type
        expected_type = type_map[column_type]

        # List to store IDs of rows with type mismatches
        mismatched_ids = []

        # Check the types in the specified column
        for idx, value in enumerate(df[column_name]):
            if not isinstance(value, expected_type):
                mismatched_ids.append(df[id_column].iloc[idx])  # Append the ID of the mismatched row

        return mismatched_ids

    """
    Merges two datasets by iterating over rows in the first dataset
    and finding the corresponding value from the second dataset.

    Args:
        first_file (str): Path to the first CSV file.
        second_file (str): Path to the second CSV file.
        new_file (str): Path to save the merged dataset.
        first_column (str): Column name in the first dataset to match with the second.
        second_column (str): Column name in the second dataset to match against.
    """

    def merge_datasets(self, first_file, second_file, new_file, first_column, second_column):
        try:
            # Load the datasets
            first_data = pd.read_csv(self.pre_path + first_file)
            second_data = pd.read_csv(self.pre_path + second_file)

            # Create a dictionary for faster lookup
            lookup_dict = dict(
                zip(second_data[second_column], second_data.drop(second_column, axis=1).to_dict(orient='records')))

            # Initialize a list for new data columns
            additional_columns = {key: [] for key in second_data.columns if key != second_column}

            # Iterate over each row in the first dataset
            for _, row in first_data.iterrows():
                key_value = row[first_column]
                # Get the corresponding row data from the second dataset
                matched_row = lookup_dict.get(key_value, None)  # None if not found

                # Append matched values or None for missing keys
                for key in additional_columns.keys():
                    additional_columns[key].append(matched_row[key] if matched_row else None)

            # Add the new columns to the first dataset
            for key, values in additional_columns.items():
                first_data[key] = values

            # Save the updated dataset
            first_data.to_csv(self.pre_path + new_file, index=False)

            print(f"Merged dataset saved as '{new_file}'")
        except Exception as e:
            print(f"An error occurred: {e}")

    """
    Creates a new dataset with a random sample of products from each category.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the sampled CSV file.
        category_column (str): Name of the category column.
        sample_size (int): Number of products to sample per category.
    """

    def sample_products_by_category(self, input_file, output_file, category_column, sample_size):
        try:
            # Load the dataset
            data = pd.read_csv(self.pre_path + input_file)

            # Group by the category column and sample from each group
            sampled_data = (
                data.groupby(category_column)
                .apply(
                    lambda x: x.sample(n=sample_size, replace=False, random_state=42) if len(x) >= sample_size else x)
                .reset_index(drop=True)
            )

            # Save the sampled dataset
            sampled_data.to_csv(self.pre_path + output_file, index=False)

            print(f"Sampled dataset saved as '{output_file}'")
        except Exception as e:
            print(f"An error occurred: {e}")

    def calculate_row_lengths(self):
        # Load the dataset
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Dictionary to store the results
        results = {}

        for column in df.columns:
            # Calculate the length of each row in the column
            row_lengths = df[column].astype(str).apply(len)

            # Calculate min, average, and max lengths
            min_length = row_lengths.min()
            avg_length = row_lengths.mean()
            max_length = row_lengths.max()

            # Store the results
            results[column] = {
                "Min Length": min_length,
                "Average Length": avg_length,
                "Max Length": max_length
            }

        return results



    """
    Splits a dataset into multiple smaller datasets, each containing a specified number of unique categories.

    Args:
        input_file (str): Path to the input CSV file.
        output_folder (str): Path to the folder where the split datasets will be saved.
        category_column (str): The column containing category names.
        products_per_category (int): Number of products per category.
        num_splits (int): Number of splits (output datasets).
    """
    def split_dataset_by_categories(self, input_file, output_folder, category_column, products_per_category, num_splits):

        try:
            # Load the dataset
            data = pd.read_csv(self.pre_path + input_file)

            # Get unique categories
            unique_categories = data[category_column].unique()

            # Ensure the output folder exists
            os.makedirs(self.pre_path + output_folder, exist_ok=True)

            # Split categories into chunks
            category_chunks = [unique_categories[i:i + len(unique_categories) // num_splits]
                               for i in range(0, len(unique_categories), len(unique_categories) // num_splits)]

            # Adjust the last chunk to ensure all categories are included
            if len(category_chunks) > num_splits:
                category_chunks[-2] = list(category_chunks[-2]) + list(category_chunks[-1])
                category_chunks = category_chunks[:-1]

            # Create datasets for each chunk
            for idx, categories in enumerate(category_chunks):
                # Filter data for the current chunk of categories
                filtered_data = data[data[category_column].isin(categories)]

                # Save to a new CSV file
                output_file = os.path.join(self.pre_path + output_folder, f'dataset_split_{idx + 1}.csv')
                filtered_data.to_csv(output_file, index=False)
                print(f"Dataset split {idx + 1} saved with {len(categories)} categories: {output_file}")

        except Exception as e:
            print(f"An error occurred: {e}")

# Example Usage
# Instantiate the DatasetMethods class by providing the (dataset_path)
DTS = DatasetMethods(dataset_path='Products2.csv')

# Merge amazon_products.csv and amazon_categories.csv based on category_id
# print(DTS.merge_datasets(
#     first_file='amazon_products.csv',
#     second_file='amazon_categories.csv',
#     new_file='Products2.csv',
#     first_column='category_id',
#     second_column='id'
# ))

# Read the csv
# print(DTS.just_read_csv())

# Find min, max, avg for a specific column
# print(DTS.find_max_min_avg_length('text'))

# Remove unwanted columns
# print(DTS.remove_columns_and_save(['asin', 'imgUrl', 'productURL', 'stars', 'reviews', 'price', 'listPrice', 'category_id', 'isBestSeller', 'boughtInLastMonth']))

# Identify the unique labels in a specific column (column_name) to understand your dataset
# DTS.display_unique_values(column_name='category_name')

# # Remove rows with empty values by providing specific column names or
# # by providing an empty array [] to check all columns
# print(DTS.remove_rows_with_empty_fields(column_names=['title','category_name']))

# Clean and standardize each row and value in your dataset
# print(DTS.standardize_and_write_csv())

# Remove empty rows after standardization
# print(DTS.remove_rows_with_empty_fields(['title','category_name']))

# # Identify the unique labels in a specific column (column_name) to understand your dataset
# DTS.display_unique_values(column_name='category_name')

# Rename an existing column
# print(DTS.rename_column_in_csv('category_name', 'category'))

# Get a sample of 100 products per category
# print(DTS.sample_products_by_category(
#     input_file='Products.csv',
#     output_file='sampled_products.csv',
#     category_column='category',
#     sample_size=20
# ))

# # Identify the unique labels in a specific column (column_name) to understand your dataset
# DTS.display_unique_values(column_name='category')

print(DTS.calculate_row_lengths())
# If the 'id' column is missing, create a new column named 'id' starting from 1
# DTS.add_id_column()

# Split datasets to 8 files containing 31 categories each and their corresponding products
# print(DTS.split_dataset_by_categories(
#     input_file='Products.csv',
#     output_folder='split_datasets',
#     category_column='category',
#     products_per_category=20,
#     num_splits=8
# ))

# # Identify the unique labels in a specific column (column_name) to understand your dataset
# DTS = DatasetMethods(dataset_path='split_datasets/dataset_split_8.csv')
# print(DTS.display_unique_values_array(column_name='category'))
