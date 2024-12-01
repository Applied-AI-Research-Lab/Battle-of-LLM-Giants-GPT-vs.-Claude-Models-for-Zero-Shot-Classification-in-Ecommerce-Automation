import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import seaborn as sns
import re
import json

class EvaluationMethods:
    def __init__(self, dataset_path=''):
        self.dataset_path = dataset_path
        self.pre_path = '../Datasets/split_datasets/'

    import json
    import os
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    def evaluate_results(self, original, prediction, model_name, dataset_name=''):
        try:
            dataset_name=self.dataset_path
            # Load dataset
            data_path = os.path.join(self.pre_path, self.dataset_path)
            data = pd.read_csv(data_path)

            # Ensure required columns are present
            if original not in data.columns or prediction not in data.columns:
                raise ValueError(f"Columns '{original}' or '{prediction}' not found in dataset.")

            # Compute metrics with zero_division handling
            accuracy = round(accuracy_score(data[original], data[prediction]), 4)
            precision = round(precision_score(data[original], data[prediction], average='weighted', zero_division=1), 4)
            recall = round(recall_score(data[original], data[prediction], average='weighted', zero_division=1), 4)
            f1 = round(f1_score(data[original], data[prediction], average='weighted', zero_division=1), 4)

            # Prepare evaluation result as a dictionary
            evaluation_result = {
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            }

            # Load the existing results (if any)
            results_path = os.path.join(self.pre_path, 'evaluation-results.json')

            # If the file doesn't exist, create it with an empty structure
            if not os.path.exists(results_path):
                with open(results_path, 'w') as json_file:
                    json.dump({}, json_file)

            # Read the existing data from the JSON file
            with open(results_path, 'r+') as json_file:
                data = json.load(json_file)

                # If the dataset doesn't exist in the JSON structure, initialize it
                if dataset_name not in data:
                    data[dataset_name] = []

                # Check if the fields in the evaluation result match the keys in the existing JSON data
                for entry in data[dataset_name]:
                    for key in evaluation_result.keys():
                        if key not in entry:
                            entry[key] = None  # Add missing fields with a default value (None)

                # Append the new result to the corresponding dataset section
                data[dataset_name].append(evaluation_result)

                # Write the updated data back to the file
                json_file.seek(0)  # Go back to the beginning of the file before writing
                json.dump(data, json_file, indent=4)

            # Return the updated list of results for the specific dataset
            return data[dataset_name]

        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None

    def scatterplot(self, original_column, prediction_column):
        df = pd.read_csv(self.pre_path + self.dataset_path)
        prediction = df[prediction_column]
        original = df[original_column]

        # Calculate Mean Absolute Error
        mae = abs(original - prediction).mean()

        # Create a scatter plot with a regression line
        sns.regplot(x=original, y=prediction, scatter_kws={'alpha': 0.5})

        plt.xlabel(original_column)
        plt.ylabel(prediction_column)

        # Save the scatterplot image to the Datasets folder
        plt.savefig(os.path.join(self.pre_path + 'Plots/', prediction_column + '.png'))

        # Show the plot
        plt.show()

        return mae

    def count_matching_rows(self, original_column, prediction_column):
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Count the number of same value rows
        matching_rows = df[df[original_column] == df[prediction_column]]

        return len(matching_rows)

    def plot_histograms(self, original_column, prediction_column):
        dataframe = pd.read_csv(self.pre_path + self.dataset_path)

        # Separate predicted probabilities by class
        predicted_probabilities_class_0 = dataframe.loc[dataframe[original_column] == 0, prediction_column]
        predicted_probabilities_class_1 = dataframe.loc[dataframe[original_column] == 1, prediction_column]

        # Plot histograms
        plt.figure(figsize=(10, 5))

        # Histogram for class 0
        plt.subplot(1, 2, 1)
        plt.hist(predicted_probabilities_class_0, bins=20, color='blue', alpha=0.7)
        plt.title('Predicted Probabilities - Class 0')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')

        # Histogram for class 1
        plt.subplot(1, 2, 2)
        plt.hist(predicted_probabilities_class_1, bins=20, color='orange', alpha=0.7)
        plt.title('Predicted Probabilities - Class 1')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, original_column, prediction_column):
        dataframe = pd.read_csv(self.pre_path + self.dataset_path)

        # Extract data from DataFrame
        y_true = dataframe[original_column]
        y_pred = dataframe[prediction_column]

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix \n('+prediction_column+')')
        plt.show()

    """
    Plot a stacked bar chart showing the distribution of labels across categories in two columns.

    Args:
    column1 (str): The name of the first column with string labels.
    column2 (str): The name of the second column with string labels.
    """

    def plot_stacked_bar_chart(self, original_column, prediction_column):
        data = pd.read_csv(self.pre_path + self.dataset_path)
        cross_tab = pd.crosstab(data[original_column], data[prediction_column])
        # Calculate row-wise percentages
        cross_tab_percent = cross_tab.apply(lambda x: x * 100 / x.sum(), axis=1)

        # Plotting the stacked bar chart
        ax = cross_tab_percent.plot(kind='bar', stacked=True, figsize=(10, 6))

        # Adding labels and title
        plt.title(f'Stacked Bar Chart of {original_column} vs. {prediction_column}')
        plt.xlabel(original_column)
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)

        # Adding percentages as text on each bar segment
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + width / 2, y + height / 2), ha='center', va='center', fontsize=8)

        plt.show()

    """
    Plot a grouped bar chart showing the relationship between labels in two columns.

    Args:
    column1 (str): The name of the first column with string labels.
    column2 (str): The name of the second column with string labels.
    """
    def plot_grouped_bar_chart(self, original_column, prediction_column):
        data = pd.read_csv(self.pre_path + self.dataset_path)
        pivot_table = data.groupby([original_column, prediction_column]).size().unstack(fill_value=0)
        pivot_table.plot(kind='bar', figsize=(10, 6))
        plt.title(f'Relationship between {original_column} and {prediction_column}')
        plt.xlabel(original_column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

    """
    Plot a heatmap showing relationships and patterns between label categories in two columns.

    Args:
    column1 (str): The name of the first column with string labels.
    column2 (str): The name of the second column with string labels.
    """
    def plot_heatmap(self, original_column, prediction_column):
        data = pd.read_csv(self.pre_path + self.dataset_path)
        cross_tab = pd.crosstab(data[original_column], data[prediction_column])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu')
        plt.title(f'Heatmap of {original_column} vs. {prediction_column}')
        plt.xlabel(prediction_column)
        plt.ylabel(original_column)
        plt.show()


# Instantiate the DatasetMethods class by providing the (dataset_path)
EVM = EvaluationMethods(dataset_path='dataset_split_8.csv')

# # Evaluate the predictions made by each model
print(f'gpt_4o_prediction: ' + str(EVM.evaluate_results('category', 'gpt_4o_prediction', 'gpt_4o_prediction')))
print(f'gpt_4o_mini_prediction: ' + str(EVM.evaluate_results('category', 'gpt_4o_mini_prediction', 'gpt_4o_mini_prediction')))
print(f'claude_3.5_sonnet_prediction: ' + str(EVM.evaluate_results('category', 'claude_3.5_sonnet_prediction', 'claude_3.5_sonnet_prediction')))
print(f'claude_3.5_haiku_prediction: ' + str(EVM.evaluate_results('category', 'claude_3.5_haiku_prediction', 'claude_3.5_haiku_prediction')))