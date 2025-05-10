import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

def visualize_data(csv_file):
    # Load data
    data = pd.read_csv(csv_file)
    
    # Plot histograms for each numeric column
    data.hist(figsize=(12, 10), bins=20, edgecolor='black')
    plt.suptitle('Histograms of Numeric Features')
    plt.show()
    
    # Plot box plots for each numeric column
    numeric_cols = data.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        data.boxplot(column=col)
        plt.title(f'Box Plot of {col}')
        plt.ylabel(col)
        plt.show()
    
    # Plot Q-Q plots for each numeric column
    for col in numeric_cols:
        plt.figure()
        stats.probplot(data[col].dropna(), dist='norm', plot=plt)
        plt.title(f'Q-Q Plot for {col}')
        plt.show()

# Example usage:
visualize_data('DateUnite.csv')


