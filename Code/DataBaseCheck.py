import pandas as pd
data = pd.read_csv('PrimaReCapturareCSV.csv')
data = pd.read_csv('ADouaRecapturareCSV.csv')
data = pd.read_csv('ATreiaReCapturareCSV.csv')
data = pd.read_csv('APatraReCapturareCSV.csv')

print(data.info())
print(data.describe(include='all'))

