import pandas as pd
import numpy as np

sheet = pd.read_excel(r"C:\Users\marvu\Desktop\GitHub\RadTACE\file\acquisition.xlsx")
cols_names = sheet.iloc[2,:].values
cols_names[3] = 'acquisition'
col_original = sheet.columns
mapper = dict(zip(sheet.columns, cols_names))
sheet = sheet.rename(columns = mapper)
sheet = sheet.iloc[3:]
sheet = sheet.drop(columns="Series UID")
Control = {}
prev_id = 'HCC_000'
for id in sheet['Subject ID']:
    if id != prev_id:
        dates = sheet[sheet['Subject ID'] == id]['Study date'].unique()
        first_date = np.min(dates)
        types = sheet[(sheet['Study date'] == first_date) & (sheet['Subject ID'] == id)]['description'].tolist()
        prev_id = id
        Control[id] = types
with open(r"C:\Users\marvu\Desktop\GitHub\RadTACE\file\phase_control.txt", 'w') as file:
    for key, values in Control.items():
        file.write(key)
        file.write(',')
        file.write(','.join(map(str, values)))
        file.write('\n')
print(0)