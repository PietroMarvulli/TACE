import pandas as pd

filename = r"C:\Users\marvu\Desktop\RadiomicsKAN_FineTuning.txt"
lines_list = []
with open(filename, 'r') as file:
    for line in file:
        lines_list.append(line.strip())

filtered_lines = lines_list
lamb_values = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09] * 6
grid_values = []
grid_repeats = [3, 5, 10, 20, 50, 100]
for val in grid_repeats:
    grid_values.extend([val] * 10)

result = pd.DataFrame({'Grid':grid_values, 'Lamb':lamb_values})
colnames = ['[12,1,1]','[12,2,1]','[12,3,1]','[12,4,1]','[12,5,1]','[12,6,1]','[12,7,1]','[12,8,1]','[12,9,1]','[12,10,1]']

for col in colnames:
    result[f"{col} train"] = float('nan')  # Aggiungere la colonna con valori NaN per train
    result[f"{col} test"] = float('nan')

col_to_fill = list(result.columns)[2:]
data = []
for j in range(0,len(colnames)):
    values_train = []
    values_test = []
    for i in range(2+(j*240),240*(j+1),4):
        string_value = filtered_lines[i]
        value_tr = float(filtered_lines[i].split(':')[-1].strip())
        values_train.append(value_tr)
        value_te = float(filtered_lines[i+1].split(':')[-1].strip())
        values_test.append(value_te)
    print(colnames[j])
    data.append(values_train)
    data.append(values_test)
for i in range(len(data)):
    result[col_to_fill[i]] = data[i]
print(0)