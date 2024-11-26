import csv
import numpy as np
import re  

def parse_array(data_str):
    
    try:
        clean_data = re.sub(r'array\(', '', data_str).rstrip(')')
        return eval(clean_data, {"__builtins__": None}, {"np": np})
    except Exception as e:
        print(f"Erreur lors du parsing de l'array : {data_str} -> {e}")
        return None

def read_csv():
    file_path = "data.csv"
    data = []

    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 3:
                try:
                    
                    array_data = parse_array(row[0])

                    if row[1].strip().lower() == "true":
                        bool_1 = True
                    elif row[1].strip().lower() == "false":
                        bool_1 = False

                    if row[2].strip().lower() == "true":
                        bool_2 = True
                    elif row[2].strip().lower() == "false":
                        bool_2 = False
                    
                    

                    
                    data.append((array_data, bool_1, bool_2))
                except Exception as e:
                    print(f"Erreur lors de la lecture de la ligne {row}: {e}")

    return data



data = read_csv()
xs = []
y = []
for i in range(0,85):
    x = data[i] 
    y .append(x[-1])
    m = x[:-1]
    xs.append(m)



print(xs)
print(y)

