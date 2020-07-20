import csv

def number_from_char(c):
    return ord(c) - ord('A')

def heat_map_from_responses():
    data_rows = []
    with open(r'C:\Users\janul\Desktop\thesis_tmp_files\odpovede_z_forms.csv', newline='', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            # print(', '.join(row))
            data_rows.append(row)


    selected_map = [[[0 for i in range(10)] for j in range(10)] for m in range(10)]

    for data_row in data_rows[1:]:
        created, *choices = data_row
        for i_task in range(10):
            selected = choices[i_task * 10: (i_task + 1) * 10]

            for i_row, columns_selected  in enumerate(selected):
                cols = columns_selected.split(", ")
                if cols == ['']:
                    continue
                for c in cols:
                    selected_map[i_task][i_row][number_from_char(c)] += 1

    # import numpy as np
    # return np.ones(dtype=np.int32, shape=(10,10,10)) # Check
    return selected_map

def main():
    for i_task, task_map in enumerate(heat_map_from_responses()):
        print()
        print("task", i_task)
        for row in task_map:
            print(row)

if __name__ == '__main__':
    main()