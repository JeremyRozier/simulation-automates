from NFA_convolution_network import *
import csv
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

x_values = []

for i in range(1,102,10):
    x_values.append(i)

print("loading statrting for ")
print(x_values)

is_automaton_found = False

# Set up the paths to the stored automata and the CSV file
csv_file_path = "simulation-automates/automaton_accuracies_vector.csv"
if os.path.exists(csv_file_path):
    with open(csv_file_path, "r") as f:
        obj = csv.reader(f)
        last_line = [line for line in obj][-1]
        name_last_automaton = last_line[1]
        size_last_automaton = int(last_line[0])
else:
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Size", "Automata Name", "Accuracies"])
    is_automaton_found = True
# Create a list of all the automata file names and their sizes
automata_files = []
size_range = range(1, 49, 5)
for size in size_range:
    folder_path = f"simulation-automates/stored_automatons/size_{size}"
    automata_names = os.listdir(folder_path)
    if (
        not is_automaton_found
        and size == size_last_automaton
        and name_last_automaton in automata_names
    ):
        is_automaton_found = True
        index_last_automaton = automata_names.index(name_last_automaton)

    elif not is_automaton_found:
        continue

    automata_files.append((size, os.path.join(folder_path, 'aut_1.pkl')))
accuracy_list = []

compteur = 0
for size, file in automata_files:
    automaton = Automaton.load_automaton(file)
    acc = []
    for v_size in x_values:
        a = Automaton2Network.get_accuracy(automaton, units = v_size)
        acc.append(a)
    file_name = os.path.basename(file)
    accuracy_list.append((size, os.path.basename(file_name), acc))
    compteur += 1
    if compteur % 10 == 0:
        print("writing")
        with open(csv_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            for accuracy in accuracy_list:
                writer.writerow(accuracy)
        accuracy_list.clear()
