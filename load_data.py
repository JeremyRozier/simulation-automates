from NFA_convolution_network import *
import csv
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


print("loading statrting")

# Set up the paths to the stored automata and the CSV file
csv_file_path = "simulation-automates/automaton_accuracies.csv"

if os.path.exists(csv_file_path):
    with open(csv_file_path, "r") as f:
        # Créer un objet csv à partir du fichier
        obj = csv.reader(f)
        last_line = [line for line in obj][-1]
        name_last_automaton = last_line[1]
        if name_last_automaton == "Automata Name":
            name_last_automaton = ""
else:
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Size", "Automata Name", "Accuracy"])
    name_last_automaton = ""
print(name_last_automaton)
# Create a list of all the automata file names and their sizes
automata_files = []
size_range = range(2, 49)
is_automaton_found = False
for size in size_range:
    name_indexes = (0, 101)
    folder_path = f"simulation-automates/stored_automatons/size_{size}"
    automata_names = os.listdir(folder_path)
    if (
        len(name_last_automaton) != 0
        and name_last_automaton not in automata_names
        and not is_automaton_found
    ):
        continue
    elif name_last_automaton in automata_names:
        is_automaton_found = True
        index_last_automaton = automata_names.index(name_last_automaton)
        name_indexes = (index_last_automaton + 1, 101)
    for name in automata_names[name_indexes[0]:name_indexes[1]]:
        automata_files.append((size, os.path.join(folder_path, name)))
# Create a list to hold the accuracy and file name for each automaton
accuracy_list = []

# Loop through all the automata files, load the automaton, train the model, and record the accuracy
compteur = 0
for size, file in automata_files:
    automaton = Automaton.load_automaton(file)
    acc = Automaton2Network.get_accuracy(automaton)
    file_name = os.path.basename(file)
    accuracy_list.append((size, os.path.basename(file_name), acc))
    compteur += 1
    print((size, os.path.basename(file_name)))
    if compteur % 10 == 0:
        print('writing')
        with open(csv_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            for accuracy in accuracy_list:
                writer.writerow(accuracy)
        accuracy_list.clear()
