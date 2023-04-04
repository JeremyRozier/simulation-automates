from NFA_convolution_network import *
import csv


print("loading statrting")
# Set up the paths to the stored automata and the CSV file
stored_automatons_path = "./simulation-automates/stored_automatons/"
csv_file_path = "automaton_accuracies.csv"

# Create a list of all the automata file names and their sizes
automata_files = []
size_range = range(1,48)
for size in size_range:
    folder_path = os.path.join(stored_automatons_path, f"size_{size}")
    automata_names = os.listdir(folder_path)
    for name in automata_names:
        automata_files.append((size, os.path.join(folder_path, name)))

# Create a list to hold the accuracy and file name for each automaton
accuracy_list = []

# Loop through all the automata files, load the automaton, train the model, and record the accuracy
for size, file in automata_files:
    automaton = Automaton.load_automaton(file)
    acc = Automaton2Network.get_accuracy(automaton)
    file_name = os.path.basename(file)
    print(size, file_name, acc)
    accuracy_list.append((size, os.path.basename(file_name), acc))

# Write the accuracy data to the CSV file
with open(csv_file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Size", "Automata Name", "Accuracy"])
    for accuracy in accuracy_list:
        writer.writerow(accuracy)
