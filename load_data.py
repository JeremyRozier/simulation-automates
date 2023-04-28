from NFA_convolution_network import *
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# print("loading statrting")

# is_automaton_found = False

# # Set up the paths to the stored automata and the CSV file
# csv_file_path = "simulation-automates/half_automaton_accuracies.csv"
# if os.path.exists(csv_file_path):
#     with open(csv_file_path, "r") as f:
#         # Créer un objet csv à partir du fichier
#         obj = csv.reader(f)
#         last_line = [line for line in obj][-1]
#         name_last_automaton = last_line[1]
#         size_last_automaton = int(last_line[0])
# else:
#     with open(csv_file_path, mode="w", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow(["Size", "Automata Name", "Accuracy"])
#     is_automaton_found = True
# # Create a list of all the automata file names and their sizes
# automata_files = []
# size_range = range(1, 49)
# for size in size_range:
#     name_indexes = (0, 101)
#     folder_path = f"simulation-automates/stored_automatons_half/stored_automatons_half/size_{size}"
#     automata_names = os.listdir(folder_path)
#     if (
#         not is_automaton_found
#         and size == size_last_automaton
#         and name_last_automaton in automata_names
#     ):
#         is_automaton_found = True
#         index_last_automaton = automata_names.index(name_last_automaton)
#         name_indexes = (index_last_automaton + 1, 101)

#     elif not is_automaton_found:
#         continue

#     for name in automata_names[name_indexes[0] : name_indexes[1]]:
#         automata_files.append((size, os.path.join(folder_path, name)))
# # Create a list to hold the accuracy and file name for each automaton
# accuracy_list = []

# # Loop through all the automata files, load the automaton, train the model, and record the accuracy
# compteur = 0
# for size, file in automata_files:
#     automaton = Automaton.load_automaton(file)
#     acc = Automaton2Network.get_accuracy(automaton)
#     file_name = os.path.basename(file)
#     accuracy_list.append((size, os.path.basename(file_name), acc))
#     compteur += 1
#     if compteur % 10 == 0:
#         print("writing")
#         with open(csv_file_path, mode="a", newline="") as file:
#             writer = csv.writer(file)
#             for accuracy in accuracy_list:
#                 writer.writerow(accuracy)
#         accuracy_list.clear()

csv_file = "simulation-automates/half_automaton_accuracies.csv"
csv_file_der = "simulation-automates/automaton_accuracies.csv"

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_der)
dt = pd.read_csv(csv_file)


# Group the data by Size and compute the min and max accuracy values for each Size
grouped_data_der = df.groupby("Size").agg({"Accuracy": ["min", "max"]})
grouped_data_der.columns = ["_".join(col) for col in grouped_data_der.columns]
grouped_data = dt.groupby("Size").agg({"Accuracy": ["min", "max"]})
grouped_data.columns = ["_".join(col) for col in grouped_data.columns]

# Plot the min and max accuracy values for each Size on a graph
plt.plot(grouped_data_der.index, grouped_data_der["Accuracy_min"], label="Min Accuracy with 10% final")
plt.plot(grouped_data_der.index, grouped_data_der["Accuracy_max"], label="Max Accuracy with 10% final")
plt.plot(grouped_data.index, grouped_data["Accuracy_min"], label="Min Accuracy with 65% final")
plt.plot(grouped_data.index, grouped_data["Accuracy_max"], label="Max Accuracy with 65% final")
plt.xlabel("Size")
plt.ylabel("Accuracy")
plt.title("Min and Max Accuracy by Size")
plt.legend()
plt.savefig('min_max_65.png')
plt.show()

grouped_data_der = df.groupby("Size").mean()
grouped_data = dt.groupby("Size").mean()

# Plot the mean accuracy for each Size on a graph
plt.plot(grouped_data_der.index, grouped_data_der["Accuracy"], label="Mean Accuracy with 10% final")
plt.plot(grouped_data.index, grouped_data["Accuracy"], label="Mean Accuracy with 65% final")
plt.xlabel("Size")
plt.ylabel("Accuracy")
plt.title("Mean Accuracy by Size")
plt.legend()
plt.savefig('mean_acc_65.png')
plt.show()

