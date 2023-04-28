from NFA_convolution_network import *
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# x_values = []

# for i in range(1,102,10):
#     x_values.append(i)

# print("loading statrting for ")
# print(x_values)

# is_automaton_found = False

# # Set up the paths to the stored automata and the CSV file
# csv_file_path = "simulation-automates/automaton_accuracies_vector.csv"
# if os.path.exists(csv_file_path):
#     with open(csv_file_path, "r") as f:
#         obj = csv.reader(f)
#         last_line = [line for line in obj][-1]
#         name_last_automaton = last_line[1]
#         size_last_automaton = int(last_line[0])
# else:
#     with open(csv_file_path, mode="w", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow(["Size", "Automata Name", "Accuracies"])
#     is_automaton_found = True
# # Create a list of all the automata file names and their sizes
# automata_files = []
# size_range = range(1, 49, 5)
# for size in size_range:
#     folder_path = f"simulation-automates/stored_automatons/size_{size}"
#     automata_names = os.listdir(folder_path)
#     if (
#         not is_automaton_found
#         and size == size_last_automaton
#         and name_last_automaton in automata_names
#     ):
#         is_automaton_found = True
#         index_last_automaton = automata_names.index(name_last_automaton)

#     elif not is_automaton_found:
#         continue

#     automata_files.append((size, os.path.join(folder_path, 'aut_1.pkl')))
# accuracy_list = []

# compteur = 0
# for size, file in automata_files:
#     automaton = Automaton.load_automaton(file)
#     acc = []
#     for v_size in x_values:
#         a = Automaton2Network.get_accuracy(automaton, units = v_size)
#         acc.append(a)
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

csv_file_path = "simulation-automates/automaton_accuracies_vector_uniform.csv"
# Charger les données à partir du fichier CSV
df = pd.read_csv(csv_file_path)

# Créer la figure et les axes
fig, ax = plt.subplots()
cmap = ['#FFFFCC', '#FFF39C', '#FFEB6F', '#FFE146', '#FFC72C', '#FFAD1C', '#FF9110', '#FF7308', '#FF5205']

# Tracer les courbes pour chaque taille
sizes = df['Size'].unique()
for size in sizes:
    accuracies = np.array(eval(df[df['Size']==size]['Accuracies'].values[0]))
    
    # Appliquer la moyenne mobile (moving average) avec une fenêtre de 5 points
    color = cmap[0]
    cmap.remove(color)
    ax.plot(range(1,102,10), accuracies, label=f'Size {size}', color = color)

# Fixer les limites de l'axe des ordonnées
min_acc = df['Accuracies'].apply(lambda x: min(eval(x))).min()
max_acc = df['Accuracies'].apply(lambda x: max(eval(x))).max()
ax.set_ylim([min_acc, max_acc])

# Ajouter les légendes et titre
ax.set_xlabel('vector size')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracies for different sizes of Automata')
ax.legend()

# Enregistrer la figure
plt.savefig('vector_influence.png')

# Afficher la figure
plt.show()


