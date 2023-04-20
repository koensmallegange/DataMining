import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from collections import Counter
import numpy as np
import re
from scipy.stats import zscore
from graphviz import Digraph
from sklearn.model_selection import train_test_split
# from fancyimpute import IterativeImputer
import sys 

# ... (keep your existing functions here) ...

def build_decision_tree(df, remaining_columns, depth=0, max_depth=3):
    # Base case: If there are no remaining columns or the DataFrame is empty, return
    # get rid of this last bit of code. Its only for visibility.
    if not remaining_columns or df.empty or depth == max_depth:
        return

    # Find the column with the lowest entropy
    attribute, lowest_entropy = find_lowest_entropy(df, remaining_columns)

    # Remove the selected column from the remaining columns
    remaining_columns = [col for col in remaining_columns if col != attribute]

    # Initialize the decision tree node
    node = {'attribute': attribute, 'children': {}}

    # Split the DataFrame based on the selected column's unique values
    unique_values = df[attribute].unique()

    for value in unique_values:
        # Apply the split on the selected column
        split_df = df[df[attribute] == value].copy()

        # Call the recursive function on the split DataFrame with increased depth
        subtree = build_decision_tree(split_df, remaining_columns, depth + 1)

        # Store the subtree in the children dictionary
        node['children'][value] = subtree

    return node
        

def find_lowest_entropy(df, remaining_columns):
    lowest_entropy = float("inf")
    selected_attribute = None
    print('remaining')
    print(remaining_columns)
    for col in remaining_columns:
        print(col)
        if col in [df.columns[2], df.columns[3], df.columns[4], df.columns[5], df.columns[6], df.columns[7], df.columns[10]]:
            value_list = [0, 1]
        elif col == df.columns[1]:
            value_list = ['AI', 'Econometrics', 'Computational Science', 'Quantitative Risk Management', 'Business Analytics', 'Computer Science', 'Finance and Technology', 'Bioinformatics', 'Exhange Programme', 'Neuroscience', 'PhD', 'Life Sciences', 'Other']
        elif col == df.columns[9]:
            value_list = ['0-50', '51-100', '101-200', '201-300', '301-400', '401-500', '501-1000', '1001+']
        elif col == df.columns[12]:
            value_list = ['0', '0-2', '2-4', '4-6', '6-10', '10-15', '15+']
        else:
            continue
        
        print(col)

        col_dict = attributes(col, df, value_list)
        weighted_entropy = entropy(col_dict)
        if weighted_entropy <= lowest_entropy:
            lowest_entropy = weighted_entropy
            selected_attribute = col
    print('selected')
    print(selected_attribute)
    return selected_attribute, lowest_entropy

# Main function to call
def decision_tree_algorithm(df):
    remaining_columns = list(df.columns)
    remaining_columns.remove("What is your stress level (0-100)?")  # Assuming this is the target variable
    remaining_columns.remove(df.columns[0])
    remaining_columns.remove(df.columns[8])   # Assuming this is the irrelevant "Birthday" column
    remaining_columns.remove(df.columns[13])  # Removing empty columns
    remaining_columns.remove(df.columns[14])
    remaining_columns.remove(df.columns[15])
    remaining_columns.remove(df.columns[16])
    decision_tree = build_decision_tree(df, remaining_columns)
    return decision_tree

def make_prediction(tree, data_point):
    if not tree['children']:
        return None

    attribute = tree['attribute']
    value = data_point[attribute]

    if value in tree['children']:
        subtree = tree['children'][value]
        if subtree is not None:
            return make_prediction(subtree, data_point)
        else:
            return None
    else:
        return None

def attributes(col,df,value_list):
    number_of_values = len(value_list)
    student_dict = {}
    for value in value_list:
        stress_levels = stress_level_evaluator(value,col,df,number_of_values)
        student_dict[value] = stress_levels
    return student_dict

def stress_level_evaluator(value,col,df,number_of_values):
    # need to first create 10 different totals of the stress 
    stress_levels = np.zeros((1,10))
    row_numbers = df[df[col] == value].index.tolist()
    print(value)
    print('row')
    print(row_numbers)
    stress = 'What is your stress level (0-100)?'
    total = 0
    for row_number in row_numbers:
        value = df.loc[row_number, stress]
        print('ja')
        # Placeholder code
        try:
            value = int(value)
        except ValueError:
            value = 1

        total = total + 1
        if value >= 0 and value < 10:
            stress_levels[0,0] = stress_levels[0,0] + 1
        if value >= 10 and value < 20:
            stress_levels[0,1] = stress_levels[0,1] + 1
        if value >= 20 and value < 30:
            stress_levels[0,2] = stress_levels[0,2] + 1
        if value >= 30 and value < 40:
            stress_levels[0,3] = stress_levels[0,3] + 1
        if value >= 40 and value < 50:
            stress_levels[0,4] = stress_levels[0,4] + 1
        if value >= 50 and value < 60:
            stress_levels[0,5] = stress_levels[0,5] + 1
        if value >= 60 and value < 70:
            stress_levels[0,6] = stress_levels[0,6] + 1
        if value >= 70 and value < 80:
            stress_levels[0,7] = stress_levels[0,7] + 1
        if value >= 80 and value < 90:
            stress_levels[0,8] = stress_levels[0,8] + 1
        if value >= 90:
            stress_levels[0,9] = stress_levels[0,9] + 1

    stress_level = stress_levels[0]

    # Laplace replacement and probability calculation.
    number_of_zeros = np.count_nonzero(stress_level == 0)
    i = 0
    # HIER NIET ZEKER OF DE LEGE LIJSTEN OOK ALLEMAAL VALUES MOETEN KRIJGEN
    # EN OOK NIET OF DE NIET-NUL VALUES OOK EEN EXTRA WAARDE MOETEN KRIJGEN MAAR NEEM AAN VAN WEL
    # VRAAG DIT AAN TA.
    for item in stress_level:
        chance_per_stress_level = (item + number_of_zeros)/(total + number_of_zeros * 10)
        stress_level[i] = chance_per_stress_level
        i = i + 1
        
    return stress_level

def weighted_average(list_for_weight,entropydict):

    new_dict = {}
    i = 0
    totalsum = sum(list_for_weight)
    weighted_entropy = 0

    for values in entropydict.values():
        weighted_entropy = weighted_entropy + (list_for_weight[i]/totalsum) * values
        i = i + 1
    return weighted_entropy

def entropy(dictionary):
    
    entropydict = {}
    list_for_weight = []
    i = 0

    for values in dictionary.values():
    
        entropyscore = 0
        
        for single_value in values:
            entropyscore = entropyscore - (single_value * np.log2(single_value))

        list_for_weight.append(sum(values))
        entropydict[list(dictionary.keys())[i]] = entropyscore
        i = i + 1
    weighted_entropy = weighted_average(list_for_weight,entropydict)

    return weighted_entropy


# plot using networkx
def plot_decision_tree(tree):
    # Create a networkx graph from the decision tree dictionary
    graph = nx.DiGraph()
    add_nodes_edges(tree, graph, parent=None)

    # Create plot layout and draw the graph
    pos = nx.drawing.nx_pydot.pydot_layout(graph, prog='dot')
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')

    # Show the plot
    plt.show()

def add_nodes_edges(tree, graph, parent=None):
    attribute = tree['attribute']
    children = tree['children']

    # Add the current attribute as a node in the graph
    graph.add_node(attribute)

    # If there is a parent, add an edge between the parent and the current attribute
    if parent is not None:
        graph.add_edge(parent, attribute)

    # Iterate through the children and add them to the graph
    for value, child in children.items():
        if child is not None:
            # Add child nodes and edges recursively
            add_nodes_edges(child, graph, parent=attribute)
        else:
            # If there is no child, add a leaf node with the value
            leaf_node = f"{attribute}={value}"
            graph.add_node(leaf_node)
            graph.add_edge(attribute, leaf_node)

df = pd.read_csv('data.csv', sep=';')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

tree = decision_tree_algorithm(train_df)

# Make predictions on the test set
predictions = []
for _, row in test_df.iterrows():
    prediction = make_prediction(tree, row)
    predictions.append(prediction)

print(predictions)

# Test decision tree algorithm
tree = decision_tree_algorithm()
# graph = plot_decision_tree(tree)
# graph.view()
print(tree)


# plot using graphviz
# def plot_decision_tree(tree, graph=None, parent=None, edge_label=None):
#     if graph is None:
#         graph = Digraph()

#     for attribute, subtree in tree.items():
#         if parent is not None:
#             graph.edge(parent, attribute, label=edge_label)

#         if isinstance(subtree, dict):
#             for value, child_tree in subtree.items():
#                 if isinstance(child_tree, dict):
#                     plot_decision_tree(child_tree, graph, parent=attribute, edge_label=str(value))
#                 else:
#                     graph.edge(attribute, f"{attribute}={value}", label=str(value))
#         else:
#             graph.edge(attribute, f"{attribute}={subtree}", label=str(subtree))

#     return graph
