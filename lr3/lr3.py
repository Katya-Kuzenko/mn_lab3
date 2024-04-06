import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
from sklearn.tree import export_graphviz
import graphviz 
from sklearn.metrics import classification_report


# Крок 1
file_name = 'dataset_2.txt' 
df = pd.read_csv(file_name, header = None) 

# Крок 2
num = df.shape
num_rows, num_columns = num
print(f"Кількість записів: {num_rows}")
print(f"Кількість полів: {num_columns}")

# Крок 3
print(f"\nПерші 10 записів:")
print(df.head(10)) 

# Крок 4
X = df.iloc[:, 2:-1] # вихідні аргументи
y = df.iloc[:, -1] # цільова характеристика
print(f'Вихідні аргументи: \n{X}')
print(f'Цільова характеристика: \n{y}')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
print(X_train.shape, X_test.shape)

# Крок 5
tree_gini = DecisionTreeClassifier(max_depth=5, random_state=1)
tree_gini.fit(X_train, y_train)
y_test_pred_gini = tree_gini.predict(X_test)

# Крок 6
dot_data_gini = export_graphviz(tree_gini, out_file=None) 
graph_gini = graphviz.Source(dot_data_gini) 
graph_gini.render("tree_gini") 

# Крок 7
tree_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=1)
tree_entropy.fit(X_train, y_train)
y_test_pred_entropy = tree_entropy.predict(X_test)
dot_data_entropy = export_graphviz(tree_entropy, out_file=None) 
graph_entropy = graphviz.Source(dot_data_entropy) 
graph_entropy.render("tree_entropy") 

classification_test_gini = classification_report(y_test, y_test_pred_gini)
classification_test_entropy = classification_report(y_test, y_test_pred_entropy)

print(f'Класифікаційні метрики для тестової вибірки (Джині): \n{classification_test_gini}')
print(f'Класифікаційні метрики для тестової вибірки (ентропія): \n{classification_test_entropy}')

plt.figure(figsize=(10,6))
plt.subplot(1, 2, 1)
plt.title('Модель на тестовій вибірці (за критерієм Джині)')
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test_pred_gini, cmap='viridis', marker='o', alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.subplot(1, 2, 2)
plt.title('Модель на тестовій вибірці (за критерієм ентропії)')
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test_pred_entropy, cmap='viridis', marker='o', alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Крок 8
leaf_sizes = range(2, 20, 2) 
train_accuracies = []
test_accuracies = []
for leaf_size in leaf_sizes:
    tree = DecisionTreeClassifier(max_leaf_nodes=leaf_size, random_state=1)
    tree.fit(X_train, y_train)
    train_accuracies.append(tree.score(X_train, y_train))
    test_accuracies.append(tree.score(X_test, y_test))
plt.figure(figsize=(10, 6))
plt.plot(leaf_sizes, train_accuracies, label='Train Accuracy')
plt.plot(leaf_sizes, test_accuracies, label='Test Accuracy')
plt.xlabel('Max Leaf Nodes')
plt.ylabel('Accuracy')
plt.title('Impact of Max Leaf Nodes on Accuracy')
plt.legend()
plt.show()

# Крок 9
importances = tree_gini.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances, align='center')
plt.xticks(range(len(importances)), X.columns, rotation=90)
plt.xlabel('Attributes')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()