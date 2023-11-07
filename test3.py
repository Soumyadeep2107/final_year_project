import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load dataset and perform data preprocessing
data = pd.read_csv('Fashion_Retail_Sales.csv', header=0)
data['Date Purchase'] = pd.to_datetime(data['Date Purchase'])
data['Month'] = data['Date Purchase'].dt.month
data['Day'] = data['Date Purchase'].dt.day

X = data.drop(columns=['Payment Method','Date Purchase','Item Purchased'])
Y = data['Payment Method']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.fit_transform(X_test)

def fitness_function(features):
    if not features:
        return 0.0
    knn = KNeighborsClassifier(n_neighbors=8)
    selected_features = X_train[:, features]
    knn.fit(selected_features, Y_train)
    Y_pred = knn.predict(X_test[:, features])
    return f1_score(Y_test, Y_pred, pos_label='Cash')

def duplicate(feature):
    set_input = set(feature)
    if len(set_input) < len(feature):
        return 1
    else:
        return 0

# PSO for Feature Selection
num_particles = 5
num_iterations = 10
num_features = X.shape[1]
population = []
fitness_values = [0.0] * num_particles

# Feature Subset Size
R = 3

# Initial feature subset generation
for x in range(num_particles):
    random_features_indices = random.sample(range(num_features), R)
    feature_subset = [i for i in random_features_indices]
    population.append(feature_subset)

# Initialize velocities
velocities = np.zeros((num_particles, R))

# Initialize personal and global best
personal_best = population.copy()
global_best = population[np.argmax(fitness_values)].copy()

# Lists to store progress data
iteration_values = []
best_fitness_values = []

# PSO parameters
inertia_weight = 0.5
c1, c2 = 2.0, 2.0

# PSO main loop
for iteration in range(num_iterations):
    for i in range(num_particles):
        for j in range(R):
            r1, r2 = random.random(), random.random()
            velocities[i, j] = (inertia_weight * velocities[i, j] +
                                c1 * r1 * (personal_best[i][j] - population[i][j]) +
                                c2 * r2 * (global_best[j] - population[i][j]))
            # Update particle position
            population[i][j] = round(population[i][j] + velocities[i, j])

    # Manage particles which go out of search space
    for p in range(num_particles):
        for s in range(R):
            x = population[p][s]
            if x < 0 or x >= num_features:
                population[p][s] = random.randint(0, num_features - 1)

    # Manage repeated feature indices with random initialization
    for p in range(num_particles):
        check = duplicate(population[p])
        if check == 1:
            random_features_indices = random.sample(range(num_features), R)
            population[p] = [i for i in random_features_indices]

    # Update fitness values
    for i in range(num_particles):
        fitness_values[i] = fitness_function(population[i])

    # Update personal best
    for i in range(num_particles):
        if fitness_values[i] > fitness_function(personal_best[i]):
            personal_best[i] = population[i].copy()

    # Update global best
    if max(fitness_values) > fitness_function(global_best):
        global_best = population[np.argmax(fitness_values)].copy()

    # Store progress data
    iteration_values.append(iteration)
    best_fitness_values.append(fitness_function(global_best))

    # Print iteration details
    print(f"Iteration {iteration + 1}: ")
    print("Population update: ")
    print(population)
    print("Corresponding fitness: ")
    print(fitness_values)
    print("Best feature subset selected: ")
    print(global_best)
    print("Best fitness of feature subset: ")
    print(fitness_function(global_best))
    print("\n")

# Plotting the progress
plt.figure(figsize=(10, 6))
plt.plot(iteration_values, best_fitness_values, marker='o', linestyle='-', color='b')
plt.title("PSO Feature Selection Progress")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness (F1 Score)")
plt.grid(True)
plt.show()

# Final selected feature subset and its accuracy
print("FINAL FEATURE SUBSET SELECTED: ")
print(global_best)
print("Accuracy of the Feature subset: ")
print(fitness_function(global_best))
