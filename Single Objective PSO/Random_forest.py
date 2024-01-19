import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Loading dataset
data = pd.read_csv('winequality-white.csv', header=0)
data = data.iloc[:, 0].str.split(';', expand=True)
data.columns = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_SO2',
    'total_SO2', 'density', 'pH', 'sulphates', 'alcohol', 'quality'
]

# Data preprocessing
X = data.drop(columns=['quality'])
y = data['quality']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputing missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Define the fitness function using Random Forest
def fitness_function(features):
    if not features:
        return 0.0

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    selected_feature_indices = [list(X.columns).index(feature) for feature in features]
    selected_features = X_train[:, selected_feature_indices]
    clf.fit(selected_features, y_train)
    y_pred = clf.predict(X_test[:, selected_feature_indices])
    return accuracy_score(y_test, y_pred)

# PSO-like Feature Selection
num_particles = 5
num_iterations = 10
num_features = X.shape[1]
population = np.random.rand(num_particles, num_features)
velocities = np.zeros((num_particles, num_features))
personal_best_positions = population.copy()
global_best_position = population[np.argmax([fitness_function([X.columns[j] for j in range(num_features) if population[i, j] > 0.5]) for i in range(num_particles)])].copy()

# Lists to store progress data
iteration_values = []
accuracy_values = []

# PSO parameters
inertia_weight = 0.5
c1, c2 = 2.0, 2.0

# PSO main loop
for iteration in range(num_iterations):
    for i in range(num_particles):
        # Randomly determine the number of selected features for each particle
        num_selected_features = np.random.randint(1, num_features + 1)

        # Randomly select features based on the determined number
        selected_features_indices = np.random.choice(num_features, num_selected_features, replace=False)
        population[i, :] = 0.0
        population[i, selected_features_indices] = 1.0

        # Update particle velocity and position
        velocities[i] = (inertia_weight * velocities[i] +
                         c1 * np.random.rand() * (personal_best_positions[i] - population[i]) +
                         c2 * np.random.rand() * (global_best_position - population[i]))

        population[i] = np.clip(population[i] + velocities[i], 0, 1)
    # Evaluate fitness for each particle
    fitness_values = [fitness_function([X.columns[j] for j in range(num_features) if population[i, j] > 0.5]) for i in range(num_particles)]

    # Update personal best
    for i in range(num_particles):
        if fitness_values[i] > fitness_function([X.columns[j] for j in range(num_features) if personal_best_positions[i, j] > 0.5]):
            personal_best_positions[i] = population[i].copy()

    # Update global best
    global_best_index = np.argmax(fitness_values)
    if fitness_values[global_best_index] > fitness_function([X.columns[j] for j in range(num_features) if global_best_position[j] > 0.5]):
        global_best_position = population[global_best_index].copy()

    # Store progress data
    iteration_values.append(iteration)
    accuracy_values.append(fitness_function([X.columns[j] for j in range(num_features) if global_best_position[j] > 0.5]))

    # Print iteration details
    best_features_iteration = [X.columns[j] for j in range(num_features) if global_best_position[j] > 0.5]
    print(f"Iteration {iteration + 1}: Best Accuracy - {accuracy_values[-1]}, Best Features - {best_features_iteration}")

# Feature selection based on the global best position
selected_feature_indices = [i for i in range(num_features) if global_best_position[i] > 0.5]
selected_features_pso = [X.columns[i] for i in selected_feature_indices]

# Plotting the progress
plt.figure(figsize=(10, 6))
plt.plot(iteration_values, accuracy_values, marker='o', linestyle='-', color='b')
plt.title("PSO-like Feature Selection Progress for Random Forest (Classification)")
plt.xlabel("Iteration")
plt.ylabel("Best Accuracy")
plt.grid(True)
plt.show()

print('Selected Features (Random Forest - Classification):', selected_features_pso)

selected_features_list = list(selected_features_pso)

X_train_selected_pso = X_train[:, [X.columns.get_loc(feature) for feature in selected_features_list]]
X_test_selected_pso = X_test[:, [X.columns.get_loc(feature) for feature in selected_features_list]]

# Training and evaluating a classifier with the selected features
clf_selected_pso = RandomForestClassifier(n_estimators=100, random_state=42)
clf_selected_pso.fit(X_train_selected_pso, y_train)
y_pred_selected_pso = clf_selected_pso.predict(X_test_selected_pso)
accuracy_pso = accuracy_score(y_test, y_pred_selected_pso)
print('Accuracy with Selected Features (Random Forest - Classification):', accuracy_pso)
