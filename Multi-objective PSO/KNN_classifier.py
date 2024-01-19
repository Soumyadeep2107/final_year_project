import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer

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

# Fitness function using F1 score
def fitness_function(features):
    selected_feature_indices = np.where(features == 1)[0]
    
    if not selected_feature_indices.any():
        return 0.0

    clf = KNeighborsClassifier(n_neighbors=5)
    selected_features = X_train[:, selected_feature_indices]
    clf.fit(selected_features, y_train)
    y_pred = clf.predict(X_test[:, selected_feature_indices])
    return f1_score(y_test, y_pred, average='weighted')

class Particle:
    def __init__(self, num_features):
        self.position = np.random.choice(2, num_features)  # Binary array indicating selected features
        self.velocity = np.random.choice([-1, 0, 1], num_features)  # Random initial velocity
        self.best_position = np.copy(self.position)
        self.fitness = fitness_function(np.where(self.position == 1)[0])

def is_pareto_dominant(sol1, sol2):
    return sol1[0] >= sol2[0] and sol1[1] < sol2[1]

def update_pareto_front(pareto_front, new_solution):
    updated_pareto_front = []
    dominated = False

    for solution in pareto_front:
        if is_pareto_dominant(solution[1], new_solution[1]):
            dominated = True
            break
        elif not is_pareto_dominant(new_solution[1], solution[1]):
            updated_pareto_front.append(solution)

    if not dominated:
        updated_pareto_front.append(new_solution)

    return updated_pareto_front

# Particle Swarm Optimization for multi-objective optimization
def pso_multiobjective(pop_size=100, num_generations=10):
    num_features = X.shape[1]
    particles = [Particle(num_features) for _ in range(pop_size)]
    global_best_position = None
    global_best_fitness = float('-inf')
    pareto_front = []

    iteration_values = []
    pareto_front_values = []

    for gen in range(num_generations):
        for particle in particles:
            # Update velocity and position
            inertia_weight = 0.7
            c1 = 1.5  # Cognitive component
            c2 = 1.5  # Social component

            if global_best_position is not None:
                particle.velocity = (inertia_weight * particle.velocity +
                                     c1 * np.random.rand() * (particle.best_position - particle.position) +
                                     c2 * np.random.rand() * (global_best_position - particle.position))
            else:
                # If global_best_position is None, set velocity to a random value
                particle.velocity = np.random.choice([-1, 0, 1], num_features)

            particle.velocity = np.clip(particle.velocity, -1, 1)  # Clip velocity to [-1, 1]
            particle.position = np.clip(particle.position + particle.velocity, 0, 1)  # Clip position to [0, 1]

            # Update fitness
            selected_features = np.where(particle.position == 1)[0]
            f1_score_value = fitness_function(selected_features)

            # Update personal best
            if f1_score_value >= particle.fitness:
                particle.fitness = f1_score_value
                particle.best_position = np.copy(particle.position)

            # Update global best
            if f1_score_value >= global_best_fitness:
                global_best_fitness = f1_score_value
                global_best_position = np.copy(particle.position)

            # Update Pareto front
            new_solution = (particle.position, (f1_score_value, len(selected_features)))
            pareto_front = update_pareto_front(pareto_front, new_solution)

        iteration_values.append(gen + 1)
        pareto_front_values.append([solution[1] for solution in pareto_front])

        print(f"Generation {gen + 1}: Global Best - {global_best_position}")
        print(f"{global_best_fitness} {len(np.where(global_best_position == 1)[0])}")

    # Display the best solutions in the Pareto front
    print("\nBest Solutions in Pareto Front:")
    for solution in pareto_front:
        selected_features = np.where(solution[0] == 1)[0]
        print(f"Selected Features: {selected_features}, F1 Score: {solution[1][0]}, Num Features: {solution[1][1]}")

    # Plotting the Pareto front
    plt.figure(figsize=(10, 6))
    for i in range(min(len(pareto_front_values[0]), len(pareto_front_values[1]))):
        objectives_values = [gen[i][0] if i < len(gen) else None for gen in pareto_front_values]
        plt.plot(iteration_values, objectives_values, label=f'Objective {i + 1}')

    plt.title("Pareto Front Progress")
    plt.xlabel("Generation")
    plt.ylabel("Objective Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# Running the PSO multi-objective optimization algorithm
pso_multiobjective()
