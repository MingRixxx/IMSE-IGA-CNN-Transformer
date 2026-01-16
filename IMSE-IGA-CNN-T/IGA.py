import torch
import torch.utils.data as Data
import random
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn, optim
from Model import EMDCNNTransformer, model_train
from joblib import dump, load
import seaborn as sns

def dataloader(batch_size, workers=2):
    try:
        train_xdata = load('emd_testX_1024_10c').to(torch.float32)
        train_ylabel = load('emd_testY_1024_10c').to(torch.int64)
        val_xdata = load('emd_testX_1024_10c').to(torch.float32)
        val_ylabel = load('emd_testY_1024_10c').to(torch.int64)
        test_xdata = load('emd_testX_1024_10c').to(torch.float32)
        test_ylabel = load('emd_testY_1024_10c').to(torch.int64)

        train_loader = Data.DataLoader(Data.TensorDataset(train_xdata, train_ylabel), batch_size=batch_size,shuffle=True, num_workers=workers, drop_last=True)
        val_loader = Data.DataLoader(Data.TensorDataset(val_xdata, val_ylabel), batch_size=batch_size, shuffle=True,num_workers=workers, drop_last=True)
        test_loader = Data.DataLoader(Data.TensorDataset(test_xdata, test_ylabel), batch_size=batch_size, shuffle=True,num_workers=workers, drop_last=True)

        return train_loader, val_loader, test_loader

    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        raise e

batch_size = 32
train_loader, val_loader, test_loader = dataloader(batch_size)
learning_rate = 0.003
hidden_dim = 128
num_layers = 4
num_heads = 2
input_dim = 7 * 8
output_dim = 10
epochs = 100
conv_archs = ((1, 64), (1, 128))

def fitness_function(params=None):
    if params is None:
        params = {
            'input_dim': 56, 'output_dim': 10, 'hidden_dim': hidden_dim,
            'num_layers': num_layers, 'num_heads': num_heads,
            'conv_archs': ((1, 64), (1, 128))
        }

    print(f"评估参数: {params}")

    try:
        model = EMDCNNTransformer(batch_size, input_dim, conv_archs, output_dim, hidden_dim, num_layers, num_heads)
        loss_function = nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

        accuracy = model_train(batch_size, epochs, model, optimizer, loss_function, train_loader, val_loader)

        if accuracy is None:
            print("Warning: model_train returned None for accuracy.")
            return 0

        print(f"验证准确率: {accuracy:.4f}")
        return accuracy

    except Exception as e:
        print(f"Error during fitness evaluation: {e}")
        return 0

class FitnessTracker:
    def __init__(self):
        self.best_fitness_per_iter = []
        self.avg_fitness_per_iter = []

    def update(self, fitness_scores):
        if fitness_scores:
            self.best_fitness_per_iter.append(max(fitness_scores))
            self.avg_fitness_per_iter.append(np.mean(fitness_scores))

    def plot_fitness_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_per_iter, label="最佳适应度", marker="o")
        plt.plot(self.avg_fitness_per_iter, label="平均适应度", linestyle="--")
        plt.title("适应度曲线")
        plt.xlabel("迭代次数")
        plt.ylabel("适应度 (验证准确率)")
        plt.legend()
        plt.grid()

        max_iterations = len(self.best_fitness_per_iter)
        max_fitness = max(self.best_fitness_per_iter)
        min_fitness = min(self.avg_fitness_per_iter)
        plt.xlim(0, max_iterations)
        plt.ylim(min_fitness - 0.01, max_fitness + 0.01)

        plt.show()

def get_best_fitness(population, fitness_scores):
    if not fitness_scores:
        print("No valid fitness scores available.")
        return None, None

    max_fitness = max(fitness_scores)
    best_individual = population[fitness_scores.index(max_fitness)]
    return best_individual, max_fitness

class ImmuneAlgorithm:
    def __init__(self, param_space, fitness_func, pop_size=20, clone_rate=0.4, max_iter=50,fitness_threshold=50, patience=5, initial_mutation_rate=0.2, concentration_threshold=0.3):
        self.param_space = param_space
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.clone_rate = clone_rate
        self.mutation_rate = initial_mutation_rate
        self.max_iter = max_iter
        self.fitness_threshold = fitness_threshold
        self.patience = patience
        self.concentration_threshold = concentration_threshold
        self.population = self.init_population()
        self.tracker = FitnessTracker()
        self.memory = []

    def init_population(self):
        return [{key: random.choice(vals) for key, vals in self.param_space.items()} for _ in range(self.pop_size)]

    def get_best_fitness(self):
        fitness_scores = [self.evaluate_fitness(individual) for individual in self.population]
        best_individual, best_fitness = get_best_fitness(self.population, fitness_scores)
        return best_individual, best_fitness

    def evaluate_fitness(self, individual):
        return self.fitness_func(individual)

    def clone_and_mutate(self, individual):
        clone = individual.copy()
        for key in clone:
            if random.random() < self.mutation_rate:
                clone[key] = random.choice(self.param_space[key])
        return clone

    def dynamic_mutation_rate(self, iteration, max_iter):
        return max(0.01, self.mutation_rate * (1 - iteration / max_iter))

    def suppress_population(self, population, fitness_scores):
        unique_population, unique_fitness = [], []
        seen = set()
        for i, individual in enumerate(population):
            is_similar = False
            for unique_ind in unique_population:
                if self.similarity(individual, unique_ind) > self.concentration_threshold:
                    is_similar = True
                    break
            if not is_similar and frozenset(individual.items()) not in seen:
                seen.add(frozenset(individual.items()))
                unique_population.append(individual)
                unique_fitness.append(fitness_scores[i])
        return unique_population, unique_fitness

    def similarity(self, ind1, ind2):
        overlap = sum(1 for k in ind1 if ind1[k] == ind2[k]) / len(ind1)
        return overlap

    def update_memory(self, individual, fitness):
        if not self.memory or fitness > self.memory[-1][1]:
            self.memory.append((individual, fitness))
            self.memory = sorted(self.memory, key=lambda x: x[1], reverse=True)[:10]

    def optimize(self):
        best_individual = None
        best_fitness = -float('inf')
        patience_counter = 0

        for iteration in range(self.max_iter):
            self.mutation_rate = self.dynamic_mutation_rate(iteration, self.max_iter)

            fitness_scores = [self.evaluate_fitness(individual) for individual in self.population]
            fitness_scores = [score for score in fitness_scores if score is not None]

            if not fitness_scores:
                print("No valid fitness scores in this iteration. Stopping optimization.")
                break

            self.tracker.update(fitness_scores)

            current_best_individual, current_best_fitness = get_best_fitness(self.population, fitness_scores)

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual
                patience_counter = 0
            else:
                patience_counter += 1

            self.update_memory(best_individual, best_fitness)

            if patience_counter >= self.patience:
                print("Early stopping due to lack of improvement.")
                break

            num_to_clone = int(self.clone_rate * self.pop_size)
            top_individuals = [self.population[i] for i in np.argsort(fitness_scores)[-num_to_clone:]]
            new_population = top_individuals.copy()

            for individual in top_individuals:
                new_population.append(self.clone_and_mutate(individual))

            new_population, _ = self.suppress_population(new_population, fitness_scores)
            self.population = new_population

            self.population.extend([mem[0] for mem in self.memory[:5]])

        return best_individual, best_fitness

def train_and_validate(best_individual):
    model = EMDCNNTransformer(**best_individual)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learn_rate = 0.0003
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    loss_function = nn.CrossEntropyLoss()

    model_train(batch_size, epochs, model, optimizer, loss_function, train_loader, val_loader)

    class_labels = []
    predicted_labels = []

    with torch.no_grad():
        for test_data, test_label in test_loader:
            model.eval()
            test_data = test_data.to(device)
            test_output = model(test_data)
            probabilities = F.softmax(test_output, dim=1)
            predicted = torch.argmax(probabilities, dim=1)

            class_labels.extend(test_label.tolist())
            predicted_labels.extend(predicted.tolist())

    confusion_mat = confusion_matrix(class_labels, predicted_labels)

    report = classification_report(class_labels, predicted_labels, digits=4)
    print(report)
    label_mapping = {
        0: "C1", 1: "C2", 2: "C3", 3: "C4", 4: "C5",
        5: "C6", 6: "C7", 7: "C8", 8: "C9", 9: "C10",
    }

    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(confusion_mat, xticklabels=label_mapping.values(), yticklabels=label_mapping.values(), annot=True,fmt='d', cmap='summer')
    plt.show()