import random
import numpy as np

class GATA:
    def __init__(self, tasks, num_resources=5, population_size=50, generations=50):
        self.tasks = tasks # DataFrame of 10 tasks
        self.num_resources = num_resources
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = 0.05
        
    def fitness_function(self, chromosome):
        """
        Eq (7): Fitness = Sum(RT + MR)
        Chromosome index = task, value = assigned resource ID
        """
        total_rt = 0
        total_cost = 0
        
        for i, resource_id in enumerate(chromosome):
            task = self.tasks.iloc[i]
            
            # Simulate: Different resources have slightly different speeds/costs
            # Resource multiplier: Res 0 is fast/expensive, Res 4 is slow/cheap
            res_speed_factor = 1 + (resource_id * 0.1) 
            res_cost_factor = 1 - (resource_id * 0.05)
            
            rt = task['RT'] * res_speed_factor
            cost = task['C'] * res_cost_factor
            
            # Fairness boost (Eq 4 and 7 logic): 
            # If task was waiting (simulated by high original RT), improve fitness score
            # Lower score is better in this specific formulation context implies minimization
            # But paper Eq 7 sums them up. 
            
            total_rt += rt
            total_cost += cost
            
        # Paper Goal: Minimize Fitness Value
        return total_rt + total_cost

    def run(self):
        # 1. Initial Population (Decimal representation as per paper)
        population = []
        for _ in range(self.pop_size):
            # Randomly assign each of the 10 tasks to one of the available resources
            chrom = [random.randint(0, self.num_resources-1) for _ in range(len(self.tasks))]
            population.append(chrom)
            
        best_solution = None
        best_fitness = float('inf')
        fitness_history = []

        for gen in range(self.generations):
            # Evaluate Fitness
            scores = [(self.fitness_function(ind), ind) for ind in population]
            scores.sort(key=lambda x: x[0]) # Ascending (Minimization)
            
            current_best_val, current_best_chrom = scores[0]
            fitness_history.append(current_best_val)
            
            if current_best_val < best_fitness:
                best_fitness = current_best_val
                best_solution = current_best_chrom
            
            # Selection (Elitism - keep top 10%)
            next_gen = [ind for score, ind in scores[:int(self.pop_size*0.1)]]
            
            # Crossover (2-point) and Mutation
            while len(next_gen) < self.pop_size:
                parent1 = random.choice(scores[:20])[1] # Pick from top 20
                parent2 = random.choice(scores[:20])[1]
                
                # 2-point crossover
                cut1 = random.randint(1, len(self.tasks)-2)
                cut2 = random.randint(cut1, len(self.tasks)-1)
                
                child = parent1[:cut1] + parent2[cut1:cut2] + parent1[cut2:]
                
                # Mutation
                if random.random() < self.mutation_rate:
                    idx = random.randint(0, len(self.tasks)-1)
                    child[idx] = random.randint(0, self.num_resources-1)
                
                next_gen.append(child)
            
            population = next_gen
            
        return best_solution, fitness_history