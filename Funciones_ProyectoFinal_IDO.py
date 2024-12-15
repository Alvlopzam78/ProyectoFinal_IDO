# Importamos librerias necesarias
import random
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# Función para cargar los datos
def load_cities_from_file(file_path):
    cities = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) == 3:
                x, y = float(parts[1]), float(parts[2])
                cities.append((x, y))
    return cities

# Clase para el algoritmo simple
class FastGeneticGreedyTSP:
    def __init__(self, cities, population_size=100, generations=50, 
                 mutation_rate=0.2, elite_rate=0.15):
        self.cities = cities
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_rate = elite_rate
        self.distance_matrix = self._generate_distance_matrix()
        
    def _generate_distance_matrix(self):
        n = len(self.cities)
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = self._calculate_distance(self.cities[i], self.cities[j])
        return matrix
    
    def _calculate_distance(self, city1, city2):
        return round(math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2))
    
    def _greedy_insertion_initial_population(self):
        def greedy_tour():
            n = len(self.cities)
            unvisited = set(range(1, n))
            tour = [0]
            
            while unvisited:
                last = tour[-1]
                next_city = min(unvisited, key=lambda x: self.distance_matrix[last][x])
                tour.append(next_city)
                unvisited.remove(next_city)
            
            return tour
        
        population = []
        for _ in range(self.population_size):
            tour = greedy_tour()
            # Añadir algo de aleatoriedad
            for _ in range(5):
                i, j = random.sample(range(len(tour)), 2)
                tour[i], tour[j] = tour[j], tour[i]
            population.append(tour)
        
        return population
    
    def _calculate_tour_length(self, tour):
        total = sum(self.distance_matrix[tour[i]][tour[i+1]] for i in range(len(tour) - 1))
        total += self.distance_matrix[tour[-1]][tour[0]]
        return total
    
    def _quick_two_opt(self, tour):
        n = len(tour)
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Calcular cambio en distancia con seguridad
                    old_distance = (
                        self.distance_matrix[tour[i-1]][tour[i]] + 
                        self.distance_matrix[tour[j]][tour[(j+1)%n]]
                    )
                    new_distance = (
                        self.distance_matrix[tour[i-1]][tour[j]] + 
                        self.distance_matrix[tour[i]][tour[(j+1)%n]]
                    )
                    
                    if new_distance < old_distance:
                        # Invertir segmento con seguridad
                        tour[i:j+1] = tour[i:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return tour
    
    def _partially_mapped_crossover(self, parent1, parent2):
        n = len(parent1)
        # Seleccionar segmento
        start, end = sorted(random.sample(range(n), 2))
        
        # Crear hijo
        child = [None] * n
        child[start:end] = parent1[start:end]
        
        # Mapear el resto
        for i in range(n):
            if child[i] is None:
                current = parent2[i]
                while current in child:
                    current = parent2[child.index(current)]
                child[i] = current
        
        return child
    
    def solve(self, verbose=False):
        # Generar población inicial con inserción voraz
        population = self._greedy_insertion_initial_population()
        
        best_tour = min(population, key=self._calculate_tour_length)
        best_length = self._calculate_tour_length(best_tour)
        
        for generation in range(self.generations):
            # Ordenar población por fitness
            population_with_fitness = [(tour, self._calculate_tour_length(tour)) for tour in population]
            population_with_fitness.sort(key=lambda x: x[1])
            
            # Selección de élite
            elite_count = int(self.population_size * self.elite_rate)
            new_population = [tour for tour, _ in population_with_fitness[:elite_count]]
            
            while len(new_population) < self.population_size:
                # Selección por torneo
                parent1 = min(random.sample(population, 3), key=self._calculate_tour_length)
                parent2 = min(random.sample(population, 3), key=self._calculate_tour_length)
                
                # Crossover
                child = self._partially_mapped_crossover(parent1, parent2)
                
                # Mutación
                if random.random() < self.mutation_rate:
                    child = self._quick_two_opt(child)
                
                new_population.append(child)
            
            population = new_population
            
            # Actualizar mejor solución
            current_best = min(population, key=self._calculate_tour_length)
            current_best_length = self._calculate_tour_length(current_best)
            
            if current_best_length < best_length:
                best_tour = current_best
                best_length = current_best_length
                
                if verbose:
                    print(f"Generación {generation}: Mejor distancia = {best_length}")
        
        best_tour.append(best_tour[0])  # Cerrar el ciclo
        return best_tour, best_length
    
# Clase para el algoritmo con clustering
class FastGeneticGreedyClusteringTSP:
    def __init__(self, cities, population_size=100, generations=50, 
                 mutation_rate=0.2, elite_rate=0.15, num_clusters=5):
        self.cities = cities
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_rate = elite_rate
        self.num_clusters = num_clusters
        self.distance_matrix = self._generate_distance_matrix()

    def _generate_distance_matrix(self):
        n = len(self.cities)
        coords = np.array(self.cities)
        dist_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :])**2).sum(axis=2))
        return np.round(dist_matrix).astype(int)

    def _calculate_distance(self, city1, city2):
        return round(math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2))

    def _calculate_tour_length(self, tour):
        total = sum(self.distance_matrix[tour[i]][tour[i+1]] for i in range(len(tour) - 1))
        total += self.distance_matrix[tour[-1]][tour[0]]
        return total

    def _quick_two_opt(self, tour):
        n = len(tour)
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    old_distance = (
                        self.distance_matrix[tour[i-1]][tour[i]] + 
                        self.distance_matrix[tour[j]][tour[(j+1)%n]]
                    )
                    new_distance = (
                        self.distance_matrix[tour[i-1]][tour[j]] + 
                        self.distance_matrix[tour[i]][tour[(j+1)%n]]
                    )
                    if new_distance < old_distance:
                        tour[i:j+1] = tour[i:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return tour

    def _greedy_tour(self, cities_subset):
        n = len(cities_subset)
        unvisited = set(range(1, n))
        tour = [0]
        while unvisited:
            last = tour[-1]
            next_city = min(unvisited, key=lambda x: self.distance_matrix[last][x])
            tour.append(next_city)
            unvisited.remove(next_city)
        return tour

    def _solve_cluster(self, cluster_indices, recursion_depth):
        cluster_cities = [self.cities[i] for i in cluster_indices]
        sub_tsp = FastGeneticGreedyClusteringTSP(cluster_cities, self.population_size, 
                                                   self.generations, self.mutation_rate, 
                                                   self.elite_rate, self.num_clusters)
        best_tour, _ = sub_tsp.solve(verbose=False, recursion_depth=recursion_depth+1)
        return [cluster_indices[i] for i in best_tour]

    def _connect_clusters(self, cluster_tours):
        merged_tour = []
        for tour in cluster_tours:
            merged_tour += tour
        return self._quick_two_opt(merged_tour)

    def solve(self, verbose=False, recursion_depth=0):
        # Límite de recursión
        MAX_RECURSION_DEPTH = 3
        if recursion_depth >= MAX_RECURSION_DEPTH:
            if verbose:
                print(f"Límite de recursión alcanzado. Usando ruta simple.")
            # Devolver una ruta simple si se alcanza el límite de recursión
            return list(range(len(self.cities))), self._calculate_tour_length(list(range(len(self.cities))))

        # Ajustar el número de clústeres si es necesario
        if len(self.cities) < self.num_clusters:
            self.num_clusters = len(self.cities)
            if verbose:
                print(f"Reduciendo el número de clústeres a {self.num_clusters} debido a la cantidad limitada de ciudades.")

        # Normalizar las coordenadas
        scaler = MinMaxScaler()
        city_coords = scaler.fit_transform(np.array(self.cities))

        # Clustering de las ciudades
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42).fit(city_coords)
        cluster_labels = kmeans.labels_

        # Resolver TSP en cada clúster
        cluster_tours = []
        for cluster_id in range(self.num_clusters):
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            if verbose:
                print(f"Resolviendo clúster {cluster_id} con {len(cluster_indices)} ciudades.")
            cluster_tour = self._solve_cluster(cluster_indices, recursion_depth)
            cluster_tours.append(cluster_tour)

        # Conectar los clústeres
        if verbose:
            print("Conectando clústeres...")
        final_tour = self._connect_clusters(cluster_tours)
        final_length = self._calculate_tour_length(final_tour)

        return final_tour, final_length
    
# Función para graficar la ruta
def plot_route(cities, route, title="TSP Route"):
    # Obtener las coordenadas de las ciudades según el orden de la ruta
    ordered_cities = [cities[i] for i in route]
    
    # Añadir la ciudad de inicio al final para formar un ciclo
    ordered_cities.append(ordered_cities[0])
    
    # Desempaquetar las coordenadas
    x, y = zip(*ordered_cities)

    # Crear la gráfica con Plotly, girando 90º (intercambiando x e y)
    fig = go.Figure()

    # Añadir la ruta
    fig.add_trace(go.Scatter(x=y, y=x, mode='lines+markers', marker=dict(size=6, color='blue'), name='Ruta'))

    # Añadir etiquetas de las ciudades
    for i, city in enumerate(ordered_cities[:-1]):  # No etiquetar el último (es el mismo que el primero)
        fig.add_trace(go.Scatter(x=[city[1]], y=[city[0]], mode='text'))

    # Configurar el título y los ejes
    fig.update_layout(
        title=title,
        xaxis_title='Coordenada X',
        yaxis_title='Coordenada Y',
        showlegend=False,
        height=700,
        width=800
    )

    # Mostrar la gráfica
    fig.show()
