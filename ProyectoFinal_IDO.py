# Importamos librerías necesarias
import pandas as pd
import time
import numpy as np
from Funciones_ProyectoFinal_IDO import FastGeneticGreedyTSP, FastGeneticGreedyClusteringTSP, load_cities_from_file, plot_route
from tabulate import tabulate

# Qatar con simple
file_path = "Qatar.txt"
qatar = load_cities_from_file(file_path)
genetic_solver = FastGeneticGreedyTSP(qatar)
start_time = time.time()
best_tour_qatar_s, best_length_qatar_s = genetic_solver.solve()
end_time = time.time()
time_qatar_s = end_time - start_time

plot_route(qatar,best_tour_qatar_s, title="Ruta óptima para Qatar con Algoritmo Simple")

# Qatar con clustering
genetic_solver = FastGeneticGreedyClusteringTSP(qatar, num_clusters=15)
start_time = time.time()
best_tour_qatar_clus, best_length_qatar_clus = genetic_solver.solve()
end_time = time.time()
time_qatar_c = end_time - start_time

plot_route(qatar,best_tour_qatar_clus, title="Ruta óptima para Qatar con Clustering")

# Uruguay
file_path = "Uruguay.txt"
uruguay = load_cities_from_file(file_path)
genetic_solver = FastGeneticGreedyClusteringTSP(uruguay, num_clusters=10) 
start_time = time.time()                                                 
best_tour_uruguay, best_length_uruguay = genetic_solver.solve()
end_time = time.time()
time_uruguay = end_time - start_time

plot_route(uruguay,best_tour_uruguay, title="Ruta óptima para Uruguay con Clustering")

# Zimbabwe
file_path = "Zimbabwe.txt"
zimbabwe = load_cities_from_file(file_path)
genetic_solver = FastGeneticGreedyClusteringTSP(zimbabwe, num_clusters=15)
start_time = time.time() 
best_tour_zimbabwe, best_length_zimbabwe = genetic_solver.solve()
end_time = time.time()
time_zimbabwe = end_time - start_time

plot_route(zimbabwe,best_tour_zimbabwe, title="Ruta óptima para Zimbabwe con Clustering")

data_res = {
    "RUTAS": ["Qatar", "Uruguay", "Zimbabwe"],
    "Algoritmo Simple": [best_length_qatar_s, np.nan, np.nan],
    "Algoritmo con Clustering": [best_length_qatar_clus, best_length_uruguay, best_length_zimbabwe],
    "Óptimo Real": [9352, 79114, 95345]
}

# Crear un DataFrame con los datos
df_res = pd.DataFrame(data_res)

# Calcular diferencias porcentuales respecto al óptimo
for column in ["Algoritmo Simple", "Algoritmo con Clustering"]:
    df_res[f"% Diferencia {column}"] = round(((df_res[column] - df_res["Óptimo Real"]) / df_res["Óptimo Real"]),2)

print(tabulate(df_res, headers="keys", tablefmt="github", showindex=False))

data_tiempo = {
    "TIEMPOS": ["Qatar", "Uruguay", "Zimbabwe"],
    "Algoritmo Simple": [str(round(time_qatar_s,2)), "-", "-"],
    "Algoritmo con Clustering": [round(time_qatar_c, 2), round(time_uruguay,2), round(time_zimbabwe,2)]
}

# Crear un DataFrame con los datos
df_tiempo = pd.DataFrame(data_tiempo)

print(tabulate(df_tiempo, headers="keys", tablefmt="github", showindex=False))