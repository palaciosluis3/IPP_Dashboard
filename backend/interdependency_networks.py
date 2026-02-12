import os
import pandas as pd
import numpy as np
import warnings

# Desactivar advertencias temporales de correlación (cuando hay varianza cero)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 1. Referencias flotantes para los archivos
def get_path(filename):
    # Encontrar la raíz del proyecto (un nivel arriba de /backend)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backend_path = os.path.join(base_path, "backend")
    
    # 1. Si es la librería PPI, se queda en backend
    if filename == 'policy_priority_inference.py':
        return os.path.join(backend_path, filename)
    
    # 2. Si el archivo es un input crudo (raw_*), debe estar en la raíz
    if filename.startswith('raw_'):
        return os.path.join(base_path, filename)
    
    # 3. Todos los demás archivos (generados o intermedios) van a Outputs
    out_dir = os.path.join(base_path, "Outputs")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)

input_file = get_path('data_indicators.xlsx')
output_file = get_path('data_network.xlsx')

if not os.path.exists(input_file):
    print(f"Error: No se encontró el archivo {input_file}. Asegúrate de correr primero indicators_preparation.py")
else:
    print(f"Leyendo desde: {input_file}")
    data = pd.read_excel(input_file)

    N = len(data)
    M = np.zeros((N, N))
    
    # Identificar columnas de años
    years = [column_name for column_name in data.columns if str(column_name).isnumeric()]
    print(f"Procesando {N} indicadores para años: {years[0]} a {years[-1]}")

    # 2. Cálculo de la matriz de interdependencia
    # La lógica busca correlaciones con rezago (lag) para identificar causalidad/influencia
    success_matrix = data[years].values.astype(float)

    for i in range(N):
        for j in range(N):
            if i != j:
                # Serie 1 (Indicador i): Cambios en t
                # Serie 2 (Indicador j): Cambios en t-1
                # Esto detecta si el cambio en j 'predice' el cambio en i
                
                # Valores del indicador i del año 1 al final
                vals_i = success_matrix[i, 1:]
                # Valores del indicador j del año 0 al penúltimo
                vals_j = success_matrix[j, :-1]
                
                # Calculamos los cambios (diferencias)
                # change_i tiene longitud len(years) - 2
                change_i = vals_i[1:] - vals_i[:-1]
                # change_j tiene longitud len(years) - 2
                change_j = vals_j[1:] - vals_j[:-1]
                
                # Verificamos que las series no sean constantes para evitar errores en corrcoef
                if np.std(change_i) > 0 and np.std(change_j) > 0:
                    correlation = np.corrcoef(change_i, change_j)[0, 1]
                    if not np.isnan(correlation):
                        M[i, j] = correlation

    # 3. Refinamiento de la red (Umbral de significancia)
    # Solo mantenemos correlaciones fuertes (abs > 0.5)
    M[np.abs(M) < 0.5] = 0

    # 4. Generación de lista de aristas (Edge List)
    ids = data.seriesCode.values
    edge_list = []
    
    # Encontramos donde hay conexiones (pesos distintos de cero)
    rows, cols = np.where(M != 0)
    for r, c in zip(rows, cols):
        edge_list.append([ids[r], ids[c], M[r, c]])

    df_network = pd.DataFrame(edge_list, columns=['origin', 'destination', 'weight'])
    
    print(f"Red generada con {len(df_network)} conexiones detectadas.")
    print(f"Guardando en: {output_file}")
    df_network.to_excel(output_file, index=False)
    print("¡Proceso de red completado!")