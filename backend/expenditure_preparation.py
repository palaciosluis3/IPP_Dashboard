import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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

input_file = get_path('raw_expenditure.xlsx')
indicators_file = get_path('data_indicators.xlsx')
output_exp_file = get_path('data_expenditure.xlsx')
output_rel_file = get_path('data_relational_table.xlsx')

if not os.path.exists(input_file) or not os.path.exists(indicators_file):
    print(f"Error: Asegúrate de que existen {input_file} y {indicators_file}")
else:
    print(f"Iniciando preparación de presupuesto y tablas relacionales...")
    
    # Cargar datos
    xl = pd.ExcelFile(input_file)
    df_budget_raw = xl.parse(xl.sheet_names[0])
    df_pop = xl.parse(xl.sheet_names[1])
    df_ipc = xl.parse(xl.sheet_names[2])
    data_indi = pd.read_excel(indicators_file)

    # Años y Limpieza inicial
    years = [str(col) for col in df_budget_raw.columns if str(col).isnumeric()]
    df_budget_raw.columns = [str(col) if str(col).isnumeric() else col for col in df_budget_raw.columns]
    
    # --- PROCESAMIENTO BASE (Precios Reales y Per Cápita) ---
    pop_map = dict(zip(df_pop.iloc[:, 0].astype(str), df_pop.iloc[:, 1]))
    ipc_map = dict(zip(df_ipc.iloc[:, 0].astype(str), df_ipc.iloc[:, 1]))
    
    # Lógica IPC base 100
    if any(v > 1 for v in ipc_map.values()):
        ipc_map = {k: v / 100 for k, v in ipc_map.items()}

    # --- VALIDACIÓN DE SINCRONIZACIÓN DE AÑOS ---
    missing_pop = [yr for yr in years if yr not in pop_map]
    missing_ipc = [yr for yr in years if yr not in ipc_map]
    
    if missing_pop or missing_ipc:
        print("\n" + "!"*60)
        print(" ERROR CRÍTICO: INCONSISTENCIA TEMPORAL (AÑOS FALTANTES) ")
        print("!"*60)
        if missing_pop:
            print(f"-> Falta información de POBLACIÓN para los años: {missing_pop}")
        if missing_ipc:
            print(f"-> Falta información de IPC para los años: {missing_ipc}")
        print("\nACCIÓN REQUERIDA:")
        print("Asegúrate de que TODOS los años presentes en los indicadores y presupuesto")
        print("también existan en las hojas de 'Población' e 'IPC' de raw_expenditure.xlsx.")
        sys.exit(1)

    processed_list = []
    for index, row in df_budget_raw.iterrows():
        new_row = {'sdg_target': row['sdg_target']}
        y_vals = []
        for yr in years:
            val = row[yr] / (pop_map[yr] * ipc_map[yr])
            new_row[yr] = val
            y_vals.append(val)
        
        # Eliminar tendencia lineal
        x = np.arange(len(years)).reshape(-1, 1)
        y = np.array(y_vals).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        tendencia = model.predict(x).flatten()
        promedio = np.mean(y)
        
        final_vals = (y.flatten() - tendencia + promedio)
        for i, yr in enumerate(years):
            new_row[yr] = max(1e-10, final_vals[i]) # Evitar ceros absolutos
        processed_list.append(new_row)

    df_processed = pd.DataFrame(processed_list)

    # --- 1. CÁLCULO AUTOMÁTICO DE T (Índice de Calibración) ---
    num_periods = len(years)
    # Buscamos un múltiplo entero tal que num_periods * calibration_index esté en [40, 60]
    # Si num_periods es 10, calibration_index sería 5 (T=50)
    calibration_index = int(np.round(50 / num_periods))
    T = num_periods * calibration_index
    print(f"Índice de calibración calculado: {calibration_index} (T total = {T})")

    # --- 2. FILTRADO Y CONTROL DE INDICADORES INSTRUMENTALES ---
    # Solo nos interesan los SDG que tienen indicadores instrumentales (instrumental == 1)
    instrumental_sdgs = data_indi[data_indi.instrumental == 1].sdg.unique()
    
    # Control crítico: Un SDG instrumental DEBE tener presupuesto asignado
    budget_sdgs = df_processed.sdg_target.unique()
    missing_budget = [int(sdg) for sdg in instrumental_sdgs if sdg not in budget_sdgs]
    
    if missing_budget:
        print("\n" + "!"*60)
        print(" ERROR CRÍTICO: FALTA DE PRESUPUESTO PARA INDICADORES INSTRUMENTALES ")
        print("!"*60)
        print(f"\nLos siguientes SDG instrumentales NO tienen presupuesto en 'raw_expenditure.xlsx':")
        print(f"SDG Targets: {missing_budget}")
        print("\nACCIÓN REQUERIDA:")
        print("1. Por favor, revisa y corrige el archivo 'raw_expenditure.xlsx'.")
        print("2. Asegúrate de que todos los SDG anteriores tengan una fila con datos presupuestarios.")
        print("3. Vuelve a ejecutar este proceso.")
        print("\nEl proceso se ha detenido para evitar errores en la simulación.")
        import sys
        sys.exit(1) # Detiene la ejecución con código de error

    # Filtrar el presupuesto procesado para solo dejar los instrumentales
    df_exp_filtered = df_processed[df_processed.sdg_target.isin(instrumental_sdgs)].copy()
    
    # --- 3. EXTENSIÓN ARTIFICIAL (Interpolación de paso) ---
    extended_rows = []
    for index, row in df_exp_filtered.iterrows():
        new_row = [int(row.sdg_target)]
        for yr in years:
            # Repetimos cada observación 'calibration_index' veces
            new_row += [float(row[yr])] * calibration_index
        extended_rows.append(new_row)

    df_exp_final = pd.DataFrame(extended_rows, columns=['sdg'] + list(range(T)))

    # --- 4. TABLA RELACIONAL (Indicator -> SDG) ---
    # Cada fila representará un indicador instrumental y su SDG (programa presupuestal) asociado
    relational_data = []
    # Usamos data_indi que ya está cargado y contiene la columna 'instrumental'
    instrum_df = data_indi[data_indi['instrumental'] == 1]

    for _, row in instrum_df.iterrows():
        # Aquí asociamos el indicador con su SDG_target
        relational_data.append([row['seriesCode'], row['sdg']])

    # Convertir a DataFrame y guardar
    df_rel = pd.DataFrame(relational_data, columns=['seriesCode', 'sdg_target'])
    df_rel.to_excel(output_rel_file, index=False)
    
    # --- GUARDAR ARCHIVOS ---
    df_exp_final.to_excel(output_exp_file, index=False)
    
    print(f"Archivos generados:")
    print(f"- {output_exp_file} (Matriz extendida {df_exp_final.shape})")
    print(f"- {output_rel_file} (Tabla relacional)")
    print(f"Valor de calibration_index guardado: {calibration_index}")
