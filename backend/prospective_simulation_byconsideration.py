import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Desactivar advertencias
warnings.filterwarnings('ignore')

# Configuración básica (Debe coincidir con prospect_simulation.py y final_report_generator.py)
INTERMEDIATE_CONVERGENCE_YEAR = 3 
YEARS_TO_FORECAST = 15

def get_path(filename):
    """Obtiene la ruta absoluta para los archivos de entrada y salida."""
    # Encontrar la raíz del proyecto (un nivel arriba de /backend)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Si el archivo es un input crudo (raw_*), debe estar en la raíz
    if filename.startswith('raw_'):
        return os.path.join(base_path, filename)
    
    # 2. Archivos generados van a la carpeta Outputs
    out_dir = os.path.join(base_path, "Outputs")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)

def generate_plots_by_consideration():
    """
    Genera 3 gráficas de barras dividiendo los indicadores según la recomendación final
    contenida en el reporte del IPP.
    """
    print("\n" + "="*50)
    print("GENERANDO GRÁFICAS POR RECOMENDACIÓN")
    print("="*50)
    
    # Definición de rutas
    file_baseline = get_path('output_baseline.xlsx')
    file_report = get_path('final_report_IPP.xlsx')
    file_raw = get_path('raw_indicators.xlsx')

    # Verificación de pre-requisitos
    if not os.path.exists(file_baseline) or not os.path.exists(file_report):
        print("ERROR: Asegúrate de haber ejecutado 'prospective_simulation.py' y 'final_report_generator.py' primero.")
        print(f"Buscando:\n - {file_baseline}\n - {file_report}")
        return

    # Cargar datos
    print("Cargando datos de simulación y reporte...")
    df_output = pd.read_excel(file_baseline)
    df_report = pd.read_excel(file_report)
    df_raw = pd.read_excel(file_raw)

    # Calcular Índice de Calibración (necesario para identificar los pasos de tiempo correctos)
    # Siguiendo la lógica de prospect_simulation.py
    historical_years = [col for col in df_raw.columns if str(col).isnumeric()]
    if len(historical_years) > 0:
        calibration_index = int(np.round(50 / len(historical_years)))
    else:
        calibration_index = 1
        print("Aviso: No se detectaron años históricos. Usando calibración por defecto (1).")

    # Unir la recomendación ('Recomendacion_Final') al output de simulación
    # Usamos 'seriesCode' como identificador único
    cols_to_merge = ['seriesCode', 'Recomendacion_Final']
    df_merged = pd.merge(df_output, df_report[cols_to_merge], on='seriesCode', how='inner')

    # Identificar las columnas que contienen la serie de tiempo (pasos 0 a T_sim-1)
    # En output_baseline.xlsx, estas columnas suelen ser numéricas o strings de números
    time_cols = [c for c in df_output.columns if isinstance(c, int) or (isinstance(c, str) and c.isdigit())]
    time_cols = sorted(time_cols, key=lambda x: int(x)) # Ordenar por paso temporal
    T_sim = len(time_cols)
    
    # Calcular índices para las flechas
    inter_idx_num = INTERMEDIATE_CONVERGENCE_YEAR * calibration_index
    if inter_idx_num >= T_sim:
        inter_idx_num = T_sim - 1
    
    # Mapeo de recomendaciones (exacto como en final_report_generator.py)
    # 1. Continuar programas
    # 2. Escalar programas
    # 3. Revisar los programas asociados
    recommendations = [
        "Continuar programas",
        "Escalar programas",
        "Revisar los programas asociados"
    ]
    
    # Sufijos para los nombres de archivo PDF
    file_suffixes = {
        "Continuar programas": "continuar",
        "Escalar programas": "escalar",
        "Revisar los programas asociados": "revisar"
    }

    # Generación de las gráficas
    for reco in recommendations:
        subset_full = df_merged[df_merged['Recomendacion_Final'] == reco]
        num_items_full = len(subset_full)
        
        if num_items_full == 0:
            print(f"[-] Sin indicadores para: '{reco}'... omitiendo.")
            continue

        # Decidir si dividir el grupo (si hay más de 50 indicadores)
        if num_items_full > 50:
            mid = num_items_full // 2
            groups = [
                (subset_full.iloc[:mid], f"{file_suffixes[reco]}_1"),
                (subset_full.iloc[mid:], f"{file_suffixes[reco]}_2")
            ]
            print(f"[+] Dividiendo {num_items_full} indicadores para: '{reco}' en dos partes.")
        else:
            groups = [(subset_full, file_suffixes[reco])]
            print(f"[+] Graficando {num_items_full} indicadores para: '{reco}'")

        for subset, suffix in groups:
            num_items = len(subset)
            # Delimitamos el ancho máximo (similar a prospective_simulation.py que usa 12, 4 para ~50 indicadores)
            fig_width = 12 if num_items >= 30 else max(8, num_items * 0.4)
            fig = plt.figure(figsize=(fig_width, 4))
            
            for idx, (_, row) in enumerate(subset.iterrows()):
                x_pos = idx
                
                # Obtener valores clave
                val_start = row[time_cols[0]]
                val_inter = row[time_cols[inter_idx_num]]
                val_final = row[time_cols[-1]]
                
                # 1. Barra base (nivel inicial)
                plt.bar(x_pos, val_start, color=row.color, width=.65, alpha=.4)
                
                # 2. Flecha progreso intermedio - Línea continua gruesa
                plt.arrow(x=x_pos, y=val_start, dx=0, dy=val_inter - val_start, 
                          color=row.color, linewidth=2.5, alpha=1, 
                          head_width=.3, head_length=.015)
                
                # 3. Flecha progreso final - Línea punteada
                plt.arrow(x=x_pos, y=val_start, dx=0, dy=val_final - val_start, 
                          color=row.color, linewidth=1.2, alpha=0.8, 
                          head_width=.3, head_length=.015, linestyle=':')
                
                # 4. Meta Real (Punto negro)
                if 'real_goal' in row:
                    plt.scatter(x_pos, row.real_goal, color='black', s=25, zorder=10)

            # Estética de la gráfica
            plt.xlim(-1, num_items)
            plt.xticks(range(num_items), subset.seriesCode, rotation=90, fontsize=7)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.ylabel('levels', fontsize=12)
            plt.xlabel('indicators', fontsize=12)
            
            plt.tight_layout()
            
            # Guardado
            out_name = f'Bars_baseline_by_consideration_{suffix}.pdf'
            save_path = get_path(out_name)
            plt.savefig(save_path)
            plt.close()
            print(f"    -> Guardada en: {out_name}")

    print("\nProceso completado exitosamente.")

if __name__ == "__main__":
    generate_plots_by_consideration()
