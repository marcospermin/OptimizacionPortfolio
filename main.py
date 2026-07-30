import os
import random
import numpy as np

from utils.datos import N_ACTIVOS, RETORNOS_ESPERADOS, MATRIZ_COVARIANZAS
from utils.funciones import (calcular_hipervolumen_2d, exportar_mejores_carteras, exportar_hipervolumen_dinamico, exportar_test_estadistico, generar_grafico_pareto)
from algoritmos.algoritmo_abc import AlgoritmoABC
from algoritmos.algoritmo_nsga_ii import NSGAII

# configuraciones
PRESUPUESTO_TOTAL = 5000
N_SEMILLAS = 30
LAMBDAS = np.linspace(0.0, 1.0, 11)

# Hiperparámetros NSGA-II
NSGA_POBLACION = 50
NSGA_PROB_CROSSOVER = 0.9           # pc: Probabilidad de que ocurra el cruce SBX
NSGA_PROB_MUTACION = 1.0 / 6        # pm: Probabilidad de mutación (suele ser 1 / n_activos)
NSGA_ETA_C = 20                     # eta_c: Índice de distribución para cruce SBX
NSGA_ETA_M = 20                     # eta_m: Índice de distribución para mutación polinomial
# El presupuesto se gasta en una sola pasada:
NSGA_GENERACIONES = PRESUPUESTO_TOTAL // NSGA_POBLACION

# Hiperparámetros ABC
ABC_POBLACION = 20
ABC_LIMITE = 50                     # Límite de estancamiento para activar abejas exploradoras
ABC_PESO_MINIMO = 0.0               # Límite inferior para la proporción de cada activo
ABC_PESO_MAXIMO = 1.0               # Límite superior para la proporción de cada activo
# El presupuesto se divide entre los 11 lambdas (hay 2 evaluaciones de fitness por cada ciclo de ABC):
EVALS_POR_LAMBDA = PRESUPUESTO_TOTAL // len(LAMBDAS)
ABC_CICLOS = EVALS_POR_LAMBDA // (2 * ABC_POBLACION) 
# ==========================================

def main():
    carpeta_salida = "resultados_comparacion"
    os.makedirs(carpeta_salida, exist_ok=True)

    # 1. Generación de Semillas Aleatorias Reproducibles
    np.random.seed(42)
    semillas = np.random.randint(1, 10000, N_SEMILLAS).tolist()

    # Almacenamiento temporal
    nsga_fronts, abc_fronts = {}, {}
    mejores_carteras_nsga, mejores_carteras_abc = [], []
    dist_sh_nsga, dist_sh_abc = [], []
    todos_los_objetivos = []

    print(f"Iniciando evaluación de {N_SEMILLAS} semillas con Presupuesto: {PRESUPUESTO_TOTAL}...")

    # 2. Ejecución de Metaheurísticas
    for i, semilla in enumerate(semillas):
        np.random.seed(semilla)
        random.seed(semilla)

        print(f"Ejecutando semilla {i+1}/{N_SEMILLAS}: {semilla}...")

        # --- NSGA-II ---
        nsga = NSGAII(n_activos=N_ACTIVOS, retornos_esperados=RETORNOS_ESPERADOS, matriz_covarianza=MATRIZ_COVARIANZAS, 
            tam_poblacion=NSGA_POBLACION, n_generaciones=NSGA_GENERACIONES,
            prob_crossover=NSGA_PROB_CROSSOVER, prob_mutacion=NSGA_PROB_MUTACION,
            eta_c=NSGA_ETA_C, eta_m=NSGA_ETA_M)
        pob_nsga, obj_nsga = nsga.ejecutar()
        
        nsga_fronts[semilla] = (pob_nsga, obj_nsga)
        todos_los_objetivos.extend(obj_nsga)

        # Extraer cartera de mayor Sharpe
        sharpes_nsga = -obj_nsga[:, 1] / np.sqrt(np.clip(obj_nsga[:, 0], 1e-10, None))
        idx_nsga = np.argmax(sharpes_nsga)
        dist_sh_nsga.append(sharpes_nsga[idx_nsga])
        mejores_carteras_nsga.append({
            'Semilla': semilla, 'Sharpe': sharpes_nsga[idx_nsga],
            'Riesgo': obj_nsga[idx_nsga, 0], 'Retorno': -obj_nsga[idx_nsga, 1],
            'Pesos': pob_nsga[idx_nsga]
        })

        print(f"  NSGA-II: Mejor Sharpe = {sharpes_nsga[idx_nsga]:.4f}, Riesgo = {obj_nsga[idx_nsga, 0]:.4f}, Retorno = {-obj_nsga[idx_nsga, 1]:.4f}")

        # --- ABC (Barrido) ---
        pob_abc, obj_abc = [], []
        for lmbda in LAMBDAS:
            abc = AlgoritmoABC(n_activos=N_ACTIVOS, 
                tamano_poblacion=ABC_POBLACION, 
                max_ciclos=ABC_CICLOS, limite=ABC_LIMITE,
                peso_minimo=ABC_PESO_MINIMO, peso_maximo=ABC_PESO_MAXIMO)
            mejor_w = abc.ejecutar(RETORNOS_ESPERADOS, MATRIZ_COVARIANZAS, lmbda)
            
            riesgo = np.dot(mejor_w.T, np.dot(MATRIZ_COVARIANZAS, mejor_w))
            retorno = -np.dot(mejor_w, RETORNOS_ESPERADOS)
            pob_abc.append(mejor_w)
            obj_abc.append([riesgo, retorno])
            
        pob_abc, obj_abc = np.array(pob_abc), np.array(obj_abc)
        abc_fronts[semilla] = (pob_abc, obj_abc)
        todos_los_objetivos.extend(obj_abc)

        # Extraer cartera de mayor Sharpe
        sharpes_abc = -obj_abc[:, 1] / np.sqrt(np.clip(obj_abc[:, 0], 1e-10, None))
        idx_abc = np.argmax(sharpes_abc)
        dist_sh_abc.append(sharpes_abc[idx_abc])
        mejores_carteras_abc.append({
            'Semilla': semilla, 'Sharpe': sharpes_abc[idx_abc],
            'Riesgo': obj_abc[idx_abc, 0], 'Retorno': -obj_abc[idx_abc, 1],
            'Pesos': pob_abc[idx_abc]
        })

        print(f"  ABC: Mejor Sharpe = {sharpes_abc[idx_abc]:.4f}, Riesgo = {obj_abc[idx_abc, 0]:.4f}, Retorno = {-obj_abc[idx_abc, 1]:.4f}")

    # 3. Punto de Referencia Dinámico
    print("\nCalculando punto de referencia dinámico y métricas...")
    todos_los_objetivos = np.array(todos_los_objetivos)
    nadir_f1 = np.max(todos_los_objetivos[:, 0])
    nadir_f2 = np.max(todos_los_objetivos[:, 1])
    pto_ref_dinamico = [nadir_f1 * 1.1, nadir_f2 + 0.001]

    # 4. Cálculo de Hipervolúmenes
    dist_hv_nsga, dist_hv_abc = [], []
    for semilla in semillas:
        _, obj_nsga = nsga_fronts[semilla]
        _, obj_abc = abc_fronts[semilla]
        dist_hv_nsga.append(calcular_hipervolumen_2d(obj_nsga, pto_ref_dinamico))
        dist_hv_abc.append(calcular_hipervolumen_2d(obj_abc, pto_ref_dinamico))

    # Encontrar la mejor semilla global (según Hipervolumen de NSGA-II)
    idx_mejor_global = np.argmax(dist_hv_nsga)
    mejor_semilla = semillas[idx_mejor_global]
    _, mejor_obj_nsga = nsga_fronts[mejor_semilla]
    _, mejor_obj_abc = abc_fronts[mejor_semilla]

    # 5. Exportaciones
    print("Exportando archivos...")
    exportar_mejores_carteras(mejores_carteras_nsga, mejores_carteras_abc, carpeta_salida)
    exportar_hipervolumen_dinamico(semillas, dist_hv_nsga, dist_hv_abc, pto_ref_dinamico, carpeta_salida)
    exportar_test_estadistico(dist_hv_nsga, dist_hv_abc, dist_sh_nsga, dist_sh_abc, carpeta_salida)
    generar_grafico_pareto(mejor_obj_nsga, mejor_obj_abc, mejor_semilla, carpeta_salida)

    print(f"Proceso completado. Revisa la carpeta '{carpeta_salida}/'.")

if __name__ == "__main__":
    main()