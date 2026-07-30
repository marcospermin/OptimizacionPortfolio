import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

def calcular_hipervolumen_2d(frente_objetivos, punto_ref):
    """Calcula el área dominada por el frente respecto a un punto de referencia."""
    frente = np.array(frente_objetivos)
    frente = frente[np.argsort(frente[:, 0])] 
    
    hv = 0.0
    ultimo_f2 = punto_ref[1]
    
    for p in frente:
        f1, f2 = p[0], p[1]
        if f1 >= punto_ref[0] or f2 >= punto_ref[1]: 
            continue
        hv += (punto_ref[0] - f1) * (ultimo_f2 - f2)
        ultimo_f2 = f2
    return hv

def exportar_mejores_carteras(carteras_nsga, carteras_abc, carpeta_salida):
    """Guarda dos CSV (uno por algoritmo) con la cartera de mayor Sharpe de cada semilla."""
    columnas = ['Semilla', 'Sharpe', 'Retorno', 'Varianza', 'COPEC', 'CTC-A', 'CAP', 'COLBUN', 'ENDESA', 'ENTEL']
    
    df_nsga = pd.DataFrame([[c['Semilla'], c['Sharpe'], c['Retorno'], c['Riesgo']] + list(c['Pesos']) for c in carteras_nsga], columns=columnas)
    df_nsga.to_csv(os.path.join(carpeta_salida, 'mejores_carteras_nsga2.csv'), index=False)

    df_abc = pd.DataFrame([[c['Semilla'], c['Sharpe'], c['Retorno'], c['Riesgo']] + list(c['Pesos']) for c in carteras_abc], columns=columnas)
    df_abc.to_csv(os.path.join(carpeta_salida, 'mejores_carteras_abc.csv'), index=False)

def exportar_hipervolumen_dinamico(semillas, hv_nsga, hv_abc, pto_ref, carpeta_salida):
    """Exporta el hipervolumen de cada semilla calculado con el punto dinámico."""
    df = pd.DataFrame({'Semilla': semillas, 'HV_NSGA_II': hv_nsga, 'HV_ABC': hv_abc})
    df['Ref_Riesgo'] = pto_ref[0]
    df['Ref_RetornoNegativo'] = pto_ref[1]
    df.to_csv(os.path.join(carpeta_salida, 'hipervolumen_dinamico.csv'), index=False)

def exportar_test_estadistico(dist_hv_nsga, dist_hv_abc, dist_sh_nsga, dist_sh_abc, carpeta_salida):
    """Ejecuta el test de Wilcoxon para el Hipervolumen y el Sharpe Máximo."""
    _, pval_hv = wilcoxon(dist_hv_nsga, dist_hv_abc)
    _, pval_sh = wilcoxon(dist_sh_nsga, dist_sh_abc)
    
    df_stats = pd.DataFrame({
        'Metrica': ['Hipervolumen (Dinámico)', 'Sharpe Máximo'],
        'Media_NSGA_II': [np.mean(dist_hv_nsga), np.mean(dist_sh_nsga)],
        'Media_ABC': [np.mean(dist_hv_abc), np.mean(dist_sh_abc)],
        'P_Valor': [pval_hv, pval_sh],
        'Diferencia_Significativa_0.05': [pval_hv < 0.05, pval_sh < 0.05]
    })
    df_stats.to_csv(os.path.join(carpeta_salida, 'test_estadistico_wilcoxon.csv'), index=False)

def generar_grafico_pareto(obj_nsga, obj_abc, semilla, carpeta_salida):
    """Grafica la comparación de frentes para la mejor semilla global."""
    plt.figure(figsize=(10, 6))
    plt.scatter(np.sqrt(obj_nsga[:, 0])*100, -obj_nsga[:, 1]*100, color='blue', alpha=0.6, label='NSGA-II')
    plt.plot(np.sqrt(obj_abc[:, 0])*100, -obj_abc[:, 1]*100, color='red', marker='o', linestyle='--', label='ABC')
    
    plt.title(f'Frentes de Pareto (Mejor Semilla Global: {semilla})')
    plt.xlabel('Riesgo (Desviación Estándar %)')
    plt.ylabel('Retorno Esperado (%)')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(carpeta_salida, 'grafico_frentes_pareto.png'), dpi=300)
    plt.close()