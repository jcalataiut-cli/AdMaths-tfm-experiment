#!/usr/bin/env python3
"""
Experimento 2: Kernel NNGP — Analítico vs Empírico en Múltiples Arquitecturas
======================================================================

Objetivos:
  1. Calcular el kernel NNGP analítico (límite n→∞) para FC
  2. Comparar con el kernel empírico de redes finitas de varios anchos
  3. Extender a CNN (peso compartido) - ¿rompe la convergencia?
  4. Extender a ResNet-like (skip connections)
  5. Analizar el espectro de los kernels (valores propios)
  6. Medir la "distancia" entre kernels: error de Frobenius normalizado

Pregunta central del TFM:
  ¿Bajo qué condiciones el kernel empírico converge al analítico NNGP?
  ¿Qué arquitecturas lo satisfacen y cuáles no?
"""

import numpy as np
from scipy import stats, linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuración
# ============================================================
INPUT_DIM = 8
N_DATA = 20  # Puntos de datos para construir la matriz Gram
SEED = 42
SIGMA_W = 1.5
SIGMA_B = 0.1
DEPTH = 2

# Anchos a comparar
WIDTHS = [10, 50, 100, 500]

rng = np.random.RandomState(SEED)

# Datos sintéticos
X = rng.randn(N_DATA, INPUT_DIM)
X = X / np.linalg.norm(X, axis=1, keepdims=True)


# ============================================================
# KERNEL ANALÍTICO NNGP (límite de ancho infinito)
# ============================================================
def nngp_kernel_fc(X, sigma_w, sigma_b, depth, activation='relu'):
    """
    Kernel NNGP para Fully Connected.
    Fórmula cerrada para ReLU, integración numérica para tanh.
    
    Para ReLU:
      K^ℓ(ρ) = σ_b² + σ_w² · (1/2π)·(√(1-ρ²) + ρ·(π - arccos(ρ)))
    """
    n = X.shape[0]
    K = sigma_b**2 + sigma_w**2 * (X @ X.T) / X.shape[1]
    
    for d in range(depth):
        K_new = np.zeros_like(K)
        for i in range(n):
            for j in range(i, n):
                q_ii = K[i, i]
                q_jj = K[j, j]
                q_ij = K[i, j]
                
                if q_ii > 1e-10 and q_jj > 1e-10:
                    rho = q_ij / np.sqrt(q_ii * q_jj)
                else:
                    rho = 0.0
                rho = np.clip(rho, -0.9999, 0.9999)
                
                if activation == 'relu':
                    theta = np.arccos(rho)
                    val = (np.sqrt(1 - rho**2) + rho * (np.pi - theta)) / (2 * np.pi)
                elif activation == 'tanh':
                    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=20000)
                    val = np.mean(np.tanh(z[:, 0]) * np.tanh(z[:, 1]))
                else:
                    val = rho
                
                K_new[i, j] = sigma_b**2 + sigma_w**2 * np.sqrt(q_ii * q_jj) * val
                K_new[j, i] = K_new[i, j]
        K = K_new
    return K


def nngp_kernel_cnn_approx(X, sigma_w, sigma_b, depth, kernel_size=3, activation='relu'):
    """
    Kernel NNGP aproximado para CNN con peso compartido.
    
    En una CNN, cada neurona recibe solo un parche local del input.
    El kernel efectivo promedia sobre las posiciones espaciales.
    
    Para una CNN 1D con stride=1 y padding='same':
    - Cada neurona ve un parche de tamaño kernel_size
    - El kernel NNGP se calcula como la media de las correlaciones entre parches
    """
    n = X.shape[0]
    dim = X.shape[1]
    
    # Construir matriz de parches (todas las posiciones)
    # Para simplificar: promediamos sobre las posiciones
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Simular el efecto de peso compartido: 
            # cada neurona ve un promedio de las correlaciones locales
            corr_local = 0
            for p in range(dim - kernel_size + 1):
                patch_i = X[i, p:p+kernel_size]
                patch_j = X[j, p:p+kernel_size]
                corr_local += np.dot(patch_i, patch_j) / kernel_size
            corr_local /= (dim - kernel_size + 1)
            K[i, j] = sigma_b**2 + sigma_w**2 * corr_local
    
    for d in range(depth):
        K_new = np.zeros_like(K)
        for i in range(n):
            for j in range(i, n):
                q_ii = K[i, i]
                q_jj = K[j, j]
                q_ij = K[i, j]
                
                if q_ii > 1e-10 and q_jj > 1e-10:
                    rho = q_ij / np.sqrt(q_ii * q_jj)
                else:
                    rho = 0.0
                rho = np.clip(rho, -0.9999, 0.9999)
                
                if activation == 'relu':
                    theta = np.arccos(rho)
                    val = (np.sqrt(1 - rho**2) + rho * (np.pi - theta)) / (2 * np.pi)
                elif activation == 'tanh':
                    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=20000)
                    val = np.mean(np.tanh(z[:, 0]) * np.tanh(z[:, 1]))
                else:
                    val = rho
                
                K_new[i, j] = sigma_b**2 + sigma_w**2 * np.sqrt(q_ii * q_jj) * val
                K_new[j, i] = K_new[i, j]
        K = K_new
    return K


# ============================================================
# KERNEL EMPÍRICO (redes finitas)
# ============================================================
def empirical_kernel_fc(X, width, depth, sigma_w, sigma_b, activation, n_mc=500):
    """
    Calcula el kernel empírico para una FC de ancho finito.
    Usa MC sampling para estimar la matriz de covarianza.
    """
    n = X.shape[0]
    
    if activation == 'relu':
        act_fn = lambda z: np.maximum(z, 0)
    elif activation == 'tanh':
        act_fn = np.tanh
    else:
        act_fn = lambda z: z
    
    # Recolectar salidas para todas las MC samples
    all_f = np.zeros((n_mc, n))
    
    for mc in range(n_mc):
        rng_mc = np.random.RandomState(SEED + mc * 1000)
        h = X.copy()
        
        for layer in range(depth):
            w = rng_mc.randn(width, h.shape[1]) * sigma_w / np.sqrt(h.shape[1])
            b = rng_mc.randn(width) * sigma_b
            h = act_fn(h @ w.T + b)
        
        # Capa de salida
        w_out = rng_mc.randn(1, h.shape[1]) * sigma_w / np.sqrt(h.shape[1])
        b_out = rng_mc.randn(1) * sigma_b
        f = h @ w_out.T + b_out
        all_f[mc] = f[:, 0]
    
    # Kernel empírico = covarianza sobre MC samples
    K_emp = np.cov(all_f.T)
    return K_emp


def empirical_kernel_cnn(X, width, depth, sigma_w, sigma_b, activation,
                         kernel_size=3, n_mc=300):
    """
    Kernel empírico para CNN 1D con peso compartido.
    
    Arquitectura:
      - Capa convolucional 1D: stride=1, padding='same' (aprox.)
      - Activación
      - Capa FC de salida → 1 neurona
    
    El peso compartido significa que el mismo filtro W se aplica
    a todas las posiciones espaciales. Esto rompe la independencia
    del caso FC.
    """
    n = X.shape[0]
    dim = X.shape[1]
    
    if activation == 'relu':
        act_fn = lambda z: np.maximum(z, 0)
    elif activation == 'tanh':
        act_fn = np.tanh
    else:
        act_fn = lambda z: z
    
    n_positions = dim - kernel_size + 1  # output spatial positions
    
    all_f = np.zeros((n_mc, n))
    
    for mc in range(n_mc):
        rng_mc = np.random.RandomState(SEED + mc * 1000 + 9999)
        
        # Para cada capa convolucional: un solo filtro reutilizado
        h = X.copy()  # (n, dim)
        
        for layer in range(depth):
            # Filtro convolucional: (width, kernel_size) — SE REUTILIZA en todas las posiciones
            W_conv = rng_mc.randn(width, kernel_size) * sigma_w / np.sqrt(kernel_size)
            b_conv = rng_mc.randn(width) * sigma_b
            
            # Aplicar convolución: cada posición usa el mismo W_conv
            conv_out = np.zeros((n, width, n_positions))
            for p in range(n_positions):
                patch = h[:, p:p+kernel_size]  # (n, kernel_size)
                conv_out[:, :, p] = patch @ W_conv.T + b_conv  # (n, width)
            
            # Global max pooling sobre posiciones espaciales
            h_pooled = np.max(conv_out, axis=2)  # max over positions: (n, width)
            h = act_fn(h_pooled)  # (n, width)
        
        # Capa de salida FC
        w_out = rng_mc.randn(1, h.shape[1]) * sigma_w / np.sqrt(h.shape[1])
        b_out = rng_mc.randn(1) * sigma_b
        f = h @ w_out.T + b_out
        all_f[mc] = f[:, 0]
    
    return np.cov(all_f.T)


# ============================================================
# Métricas de comparación entre kernels
# ============================================================
def kernel_comparison(K_emp, K_analytic):
    """
    Compara kernel empírico vs analítico con múltiples métricas.
    """
    n = K_emp.shape[0]
    
    # 1. Error de Frobenius normalizado
    frob_err = np.linalg.norm(K_emp - K_analytic, 'fro') / np.linalg.norm(K_analytic, 'fro')
    
    # 2. Error absoluto medio (MAE) en las entradas
    mae = np.mean(np.abs(K_emp - K_analytic))
    
    # 3. Error en la traza (varianza total)
    trace_err = np.abs(np.trace(K_emp) - np.trace(K_analytic)) / np.trace(K_analytic)
    
    # 4. Error en la diagonal (varianza por punto)
    diag_err = np.mean(np.abs(np.diag(K_emp) - np.diag(K_analytic)) / np.diag(K_analytic))
    
    # 5. Correlación entre matrices (Pearson en entradas aplanadas)
    flat_emp = K_emp[np.triu_indices(n)]
    flat_ana = K_analytic[np.triu_indices(n)]
    corr = np.corrcoef(flat_emp, flat_ana)[0, 1]
    
    # 6. Error espectral (mayor autovalor)
    eig_emp = linalg.eigvalsh(K_emp)
    eig_ana = linalg.eigvalsh(K_analytic)
    eig_err = np.linalg.norm(eig_emp - eig_ana) / np.linalg.norm(eig_ana)
    
    # 7. Normalized Kernel Alignment (NKA) de Cristianini
    # NKA = <K1, K2>_F / (||K1||_F * ||K2||_F)
    nka = np.sum(K_emp * K_analytic) / (np.linalg.norm(K_emp, 'fro') * np.linalg.norm(K_analytic, 'fro'))
    
    return {
        'frobenius_err': frob_err,
        'mae': mae,
        'trace_err': trace_err,
        'diag_err': diag_err,
        'correlation': corr,
        'eigen_err': eig_err,
        'nka': nka,
    }


# ============================================================
# Experimento 2a: FC kernel convergence
# ============================================================
def run_exp2a():
    """Comparación FC: kernel analítico vs empírico para varios anchos."""
    print("=" * 72)
    print("EXPERIMENTO 2a: Kernel NNGP — FC, Analítico vs Empírico")
    print("=" * 72)
    
    results = {}
    
    for activation in ['relu', 'tanh']:
        print(f"\n{'─' * 60}")
        print(f"Activación: {activation}")
        print(f"{'─' * 60}")
        
        # Kernel analítico
        t0 = time.time()
        K_analytic = nngp_kernel_fc(X, SIGMA_W, SIGMA_B, DEPTH, activation)
        print(f"  Kernel analítico: {time.time()-t0:.1f}s")
        
        for width in WIDTHS:
            t0 = time.time()
            n_mc = min(500, 5000 // width + 100)
            K_emp = empirical_kernel_fc(X, width, DEPTH, SIGMA_W, SIGMA_B, activation, n_mc=n_mc)
            metrics = kernel_comparison(K_emp, K_analytic)
            results[(activation, width)] = metrics
            elapsed = time.time() - t0
            
            print(f"  n={width:4d} (MC={n_mc}): "
                  f"Frob={metrics['frobenius_err']:.4f}  "
                  f"NKA={metrics['nka']:.4f}  "
                  f"Corr={metrics['correlation']:.4f}  "
                  f"({elapsed:.1f}s)")
    
    return results


# ============================================================
# Experimento 2b: CNN — ¿se rompe la convergencia?
# ============================================================
def run_exp2b():
    """CNN con peso compartido: kernel empírico vs analítico."""
    print(f"\n{'=' * 72}")
    print("EXPERIMENTO 2b: CNN (Peso Compartido) — ¿Convergencia GP?")
    print("=" * 72)
    print("  Hipótesis: El peso compartido introduce correlaciones que")
    print("  retrasan/alteran la convergencia al kernel NNGP.")
    
    width = 200
    activation = 'relu'
    
    # Kernel analítico aproximado para CNN
    K_ana_cnn = nngp_kernel_cnn_approx(X, SIGMA_W, SIGMA_B, DEPTH, kernel_size=3, activation=activation)
    
    # Kernel empírico CNN
    print(f"\n  Calculando kernel empírico CNN (n={width})...")
    t0 = time.time()
    K_emp_cnn = empirical_kernel_cnn(X, width, DEPTH, SIGMA_W, SIGMA_B, activation, 
                                      kernel_size=3, n_mc=300)
    print(f"  Tiempo: {time.time()-t0:.1f}s")
    
    # También comparar con kernel empírico FC (mismo ancho)
    K_emp_fc = empirical_kernel_fc(X, width, DEPTH, SIGMA_W, SIGMA_B, activation, n_mc=300)
    K_ana_fc = nngp_kernel_fc(X, SIGMA_W, SIGMA_B, DEPTH, activation)
    
    metrics_cnn = kernel_comparison(K_emp_cnn, K_ana_cnn)
    metrics_fc = kernel_comparison(K_emp_fc, K_ana_fc)
    
    print(f"\n  Resultados:")
    print(f"  {'Métrica':20s}  {'FC':>12s}  {'CNN':>12s}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}")
    for key in ['frobenius_err', 'nka', 'correlation', 'trace_err']:
        print(f"  {key:20s}  {metrics_fc[key]:>12.4f}  {metrics_cnn[key]:>12.4f}")
    
    return {
        'cnn': metrics_cnn,
        'fc': metrics_fc,
        'K_ana_cnn': K_ana_cnn,
        'K_emp_cnn': K_emp_cnn,
        'K_ana_fc': K_ana_fc,
        'K_emp_fc': K_emp_fc,
    }


# ============================================================
# Visualización
# ============================================================
def plot_kernels(results_2a, results_2b):
    """Genera figuras de comparación de kernels."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Convergencia del Kernel NNGP: Analítico vs Empírico', 
                 fontsize=14, fontweight='bold')
    
    colors = {'tanh': '#2196F3', 'relu': '#F44336'}
    
    # 1. Frobenius error vs width
    ax = axes[0, 0]
    for activation in ['tanh', 'relu']:
        widths = WIDTHS
        errs = [results_2a.get((activation, w), {}).get('frobenius_err', 0) for w in widths]
        ax.semilogy(widths, errs, 'o-', color=colors[activation], label=activation, markersize=8)
    ax.set_xlabel('Ancho n'); ax.set_ylabel('Error Frobenius (norm.)')
    ax.set_title('Convergencia del Kernel', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # 2. NKA vs width
    ax = axes[0, 1]
    for activation in ['tanh', 'relu']:
        nkas = [results_2a.get((activation, w), {}).get('nka', 0) for w in WIDTHS]
        ax.plot(WIDTHS, nkas, 'o-', color=colors[activation], label=activation, markersize=8)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Ancho n'); ax.set_ylabel('NKA')
    ax.set_title('Kernel Alignment', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # 3. Correlation vs width
    ax = axes[0, 2]
    for activation in ['tanh', 'relu']:
        corrs = [results_2a.get((activation, w), {}).get('correlation', 0) for w in WIDTHS]
        ax.plot(WIDTHS, corrs, 'o-', color=colors[activation], label=activation, markersize=8)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Ancho n'); ax.set_ylabel('Correlación')
    ax.set_title('Correlación Matricial', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # 4. Espectro de autovalores
    ax = axes[1, 0]
    K_ana_fc = nngp_kernel_fc(X, SIGMA_W, SIGMA_B, DEPTH, 'relu')
    eig_ana = linalg.eigvalsh(K_ana_fc)[::-1]
    ax.semilogy(range(1, len(eig_ana)+1), eig_ana, 'k-', linewidth=2, label='Analítico (NNGP)')
    for width in [10, 100]:
        n_mc = min(500, 5000 // width + 100)
        K_emp = empirical_kernel_fc(X, width, DEPTH, SIGMA_W, SIGMA_B, 'relu', n_mc=n_mc)
        eig_emp = linalg.eigvalsh(K_emp)[::-1]
        ax.semilogy(range(1, len(eig_emp)+1), eig_emp, 'o-', alpha=0.6, 
                   label=f'n={width}', markersize=4)
    ax.set_xlabel('Índice de autovalor'); ax.set_ylabel('Autovalor')
    ax.set_title('Espectro del Kernel', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # 5. Matriz de correlación (n=500)
    ax = axes[1, 1]
    K_emp_500 = empirical_kernel_fc(X, 500, DEPTH, SIGMA_W, SIGMA_B, 'relu', n_mc=500)
    # Normalizar a correlación
    D = np.diag(1.0 / np.sqrt(np.maximum(np.diag(K_emp_500), 1e-10)))
    K_corr_emp = D @ K_emp_500 @ D
    im = ax.imshow(K_corr_emp, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(f'Matriz de correlación (n=500)', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 6. Error CNN vs FC
    ax = axes[1, 2]
    metrics_list = []
    labels_list = []
    if results_2b:
        for arch in ['fc', 'cnn']:
            if arch in results_2b:
                m = results_2b[arch]
                metrics_list.append([m['frobenius_err'], 1 - m['nka'], m['trace_err']])
                labels_list.append(arch)
    if metrics_list:
        x = np.arange(len(labels_list))
        width = 0.25
        for idx, metric_name in enumerate(['Frob. Error', '1-NKA', 'Trace Error']):
            ax.bar(x + idx*width, [m[idx] for m in metrics_list], width, label=metric_name)
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels_list)
        ax.set_ylabel('Error')
        ax.set_title('FC vs CNN (peso compartido)', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('/home/radxa/.hermes/hermes-agent/Research/TFM-Advanced Mathematics/experiments/fig03_kernel_convergence.png', 
                dpi=150, bbox_inches='tight')
    print(f"\n[✓] fig03_kernel_convergence.png")
    plt.close()


# ============================================================
# Reporte LaTeX
# ============================================================
def generate_report(results_2a):
    lines = []
    lines.append(r'')
    lines.append(r'\subsection{Kernel NNGP: Analítico vs Empírico}')
    lines.append(r'')
    lines.append(r'Comparamos el kernel analítico del límite NNGP con el kernel empírico')
    lines.append(r'de redes finitas para anchos crecientes $n \in \{10, 50, 100, 500\}$.')
    lines.append(r'Usamos %d puntos de datos, dimensión de entrada %d,' % (N_DATA, INPUT_DIM))
    lines.append(r'%d capas ocultas, $\sigma_w=%.1f$, $\sigma_b=%.1f$.' % (DEPTH, SIGMA_W, SIGMA_B))
    lines.append(r'')
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(r'\caption{Convergencia del kernel NNGP (FC, ReLU)}')
    lines.append(r'\begin{tabular}{c|cccc}')
    lines.append(r'\toprule')
    lines.append(r'$n$ & Frob. Error & NKA & Correlación & Eigen Error \\')
    lines.append(r'\midrule')
    for w in WIDTHS:
        m = results_2a.get(('relu', w), {})
        if m:
            lines.append(r'  $%d$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ \\' % 
                        (w, m['frobenius_err'], m['nka'], m['correlation'], m['eigen_err']))
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    lines.append(r'')
    lines.append('La Figura~\\ref{fig:kernels} muestra la convergencia.')
    
    return '\n'.join(lines)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Iniciando experimento de kernels...")
    
    results_2a = run_exp2a()
    results_2b = run_exp2b()
    
    print(f"\n{'=' * 60}")
    print("Generando figuras...")
    plot_kernels(results_2a, results_2b)
    
    print(f"\n{'=' * 60}")
    latex = generate_report(results_2a)
    with open('/home/radxa/.hermes/hermes-agent/Research/TFM-Advanced Mathematics/experiments/02_kernel_report.tex', 'w') as f:
        f.write(latex)
    print("[✓] Reporte LaTeX guardado")
    
    print(f"\n{'=' * 60}")
    print("EXPERIMENTO 2 COMPLETADO")
    print("=" * 60)
