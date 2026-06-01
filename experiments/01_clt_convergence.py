#!/usr/bin/env python3
"""
Experimento 1: Verificación numérica del Teorema Central del Límite en RNs
(Versión híbrida: vectorizada para anchos pequeños, secuencial para grandes)
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuración
# ============================================================
INPUT_DIM = 10
N_TEST_POINTS = 5
SEED = 42
SIGMA_W = 1.5
SIGMA_B = 0.1
DEPTH = 3

# Anchos y muestras MC (más MC para anchos pequeños)
WIDTHS_AND_MC = [
    (5, 3000),
    (10, 2000),
    (50, 1500),
    (100, 1000),
    (500, 500),
    (1000, 150),
    (2000, 0),  # 0 = skip (solo analítico)
    (5000, 0),  # solo analítico
]

rng = np.random.RandomState(SEED)

ACTIVATIONS = ['tanh', 'relu']

def get_activation_fn(name):
    if name == 'tanh':
        return lambda z: np.tanh(z)
    elif name == 'relu':
        return lambda z: np.maximum(z, 0)
    elif name == 'gelu':
        return lambda z: z * stats.norm.cdf(z)
    return lambda z: z  # linear


# ============================================================
# NNGP kernel analítico
# ============================================================
def nngp_kernel_analytic(X, sigma_w, sigma_b, activation='relu', depth=1):
    n = X.shape[0]
    K = sigma_b**2 + sigma_w**2 * (X @ X.T) / X.shape[1]
    
    for _ in range(depth):
        K_new = np.zeros_like(K)
        for i in range(n):
            for j in range(n):
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
                    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=50000)
                    val = np.mean(np.tanh(z[:, 0]) * np.tanh(z[:, 1]))
                elif activation == 'gelu':
                    # GELU analítico cerrado: E[gelu(u) gelu(v)]
                    # gelu(x) = x * Phi(x), donde Phi es la CDF normal
                    # Usamos integración numérica directa
                    n_grid = 200
                    x = np.linspace(-5, 5, n_grid)
                    dx = x[1] - x[0]
                    wx = stats.norm.pdf(x, 0, 1)  # marginal N(0,1)
                    # grid 2D
                    X_grid, Y_grid = np.meshgrid(x, x)
                    w2d = np.exp(-0.5 * (X_grid**2 + Y_grid**2 - 2*rho*X_grid*Y_grid) / (1 - rho**2))
                    w2d = w2d / (2 * np.pi * np.sqrt(1 - rho**2))
                    gelu_x = X_grid * stats.norm.cdf(X_grid)
                    gelu_y = Y_grid * stats.norm.cdf(Y_grid)
                    val = np.sum(gelu_x * gelu_y * w2d) * dx**2
                else:
                    val = rho
                
                K_new[i, j] = sigma_b**2 + sigma_w**2 * np.sqrt(q_ii * q_jj) * val
        K = K_new
    return K


# ============================================================
# Red neuronal de ancho finito (2 modos: vectorizado y secuencial)
# ============================================================
def finite_width_network_vectorized(X, width, n_mc, depth, sigma_w, sigma_b, activation_name, seed):
    """Para anchos pequeños/medianos: vectorizado. Para anchos grandes: secuencial."""
    act_fn = get_activation_fn(activation_name)
    n_samples = X.shape[0]
    
    # Umbral: anchos > 2000 usan modo secuencial para evitar OOM
    if width > 2000:
        return _finite_width_sequential(X, width, n_mc, depth, sigma_w, sigma_b, act_fn, seed)
    
    if width <= 300:
        # Procesar todo de una vez
        return _run_vectorized_batch(X, width, n_mc, depth, sigma_w, sigma_b, act_fn, seed)
    else:
        # Anchoss grandes: dividir en batches pequeños para no saturar memoria
        batch_size = max(10, min(50, n_mc))
        all_out = []
        remaining = n_mc
        batch_id = 0
        while remaining > 0:
            this_batch = min(batch_size, remaining)
            out = _run_vectorized_batch(X, width, this_batch, depth, sigma_w, sigma_b, act_fn, seed + batch_id * 10000)
            all_out.append(out)
            remaining -= this_batch
            batch_id += 1
        return np.concatenate(all_out, axis=1)


def _run_vectorized_batch(X, width, n_mc, depth, sigma_w, sigma_b, act_fn, seed):
    """Ejecuta un batch de MC samples completamente vectorizado."""
    rng_local = np.random.RandomState(seed)
    n_samples = X.shape[0]
    h = np.tile(X[None, :, :], (n_mc, 1, 1))  # (n_mc, n_samples, dim)
    
    for layer in range(depth):
        in_dim = h.shape[2]
        # Usar float32 para ahorrar memoria
        W = rng_local.randn(n_mc, width, in_dim).astype(np.float32) * sigma_w / np.sqrt(in_dim)
        b = rng_local.randn(n_mc, width).astype(np.float32) * sigma_b
        z = np.matmul(W, h.transpose(0, 2, 1)) + b[:, :, None]
        h = act_fn(z).transpose(0, 2, 1)
    
    # Capa de salida
    in_dim = h.shape[2]
    W_out = rng_local.randn(n_mc, 1, in_dim).astype(np.float32) * sigma_w / np.sqrt(in_dim)
    b_out = rng_local.randn(n_mc, 1).astype(np.float32) * sigma_b
    f = np.matmul(W_out, h.transpose(0, 2, 1)) + b_out[:, :, None]
    return f[:, 0, :].T  # (n_samples, n_mc)


def _finite_width_sequential(X, width, n_mc, depth, sigma_w, sigma_b, act_fn, seed):
    """Para anchos muy grandes: procesa una MC sample a la vez (bajo consumo de memoria)."""
    n_samples = X.shape[0]
    outputs = np.zeros((n_samples, n_mc), dtype=np.float32)
    rng_local = np.random.RandomState(seed)
    
    for mc in range(n_mc):
        h = X.copy().astype(np.float32)  # (n_samples, dim)
        
        for layer in range(depth):
            in_dim = h.shape[1]
            W = rng_local.randn(width, in_dim).astype(np.float32) * sigma_w / np.sqrt(in_dim)
            b = rng_local.randn(width).astype(np.float32) * sigma_b
            z = W @ h.T + b[:, None]  # (width, n_samples)
            h = act_fn(z).T  # (n_samples, width)
        
        # Capa de salida
        in_dim = h.shape[1]
        w_out = rng_local.randn(1, in_dim).astype(np.float32) * sigma_w / np.sqrt(in_dim)
        b_out = rng_local.randn(1).astype(np.float32) * sigma_b
        f = (w_out @ h.T + b_out[:, None])[0]  # (n_samples,)
        outputs[:, mc] = f
    
    return outputs


# ============================================================
# Métricas de convergencia
# ============================================================
def convergence_metrics(samples, analytic_std):
    z_scores = (samples - np.mean(samples)) / np.std(samples)
    
    if len(z_scores) < 5000:
        _, shapiro_p = stats.shapiro(z_scores)
    else:
        _, shapiro_p = stats.normaltest(z_scores)
    
    ks_stat, ks_p = stats.kstest(z_scores, 'norm')
    
    sorted_samples = np.sort(z_scores)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.001, 0.999, len(sorted_samples)))
    wasserstein_dist = np.mean(np.abs(sorted_samples - theoretical_quantiles))
    
    hist_emp, bins = np.histogram(z_scores, bins=50, density=True, range=(-4, 4))
    hist_theor = stats.norm.pdf((bins[:-1] + bins[1:]) / 2)
    hist_theor = hist_theor / np.sum(hist_theor)
    hist_emp = hist_emp / np.sum(hist_emp) + 1e-10
    js_div = jensenshannon(hist_emp, hist_theor)
    
    std_ratio = np.std(samples) / max(analytic_std, 1e-10)
    
    return {
        'shapiro_p': shapiro_p,
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        'wasserstein': wasserstein_dist,
        'js_divergence': js_div,
        'std_ratio': std_ratio,
    }


# ============================================================
# Experimento principal
# ============================================================
def run_experiment():
    print("=" * 72)
    print("EXPERIMENTO 1: Convergencia CLT en Redes Neuronales")
    print("=" * 72)
    
    X_test = rng.randn(N_TEST_POINTS, INPUT_DIM)
    X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
    
    all_results = {}
    
    for activation_name in ACTIVATIONS:
        print(f"\n{'─' * 60}")
        print(f"Activación: {activation_name.upper()}")
        print(f"{'─' * 60}")
        
        t0 = time.time()
        K_analytic = nngp_kernel_analytic(X_test, SIGMA_W, SIGMA_B, activation_name, DEPTH)
        analytic_stds = np.sqrt(np.diag(K_analytic))
        print(f"  Kernel analítico en {time.time()-t0:.1f}s")
        
        results = {}
        for width, n_mc in WIDTHS_AND_MC:
            if n_mc == 0:
                print(f"  n={width:5d}  (skip, solo analítico)  ", end='')
                # Añadir NaN para las métricas
                results[width] = {'wasserstein': np.nan, 'js_divergence': np.nan,
                                  'ks_p': np.nan, 'std_ratio': 1.0, 'shapiro_p': np.nan}
                print()
                continue
            t_start = time.time()
            print(f"  n={width:5d} MC={n_mc:4d}  ", end='', flush=True)
            
            outputs = finite_width_network_vectorized(
                X_test, width, n_mc, DEPTH, SIGMA_W, SIGMA_B, activation_name,
                seed=SEED + width
            )
            
            point_metrics = []
            for i in range(N_TEST_POINTS):
                metrics = convergence_metrics(outputs[i], analytic_stds[i])
                point_metrics.append(metrics)
            
            avg_metrics = {k: np.mean([m[k] for m in point_metrics]) for k in point_metrics[0]}
            results[width] = avg_metrics
            
            elapsed = time.time() - t_start
            print(f"Wass={avg_metrics['wasserstein']:.5f}  "
                  f"KS-p={avg_metrics['ks_p']:.4f}  "
                  f"σ-ratio={avg_metrics['std_ratio']:.4f}  "
                  f"({elapsed:.1f}s)")
        
        all_results[activation_name] = results
    
    return all_results, X_test


# ============================================================
# Visualización
# ============================================================
def plot_convergence(all_results, X_test):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Convergencia CLT: Salida → Gaussiana cuando n → ∞', 
                 fontsize=14, fontweight='bold')
    
    colors = {'tanh': '#2196F3', 'relu': '#F44336', 'gelu': '#4CAF50'}
    
    # 1. Wasserstein distance
    ax = axes[0, 0]
    power_laws = {}
    for act_name, results in all_results.items():
        widths = sorted(results.keys())
        ws = [results[w]['wasserstein'] for w in widths]
        ax.loglog(widths, ws, 'o-', color=colors[act_name], label=act_name, markersize=6)
        coeffs = np.polyfit(np.log(widths), np.log(ws), 1)
        power_laws[act_name] = coeffs[0]
        ax.loglog(widths, np.exp(np.polyval(coeffs, np.log(widths))), 
                  '--', color=colors[act_name], alpha=0.3,
                  label=f'$n^{{{coeffs[0]:.2f}}}$')
    ax.axhline(0.01, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Ancho n'); ax.set_ylabel('Wasserstein-1')
    ax.set_title('Convergencia a la Normal', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # 2. Std ratio
    ax = axes[0, 1]
    for act_name, results in all_results.items():
        widths = sorted(results.keys())
        ax.semilogx(widths, [results[w]['std_ratio'] for w in widths], 
                   'o-', color=colors[act_name], label=act_name, markersize=6)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Ancho n'); ax.set_ylabel('σ_emp / σ_anal')
    ax.set_title('Precisión de la Varianza', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # 3. KS p-value
    ax = axes[0, 2]
    for act_name, results in all_results.items():
        widths = sorted(results.keys())
        ax.semilogx(widths, [results[w]['ks_p'] for w in widths], 
                   'o-', color=colors[act_name], label=act_name, markersize=6)
    ax.axhline(0.05, color='red', linestyle=':', alpha=0.5)
    ax.set_xlabel('Ancho n'); ax.set_ylabel('KS p-valor')
    ax.set_title('Test de Normalidad', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # 4. QQ-plot n=10 vs n=1000
    ax = axes[1, 0]
    X1 = X_test[:1]
    for act_name in ['relu', 'tanh']:
        for w, style in [(10, ':'), (1000, '-')]:
            n_mc = 200 if w == 1000 else 2000
            out = finite_width_network_vectorized(X1, w, n_mc, DEPTH, SIGMA_W, SIGMA_B, act_name, SEED + w)
            z = np.sort((out[0] - np.mean(out[0])) / np.std(out[0]))
            q = stats.norm.ppf(np.linspace(0.001, 0.999, len(z)))
            ax.plot(q, z, style, color=colors.get(act_name, 'gray'), alpha=0.7, lw=1,
                   label=f'{act_name} n={w}')
    ax.plot([-4, 4], [-4, 4], 'k--', alpha=0.5)
    ax.set_xlabel('Cuantiles N(0,1)'); ax.set_ylabel('Cuantiles empíricos')
    ax.set_title('QQ-plot', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # 5. Histogram (n=1000)
    ax = axes[1, 1]
    for act_name in ['relu', 'tanh']:
        out = finite_width_network_vectorized(X1, 1000, 200, DEPTH, SIGMA_W, SIGMA_B, act_name, SEED + 1000)
        z = (out[0] - np.mean(out[0])) / np.std(out[0])
        ax.hist(z, bins=40, density=True, alpha=0.3, color=colors[act_name], label=f'{act_name}')
    x = np.linspace(-4, 4, 200)
    ax.plot(x, stats.norm.pdf(x), 'k-', lw=2, label='N(0,1)')
    ax.set_xlabel('Salida estandarizada'); ax.set_ylabel('Densidad')
    ax.set_title('Histograma n=5000', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # 6. Power law summary
    ax = axes[1, 2]; ax.axis('off')
    table_data = []
    for a in ['tanh', 'relu', 'gelu']:
        if a not in all_results:
            continue
        res = all_results[a]
        widths = sorted(res.keys())
        ws = [res[w]['wasserstein'] for w in widths]
        coeffs = np.polyfit(np.log(widths), np.log(ws), 1)
        alpha = coeffs[0]
        w50 = res.get(50, {}).get('wasserstein', 0)
        if alpha < -0.01:
            log_n_crit = (np.log(0.01) - coeffs[1]) / coeffs[0]
            n_crit = f'{int(np.exp(log_n_crit))}'
        else:
            n_crit = '$\\infty$'
        table_data.append([f'$n^{{{alpha:.2f}}}$', f'{w50:.4f}', n_crit])
    row_labels = ['tanh', 'relu', 'gelu'][:len(table_data)]
    if table_data:
        ax.table(cellText=table_data, rowLabels=row_labels,
                colLabels=['Ley', 'W(n=50)', 'n_crit'], loc='center', cellLoc='center')
        ax.set_title('Scaling Laws', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/radxa/.hermes/hermes-agent/Research/TFM-Advanced Mathematics/experiments/fig01_clt_convergence.png', 
                dpi=150, bbox_inches='tight')
    print(f"\n[✓] fig01_clt_convergence.png")
    plt.close()
    
    # FIG 2: Covariance convergence
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Covarianza Empírica → Kernel NNGP', fontsize=13, fontweight='bold')
    
    act = 'relu'
    im = None
    
    for idx, (width, n_mc) in enumerate([(10, 1500), (100, 500), (500, 300)]):
        ax = axes[idx]
        all_out = []
        for i in range(N_TEST_POINTS):
            out_i = finite_width_network_vectorized(
                X_test[i:i+1], width, n_mc, DEPTH, 
                SIGMA_W, SIGMA_B, act, SEED + i * 10000 + width)
            all_out.append(out_i[0])
        outputs = np.array(all_out)
        
        K_emp = np.cov(outputs)
        K_ana = nngp_kernel_analytic(X_test, SIGMA_W, SIGMA_B, act, DEPTH)
        
        # Normalizar a matrices de correlación
        D_emp = np.diag(1.0 / np.sqrt(np.maximum(np.diag(K_emp), 1e-10)))
        K_emp_norm = D_emp @ K_emp @ D_emp
        D_ana = np.diag(1.0 / np.sqrt(np.maximum(np.diag(K_ana), 1e-10)))
        K_ana_norm = D_ana @ K_ana @ D_ana
        
        error = np.abs(K_emp_norm - K_ana_norm)
        im = ax.imshow(error, cmap='YlOrRd', vmin=0, vmax=0.3)
        for i in range(N_TEST_POINTS):
            for j in range(N_TEST_POINTS):
                ax.text(j, i, f'{K_emp_norm[i,j]:.2f}', ha='center', va='center',
                       fontsize=9, color='black' if error[i,j] < 0.15 else 'white')
        ax.set_title(f'n = {width}', fontweight='bold')
        ax.set_xlabel('Input j'); ax.set_ylabel('Input i')
    
    if im is not None:
        plt.colorbar(im, ax=axes, label='Error de correlación', shrink=0.8)
    plt.tight_layout()
    plt.savefig('/home/radxa/.hermes/hermes-agent/Research/TFM-Advanced Mathematics/experiments/fig02_joint_convergence.png', 
                dpi=150, bbox_inches='tight')
    print(f"[✓] fig02_joint_convergence.png")
    plt.close()


# ============================================================
# Reporte LaTeX
# ============================================================
def generate_latex_report(all_results):
    lines = []
    lines.append(r'\section{Verificación Numérica del Límite NNGP}')
    lines.append(r'\subsection{Convergencia por el Teorema Central del Límite}')
    lines.append('')
    lines.append('Para verificar la convergencia de la salida de una red neuronal')
    lines.append('a un proceso gaussiano, simulamos una red feedforward de %d capas ocultas' % DEPTH)
    lines.append('con anchos crecientes. Las simulaciones usan la versión vectorizada')
    lines.append('(todas las muestras Monte Carlo simultáneas).')
    lines.append('')
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(r'\caption{Métricas de convergencia a la normal (ReLU)}')
    lines.append(r'\label{tab:convergence}')
    lines.append(r'\begin{tabular}{c|cccc}')
    lines.append(r'\toprule')
    lines.append(r'$n$ & Wasserstein & KS $p$ & $\sigma_{\text{emp}}/\sigma_{\text{anal}}$ & JS \\')
    lines.append(r'\midrule')
    if 'relu' in all_results:
        for w in sorted(all_results['relu'].keys()):
            m = all_results['relu'][w]
            lines.append(r'  $%d$ & $%.5f$ & $%.4f$ & $%.4f$ & $%.5f$ \\' % 
                        (w, m['wasserstein'], m['ks_p'], m['std_ratio'], m['js_divergence']))
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')
    lines.append('Las figuras muestran la convergencia empírica.')
    return '\n'.join(lines)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Iniciando experimento...")
    print(f"  Inputs: {N_TEST_POINTS} pts, dim={INPUT_DIM}, σ_w={SIGMA_W}, σ_b={SIGMA_B}")
    print()
    
    all_results, X_test = run_experiment()
    
    print(f"\n{'=' * 60}")
    print("Generando figuras...")
    plot_convergence(all_results, X_test)
    
    print(f"\n{'=' * 60}")
    latex = generate_latex_report(all_results)
    with open('/home/radxa/.hermes/hermes-agent/Research/TFM-Advanced Mathematics/experiments/01_clt_report.tex', 'w') as f:
        f.write(latex)
    print("[✓] Reporte LaTeX guardado")
    
    print(f"\n{'=' * 60}")
    print("EXPERIMENTO COMPLETADO")
    print("=" * 60)
