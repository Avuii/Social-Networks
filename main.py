import os
import csv
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# konfiguracja
N_VALUES = [50, 100, 200, 500, 1000]
N_SMALL = [10, 20, 50, 100, 200]

# z ilu uśrednianie
N_RUNS = 30

# parametry modeli
ER_P = 0.05

WS_K = 4
WS_P_REWIRE = 0.1
WS_P_VALUES = [0.0, 0.01, 0.05, 0.1, 0.5]  # do zbadania

BA_M = 2

COLORS = {"ER": "tab:blue", "WS": "tab:orange", "BA": "tab:green"}

random.seed(0)
np.random.seed(0)

# ======================================================================================================================
# zapis i logi

PLOTS_DIR = "plots"
DATA_DIR = "data"

for sub in ["ER", "WS", "BA", "compare"]:
    os.makedirs(os.path.join(PLOTS_DIR, sub), exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def plot_path(model, fname):
    return os.path.join(PLOTS_DIR, model, fname)


def log(msg):
    print(f"[LOG] {msg}")


def log_saved(path):
    log(f"Zapisano -> {os.path.abspath(path)}")


# ======================================================================================================================

def largest_connected_component(G: nx.Graph) -> nx.Graph:
    if nx.is_connected(G):
        return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def compute_metrics(G: nx.Graph):
    Gc = largest_connected_component(G)
    return {
        "avg_clustering": nx.average_clustering(G),
        "diameter": nx.diameter(Gc),
        "radius": nx.radius(Gc),
        "avg_degree": float(np.mean([d for _, d in G.degree()])),
    }


def degree_distribution(G: nx.Graph) -> Counter:
    return Counter([d for _, d in G.degree()])


def averaged_degree_distribution(model_fn, runs: int, *args):
    acc = Counter()
    for _ in range(runs):
        acc += degree_distribution(model_fn(*args))

    total = sum(acc.values())
    k = np.array(sorted(acc.keys()))
    pk = np.array([acc[ki] for ki in k], dtype=float) / float(total)
    return k, pk


def run_experiment(model_fn, *args):
    acc = {k: [] for k in ["avg_clustering", "diameter", "radius", "avg_degree"]}
    for _ in range(N_RUNS):
        m = compute_metrics(model_fn(*args))
        for key in acc:
            acc[key].append(m[key])

    return {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in acc.items()}


# ======================================================================================================================
# grafy modeli sieci

def er_graph(N, p):
    return nx.erdos_renyi_graph(N, p)


def ba_graph(N, m):
    return nx.barabasi_albert_graph(N, m)


def ws_graph(N, k, p):
    G = nx.Graph()
    G.add_nodes_from(range(N))

    # ring
    for i in range(N):
        for j in range(1, k // 2 + 1):
            G.add_edge(i, (i + j) % N)

    # rewiring
    for i in range(N):
        for j in range(1, k // 2 + 1):
            if random.random() < p:
                old = (i, (i + j) % N)
                neighbors = set(G.neighbors(i))
                possible = list(set(range(N)) - {i} - neighbors)
                if possible:
                    G.remove_edge(*old)
                    G.add_edge(i, random.choice(possible))

    return G


# ======================================================================================================================
# WYKRESY

def plot_graph_visual(G, model, title, fname, seed=0, use_lcc=True):
    G_draw = G

    n_comp = nx.number_connected_components(G_draw)
    n_iso = nx.number_of_isolates(G_draw)

    if use_lcc and n_comp > 1:
        G_draw = G_draw.subgraph(max(nx.connected_components(G_draw), key=len)).copy()

    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G_draw, seed=seed, iterations=300)
    nx.draw(
        G_draw, pos,
        node_size=25,
        width=0.6,
        node_color=COLORS[model],
        edge_color="gray",
        alpha=0.85
    )
    plt.title(title)
    plt.axis("off")

    out = plot_path(model, fname)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    if use_lcc and n_comp > 1:
        log(f"{model}: components={n_comp}, isolates={n_iso} -> rysuję LCC: |V|={G_draw.number_of_nodes()}, |E|={G_draw.number_of_edges()}")
    else:
        log(f"{model}: components={n_comp}, isolates={n_iso} -> rysuję cały graf: |V|={G_draw.number_of_nodes()}, |E|={G_draw.number_of_edges()}")

    log_saved(out)


def plot_degree_histogram_avg(model_fn, model, N, title, *args):
    degrees_all = []
    for _ in range(N_RUNS):
        G = model_fn(N, *args)
        degrees_all.extend([d for _, d in G.degree()])

    plt.figure()
    plt.hist(degrees_all, bins=30, color=COLORS[model], alpha=0.9)
    plt.yscale("log")
    plt.xlabel("stopień k")
    plt.ylabel("liczba wierzchołków (log)")
    plt.title(f"{title} (N={N}, {N_RUNS} realizacji)")

    out = plot_path(model, f"{model}_histogram_N{N}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log_saved(out)


def plot_pk_loglog_avg(model_fn, model, N, title, *args):
    k, pk = averaged_degree_distribution(model_fn, N_RUNS, N, *args)

    plt.figure()
    plt.scatter(k, pk, s=18, color=COLORS[model], alpha=0.9)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.title(f"{title} (N={N}, {N_RUNS} realizacji)")
    plt.grid(alpha=0.25, which="both")

    out = plot_path(model, f"{model}_pk_loglog_N{N}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log_saved(out)


def plot_pk_compare(N):
    plt.figure()

    for fn, args, label in [
        (er_graph, (ER_P,), "ER"),
        (ws_graph, (WS_K, WS_P_REWIRE), "WS"),
        (ba_graph, (BA_M,), "BA"),
    ]:
        k, pk = averaged_degree_distribution(fn, N_RUNS, N, *args)
        plt.scatter(k, pk, s=18, alpha=0.9, label=label, color=COLORS[label])

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.title(f"Porównanie P(k) (N={N}, {N_RUNS} realizacji)")
    plt.legend()
    plt.grid(alpha=0.25, which="both")

    out = plot_path("compare", f"Pk_compare_N{N}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log_saved(out)


def plot_metric_vs_N(results, key, ylabel, fname):
    plt.figure()
    for model, res in results.items():
        N = list(res.keys())
        means = [res[n][key][0] for n in N]
        stds = [res[n][key][1] for n in N]

        plt.errorbar(
            N, means, yerr=stds,
            fmt="o", markersize=6,
            capsize=4, elinewidth=1.2,
            label=model, color=COLORS[model], alpha=0.9
        )

    plt.xlabel("N")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)

    out = plot_path("compare", fname)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log_saved(out)


def plot_ws_clustering_vs_p(N):
    means = []
    stds = []
    for p in WS_P_VALUES:
        res = run_experiment(ws_graph, N, WS_K, p)
        means.append(res["avg_clustering"][0])
        stds.append(res["avg_clustering"][1])

    plt.figure()
    plt.errorbar(
        WS_P_VALUES, means, yerr=stds,
        fmt="o", markersize=6,
        capsize=4, elinewidth=1.2,
        color=COLORS["WS"], alpha=0.9
    )

    plt.xlabel(r"$p_{\mathrm{rew}}$")
    plt.ylabel("⟨C⟩")
    plt.title(f"WS: ⟨C⟩ vs p (N={N}, {N_RUNS} realizacji)")
    plt.grid(alpha=0.3)

    out = plot_path("WS", f"WS_C_vs_p_N{N}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log_saved(out)


# ======================================================================================================================
# .CSV

def save_metrics_csv(model, res_dict):
    out = os.path.join(DATA_DIR, f"{model}_metrics.csv")
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N",
            "avg_clustering_mean", "avg_clustering_std",
            "diameter_mean", "diameter_std",
            "radius_mean", "radius_std",
            "avg_degree_mean", "avg_degree_std",
        ])
        for N, m in res_dict.items():
            writer.writerow([
                N,
                m["avg_clustering"][0], m["avg_clustering"][1],
                m["diameter"][0], m["diameter"][1],
                m["radius"][0], m["radius"][1],
                m["avg_degree"][0], m["avg_degree"][1],
            ])
    log_saved(out)


# ======================================================================================================================
# MAIN
# ======================================================================================================================

if __name__ == "__main__":
    log("START")

    N_VIS = 100
    log(f"Rysowanie przykładowych sieci (N={N_VIS})")
    plot_graph_visual(er_graph(N_VIS, ER_P), "ER", f"ER (N={N_VIS}, p={ER_P})", f"ER_graph_N{N_VIS}.png")
    plot_graph_visual(ws_graph(N_VIS, WS_K, WS_P_REWIRE), "WS", f"WS (N={N_VIS}, k={WS_K}, p={WS_P_REWIRE})", f"WS_graph_N{N_VIS}.png")
    plot_graph_visual(ba_graph(N_VIS, BA_M), "BA", f"BA (N={N_VIS}, m={BA_M})", f"BA_graph_N{N_VIS}.png")

    log("Histogramy stopni (ER/WS/BA)")
    for N in [100, 500, 1000]:
        plot_degree_histogram_avg(er_graph, "ER", N, "ER – histogram stopni", ER_P)
        plot_degree_histogram_avg(ws_graph, "WS", N, "WS – histogram stopni", WS_K, WS_P_REWIRE)
        plot_degree_histogram_avg(ba_graph, "BA", N, "BA – histogram stopni", BA_M)

    log("P(k) log-log (osobno ER/WS/BA) + compare")
    for N in N_SMALL:
        plot_pk_loglog_avg(er_graph, "ER", N, "ER: P(k)", ER_P)
        plot_pk_loglog_avg(ws_graph, "WS", N, "WS: P(k)", WS_K, WS_P_REWIRE)
        plot_pk_loglog_avg(ba_graph, "BA", N, "BA: P(k)", BA_M)
        plot_pk_compare(N)

    log("Metryki vs N (⟨C⟩, średnica, promień)")
    results = {"ER": {}, "WS": {}, "BA": {}}
    for N in N_VALUES:
        results["ER"][N] = run_experiment(er_graph, N, ER_P)
        results["WS"][N] = run_experiment(ws_graph, N, WS_K, WS_P_REWIRE)
        results["BA"][N] = run_experiment(ba_graph, N, BA_M)

    plot_metric_vs_N(results, "avg_clustering", "⟨C⟩", "CC_vs_N.png")
    plot_metric_vs_N(results, "diameter", "średnica", "diameter_vs_N.png")
    plot_metric_vs_N(results, "radius", "promień", "radius_vs_N.png")

    log("WS: zależność ⟨C⟩ od p (rewiring)")
    plot_ws_clustering_vs_p(N=200)

    log("CSV z metrykami (mean + std)")
    for model in ["ER", "WS", "BA"]:
        save_metrics_csv(model, results[model])

    log("KONIEC")
