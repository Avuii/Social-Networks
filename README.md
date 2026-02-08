# 🌐 Social Networks — ER / WS / BA (Generation & Analysis)

**Random network models in Python**: **Erdős–Rényi (ER)**, **Watts–Strogatz (WS)** and **Barabási–Albert (BA)**.  
The project generates networks, computes core graph metrics, and produces **plots + CSV summaries** for comparison.

📄 **Report (PDF):** [Report.pdf](Report.pdf)  
📊 **Outputs:** saved to `plots/` (figures) and `data/` (CSV tables)

---

## 🎯 Goal
This project focuses on:
- generating three classic random network models (**ER / WS / BA**),
- comparing their structure using:
  - **degree distribution** \(P(k)\),
  - **clustering coefficient**,
  - **radius & diameter** (computed on the **Largest Connected Component**, LCC),
- analyzing how metrics change with the number of nodes **N** and selected model parameters.

---

## 🧠 Models
### 🔹 Erdős–Rényi (ER)
Random graph \(G(N,p)\): each pair of nodes is connected independently with probability **p**.

### 🔹 Watts–Strogatz (WS)
Small-world model: start from a ring lattice (each node connected to **k** nearest neighbors) and **rewire** edges with probability **β**.  
✅ In this project, **WS is implemented manually**.

### 🔹 Barabási–Albert (BA)
Growing network with **preferential attachment**: new nodes connect to existing ones with probability proportional to node degree (hubs emerge).

---

## ⚙️ Experiment setup
- Language: **Python**
- Libraries: **NetworkX**, **NumPy**, **Matplotlib**
- Multiple independent runs per configuration to report **mean ± std**
- Radius/diameter computed on **LCC** to handle disconnected graphs

---

## 📂 Repository structure
```text
.
├─ main.py
├─ Report.pdf
├─ plots/
│  ├─ ER/
│  ├─ WS/
│  ├─ BA/
│  └─ compare/
└─ data/
   ├─ ER_metrics.csv
   ├─ WS_metrics.csv
   └─ BA_metrics.csv
```

---

## ▶️ Running the project
Install dependencies:  
```
pip install numpy networkx matplotlib
```
Run:  
```
python main.py
```
Outputs will be generated automatically in:  
* plots/ — figures
* data/ — CSV metric tables

---

## ✨ Outputs
The project produces:    
* example network visualizations,    
* degree histograms,
* P(k) plots (log–log),
* comparison plots across models,
* CSV tables with aggregated metrics (mean ± std).

---

### 🧑‍💻 Author

Created by Avuii
