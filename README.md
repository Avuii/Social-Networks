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

## 🖼️ Quick preview

### Example networks (N=100)  
<table> <tr> <td align="center"><b>ER</b></td> <td align="center"><b>WS</b></td> <td align="center"><b>BA</b></td> </tr> <tr> <td><img src="plots/ER/ER_graph_N100.png" width="280"/></td> <td><img src="plots/WS/WS_graph_N100.png" width="280"/></td> <td><img src="plots/BA/BA_graph_N100.png" width="280"/></td> </tr> </table>

### Degree histograms (N=1000)
<table> <tr> <td align="center"><b>ER</b></td> <td align="center"><b>WS</b></td> <td align="center"><b>BA</b></td> </tr> <tr> <td><img src="plots/ER/ER_histogram_N1000.png" width="280"/></td> <td><img src="plots/WS/WS_histogram_N1000.png" width="280"/></td> <td><img src="plots/BA/BA_histogram_N1000.png" width="280"/></td> </tr> </table>

### Degree distribution comparison P(k) (log–log)
<table> <tr> <td align="center"><b>N=10</b></td> <td align="center"><b>N=50</b></td> <td align="center"><b>N=100</b></td> <td align="center"><b>N=200</b></td> </tr> <tr> <td><img src="plots/compare/Pk_compare_N10.png" width="230"/></td> <td><img src="plots/compare/Pk_compare_N50.png" width="230"/></td> <td><img src="plots/compare/Pk_compare_N100.png" width="230"/></td> <td><img src="plots/compare/Pk_compare_N200.png" width="230"/></td> </tr> </table>

### Clustering / Radius / Diameter vs N
<table> <tr> <td align="center"><b>Clustering vs N</b></td> <td align="center"><b>Radius vs N</b></td> <td align="center"><b>Diameter vs N</b></td> </tr> <tr> <td><img src="plots/compare/CC_vs_N.png" width="280"/></td> <td><img src="plots/compare/radius_vs_N.png" width="280"/></td> <td><img src="plots/compare/diameter_vs_N.png" width="280"/></td> </tr> </table>

---

### 🧑‍💻 Author

Created by Avuii
