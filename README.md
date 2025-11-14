Here’s a tightened, more “paper-supplement” style README.md:

# Matrix-Weighted Graph Operators

Code supplement for the manuscript  
**“Static Loewner Order, Dynamic Response, and Optimization Rates.”**

This repository contains a single, self-contained Python script that reproduces the figures in the manuscript based on simple graph Laplacians and shifted operators \(A = L + mI\).

---

## Contents

- `Repocode.py`  
  - Constructors for Laplacians of:
    - Path graphs
    - Cycle graphs
    - 2D grid graphs  
  - Routines to generate three figures:
    1. \(\lambda_{\min}(A)\) vs. graph size \(n\) for several graph families and shifts \(m\)
    2. Gradient descent convergence on a grid graph
    3. Mode-wise dynamic response on a cycle graph

- `MatrixWeightedGraphOperators_v14_polished.docx`  
  Polished version of the manuscript.

---

## Requirements

- Python ≥ 3.8  
- NumPy  
- Matplotlib  

Install dependencies, e.g.:

```bash
pip install numpy matplotlib


⸻

Usage

Clone and run:

git clone https://github.com/mjs1310/Matrix-Weighted-Graph-Operators.git
cd Matrix-Weighted-Graph-Operators
python Repocode.py

This will generate:
	•	fig1_lambda_vs_n.png
	•	fig2_gd_grid50.png
	•	fig3_modes_cycle.png

To call individual routines from Python:

import Repocode as mwgo

mwgo.figure1_lambda_vs_n("fig1_lambda_vs_n.png")
mwgo.figure2_gradient_descent("fig2_gd_grid50.png")
mwgo.figure3_modewise_cycle("fig3_modes_cycle.png")


⸻

Reproducibility
	•	All computations use standard NumPy linear algebra.
	•	Random quantities (where present) are seeded via np.random.seed(0) for deterministic output.
	•	Operators are of the form (A = L + mI) with scalar node weights ((d = 1)).

For full definitions, proofs, and context, see
MatrixWeightedGraphOperators_v14_polished.docx.

⸻

Citation

If you use this code or figures, please cite the accompanying manuscript:

Static Loewner Order, Dynamic Response, and Optimization Rates
(see the included manuscript for full bibliographic details).

⸻

License

No explicit license is currently provided.
Until one is added, please treat this code as all rights reserved and contact the author for uses beyond private or academic study.
