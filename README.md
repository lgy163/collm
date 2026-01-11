# Co-LLM (Cooperative LLM) for Chiller-Plant Control

## Highlights

- **Retrieval-based warm-start**: reuse verified state–policy pairs to reduce unstable early exploration.
- **Selective Recalibration (SRC)**: only re-optimize sensitive sub-policies under small perturbations.
- **Energy–Comfort evaluation**: unified composite metric to guide iterative improvement.
- **Reproducible comparisons**: scripts and baselines for RBC / model-based / DQN comparisons.

---

## Repository Structure

The current repo contains the following main entry files and modules:

- `collm.py` — main Co-LLM logic / pipeline entry (LLM + RAG + SRC + evaluation).  
- `Simulation_Environment.py` — simulation environment for chiller-plant control evaluation.  
- `compare_experiment/` — comparative experiments with baselines (RBC / MBC / DQN).  
- `GBM_model/` — surrogate / regression models (if used in your pipeline).
- 
## RAG Knowledge Base (LightRAG) Setup (Minimal)

Co-LLM uses a lightweight RAG pipeline to inject chiller-domain documents (manuals, standards, expert logs) into the controller context. We adopt **LightRAG** as the retrieval backend.  
You only need to (1) install LightRAG, and (2) insert your documents once.

### 1) Install LightRAG
```bash
pip install lightrag


## Requirements

- Python >= 3.9 (recommended 3.10+)
- OS: Linux / macOS / Windows (simulation dependencies may vary)
- Lightrag lastest one
