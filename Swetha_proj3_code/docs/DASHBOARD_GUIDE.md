# Dashboard Guide

## Launch

```bash
cd proj3_code
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`

## Pages

| Page | Description |
|------|-------------|
| Overview | Project summary, architecture KPIs, scenario distribution |
| Data Exploration | Interactive histograms, heatmaps, CDFs, scenario box plots |
| Agent Evaluation | Per-agent train/val/test metrics, confusion matrix, ROC, learning curves |
| Train/Val/Test | All agents performance comparison |
| Digital Twin | Live KPI time series, cell map, gNB heatmap |
| Chatbot | RAG Q&A over project knowledge base |

## Prerequisites

Run `python run_pipeline.py` first to generate models, metrics, and plots.

## Customization

- Theme colors: Nokia Blue `#124191`, Teal `#00C9FF` in `dashboard/app.py`
- Metrics path: `outputs/metrics/all_metrics.json`
- Plots path: `outputs/plots/`
