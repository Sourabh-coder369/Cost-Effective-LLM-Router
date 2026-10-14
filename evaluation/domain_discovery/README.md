# Semantic domain discovery

`cluster_domains.py` runs KMeans and seed-similarity analyses over cached prompt
embeddings. It reads the local training split and embedding cache under `data/`
and saves generated tables under `artifacts/domain_discovery/clusters/`.

Run it from the project root:

```powershell
.\.venv\Scripts\python.exe evaluation\domain_discovery\cluster_domains.py
```
