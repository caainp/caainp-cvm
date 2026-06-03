# COEX view pipeline artifacts

These files are the COEX 3F view-localization artifacts used by the CVM/value-map flow.

## Layout

- `source/`: manually maintained source files.
- `graph/`: derived graph/value-map CSV used at runtime.
- `localization/`: derived view index and embedding files used at runtime.
- `reports/`: local validation and smoke-test outputs. Ignored by git.

## Sources

- `source/coex_nodemap_m11.xlsx`: graph/node/edge source of truth. Read only.
- `view/3F`: COEX view image root.

## Regenerate views.csv

```powershell
python -m scripts.build_coex_views --root_dir .
```

Outputs:

- `data/coex/localization/views.csv`
- `data/coex/reports/view_validation_report.csv`
- `data/coex/reports/view_validation_report.md`

## Regenerate view embeddings

```powershell
python -m scripts.build_view_embeddings --root_dir . --device auto
```

Outputs:

- `data/coex/localization/view_embeddings.npy`
- `data/coex/localization/view_embedding_index.csv`
- `data/coex/reports/view_embedding_report.md`

## Regenerate graph CSV

```powershell
python -m scripts.build_coex_graph_csv --root_dir .
```

Outputs:

- `data/coex/graph/coex_nodemap.csv`
- `data/coex/reports/coex_nodemap_report.csv`
- `data/coex/reports/coex_nodemap_report.md`

## COEX CVM step wrapper

```powershell
python -m scripts.run_coex_cvm_step --root_dir . --image view\3F\3132\3F_3132_context_01.jpg --target_node 3142
```

By default, only `use_for_localization=true` views are searched. Use
`--include_all_views` to include destination or entrance confirmation views.

The sources stay separated:

- `graph/coex_nodemap.csv` supplies graph and edge structure.
- `localization/views.csv` and `localization/view_embeddings.npy` supply localization candidates.
- The adapter passes aggregated `node_id` candidates to the value map.

The wrapper returns node-based output for AR/CSM integration:

- `current_node`
- `next_node`
- `route_nodes`
- `target_nodes`
- `move_instruction`
- `route_summary`

It is intentionally thin and COEX-specific. The existing school-4F
`run_cvm_step.py` remains unchanged.
