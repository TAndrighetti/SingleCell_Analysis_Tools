# sctools conventions

## Function names
- Every function uses PascalCase, like `LoadSample`, `PlotQcViolins`, `RunHarmonyIntegration`. Never snake_case like `load_sample`.
- Private helpers keep the leading underscore, then PascalCase: `_RequireExists`, `_MadBounds`.
- Acronyms follow the existing style: `Qc`, `Umap`, `Pca`, `Hvg`, `Knn` (not `QC`, `UMAP`).
- Function arguments and variables stay snake_case.
- Local closures defined inside a function may stay snake_case.
- When creating or renaming a function, check for name collisions in the package first.
