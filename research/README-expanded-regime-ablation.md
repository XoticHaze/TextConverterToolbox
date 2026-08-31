# Expanded regime ablation (research-only)

This disposable public-compute harness compares the existing 20-feature Foundry MNQ benchmark against broader causally safe indicator, regime, market-ID, and cross-market context feature sets.

It is research-only and carries no MM-IBKR runtime, StrategySpec, corpus-promotion, model-promotion, or trading authority.

Causal admission deliberately excludes known forward-aligned transforms such as `chikou_span`. Candidate selection uses only the first three chronological OOS folds; fold four is held out from selection. Thresholded move targets are ternary (down / neutral / up), so no future-known materiality filter is used to select inference rows.

The source market data is read directly from the original pinned public upstream repository and is not copied into this repository. Only result metrics are uploaded.
