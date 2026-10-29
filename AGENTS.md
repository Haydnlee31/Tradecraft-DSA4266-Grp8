# Tradecraft: active project scope

Read this before changing architecture, dependencies, study boundaries or results. The project now focuses on **centralized versus federated learning and the quality/communication cost of closing their performance gap**. The earlier heavy/light outline is historical context, not the default run protocol.

## Primary study

- Downloaded official CICIoT2023 **39 numeric features**, eight classes. The partial Kaggle 46-feature study remains separate. Never zero-fill columns or reuse its scaler/checkpoints to combine representations.
- 500,000 fitting rows, 399,552 development rows, 398,502 confirmation rows, derived deterministically from the historical 2M-row official training view. Models do not train on all 2M rows. See `docs/data-protocol.md`.
- One 39 -> 64 -> 32 -> 8 LayerNorm MLP; 20 fixed non-IID virtual clients; full participation; fitting-population aggregation weights.
- Main narrative: Central Adam, FedAvg-Adam, Shared Adam, block128 four-bit bidirectional residual-feedback compressed Shared Adam. Supporting methods: FedProx, SCAFFOLD, full MIME, with SGD/MIME-base optimizer controls.
- Frozen primary protocol: `reports/methodology_comparison/plan.json`; 11 configurations, seeds 7/17/27, 21.6M backward examples each.
- Report macro-F1, all class recalls, actual benign false alerts, application bytes, rounds and logical barriers. A common calibration target does not imply identical realized FAR.
- Confirmation was already evaluated. No new independent-test claim, threshold refitting on confirmation, favorable-seed selection or silent budget extension.
- MimeLite is a separate 23.1M-budget, development-only supplement. Never merge its endpoints into the primary confirmation table.

## Scientific boundaries

Federation is simulated on one machine with client-owned rows and explicit aggregation. The main runner uses custom PyTorch updates and some Flower helpers; it is not a physically distributed Flower deployment. Public flow features have no device identity: no device graph or per-device vulnerability claims. No physical-edge, power, privacy or measured wide-area-network claims. Compression reduces bytes, not synchronization frequency. Rare-class detection remains unresolved. Shared Adam is not claimed to be a new optimizer theorem.

Earlier cloud runs used NVIDIA Tesla T4. Primary runs used local macOS ARM CPU. Never relabel hardware based on similar scores. The compressed portable route reports cross-platform differences; it does not establish all-method GPU equivalence. Read `docs/reproduction.md` before training.

## Engineering conventions

- Use `scripts/study.py` for the supported primary workflow. It delegates to frozen implementations rather than replacing numerical behavior.
- Do not modify hash-pinned sources/plans/results or update old hashes to make a refactor pass. Use a versioned adapter and prove endpoint equivalence.
- Older `src/` modules can remain active dependencies. Check imports, source contracts and report references before archiving.
- Preserve explanatory comments and unfavorable results. Use archive manifests and restoration tests. Preserve unrelated working drafts.
- Stream raw CSV preparation; never load all ~46M occurrences into RAM. Preserve provenance, feature-group membership and row ordering.
- Fit scaler/class weights on fitting rows only. Keep float32 collision checks. Label ordering/mapping lives in `src/data/label_map.py`.
- Keep raw inputs/checkpoints/caches out of Git. Use checksum-verified prepared release assets for default reproduction.
- Synthetic tests are not real-data training evidence. Distinguish report reconstruction, fresh training and raw-data reconstruction.

SHAP remains supporting feature-level interpretation if required by the course. No new dashboard, LLM router, classifier search or autonomous security actions are needed. See `studies/historical/README.md` and `archive/README.md` for preserved history and scope documents.
