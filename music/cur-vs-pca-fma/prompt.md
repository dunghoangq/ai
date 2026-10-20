# Project brief


PROJECT: Music Similarity Search — CUR vs PCA

CONTEXT

I am studying StanfordOnline CSX0002 Mining Massive Datasets. I want to understand CUR decomposition, dimensionality reduction, and similarity search from first principles by deriving ideas, implementing methods, testing them, and analysing actual results.

This is a five-day, learning-focused project inside my existing GitHub AI repository. The final deliverable must be both:
1. A controlled mathematical experiment comparing representations.
2. A small, usable music similarity-search prototype.

Do not turn this into a large production system. Prioritise understanding, reproducibility, and a working end-to-end result.

PROBLEM

Build a local application that accepts a track from a music catalogue and returns the most similar tracks according to their numerical audio features.

Investigate whether reducing the number of features using CUR or PCA can reduce representation size or search cost while preserving useful retrieval results.

IMPORTANT LIMITATION

Similarity in a precomputed feature space is not necessarily the same as perceived musical similarity. Genre agreement, if used, is only a weak relevance proxy. Do not claim that the system understands musical similarity unless the data and evaluation support that claim.

DATASET

Use the Free Music Archive (FMA) dataset and its available metadata/precomputed numerical features.

Official repository:
https://github.com/mdeff/fma

Metadata archive:
https://os.unil.cloud.switch.ch/fma/fma_metadata.zip

Inspect the actual files and schemas before writing data-loading code. Do not assume a particular CSV layout, feature set, track count, or column name without verifying it.

Prefer metadata and precomputed features. Do not download the full audio collection unless genuinely necessary.

If the download is too large or unavailable, investigate the repository's documented alternatives and report the limitation. Do not silently replace the dataset with fabricated or unrelated data.

CORE APPLICATION

A user selects a track from the catalogue. The application returns the top-k most similar tracks, excluding the query track itself.

For each result, display the track ID and whatever useful metadata is actually available, such as title, artist, album, or genre.

A simple command-line interface is acceptable for the minimum viable application. A small Streamlit interface is optional only after the end-to-end search works.

METHODS TO COMPARE

1. Full-feature baseline: use all eligible numerical features.
2. CUR-based representation: select a subset of original feature columns using a documented CUR-compatible column-selection method.
3. PCA-based representation: project features into a lower-dimensional space.
4. Optional: truncated SVD or a simple random-feature baseline, only if time permits.

Do not assume CUR will win. The experiment is valuable even if the full-feature baseline or PCA performs better.

MATHEMATICAL CONVENTIONS

Represent the data as X ∈ R^(N×d), where rows are tracks and columns are numerical features.

For a matrix A, CUR approximates A ≈ CUR:
- C contains selected columns of A.
- R contains selected rows of A.
- U links the selected columns and rows.
- A common least-squares choice is U = C⁺AR⁺, where ⁺ denotes the Moore–Penrose pseudoinverse.

For this application, the selected CUR columns define a reduced feature representation for tracks. Explain how those columns are selected and distinguish this retrieval representation from reconstructing the entire matrix using CUR.

For PCA, fit the transformation on the catalogue/training portion only, then apply the fitted transformation to query tracks.

DATA AND EVALUATION RULES

- Inspect the data before deciding which features and tracks to retain.
- Exclude identifiers, titles, artist names, album names, and target labels from numerical distance calculations unless there is a separately justified experiment.
- Handle missing values and scaling consistently. Fit any learned imputation or scaling on the catalogue/training data only.
- Save the exact track IDs used for the catalogue and evaluation queries.
- Exclude each query track from its own candidate results.
- Use the same catalogue, queries, distance metric, and candidate restrictions when comparing methods.
- Choose representation sizes on validation data where possible. Keep a final evaluation set untouched until the choices are fixed.
- Do not describe genre agreement as ground-truth sonic similarity. If genre labels are available, report results as retrieval against a genre-based proxy.
- If no defensible relevance labels are available, report descriptive and qualitative evaluation honestly rather than inventing relevance metrics.

EVALUATION

Measure, as applicable:
- Retrieval quality using Precision@k, Recall@k, or NDCG@k against a clearly defined relevance proxy.
- Representation dimensionality and approximate storage requirements.
- Search latency under a reproducible setup.
- The trade-off between retrieval quality, feature count, and latency.
- A few qualitative examples, including failures and limitations.

Explain the relevance definition, denominator, candidate set, and query protocol. Do not compare numbers produced by different protocols as if they were directly comparable.

CUR CODE REUSE AND PROJECT INDEPENDENCE

Project B must be executable independently of Project A.

If the tested shared implementation exists at ai/common/cur.py:
- Inspect the actual source, API, dependencies, and tests before using it.
- Reuse only functionality that is appropriate for this experiment.
- Do not assume a function name, signature, or selection method.

If the module is unavailable:
- Continue with data inspection, baseline search, and the application.
- When CUR becomes necessary, implement the minimum compatible functionality in Project B or create the shared module.
- Test the implementation and document its exact method.
- Do not block independent work while waiting for Project A.

Keep Project B's dataset-specific code, configuration, saved splits, metrics, and report in its own directory.

TARGET REPOSITORY

ai/music/cur-vs-pca-fma/

A possible structure is:

ai/
├── common/
│   └── cur.py
└── music/
    └── cur-vs-pca-fma/
        ├── README.md
        ├── data/              # downloaded data or instructions; do not commit large data
        ├── notebooks/
        ├── src/
        ├── tests/
        ├── results/
        └── app/

Adapt this structure to the existing repository rather than duplicating files or overwriting unrelated work.

TEACHING STYLE

Teach me from first principles using the Feynman technique. I want to understand, not merely execute code.

For each major concept:
1. State the problem being solved.
2. Explain the assumptions and constraints.
3. Derive the core idea or equation.
4. Work through a tiny numerical example where useful.
5. Implement and test it.
6. Explain how it integrates into the application.
7. Discuss failure cases and trade-offs.

Use British English. Be direct and technically precise. Avoid long, disconnected lectures and avoid repeating concepts I have already demonstrated understanding of.

IMPLEMENTATION RULES

- Inspect the existing repository before modifying it.
- Prefer small, readable, testable functions.
- Use fixed random seeds where randomness is involved.
- Assert important shapes, ID alignments, and invariants.
- Avoid data leakage and duplicate query results.
- Do not overwrite previous work without checking it.
- Do not fabricate outputs, metrics, tests, or file paths.
- Run the relevant code and tests where execution is available. Clearly distinguish executed results from proposed code.

FIVE-DAY PLAN

B1 — Audit the data, define the retrieval protocol, build the full-feature baseline.
B2 — Understand and implement the CUR-based representation.
B3 — Implement PCA and compare reduced representations.
B4 — Evaluate retrieval quality, efficiency, and failure cases.
B5 — Finish the usable application, tests, and portfolio case study.

At the end of every step, provide a HANDOFF MANIFEST containing:
- Step completed and files created/modified.
- Actual dataset files, shapes, track IDs, and saved split paths.
- Actual implementation choices and important parameters.
- Tests and commands actually run, with their outcomes.
- Actual results, including the path to saved results.
- Decisions made and why.
- Known issues and unresolved questions.
- Exact instructions for the next step.

Do not claim a step is complete if required work has not been executed or verified.

# Step B1 — ta audit and full-feature baseline


Use the Project Brief above.

TASK

Guide me through Step B1: inspect the FMA data, define a defensible music retrieval protocol, and build a working full-feature similarity-search baseline. Complete this step from start to finish in one response.

INDEPENDENCE

This step must not depend on Project A, its code, or its handoff manifest. Do not wait for the CUR implementation.

REQUIREMENTS

1. Inspect the existing repository and determine where this project belongs. Preserve unrelated work.
2. Locate or download the appropriate FMA metadata/features files. Inspect the actual archive contents and schemas before coding.
3. Identify the available numerical features, track identifiers, metadata, genre labels (if present), missing values, duplicate IDs, and usable track counts.
4. Select a manageable subset if needed. Base the selection on observed data and explain the trade-off. Do not invent dataset properties.
5. Define the retrieval problem precisely: catalogue, query tracks, candidate tracks, distance metric, and k.
6. Define a defensible evaluation protocol. If using genre agreement as relevance, label it explicitly as a weak genre-based proxy rather than ground-truth sonic similarity. If the available data do not support a credible relevance proxy, explain what can and cannot be evaluated.
7. Create a reproducible catalogue/query split using track IDs. Save the exact IDs and configuration so later steps reuse them without resampling.
8. Build the full-feature baseline:
   - Apply the chosen missing-value handling and feature scaling.
   - Fit any learned preprocessing on the catalogue/training portion only.
   - Compute nearest neighbours for a query track.
   - Exclude the query itself from candidates.
   - Return track IDs and available metadata.
9. Write tests for feature/ID alignment, query exclusion, result count, deterministic behaviour where expected, and valid neighbour ordering.
10. Run the baseline and save actual outputs and timing measurements. Record the environment and timing procedure; do not make unsupported performance claims.
11. Keep large downloaded data out of Git unless the repository already has a deliberate data-management policy.
12. Update the project README with setup and run instructions.

FIRST-PRINCIPLES TEACHING

Explain what a feature vector represents, why scaling affects distance, how nearest-neighbour retrieval works, and why query/candidate separation matters. Use a tiny worked example before the general implementation where helpful.

Do not build CUR or PCA in this step. Do not build an elaborate interface yet. The goal is a trustworthy, executable baseline.

DELIVERABLES

- Data-audit notes.
- Reproducible saved track-ID splits and configuration.
- Full-feature retrieval implementation.
- Tests and baseline results.
- A minimal way to run a query.
- HANDOFF MANIFEST for Step B2.

Use only observed values in the report. If a required file cannot be downloaded or executed, state the blocker precisely and finish all independent work possible.

# Step B2 — Implement and integrate CUR


Use the Project Brief above and the exact data, track-ID splits, preprocessing configuration, and full-feature baseline saved in Step B1.

TASK

Guide me through Step B2: understand CUR's mathematics, implement or integrate the CUR-based feature-selection method, and add it as an alternative representation in the music similarity-search system. Complete this step from start to finish in one response.

REQUIREMENTS

1. Inspect the actual Step B1 files and HANDOFF MANIFEST. Reuse the saved catalogue and query IDs. Do not repeat the data audit or rebuild the baseline unnecessarily.
2. Inspect ai/common/cur.py if it exists. Read its actual implementation and tests before using it. Do not assume its API or mathematical method.
3. If a tested, compatible implementation is available, reuse it. Otherwise, implement the minimum necessary functionality and test it. Continue independent tasks rather than waiting for Project A.
4. Explain CUR from first principles:
   - Why a matrix can sometimes be approximated using selected rows and columns.
   - The dimensions and roles of C, U, and R.
   - The relationship A ≈ CUR.
   - The meaning of U = C⁺AR⁺ and the assumptions/limitations involved.
   - The distinction between reconstructing a matrix and selecting columns to represent unseen query tracks.
5. Choose and document a defensible column-selection method. Inspect whether the available implementation uses uniform sampling, leverage scores, or another method. Do not silently change methods between experiments.
6. Use only the catalogue/training portion to determine selected feature columns. Never use query labels or final evaluation outcomes to select features.
7. Construct the CUR-based retrieval representation by selecting the chosen original feature columns for both catalogue and query tracks. Apply the same feature scaling policy as the baseline, fitting learned preprocessing on catalogue data only.
8. If useful, separately demonstrate matrix reconstruction on a small matrix or the catalogue matrix. Keep reconstruction error separate from retrieval quality: good reconstruction does not automatically imply good nearest-neighbour results.
9. Test:
   - Matrix and representation dimensions.
   - Selected-column indices and their validity.
   - Deterministic behaviour with fixed seeds, where applicable.
   - Catalogue/query feature alignment.
   - Exclusion of the query track from its own results.
   - No leakage from evaluation queries into feature selection.
10. Integrate CUR into the existing retrieval code without duplicating the whole application. Add configuration for the number of selected features.
11. Evaluate the CUR version on validation queries using the same catalogue, candidate rules, distance metric, and retrieval protocol as Step B1.
12. Save the actual selected feature indices, configuration, test outcomes, retrieval results, and timing measurements.

IMPORTANT

Do not assume CUR will improve retrieval quality or speed. If its feature-selection method is weak or the selected features perform poorly, report that honestly. Do not tune on the final evaluation set.

Do not re-download the dataset, resample the splits, or repeat completed work unless a verified defect requires it.

DELIVERABLES

- Tested CUR-based representation.
- Explanation of the actual selection method and implementation.
- Baseline-versus-CUR validation comparison.
- Saved configuration and selected feature indices.
- Updated tests and run instructions.
- HANDOFF MANIFEST for Step B3.


# Step B3 — Implement PCA and compare representations


Use the Project Brief above and the exact saved data, track-ID splits, preprocessing configuration, and baseline from Steps B1–B2.

TASK

Guide me through Step B3: implement PCA as a competing low-dimensional representation, compare it fairly with the full-feature baseline and CUR, and integrate the alternatives into the existing retrieval system. Complete this step from start to finish in one response.

REQUIREMENTS

1. Inspect the actual code, results, and HANDOFF MANIFESTS from Steps B1–B2. Reuse them. Do not repeat completed data loading or baseline experiments unnecessarily.
2. Explain PCA from first principles:
   - What variance means in this feature matrix.
   - How centring and scaling affect PCA.
   - What principal components represent.
   - Why high-variance directions do not necessarily preserve the information relevant to retrieval.
   - How PCA differs from selecting actual columns using CUR.
3. Implement PCA using a suitable numerical library unless a from-scratch implementation is necessary for understanding. If using a library, explain the underlying calculation and validate the output with small tests.
4. Fit preprocessing and PCA on catalogue/training data only. Apply the fitted transformation to both catalogue and query tracks. Do not fit PCA on the combined catalogue and query data.
5. Use the same saved catalogue/query IDs, candidate restrictions, distance metric, and evaluation protocol as previous steps.
6. Compare:
   - Full-feature baseline.
   - CUR-selected original features.
   - PCA representations.
7. Evaluate several reasonable representation sizes. Choose a manageable range based on the actual feature count; do not assume dimensions that exceed the data's limits.
8. Choose the final representation size using validation results and practical constraints. Do not use final evaluation results to select it.
9. Record retrieval quality, representation dimensions, approximate storage requirements, and measured search latency. Explain the limits of the timing method.
10. Add tests for transformation dimensions, fitted component count, reproducibility where expected, query/catalogue alignment, and prevention of fitting on evaluation queries.
11. If time permits, consider truncated SVD as an additional comparison. Treat it as optional and do not let it prevent completion of the main experiment.
12. Save all configurations and actual results. Update the README with commands for selecting the representation used by the retrieval system.

FIRST-PRINCIPLES ANALYSIS

Use a small numerical example to make PCA's projection concrete. Then connect the mathematics to the actual FMA feature matrix and explain any surprising results.

Do not claim that a lower-dimensional representation is better simply because it is smaller. Discuss the quality-efficiency trade-off.

DELIVERABLES

- Tested PCA representation.
- Unified retrieval interface supporting full features, CUR, and PCA.
- Fair validation comparison using saved splits.
- Results table with actual values and a clear metric definition.
- Updated tests and instructions.
- HANDOFF MANIFEST for Step B4.


# Step B4 — Evaluate the trade-offs and investigate failures


Use the Project Brief above and the actual code, saved track-ID splits, configurations, and results from Steps B1–B3.

TASK

Guide me through Step B4: conduct a controlled evaluation of full-feature, CUR, and PCA-based music similarity search; analyse failure cases; and decide which representation, if any, is appropriate for the final prototype. Complete this step from start to finish in one response.

REQUIREMENTS

1. Inspect the actual outputs and HANDOFF MANIFESTS from previous steps. Reuse existing work and identify missing outputs before attempting to regenerate anything.
2. Verify that all methods use the same query tracks, catalogue, candidate restrictions, distance metric, and relevance definition.
3. Verify that every query track is excluded from its own candidate results.
4. Confirm that learned preprocessing and feature selection were fitted only on catalogue/training data. Check that final evaluation results have not been used to select representation sizes or parameters.
5. If Step B1 defined a defensible validation/final evaluation split, freeze the chosen configurations using validation results and evaluate them on the untouched final queries now.
6. If no untouched final evaluation set exists, do not pretend that validation results are held-out test results. State the limitation and either use a remaining valid split or report the results as exploratory.
7. Compare full features, CUR, and PCA using the metrics supported by the actual dataset and protocol:
   - Precision@k, Recall@k, or NDCG@k if a relevance definition exists.
   - The exact relevance definition and candidate set.
   - Representation dimensionality and approximate storage.
   - Measured search latency using a documented timing method.
   - Any other justified metric, with its definition.
8. If genre labels are used to define relevance, explicitly call this genre-proxy evaluation. Discuss why same-genre tracks may sound different and different-genre tracks may sound similar.
9. Analyse actual query examples:
   - Queries where methods return similar neighbours.
   - Queries where CUR and PCA disagree with the full-feature baseline.
   - Cases where the result list seems unhelpful, based on available evidence.
   - Missing metadata or data-quality problems that affect interpretation.
10. Save query examples and actual result lists so the analysis is reproducible.
11. Check whether apparent speed improvements survive repeated timing runs. Distinguish representation storage from total application memory and search latency from preprocessing time.
12. Make a reasoned recommendation based on the measured trade-offs. It is acceptable to conclude that full features are best, that PCA is preferable, or that CUR has a useful niche.
13. Do not invent numbers, infer unsupported causality, or describe a proxy metric as a human judgement of musical similarity.

FIRST-PRINCIPLES QUESTIONS

Explain:
- Why reconstruction error and retrieval quality can disagree.
- Why variance retention does not guarantee preservation of nearest neighbours.
- Why selecting actual columns may improve interpretability while sacrificing predictive or retrieval performance.
- How the number of retained dimensions changes distance calculations.
- What additional evidence would be needed before claiming that people perceive the returned tracks as similar.

DELIVERABLES

- Final controlled evaluation, with actual values and metric definitions.
- Saved query examples and failure analysis.
- A concise trade-off table.
- A recommendation for the final prototype, justified by evidence.
- Updated tests and results documentation.
- HANDOFF MANIFEST for Step B5.


# Step B5 — Finish the application and write the portfolio case study


Use the Project Brief above and the actual code, evaluation results, configurations, tests, and HANDOFF MANIFESTS from Steps B1–B4.

TASK

Guide me through Step B5: finish the music similarity-search prototype, verify that another person can run it, and document the project as a concise, technically honest portfolio case study. Complete this step from start to finish in one response.

REQUIREMENTS

1. Inspect the existing repository and previous handoff manifests. Reuse completed work; do not rebuild the pipeline from scratch or repeat experiments without a verified reason.
2. Finish a minimal end-to-end application:
   - Let a user choose a track by ID or searchable metadata.
   - Return the top-k nearest tracks, excluding the query itself.
   - Display available track metadata and the selected representation method.
   - Allow switching between full features, CUR, and PCA if those methods have been implemented successfully.
   - Clearly communicate missing metadata or unsupported operations.
3. Prefer a reliable command-line interface if that is already sufficient. Build a small Streamlit interface only if the core search is working and the extra interface can be completed within the remaining scope.
4. Add or complete tests for:
   - Valid and invalid query IDs.
   - Query self-exclusion.
   - Correct number and ordering of returned results.
   - Consistent feature transformations.
   - Stable results under fixed configurations where expected.
   - Missing data and empty or invalid inputs.
5. Provide reproducible setup and run instructions. Record dependencies, expected input files, configuration, and commands. Do not include large downloaded datasets in Git unless the repository already follows an appropriate data policy.
6. Run the application and relevant tests if execution is available. Report actual outcomes. If something cannot be executed, identify the exact limitation rather than claiming it works.
7. Verify that the README describes the implementation that actually exists, not an earlier plan. Include the dataset source and relevant attribution/licensing considerations; do not make unsupported claims about dataset permissions.
8. Write a concise portfolio case study containing:
   - Problem and intended user.
   - Dataset and observed data limitations.
   - Why similarity search was chosen.
   - Methods and mathematical intuition.
   - Experimental protocol and leakage controls.
   - Actual results in a compact table.
   - A few concrete query examples.
   - What worked, what failed, and what remains uncertain.
   - Engineering choices, tests, and reproducibility.
   - Practical value and limitations.
   - What I would investigate next.
9. Keep the claims technically honest. Distinguish genre-proxy retrieval quality from genuine perceptual similarity. Do not claim production readiness or user benefit without evidence.
10. Do not invent metrics or copy values from planned examples. Load saved results and use the actual values. If a necessary output is missing, identify the file or command needed to produce it, and do not fill the gap with a guess.
11. If actual findings contradict the original hypothesis, make that contradiction central to the analysis instead of hiding it.

FINAL DELIVERABLES

- A working, tested local music similarity-search prototype.
- Reproducible setup and usage instructions.
- Final source code, configuration, and relevant tests.
- Saved results and query examples.
- A concise technical portfolio case study suitable for the repository README.
- A final project summary listing completed work, verified test outcomes, actual findings, known limitations, and possible next experiments.

