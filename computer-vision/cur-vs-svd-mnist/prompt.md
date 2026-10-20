# Project brief

PROJECT: CUR vs SVD for Compact Image Representations

CONTEXT
I am studying CUR decomposition from first principles. This is a learning-focused, five-day project in my existing GitHub AI repository. I want to understand the mathematics by implementing, testing, visualising, and comparing methods rather than merely calling libraries.

PROBLEM
Given handwritten digit images, reduce their representation while preserving:
1. Image information, measured by reconstruction error.
2. Digit classification performance, measured on unseen data.
3. Computational efficiency, measured by runtime and representation size.

INDUSTRY MOTIVATION
Compact representations may reduce memory traffic and downstream computation in constrained computer-vision systems. MNIST is a controlled educational benchmark, not proof of industrial defect-detection performance.

DATA
MNIST: 28×28 grayscale images, 10 classes.
Official information: http://yann.lecun.com/exdb/mnist/
Convenient loader: torchvision.datasets.MNIST or Keras MNIST.
Do not scrape data or invent dataset contents.

TIME LIMIT
Maximum five working days across all steps. Each step must be achievable in one focused session. Avoid unnecessary abstractions, excessive hyperparameter searches, and unrelated features.

METHODS
- Full-pixel logistic regression baseline.
- CUR decomposition and CUR-selected pixel features.
- Truncated SVD for reconstruction and dimensionality reduction.
- Random projection.
- SelectKBest as a supervised feature-selection baseline.

MATHEMATICAL CONVENTION
Represent images as rows and pixels as columns:
X ∈ R^(N×784).

For CUR:
C consists of selected columns of X;
R consists of selected rows of X;
U = C⁺ X R⁺;
X_hat = C U R.

Here + denotes the Moore–Penrose pseudoinverse.
Use training data only to choose CUR rows/columns and fit learned transformations. Never use test data to choose features, fit preprocessing, or tune hyperparameters.

EXPERIMENTAL RULES
- Use reproducible random seeds.
- Keep the same training, validation and test examples across methods.
- Start with a manageable stratified training subset; use the official test set for final evaluation.
- Fit preprocessing on training data only.
- Compare classification and reconstruction separately.
- Do not fabricate results or expected numerical metrics.
- Distinguish code that was executed from code that was merely written.

REPOSITORY
My existing GitHub repository is named ai. Inspect its actual structure if available. Prefer:
computer-vision/cur-vs-svd-mnist/
Use src/, notebooks/, tests/, outputs/ or equivalent only when useful.
Do not commit large raw datasets or unnecessary generated files.

TEACHING STYLE
Explain from first principles:
problem → assumptions → derivation → small numerical example → implementation → tests → interpretation.
Define symbols and matrix dimensions. Explain why each operation is needed.
I want to understand and eventually implement the algorithm independently.

DELIVERY
Complete the requested step in one response without asking follow-up questions.
Provide runnable code, file paths, commands, assertions, expected output structure, debugging advice, and a concise explanation of common mistakes.
Do not silently repeat completed work.

At the end, provide a HANDOFF MANIFEST for the next step, listing:
1. Completed work.
2. Files created or modified.
3. Dataset shapes and exact split definitions.
4. Random seeds and algorithm parameters.
5. Actual measured results, if executed.
6. Outstanding issues.
7. Exact inputs required by the next step.

# Step A1 — Load data and establish the baseline


TASK: Complete Step A1 from start to finish.

INPUT
Project Brief A above. No prior project steps have been completed.

OBJECTIVES
1. Inspect the available repository structure, if accessible.
2. Download/load MNIST through a documented loader.
3. Inspect image shapes, pixel ranges, class frequencies and example images.
4. Flatten each image to 784 features and scale pixel values consistently.
5. Define reproducible stratified training, validation and test sets.
6. For a manageable five-day project, target approximately 10,000 training examples, 2,000 validation examples and the official 10,000 test examples. Explain the exact sampling procedure and ensure the splits are disjoint.
7. Train a logistic regression baseline using all 784 pixels.
8. Measure validation and test accuracy, macro-F1, training time, inference time and model size where practical. Do not tune on the test set.
9. Save split indices and preprocessing metadata so subsequent chats reuse exactly the same data partition.

OUTPUTS
- A notebook named 01_data_and_baseline.ipynb, or equivalent Python scripts.
- A reproducible split manifest in outputs/.
- Baseline metrics saved to outputs/baseline_metrics.json or CSV.
- A data summary and class-distribution plot.
- A HANDOFF MANIFEST for Step A2.

IMPORTANT
Do not implement CUR yet. Do not fabricate measured results. If execution is unavailable, provide runnable code and clearly label results as not yet measured.

# Step A2 — Implement CUR from first principles

TASK: Complete Step A2. Reuse Step A1's exact saved data and split indices. Do not reload and independently resplit the data.

INPUT
- Project Brief A.
- Step A1 HANDOFF MANIFEST.
- Existing source files and saved split artifacts, if available.

OBJECTIVES
1. Explain CUR geometrically and derive the dimensions of C, U and R.
2. Implement a simple CUR decomposition in NumPy:
   - Select c columns and r rows from the training matrix.
   - Construct C and R from the selected original data.
   - Compute U = pinv(C) @ X_train @ pinv(R).
   - Reconstruct X_hat = C @ U @ R.
3. Start with uniform random sampling. Add norm-based sampling if feasible.
4. Explain how sampling with replacement differs from sampling without replacement, and handle duplicate indices safely.
5. Explain the numerical limitations of pseudoinverses and why poor selections can produce unstable approximations.
6. Implement relative Frobenius reconstruction error and test it on a tiny hand-checkable matrix.
7. Run a small experiment with a few c and r settings on training data only. Never use validation or test data to select rows or columns.
8. Separately evaluate a CUR-selected-pixel classifier: select pixel columns using training data only, train logistic regression on those pixels, and evaluate on the unchanged validation set.
9. Explain why the CUR reconstruction and selected-feature classification experiments answer different questions.
10. Add assertions for dimensions, finite values, valid indices and reconstruction shape.

OUTPUTS
- A tested CUR implementation, preferably in a reusable source file such as common/cur.py if the repository layout supports it.
- A notebook/script named 02_cur_from_scratch.
- Saved selected row/column indices and experiment parameters.
- CUR reconstruction and validation-classification metrics.
- A HANDOFF MANIFEST for Step A3.

Do not implement SVD, random projection or SelectKBest in this step. Do not claim code was executed unless it was.

HANDOFF MANIFEST — Step A1 to Step A2
1. Completed work
Implementation specified for MNIST loading, data inspection, deterministic stratified sampling, pixel scaling, full-pixel logistic regression, evaluation and metadata export. Execution has not been verified in this chat.

2. Files created or modified
Proposed files — create them locally using the code above.
requirements.txt, src/step_a1.py, tests/test_step_a1.py, plus six expected files under outputs/.

3. Dataset shapes and split definitions
Original train: (60000, 28, 28). Original test: (10000, 28, 28). Flattened training: (10000, 784). Validation: (2000, 784). Test: (10000, 784).
Training and validation are disjoint subsets of the original training partition. The test partition is the entire official MNIST test set. Split indices are zero-based indices into the original training set.

4. Seeds and parameters
Seed: 42. Split method: two-stage stratified train_test_split. Pixel transform: float32(pixel) / 255.0. Classifier: scikit-learn LogisticRegression(solver="lbfgs", max_iter=300, random_state=42).

5. Actual measured results
Not yet measured
No accuracy, macro-F1, runtime or model-size results are claimed. Populate outputs/baseline_metrics.json by executing the script.

6. Outstanding issues
Run the script and tests; verify the MNIST download; inspect convergence warnings; review all generated files; record hardware details if runtime comparisons will be important.

7. Exact inputs required by Step A2
1. outputs/split_manifest.json — authoritative partition indices.
2. outputs/preprocessing_metadata.json — pixel scaling and feature ordering.
3. outputs/data_summary.json — original data dimensions and class frequencies.
4. outputs/baseline_metrics.json — measured baseline metrics, including software versions and model parameters.

# Step A3 — Implement alternative methods

TASK: Complete Step A3 without redoing Steps A1 or A2.

INPUT
- Project Brief A.
- Step A2 HANDOFF MANIFEST.
- The saved splits and tested CUR implementation.

OBJECTIVES
Implement and explain these alternatives:
1. Truncated SVD: explain the best rank-k approximation property under Frobenius norm.
2. Random projection: use a fixed random seed and a documented projection matrix.
3. SelectKBest: use training labels to select pixels; fit selection on training data only.
4. Keep the full-pixel logistic regression baseline from Step A1.

Use a consistent dimensionality budget where applicable, such as 16, 32, 64 and 128 features/components. Fit each method on training data and evaluate on validation data.

For CUR, distinguish selected original pixel coordinates from a CUR low-rank reconstruction. For SVD, distinguish the reconstructed image from the lower-dimensional classifier input. Do not pretend these are identical representations.

Use the same classifier and comparable preprocessing wherever appropriate. Explain any unavoidable differences.

OUTPUTS
- A notebook/script named 03_baseline_methods.
- Tested implementations or documented library calls for SVD, random projection and SelectKBest.
- Validation metrics in one tidy CSV.
- A plot of validation accuracy against representation size.
- A short explanation of which methods preserve original feature meanings and which create transformed coordinates.
- A HANDOFF MANIFEST for Step A4.

Do not run a large hyperparameter search. Do not use test results to select a method or dimensionality.

HANDOFF MANIFEST — Step A2 to Step A3
1. Implementation specified
CUR decomposition, uniform and norm-based sampling, relative Frobenius error, and separate reconstruction/classification experiments have been specified.

2. Files to create
- src/cur.py
- tests/test_cur.py
- 02_cur_from_scratch.py
The experiment should save outputs/cur_reconstruction_metrics.json and outputs/cur_validation_metrics.json.

3. Dataset and split definitions
Intended shapes: training (10000, 784), validation (2000, 784), test (10000, 784). Training and validation are subsets of the official training partition; test is the official test partition. The exact indices must be read from A1 artefacts.

4. Seeds and parameters
Seed 42; pixel scaling float32(pixel) / 255.0 from A1; uniform and squared-norm sampling without replacement; initial grid (c, r) of (25, 100), (50, 200), (100, 400). Logistic regression uses lbfgs, max_iter=300, random_state=42.

5. Actual measured results
None verified in this chat. Do not report the proposed settings as measured results. Populate the JSON outputs by running the implementation locally.

6. Outstanding issues
- Confirm A1 filenames and metadata schema.
- Ensure the script consumes the exact saved splits.
- Run tests and experiments.
- Check runtime and pseudoinverse conditioning.
- Confirm the official test set remains untouched.

7. Exact inputs required by Step A3
- Step A1's split_manifest.json and preprocessing metadata.
- A1 baseline metrics and data summary.
- The tested src/cur.py implementation.
- The selected row and column indices and parameters.
- Actual reconstruction and validation-classification JSON outputs.
- Test logs and any convergence or numerical-stability warnings.

# Step A4 — Controlled experiments and error analysis


TASK: Complete Step A4 using the existing code, data splits and saved outputs.

OBJECTIVES
1. Validate that all methods use the same examples and evaluation protocol.
2. Compare CUR, truncated SVD, random projection, SelectKBest and the full-pixel baseline.
3. Plot:
   - Validation accuracy versus feature/component budget.
   - Reconstruction error versus selected rank or CUR dimensions, where applicable.
   - Runtime or representation size versus budget where measurements are reliable.
4. Evaluate the chosen configurations once on the official test set, using selection decisions made from validation results only.
5. Inspect misclassified digits and CUR-selected pixel locations.
6. Explain variance, redundancy, sampling instability and why the lowest reconstruction error need not give the highest classification accuracy.
7. Record uncertainty across a small number of CUR random seeds if time permits.
8. Identify the best method for each objective rather than declaring one universal winner.

OUTPUTS
- A notebook/script named 04_evaluation_and_analysis.
- A final comparison CSV with actual metrics.
- Plots saved in outputs/figures/.
- A short results interpretation with limitations.
- A HANDOFF MANIFEST for Step A5.

Do not fabricate results, change the test set, or rerun the whole project unnecessarily. If a measurement is too expensive, explain and omit it.

# Step A5 — Final report and reusable implementation


TASK: Finish Project A as a compact, reproducible learning and portfolio project.

INPUT
- Project Brief A.
- Step A4 HANDOFF MANIFEST.
- Existing source code, plots, metrics and tests.

OBJECTIVES
1. Audit the project for reproducibility, data leakage, inconsistent splits and misleading comparisons.
2. Ensure the CUR implementation has meaningful tests and clear documentation.
3. Finalise the source structure without unnecessary refactoring.
4. Write README.md covering:
   - Industry-motivated problem and the limitations of MNIST as a proxy.
   - Dataset and setup instructions.
   - CUR derivation and implementation choices.
   - Comparison methods and experimental protocol.
   - Actual results, plots and interpretation.
   - Limitations and next experiments.
5. Explain when CUR is useful, when SVD is preferable, and why task-specific feature selection may outperform both.
6. Prepare a reusable CUR module for Project B. Prefer a shared module in the repository if practical; otherwise document the exact tested source file to reuse.
7. Give a concise list of the mathematical lessons learned.

OUTPUTS
- Final README.md.
- Clean source files, tests and saved comparison results.
- A clear run command.
- A HANDOFF MANIFEST for Project B, Step B1, specifying the exact CUR module path, its public API, dependencies, tests and known limitations.

Do not invent performance numbers or claim industrial validation. Finish within the five-day project scope.
