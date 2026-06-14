# Project-2 legacy ROM alignment audit

This note records the remaining alignment-sensitive details between the professional repository workflow and `THERMOMECHANICAL_FOM_ROM.py`.

The notebook methods use deterministic seeds, an 80/10/10 train-validation-test design, internal `StandardScaler` objects fitted only on training data, and PyTorch CPU/thread limiting.  The repository mirrors the method-level hyperparameters for PODI, POD-GPR, POD-NN, and POD-AE.

Important implementation details:

- POD-NN field-aware fine tuning must reconstruct centered POD artifacts as `C @ basis.T + mean`.  Without adding the POD mean, the field loss optimizes the wrong physical target and can destroy a good coefficient model.
- POD-AE includes the notebook-style end-to-end online-path fine tuning stage controlled by `end_to_end_finetune`, `e2e_epochs`, `e2e_lr`, `e2e_field_weight`, and `e2e_coeff_weight`.
- A fully exact intrusive POD-Galerkin match still requires the legacy FEniCS residual/Jacobian wrapping and remains intentionally disabled by default in the pure-data suite runner.
- A numerically exact POD match to the notebook requires the same POD construction convention: Solution/W or projected-CG space, no mean-centering in the RBNiCS basis, and the same inner product.  The NumPy preprocessing remains a portable approximation unless a FEniCS/RBNiCS-backed POD builder is added.
