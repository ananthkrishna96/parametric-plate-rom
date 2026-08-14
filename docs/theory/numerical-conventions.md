# Numerical conventions

## Plate kinematics and signs

The plate midsurface is `Omega = (0, 2) x (0, 1)` m in the thesis campaigns unless a configuration states otherwise. Transverse displacement and load are upward-positive. Downward pressure is therefore negative. The regularized unilateral foundation uses the compression stiffness for `w < 0` and the uplift-side factor for `w >= 0`.

## Interior facets and physical seams

Element facets strictly inside one physical panel carry the symmetric `C0-IPG` consistency, symmetry, and stabilization terms. A seam between independent panel fields is a physical interface and carries separate displacement and normal-rotation penalties. The implementation does not use the seam penalties as numerical interior-facet stabilization.

## Thermomechanical output

The thermal driver is

```text
theta(x, y) = 12 / h^3 * integral_{-h/2}^{h/2} z * Delta T(x, y, z) dz.
```

Its unit is K/m. It is not the three-dimensional temperature and is never concatenated with displacement as one ROM target. The FOM remains coupled through displacement-dependent lower-face heat exchange and thermal bending moments.

## Transient regime

One effective foundation branch is selected before a complete trajectory is advanced. The semidiscrete model retains transverse translational inertia and a consistent mass matrix. Panelized mixed spaces are restricted to the common dynamically active degrees of freedom before time integration. The thesis campaign uses average-acceleration Newmark with `beta = 1/4`, `gamma = 1/2`, `dt = 1e-3` s, and 41 stored states on `[0, 0.20]` s.

## Reduced outputs and metrics

POD bases, preprocessors, and predictors are fitted only on training data. Static and steady outputs use their declared finite-element metrics. The transient workflow first performs an uncentered Euclidean SVD on training trajectories and then reorthonormalizes the retained displacement modes in the finite-element metric. POD–Proj is a compression diagnostic requiring an available full-order field; it is not a predictive online method.
