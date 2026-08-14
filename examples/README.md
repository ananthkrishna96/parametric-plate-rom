# Lightweight examples

The examples exercise the software path without reproducing the dissertation campaigns. The bundled arrays are deterministic synthetic fields, not numerical results reported in the thesis.

```bash
python examples/static_navier_smoke.py
python examples/thermal_driver_smoke.py
python examples/newmark_smoke.py
python examples/rom_smoke.py --study mechanical --method podi-rbf --rank 4
python examples/rom_smoke.py --study thermomechanical --output-field thermal_driver --method pod-proj --rank 2
python examples/rom_smoke.py --study dynamics --method podi-linear --rank 3
```

Regenerate the small archives with `python examples/generate_smoke_data.py`.
