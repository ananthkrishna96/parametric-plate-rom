# Figure provenance

The three PNG files in this directory are reduced-size copies of original computational figures contained in the final thesis source archive.

- `static_verification.png`: Chapter 4 monolithic verification summary, comparing the `C0-IPG` FOM with the truncated Navier reference and an independent three-dimensional solid result.
- `thermomechanical_geometry.png`: Chapter 6 thermomechanical plate, foundation, localized load, solar input, top exchange, and lower contact-gap exchange.
- `transient_geometry.png`: Chapter 9 monolithic transient setting with prescribed effective foundation regime, localized load, and response probe.

They are included for orientation, not as substitute data. Plotting functions in `paramplate.postprocessing` generate new figures from numerical arrays; full thesis figure regeneration requires the corresponding external arrays and configurations.
