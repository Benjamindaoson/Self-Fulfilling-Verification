# Gate-0 Probe Calibration Receipt

Calibration used only seeds 10000–10039. Confirmatory main seeds 20000–20499 were not inspected. The success predicate and margin bins remained unchanged throughout. This receipt supersedes the earlier fixed-contact calibration because the frozen model now samples contact time constant, damping ratio, and impedance width before task execution.

| Current debug run | Probe strengths | VIC V2/V3/V4 | VID V2/V3/V4 | Mean depth effect V2/V3/V4 |
|---|---|---|---|---|
| `debug-contact-calibration-v2` | 0.02 / 0.08 / 0.10 | 0.029 / 0.088 / 0.088 | 0.000 / 0.000 / 0.000 | 0.499 / 1.504 / 1.717 mm |
| `debug-contact-calibration-v3` | 0.02 / 0.08 / 0.12 | 0.029 / 0.088 / 0.088 | 0.000 / 0.000 / 0.000 | 0.499 / 1.504 / 1.925 mm |

The frozen main strengths are **0.02 / 0.08 / 0.12**. V4=0.12 was accepted because it increased the V3→V4 mean depth-effect separation from 0.214 mm to 0.421 mm without increasing VIC or causing any disruption in the six debug pre-successes. All three V4 completions remained in the frozen very-near-boundary bin; this was recorded as a main-run falsification risk, not used to alter thresholds, bins, or sampling. This calibration is engineering evidence only and is excluded from the confirmatory Gate result.
