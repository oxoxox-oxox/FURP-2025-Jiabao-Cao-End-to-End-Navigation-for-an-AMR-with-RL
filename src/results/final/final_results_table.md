# Final Results Table

Evaluation protocol:

- Hard scenarios: U-trap, Double-U, narrow door, symmetric corridor.
- Each hard scenario uses 3 seeds x 12 perturbed starts.
- Standard environment uses an independent 100-episode evaluation.

| Method | Standard | U-trap | Double-U | Narrow Door | Corridor | Scenario Avg. |
|---|---:|---:|---:|---:|---:|---:|
| CNNTD3 baseline | 87% | 0±0% | 69±4% | 100±0% | 100±0% | 67% |
| NeuPAN | 0% | 0±0% | 0±0% | 0±0% | 0±0% | 0% |
| **STPS v2** | **88%** | **75±7%** | **100±0%** | **100±0%** | **100±0%** | **94%** |

Notes:

- The earlier CNNTD3 baseline value of 92% came from training-time evaluation with 10 episodes per epoch. The final standard value here uses an independent 100-episode evaluation.
- The NeuPAN result reflects this compact 10 x 10 m benchmark and the tested forward-only / safety-margin configuration. It should be reported as a domain/configuration mismatch result, not as a general claim that NeuPAN fails.
- STPS v2 switches between a precision-oriented policy and an exploration-oriented policy using stall and oscillation detection.

