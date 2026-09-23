# Tool-calling governance benchmark

## Model: jev

### call_approval

- n: 40 · accuracy: **70.0%**
- Brier: 0.198
  - p ≥ 0.5: coverage 65%, precision 65.4%
  - p ≥ 0.7: coverage 52%, precision 81.0%
  - p ≥ 0.9: coverage 32%, precision 76.9%

### arg_validation

- n: 40 · accuracy: **100.0%**
- Brier: 0.019
  - p ≥ 0.5: coverage 50%, precision 100.0%
  - p ≥ 0.7: coverage 45%, precision 100.0%
  - p ≥ 0.9: coverage 10%, precision 100.0%

### injection_risk

- n: 40 · accuracy: **100.0%**
- Brier: 0.013
  - p ≥ 0.5: coverage 55%, precision 100.0%
  - p ≥ 0.7: coverage 48%, precision 100.0%
  - p ≥ 0.9: coverage 45%, precision 100.0%

### tool_selection

- n: 40 · accuracy: **85.0%**
  - p ≥ 0.5: coverage 85%, precision 100.0%
  - p ≥ 0.7: coverage 82%, precision 100.0%
  - p ≥ 0.9: coverage 75%, precision 100.0%

Tool-selection accuracy by catalog size:

- 5 options: 100.0%
- 20 options: 90.0%
- 50 options: 70.0%
- 200 options: 80.0%

## Model: laya

### call_approval

- n: 40 · accuracy: **32.5%**
- Brier: 0.547
  - p ≥ 0.5: coverage 18%, precision 0.0%
  - p ≥ 0.7: coverage 10%, precision 0.0%

### arg_validation

- n: 40 · accuracy: **65.0%**
- Brier: 0.238
  - p ≥ 0.5: coverage 45%, precision 66.7%
  - p ≥ 0.7: coverage 35%, precision 71.4%
  - p ≥ 0.9: coverage 5%, precision 100.0%

### injection_risk

- n: 40 · accuracy: **100.0%**
- Brier: 0.014
  - p ≥ 0.5: coverage 55%, precision 100.0%
  - p ≥ 0.7: coverage 45%, precision 100.0%
  - p ≥ 0.9: coverage 42%, precision 100.0%

### tool_selection

- n: 40 · accuracy: **60.0%**
  - p ≥ 0.5: coverage 55%, precision 100.0%
  - p ≥ 0.7: coverage 55%, precision 100.0%
  - p ≥ 0.9: coverage 42%, precision 100.0%

Tool-selection accuracy by catalog size:

- 5 options: 100.0%
- 20 options: 100.0%
- 50 options: 40.0%
- 200 options: 0.0%
