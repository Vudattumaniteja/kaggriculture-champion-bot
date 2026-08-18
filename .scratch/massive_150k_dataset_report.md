# 14-Core Domain-Randomized $150k+ Grandmaster Dataset Report

**Generation Date**: 2026-08-17 23:04:39

**Total Transitions**: 359,500

**Full Match Episodes**: 250

**CPU Workers**: 14 Cores

**Total Generation Duration**: 65.56s (5483.2 steps/s)

**NPZ Compressed File Size**: 20.71 MB

**Saved File Path**: `C:\Users\Manit\Desktop\kaggle\data\massive_150k_grandmaster_dataset.npz`

## 1. Tensor Specifications & Shapes

| Tensor Key | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `grids` | `(359500, 11, 10, 10)` | `float32` | 11-channel 10x10 spatial farm state |
| `scalars` | `(359500, 32)` | `float32` | 32-dim global economic & inventory feature vector |
| `actions` | `(359500,)` | `int64` | Inferred high-level macro-action index (0-9) |
| `policies` | `(359500, 10)` | `float32` | 10-dim policy action distribution target |
| `values_1001` | `(359500, 1001)` | `float32` | 1001-Bin Two-Hot Symlog Value Distribution in `[-20.0, +20.0]` |
| `future_scalars` | `(359500, 32)` | `float32` | $s_{t+4}$ 32-dim forward dynamics state embedding |

## 2. Match Score Distribution ($ Final Bank)

| Statistic | Value ($) |
| :--- | :--- |
| **Min Score** | `$0.0` |
| **10th Percentile (p10)** | `$1,331.8` |
| **25th Percentile (p25)** | `$3,322.0` |
| **Median Score (p50)** | `$12,976.5` |
| **Mean Score** | `$19,241.0` |
| **75th Percentile (p75)** | `$28,107.8` |
| **90th Percentile (p90)** | `$49,384.8` |
| **Peak Max Score** | `$94,032.0` |
| **% >= $50,000** | `9.8%` |
| **% >= $100,000** | `0.0%` |
| **% >= $150,000** | `0.0%` |

## 3. Macro-Action Distribution

| Macro Action Class | Sample Count | Percentage |
| :--- | :--- | :--- |
| `FARM_CARROTS_INTENSIVE` | 119,819 | 33.33% |
| `FARM_WHEAT_EXPANSION` | 3,134 | 0.87% |
| `FARM_DIVERSIFIED` | 766 | 0.21% |
| `BUY_LAND_EXPANSION` | 1,034 | 0.29% |
| `HIRE_EXTRA_LABOR` | 11,391 | 3.17% |
| `BUILD_GOOSE_COOP` | 177 | 0.05% |
| `BUILD_PASTURE_LIVESTOCK` | 28,113 | 7.82% |
| `HARVEST_AND_LIQUIDATE_ALL` | 35,500 | 9.87% |
| `MARKET_ARBITRAGE_TRADE` | 8,372 | 2.33% |
| `LIVESTOCK_CARE_FEED` | 151,194 | 42.06% |

## 4. Continuous Domain Randomization Pipeline

- **Seeds**: Continuous uniform draw in `[0, 10^9]` per match.
- **Dirichlet Continuous Crop Allocation**: Vector drawn from $\text{Dir}(\alpha)$ with $\alpha \in [0.5, 3.0]^5$.
- **Multivariate Gaussian Livestock Portfolio**: 3D distribution $\mathcal{N}(\mu=[26, 24, 2], \Sigma)$ parameterized across Sheep, Cows, Geese.
- **Expansion Timings**: Randomized quadrant unlock thresholds across NE (Days 5-7), SW (Days 7-9), SE (Days 9-11).
- **Stochastic Hungarian Utility Weights**: Continuous utility scaling for feed ($1100-1400$), care ($1000-1250$), fertilizer ($900-1150$), harvest ($800-1000$).
- **Dynamic Labor Curve**: Labor scaled dynamically from 8 to 14 workers during compound growth phase with lean endgame tapering.
