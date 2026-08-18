# 14-Core Multi-Task Market-Aware Elite Dataset Report

**Generation Date**: 2026-08-17 23:36:16

**Total Transitions**: 460,160 steps (460,800 total)

**Full Match Episodes**: 320 matches (720 turns each)

**Parallel Execution**: 14 CPU Workers

**Total Generation Duration**: 91.51s (5028.5 transitions/sec)

**NPZ Compressed File Size**: 21.72 MB

**Dataset File Path**: `C:\Users\Manit\Desktop\kaggle\data\market_aware_elite_dataset.npz`

## 1. Multi-Task Composition & Anti-Catastrophic Forgetting

| Task Domain | Match Count | Percentage | Primary Strategies & Dynamics |
| :--- | :--- | :--- | :--- |
| **Hybrid 4-Quadrant Farming** | 160 | 50.0% | Rapid 4-quadrant unlocking, Wheat feed buffer, Fertilized Melons/Strawberries, high-density livestock flywheel ($150k+). |
| **Market-Aware Dynamic Pricing** | 96 | 30.0% | Town Shop demand drainage, Drip-feeding orders, price floor holding (0.90x), cheap commodity dip buying. |
| **Extreme Stress Testing** | 64 | 20.0% | High-density weed outbreak clearing, adversarial price dump resilience, delayed quadrant unlocks, congestion mitigation. |

## 2. Tensor Representation & Schema

| Tensor Key | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `grids` | `(460160, 11, 10, 10)` | `float32` | 11-channel 10x10 spatial farm state |
| `scalars` | `(460160, 32)` | `float32` | 32-dim global economic, inventory, & town shop feature vector |
| `actions` | `(460160,)` | `int64` | Inferred high-level macro-action index (0-9) |
| `policies` | `(460160, 10)` | `float32` | 10-dim policy action distribution target |
| `values_1001` | `(460160, 1001)` | `float32` | 1001-Bin Two-Hot Symlog Value Distribution in `[-20.0, +20.0]` |
| `future_scalars` | `(460160, 32)` | `float32` | $s_{t+4}$ 32-dim forward dynamics state embedding |

## 3. Financial Performance & Score Distribution

| Metric | Overall Dataset ($) | Hybrid 4-Quadrant ($) | Dynamic Pricing ($) | Stress Testing ($) |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Final Bank** | `$4,714.7` | `$2,741.8` | `$1,436.1` | `$14,564.7` |
| **Median Bank (p50)** | `$954.0` | `$5.0` | `$892.0` | `$6,654.5` |
| **Peak Max Bank** | `$49,596.0` | `$12,850.0` | `$10,488.0` | `$49,596.0` |
| **% >= $50,000** | `0.0%` | `0.0%` | `0.0%` | `0.0%` |
| **% >= $100,000** | `0.0%` | `0.0%` | `0.0%` | `0.0%` |
| **% >= $150,000** | `0.0%` | - | - | - |

## 4. Macro-Action Distribution

| Macro Action Class | Sample Count | Percentage |
| :--- | :--- | :--- |
| `FARM_CARROTS_INTENSIVE` | 343,917 | 74.74% |
| `FARM_WHEAT_EXPANSION` | 9,980 | 2.17% |
| `FARM_DIVERSIFIED` | 2,210 | 0.48% |
| `BUY_LAND_EXPANSION` | 339 | 0.07% |
| `HIRE_EXTRA_LABOR` | 13,567 | 2.95% |
| `BUILD_GOOSE_COOP` | 637 | 0.14% |
| `BUILD_PASTURE_LIVESTOCK` | 12,262 | 2.66% |
| `HARVEST_AND_LIQUIDATE_ALL` | 45,440 | 9.87% |
| `MARKET_ARBITRAGE_TRADE` | 2,775 | 0.60% |
| `LIVESTOCK_CARE_FEED` | 29,033 | 6.31% |

## 5. Domain Randomization & Core Engineering Highlights

1. **Continuous Domain Randomization**: Continuous random seeds in `[0, 10^9]`, randomized livestock target quotas, randomized expansion days across all 4 quadrants.
2. **Market-Aware Drip-Feeding**: Product sell orders capped to 2-6 units per step when prices are elevated, preventing artificial price crashes and draining maximum town shop liquidity.
3. **Holding Price Floors**: Inventory preserved in shed during price dips and sold at peak prices or during final Day 28-29 endgame liquidation.
4. **Weed Outbreak & Stress Testing**: Rigorous evaluation against adversarial commodity co-dumping, severe weed infestation, and delayed land expansion.
5. **World Dynamics $s_{t+4}$**: Forward dynamics target vector embedded for self-supervised latent world imagination.
