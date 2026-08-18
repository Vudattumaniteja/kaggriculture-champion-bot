# Tournament Evaluation: rl_champion_bot vs ssl_bot

**Total Episodes:** 5 (720 turns/episode)
**Win Rate (rl_champion_bot):** 60.0% (3W / 2L / 0T)
**Mean Final Bank:** $2,838.8 vs $1,456.0 (Margin: $+1,382.8)
**Mean Trapped Deadweight:** $1,374.0 (rl_champion_bot) vs $750.0 (ssl_bot)

## Aggregate Statistics

| Agent | Win Rate | Wins | Losses | Ties | Mean Final Bank | Min Bank | Max Bank | Mean Deadweight | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **rl_champion_bot** | **60.0%** | 3 | 2 | 0 | $2,838.8 | $24.0 | $6,526.0 | $1,374.0 | $2,604.4 |
| **ssl_bot** | **40.0%** | 2 | 3 | 0 | $1,456.0 | $268.0 | $2,239.0 | $750.0 | $655.5 |

## Match-by-Match Breakdown

| Ep | Position | rl_champion_bot Bank | ssl_bot Bank | rl_champion_bot DW | ssl_bot DW | Winner | Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| #1 | Direct | $895.0 | $268.0 | $800.0 | $500.0 | **rl_champion_bot** | 3.76s |
| #2 | Direct | $6,526.0 | $1,779.0 | $1,260.0 | $640.0 | **rl_champion_bot** | 3.62s |
| #3 | Direct | $24.0 | $2,239.0 | $2,130.0 | $1,070.0 | **ssl_bot** | 3.73s |
| #4 | Direct | $1,369.0 | $1,406.0 | $500.0 | $460.0 | **ssl_bot** | 3.87s |
| #5 | Direct | $5,380.0 | $1,588.0 | $2,180.0 | $1,080.0 | **rl_champion_bot** | 4.71s |