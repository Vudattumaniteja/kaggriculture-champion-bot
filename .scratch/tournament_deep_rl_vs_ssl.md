# Tournament Evaluation: rl_champion_bot vs ssl_bot

**Total Episodes:** 10 (720 turns/episode)
**Win Rate (rl_champion_bot):** 60.0% (6W / 4L / 0T)
**Mean Final Bank:** $3,773.6 vs $3,243.4 (Margin: $+530.2)
**Mean Trapped Deadweight:** $1,160.0 (rl_champion_bot) vs $1,793.0 (ssl_bot)

## Aggregate Statistics

| Agent | Win Rate | Wins | Losses | Ties | Mean Final Bank | Min Bank | Max Bank | Mean Deadweight | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **rl_champion_bot** | **60.0%** | 6 | 4 | 0 | $3,773.6 | $2,147.0 | $5,340.0 | $1,160.0 | $1,212.4 |
| **ssl_bot** | **40.0%** | 4 | 6 | 0 | $3,243.4 | $1,690.0 | $4,849.0 | $1,793.0 | $852.1 |

## Match-by-Match Breakdown

| Ep | Position | rl_champion_bot Bank | ssl_bot Bank | rl_champion_bot DW | ssl_bot DW | Winner | Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| #1 | Direct | $2,240.0 | $2,250.0 | $915.0 | $1,380.0 | **ssl_bot** | 6.07s |
| #2 | Direct | $5,302.0 | $2,709.0 | $670.0 | $2,420.0 | **rl_champion_bot** | 6.12s |
| #3 | Direct | $5,340.0 | $3,533.0 | $840.0 | $1,400.0 | **rl_champion_bot** | 5.89s |
| #4 | Direct | $3,506.0 | $3,154.0 | $550.0 | $1,960.0 | **rl_champion_bot** | 4.58s |
| #5 | Direct | $4,982.0 | $1,690.0 | $1,720.0 | $1,730.0 | **rl_champion_bot** | 4.92s |
| #6 | Direct | $2,493.0 | $3,487.0 | $1,720.0 | $1,620.0 | **ssl_bot** | 3.69s |
| #7 | Direct | $3,023.0 | $4,085.0 | $1,420.0 | $1,840.0 | **ssl_bot** | 3.93s |
| #8 | Direct | $4,856.0 | $4,849.0 | $680.0 | $2,190.0 | **rl_champion_bot** | 3.75s |
| #9 | Direct | $2,147.0 | $3,584.0 | $2,105.0 | $1,620.0 | **ssl_bot** | 3.63s |
| #10 | Direct | $3,847.0 | $3,093.0 | $980.0 | $1,770.0 | **rl_champion_bot** | 3.88s |