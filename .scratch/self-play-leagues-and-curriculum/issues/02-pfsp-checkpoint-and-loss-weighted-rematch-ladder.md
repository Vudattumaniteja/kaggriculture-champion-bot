# PFSP Checkpoint and Loss-Weighted Rematch Ladder

Type: grilling
Status: open
Blocked by: none

## Question

How should the `PFSPLadder` track rolling historical checkpoints, compute loss-weighted sampling probabilities $P(i) \propto \max(0.05, (1 - \text{WinRate}_i)^{1.5})$, and update win rates through background evaluation matches to automate rematches against weak and losing matchups?
