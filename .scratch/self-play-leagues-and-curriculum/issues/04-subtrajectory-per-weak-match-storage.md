# Sub-Trajectory Prioritized Experience Replay for Weak Matches

Type: grilling
Status: open
Blocked by: 02, 03

## Question

How should sub-trajectories exhibiting high TD-error or resulting in game losses be flagged, weighted with an error-proportional priority ($p_i = (|\delta_i| + \epsilon)^{0.6} \times \text{loss\_boost}$), and sampled from a 500k Binary SumTree Prioritized Replay Buffer during neural gradient updates?
