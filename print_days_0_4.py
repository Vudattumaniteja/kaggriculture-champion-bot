import json
from detailed_forensic_93946801 import parse_day_by_day

d0 = parse_day_by_day(0)
d1 = parse_day_by_day(1)

for day in range(5):
    p0_d = d0[day]
    p1_d = d1[day]
    print(f"--- DAY {day:02d} ---")
    print(f"P0: Start=${p0_d['start_money']:.0f}, End=${p0_d['end_money']:.0f} | Hires={p0_d['hires']} | Plants={dict(p0_d['plantings'])} | Sales={dict(p0_d['sales'])} | SeedBuys={dict(p0_d['seed_buys'])} | Land={p0_d['land_buys']} | Structs={dict(p0_d['structures'])} | AnimBuys={dict(p0_d['animals_bought'])}")
    print(f"P1: Start=${p1_d['start_money']:.0f}, End=${p1_d['end_money']:.0f} | Hires={p1_d['hires']} | Plants={dict(p1_d['plantings'])} | Sales={dict(p1_d['sales'])} | SeedBuys={dict(p1_d['seed_buys'])} | Land={p1_d['land_buys']} | Structs={dict(p1_d['structures'])} | AnimBuys={dict(p1_d['animals_bought'])}")
