import json

with open(".scratch/move_audit_data.json") as f:
    data = json.load(f)

for name in ["Match_2_Grandmaster", "Match_8_AlphaZero"]:
    m = data[name]
    print("=" * 80)
    print(f"MATCH: {name} (Final ssl_bot Bank: ${m['ssl_reward']}, Starter: ${m['starter_reward']})")
    print("=" * 80)
    print("CASH PROGRESSION:")
    print("  " + " | ".join([f"{k}: ${v:.0f}" for k, v in m["cash_by_day"].items() if k in ["Day 0", "Day 5", "Day 10", "Day 15", "Day 20", "Day 25", "Day 29", "Day 30 (Final)"]]))
    
    print("\n--- PHASE 1 (Days 0-9) ---")
    p1 = m["phase_1"]
    print(f"  Spend: Seed=${p1['seed_spend']}, Animal=${p1['animal_spend']}, Labor=${p1['labor_spend']}, Land=${p1['land_spend']}")
    print(f"  Seeds Bought: {p1['seeds_bought']}")
    print(f"  Animals Bought: {p1['animals_bought']}")
    print(f"  Sales Revenue: ${p1['sales_revenue']} (Items: {p1['items_sold']})")
    print(f"  Structures Built: {p1['structures_built']}")
    print(f"  Macro Actions: {p1['macro_actions']}")

    print("\n--- PHASE 2 (Days 10-19) ---")
    p2 = m["phase_2"]
    print(f"  Spend: Seed=${p2['seed_spend']}, Animal=${p2['animal_spend']}, Labor=${p2['labor_spend']}, Land=${p2['land_spend']}")
    print(f"  Seeds Bought: {p2['seeds_bought']}")
    print(f"  Animals Bought: {p2['animals_bought']}")
    print(f"  Sales Revenue: ${p2['sales_revenue']} (Items: {p2['items_sold']})")
    print(f"  Structures Built: {p2['structures_built']}")
    print(f"  Macro Actions: {p2['macro_actions']}")

    print("\n--- PHASE 3 (Days 20-29) ---")
    p3 = m["phase_3"]
    print(f"  Spend: Seed=${p3['seed_spend']}, Animal=${p3['animal_spend']}, Labor=${p3['labor_spend']}, Land=${p3['land_spend']}")
    print(f"  Seeds Bought: {p3['seeds_bought']}")
    print(f"  Animals Bought: {p3['animals_bought']}")
    print(f"  Sales Revenue: ${p3['sales_revenue']} (Items: {p3['items_sold']})")
    print(f"  Structures Built: {p3['structures_built']}")
    print(f"  Macro Actions: {p3['macro_actions']}")

    print("\n--- TRAPPED ASSETS AT TURN 719 ---")
    ta = m["trapped_assets"]
    print(f"  Living Animals: {ta['living_animals']}")
    print(f"  Structures: Coops={ta['total_coops']} (empty={ta['empty_coops']}), Pastures={ta['total_pastures']} (empty={ta['empty_pastures']})")
    print(f"  Crops in Ground: {ta['crops_in_ground']}")
    print(f"  Shed: {ta['shed_inventory']}")
    print(f"  Seeds in Pocket: {ta['seeds_inventory']}")
    print(f"  Unlocked Quads: {ta['unlocked_quads']}")
    print(f"  Financial Deadweight: {ta['financial_impact']}")
    print("\n")
