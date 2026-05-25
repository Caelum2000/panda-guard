from importlib.metadata import entry_points

for group in ["panda_guard.judges", "panda_guard.judge_configs"]:
    print(f"\n[{group}]")
    for ep in entry_points(group=group):
        print(f"{ep.name} = {ep.value}")