import os
import sys
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm


JBB_DIR = "../benchmarks/jbb_judged"

jailbreak_data = pd.read_csv("../data/jbb_expanded.csv")

goal2category = {}

for i, row in jailbreak_data.iterrows():
    goal2category[row["Goal"]] = row["Category"]

categories = []
categories.extend(list(set(goal2category.values())))

category_count = {}
for k, v in goal2category.items():
    if v not in category_count:
        category_count[v] = 0
    category_count[v] += 1


judge_cls = {
    "GCG": 0,
    "PAIR_gpt-4o-2024-11-20": 0,
    "PAIR_Qwen_Qwen2.5-72B-Instruct": 0,
    "PAIR_meta-llama_Llama-3.3-70B-Instruct": 0,
}


df = pd.DataFrame(
    columns=[
        "model_name",
        "attack_method",
        "jailbreak_type",
        *judge_cls.keys(),
    ]
)


for root, dirs, files in os.walk(JBB_DIR):
    # if "NoneDefender" not in root:
    #     continue
    # print(root)
    for file in files:
        if file.endswith(".json"):
            # model/attack/defense/results.json
            model_path = root.split("/")[-3]
            model_name = model_path
            if model_name.endswith("_"):
                model_name = model_name[:-1]
            elif model_name.startswith("aihubmix-"):
                model_name = model_name[9:]
            elif model_name.startswith("ahm-"):
                model_name = model_name[4:]

            attack_path = root.split("/")[-2]
            attack_method = '_'.join(attack_path.split("_")[1:])

            defense_path = root.split("/")[-1]
            defense_method = defense_path

            asr = {k: deepcopy(judge_cls) for k in categories}
            prompt_tokens = {k: 0 for k in categories}
            completion_tokens = {k: 0 for k in categories}

            with open(os.path.join(root, file), "r") as f:
                results = json.load(f)["results"]

            # alpaca_json_path = f"{ALPACA_DIR}/{model_path}/{defense_path}/alpaca_eval_llama3_70b_fn/leaderboard.csv"
            # # print(alpaca_json_path)
            # if os.path.exists(alpaca_json_path):
            #     with open(alpaca_json_path, "r") as f:
            #         alpaca_result = pd.read_csv(f)
            #     alpaca_winrate = alpaca_result["win_rate"].values[0]
            #     alpaca_lc_winrate = alpaca_result["length_controlled_winrate"].values[0]
            # else:
            #     if "None" in defense_path:
            #         alpaca_winrate = 50.0
            #         alpaca_lc_winrate = 50.0
            #     else:
            #         alpaca_winrate = np.nan
            #         alpaca_lc_winrate = np.nan
            #         print(f"File not found: {alpaca_json_path}")

            for res in results:
                ratio = res["jailbroken"]
                goal = res["goal"]
                for k, v in ratio.items():
                    if k == 'PAIR_Llama-3.3-70B-Instruct' or k == 'PAIR_aihubmix-Llama-3-3-70B-Instruct':
                        k = 'PAIR_meta-llama_Llama-3.3-70B-Instruct'
                    asr[goal2category[goal]][k] += (v == 10)
                    # asr[f"{goal2category[goal]}_{k}"] += (v == 10)
                    # asr[f"All_{k}"] += (v == 10)
                p_tk = res["data"][0]["usage"]["defender"]["prompt_tokens"]
                c_tk = res["data"][0]["usage"]["defender"]["completion_tokens"]

                prompt_tokens[goal2category[goal]] += p_tk / 10
                completion_tokens[goal2category[goal]] += c_tk / 10

            # asr = {k: v * 1. if "All" in k else v * 10. for k, v in asr.items()}

            for goal_cat in categories:
                asr[goal_cat] = {k: v * 10. for k, v in asr[goal_cat].items()}
                new_row = pd.DataFrame([{
                    "model_name": model_name,
                    "attack_method": attack_method,
                    "defense_method": defense_method,
                    "jailbreak_type": goal_cat,
                    "prompt_tokens": prompt_tokens[goal_cat],
                    "completion_tokens": completion_tokens[goal_cat],
                    # "alpaca_winrate": alpaca_winrate,
                    # "alpaca_lc_winrate": alpaca_lc_winrate,
                    **asr[goal_cat]
                }])
                # new_row = new_row.dropna(axis=1, how='all')
                df = pd.concat([df, new_row], ignore_index=True)

df.to_csv("./ana.csv")