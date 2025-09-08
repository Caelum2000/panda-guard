
import os


root_dir = "../../benchmarks/jbb"


attacks = [
    "TransferAttacker_AIM",
    "TransferAttacker_ANTI_GPT_V2",
    "TransferAttacker_BETTER_DAN",
    "TransferAttacker_DEV_MODE_Ranti",
    "TransferAttacker_DEV_MODE_V2",
    "TransferAttacker_Goal",
    "TransferAttacker_new_art_prompt",
    "TransferAttacker_new_autodan",
    "TransferAttacker_new_cold",
    "TransferAttacker_new_deepinception",
    "TransferAttacker_new_gcg",
    "TransferAttacker_new_gpt4_cipher",
    "TransferAttacker_new_gptfuzzer",
    "TransferAttacker_new_ica",
    "TransferAttacker_new_pair",
    "TransferAttacker_new_random_search",
    "TransferAttacker_new_renellm",
    "TransferAttacker_new_scav",
    "TransferAttacker_new_tap",
    "TransferAttacker_past_tense",
    "TransferAttacker_tense_future",
]

defenses = [
    "GoalPriorityDefender",
    "IclDefender",
    "NoneDefender",
    "ParaphraseDefender",
    "PerplexityFilterDefender",
    "RPODefender",
    "SelfDefenseDefender",
    "SelfReminderDefender",
    "SemanticSmoothLLMDefender",
    "SmoothLLMDefender"
]

total_missing = 0

print(os.listdir(root_dir))

for model_name in os.listdir(root_dir):
    model_path = os.path.join(root_dir, model_name)
    if not os.path.isdir(model_path):
        continue  # 跳过非文件夹项

    completed = 0
    missing = 0
    missing_experiments = []

    for attack in attacks:
        for defense in defenses:
            result_path = os.path.join(model_path, attack, defense, "results.json")
            if os.path.isfile(result_path):
                completed += 1
            else:
                missing += 1
                missing_experiments.append((attack, defense))

    total = completed + missing

    # 输出结果
    print(f"\n模型 {model_name}：实验共 {total} 个")
    print(f"已完成实验 {completed} 个，缺失实验 {missing} 个")
    if missing_experiments:
        print("缺失实验如下：")
        for attack, defense in missing_experiments:
            print(f" {attack} -> {defense} ")
    total_missing += missing

print(f"\n\n缺失实验总数： {total_missing} ")
