import os
import re
import sys
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from matplotlib import pyplot as plt
import seaborn as sns
from adjustText import adjust_text

ROOT_DIR = "/mnt/home/tonghaibo/panda-guard"
JBB_DIR = os.path.join(ROOT_DIR, "benchmarks", "jbb_judged")
DATA_DIR = os.path.join(ROOT_DIR, "data")
ALPACA_DIR = os.path.join(ROOT_DIR, "benchmarks", "alpaca_eval_judged")

jailbreak_data = pd.read_csv(os.path.join(DATA_DIR, "jbb_expanded.csv"))
goal2category = {}
for i, row in jailbreak_data.iterrows():
    goal2category[row["Goal"]] = row["Category"]
categories = []
categories.extend(list(set(goal2category.values())))


# category_count = {}
# for k, v in goal2category.items():
#     if v not in category_count:
#         category_count[v] = 0
#     category_count[v] += 1
# print(categories)
# print(category_count)

def get_data_df():
    judge_cls = {
        "GCG": 0,
        "PAIR_gpt-4o-2024-11-20": 0,
        "PAIR_Qwen_Qwen2.5-72B-Instruct": 0,
    }

    usage_cls = {"prompt_tokens": 0, "completion_tokens": 0}

    asr_df = pd.DataFrame(
        columns=[
            "model_name",
            "attack_method",
            "defence_method",
            "jailbreak_type",
            *judge_cls.keys(),
        ]
    )

    usage_df = pd.DataFrame(
        columns=[
            "model_name",
            "attack_method",
            "defence_method",
            "jailbreak_type",
            "prompt_tokens",
            "completion_tokens",
        ]
    )

    for root, dirs, files in os.walk(JBB_DIR):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    results = json.load(f)["results"]
                model_name = root.split("/")[-3]
                # if model_name.endswith("_"):
                #     model_name = model_name[:-1]
                # elif model_name.startswith("aihubmix-"):
                #     model_name = model_name[9:]
                # elif model_name.startswith("ahm-"):
                #     model_name = model_name[4:]
                attack_method = root.split("/")[-2]
                if attack_method == "TransferAttacker_PAIR":
                    attack_method = "T_PAIR"
                else:
                    attack_method = attack_method.split("_")[-1]
                defence_method = root.split("/")[-1].removesuffix('Defender')

                asr = {k: deepcopy(judge_cls) for k in categories}

                usage = {k: deepcopy(usage_cls) for k in categories}

                for res in results:
                    ratio = res["jailbroken"]
                    goal = res["goal"]
                    for k, v in ratio.items():
                        if k in judge_cls:
                            asr[goal2category[goal]][k] += (v == 10) * 10.0  # /10*100百分比值
                        elif k == "PAIR_qwen2.5-72b-instruct":
                            asr[goal2category[goal]]["PAIR_Qwen_Qwen2.5-72B-Instruct"] += (v == 10) * 10.0

                    tokens = res["data"][0]["usage"]["defender"]
                    for k, v in tokens.items():
                        if k in judge_cls:
                            usage[goal2category[goal]][k] += v * 0.1  # 每类10条，累加前0.1则得到结果是均值
                        elif k == "PAIR_qwen2.5-72b-instruct":
                            asr[goal2category[goal]]["PAIR_Qwen_Qwen2.5-72B-Instruct"] += v * 0.1

                # asr = {k: v * 1. if "All" in k else v * 10. for k, v in asr.items()}

                for goal_cat in categories:
                    # asr[goal_cat] = {k: v * 1.0 for k, v in asr[goal_cat].items()}
                    asr_new_row = pd.DataFrame([{
                        "model_name": model_name,
                        "attack_method": attack_method,
                        "defence_method": defence_method,
                        "jailbreak_type": goal_cat,
                        **asr[goal_cat]
                    }])
                    # new_row = new_row.dropna(axis=1, how='all')
                    asr_df = pd.concat([asr_df, asr_new_row], ignore_index=True)

                    usage_new_row = pd.DataFrame([{
                        "model_name": model_name,
                        "attack_method": attack_method,
                        "defence_method": defence_method,
                        "jailbreak_type": goal_cat,
                        **usage[goal_cat]
                    }])
                    usage_df = pd.concat([usage_df, usage_new_row], ignore_index=True)
    return asr_df, usage_df


def plot_asr(asr_df):
    numeric_columns = asr_df.select_dtypes(include=["number"]).columns
    asr_total = asr_df.groupby(["model_name", "defence_method", "attack_method", ])[numeric_columns].mean()
    print(asr_total)
    asr_avg = asr_total.groupby(["model_name", "defence_method"]).mean()
    print(asr_avg)
    for model in asr_df["model_name"].unique():
        df_pivoted = asr_total.loc[model].pivot_table(
            index='defence_method', columns='attack_method',
            values=['GCG', 'PAIR_Qwen_Qwen2.5-72B-Instruct', 'PAIR_gpt-4o-2024-11-20'])
        plt.figure(figsize=(6, 6))
        sns.heatmap(
            data=df_pivoted['GCG'],
            cmap='Blues',
            annot=True,
            fmt=".1f",
            linewidths=0.5,
        )
        plt.ylabel("Defence Method")
        plt.xlabel("Attack Method")
        plt.title(f'{model} ASR by GCG Judge')
        plt.tight_layout()
        plt.savefig(os.path.join(ROOT_DIR, "results", f"{model}_asr_by_gcg_judge.png"))
        # plt.show()

        plt.figure(figsize=(6, 6))
        sns.heatmap(
            data=df_pivoted['PAIR_Qwen_Qwen2.5-72B-Instruct'],
            cmap='Blues',
            annot=True,
            fmt=".1f",
            linewidths=0.5,
        )
        plt.ylabel("Defence Method")
        plt.xlabel("Attack Method")
        plt.title(f'{model} ASR by Pair Judge Qwen2.5-72B')
        plt.tight_layout()
        plt.savefig(os.path.join(ROOT_DIR, "results", f"{model}_asr_by_pair_qwen_judge.png"))
        # plt.show()

        plt.figure(figsize=(6, 6))
        sns.heatmap(
            data=df_pivoted['PAIR_gpt-4o-2024-11-20'],
            cmap='Blues',
            annot=True,
            fmt=".1f",
            linewidths=0.5,
        )
        plt.ylabel("Defence Method")
        plt.xlabel("Attack Method")
        plt.title(f'{model} ASR by Pair Judge GPT-4o-2024-11-20')
        plt.tight_layout()
        plt.savefig(os.path.join(ROOT_DIR, "results", f"{model}_asr_by_pair_gpt4o_judge.png"))

        df_melted = asr_df[asr_df["model_name"] == model].melt(
            id_vars=['defence_method'],
            value_vars=['PAIR_Qwen_Qwen2.5-72B-Instruct', 'PAIR_gpt-4o-2024-11-20'],
            var_name='Metric',
            value_name='Value').sort_values('defence_method')
        print(df_melted)
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df_melted, x='defence_method', y='Value', hue='Metric', palette="Blues")
        plt.ylabel("Value")
        plt.xlabel("Defence Method")
        plt.title(f'{model} average ASR')
        plt.tight_layout()
        plt.savefig(os.path.join(ROOT_DIR, "results", f"{model}_average_asr.png"))
        # plt.show()


def plot_usage(usage_df):
    numeric_columns = usage_df.select_dtypes(include=["number"]).columns
    usage_avg = usage_df.groupby(["model_name", "defence_method", "attack_method", ])[numeric_columns].mean()
    print(usage_avg)
    usage_avg_across_attack = usage_avg.groupby(["model_name", "defence_method"]).mean()
    print(usage_avg_across_attack)

    for model in usage_df["model_name"].unique():
        df_pivoted = usage_avg.loc[model].pivot_table(
            index='defence_method', columns='attack_method',
            values=['prompt_tokens', 'completion_tokens'])
        # none_row = df_pivoted['prompt_tokens'].loc['None'].copy()
        # for _, row in df_pivoted['prompt_tokens'].iterrows():
        #     row -= none_row
        plt.figure(figsize=(6, 6))
        sns.heatmap(
            data=df_pivoted['prompt_tokens'],
            cmap='Blues',
            annot=True,
            fmt=".1f",
            linewidths=0.5,
        )
        plt.ylabel("Defence Method")
        plt.xlabel("Attack Method")
        plt.title(f'{model} Prompt Tokens')
        plt.tight_layout()
        plt.savefig(os.path.join(ROOT_DIR, "results", f"{model}_prompt_tokens.png"))
        plt.show()

        plt.figure(figsize=(6, 6))
        sns.heatmap(
            data=df_pivoted['completion_tokens'],
            cmap='Blues',
            annot=True,
            fmt=".1f",
            linewidths=0.5,
        )
        plt.ylabel("Defence Method")
        plt.xlabel("Attack Method")
        plt.title(f'{model} Completion Tokens')
        plt.tight_layout()
        plt.savefig(os.path.join(ROOT_DIR, "results", f"{model}_completion_tokens.png"))
        plt.show()

        df_melted = usage_df[usage_df["model_name"] == model].melt(
            id_vars=['defence_method'],
            value_vars=['prompt_tokens', 'completion_tokens'],
            var_name='Metric',
            value_name='Value').sort_values('defence_method')
        print(df_melted)
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df_melted, x='defence_method', y='Value', hue='Metric', palette="Blues")
        plt.ylabel("Value")
        plt.xlabel("Defence Method")
        plt.title(f'{model} average usage')
        plt.tight_layout()
        plt.savefig(os.path.join(ROOT_DIR, "results", f"{model}_average_usage.png"))
        plt.show()


def tom_stage_analysis():
    stage_df = pd.DataFrame(
        columns=[
            "model_name",
            "attack_method",
            "defence_method",
            "jailbreak_type",
            "prompt",
            "response",
            "undetected_safe",
            "undetected_danger"
        ]
    )
    for root, dirs, files in os.walk(JBB_DIR):
        for file in files:
            if file.endswith(".json"):
                model_name = root.split("/")[-3]
                attack_method = root.split("/")[-2]
                if attack_method == "TransferAttacker_PAIR":
                    attack_method = "T_PAIR"
                else:
                    attack_method = attack_method.split("_")[-1]
                defence_method = root.split("/")[-1].removesuffix('Defender')
                if defence_method in {"Tom", "Tom2"}:
                    with open(os.path.join(root, file), "r") as f:
                        results = json.load(f)["results"]

                    stage = {k: {'prompt': 0, 'response': 0, 'undetected_safe': 0, 'undetected_danger': 0} for k in categories}

                    for res in results:
                        goal = res["goal"]
                        response = res['data'][0]['messages'][-1]['content']
                        jailbroken = res["jailbroken"]
                        try:
                            s = re.search(r'unsafe request inferred from (.*?)\.', response).group(1)
                            assert s in ('prompt', 'response')
                        except AttributeError:
                            if jailbroken['PAIR_Qwen_Qwen2.5-72B-Instruct'] == 10:
                                s = "undetected_danger"
                            else:
                                s = "undetected_safe"
                        stage[goal2category[goal]][s] += 1

                    for goal_cat in categories:
                        stage_new_row = pd.DataFrame([{
                            "model_name": model_name,
                            "attack_method": attack_method,
                            "defence_method": defence_method,
                            "jailbreak_type": goal_cat,
                            **stage[goal_cat]
                        }])
                        stage_df = pd.concat([stage_df, stage_new_row], ignore_index=True)
    return stage_df


def plot_stage(stage_df):
    usage_total = stage_df.groupby(["model_name", "defence_method", "attack_method", ])[
        ['prompt', 'response', 'undetected_safe', 'undetected_danger']].sum()
    print(usage_total)
    for model in stage_df["model_name"].unique():
        for defence_method in stage_df["defence_method"].unique():
            usage_model = usage_total.loc[model, defence_method].reset_index()
            print(usage_model)
            pie_num = usage_model.shape[0]
            colors = sns.color_palette("Blues", 4)
            plt.figure(figsize=(20, 6))
            # plt.title(f"{model} Tom Stage Analysis")
            for i, row in usage_model.iterrows():
                plt.subplot(1, pie_num, i + 1)
                plt.pie([row['prompt'], row['response'], row['undetected_safe'], row['undetected_danger']],
                        colors=colors, autopct='%1.1f%%', pctdistance=1.2, )
                plt.title(row['attack_method'], y=-0.2)
            plt.legend(labels=['prompt', 'response', 'undetected_safe', 'undetected_danger'], loc='lower right')
            plt.suptitle(f"{model} {defence_method} Stage Analysis")
            plt.tight_layout()
            plt.savefig(os.path.join(ROOT_DIR, "results", f"{model}_{defence_method}_stage.png"))
            plt.show()
    # for model in stage_df["model_name"].unique():
    #     model_df = stage_df[stage_df["model_name"] == model]
    #     pie_num = len(stage_df["attack_method"].unique())
    #     plt.figure(figsize=(30, 4))
    #     for i, attack in enumerate(stage_df["attack_method"].unique()):
    #         attack_df = model_df[model_df["attack_method"] == attack]
    #         plt.subplot(1, pie_num, i + 1)
    #         df_melted = attack_df.melt(
    #             id_vars='jailbreak_type',
    #             value_vars=['prompt', 'response', 'undetected'],
    #             var_name='Stage',
    #             value_name='Value')
    #         sns.barplot(x='Value', y='jailbreak_type', hue='Stage', data=df_melted, orient='h', ci=None)
    #     plt.tight_layout
    #     plt.show()


def alpaca_analysis():
    alpaca_df = pd.DataFrame(
        columns=[
            "model_name",
            "defence_method",
            "win_rate",
            "std_error",
            "length",
        ]
    )
    for root, dirs, files in os.walk(ALPACA_DIR):
        for file in files:
            if file.endswith(".csv"):
                result = pd.read_csv(os.path.join(root, file))
                model_name = root.split("/")[-3]
                defence_method = root.split("/")[-2].removesuffix('Defender')
                print(result)
                new_row = pd.DataFrame([{
                    "model_name": model_name,
                    "defence_method": defence_method,
                    "win_rate": result.loc[0, 'win_rate'],
                    "std_error": result.loc[0, 'standard_error'],
                    "length": result.loc[0, 'avg_length'],
                }])
                alpaca_df = pd.concat([alpaca_df, new_row], ignore_index=True)
    print(alpaca_df)
    return alpaca_df
    # plt.figure(figsize=(10,6))
    # plt.axis('off')
    # table = plt.table(
    #     cellText=alpaca_df.values,
    #     colLabels=alpaca_df.columns,
    #     cellLoc='center',  # 内容居中
    #     loc='center'
    # )
    # table.auto_set_font_size(False)
    # table.set_fontsize(10)
    # table.scale(1.2, 1.5)  # 宽、高缩放
    #
    # plt.savefig("alpaca_table.png", dpi=200, bbox_inches='tight')  # 保存为图片
    # plt.show()


def plot_alpaca_asr(asr_df, alpaca_df):
    print(asr_df)
    asr_avg = asr_df.groupby(['model_name', 'defence_method'])['PAIR_gpt-4o-2024-11-20'].mean().reset_index()
    print(asr_avg)
    print(alpaca_df)
    asr_alpaca_df = pd.DataFrame(columns=["defence_method", "asr", "win_rate"])
    for model in alpaca_df["model_name"].unique():
        asrs = asr_avg[asr_avg['model_name'] == model]
        for defence_method in alpaca_df["defence_method"].unique():
            defence_asr = asrs[asrs['defence_method'] == defence_method]['PAIR_gpt-4o-2024-11-20'].values[0]
            defence_win_rate = alpaca_df[alpaca_df['defence_method'] == defence_method]['win_rate'].values[0]
            new_row = pd.DataFrame([{
                "defence_method": defence_method,
                "asr": defence_asr,
                "win_rate": defence_win_rate,
            }])
            asr_alpaca_df = pd.concat([asr_alpaca_df, new_row], ignore_index=True)
        asr_none = asrs[asrs['defence_method'] == 'None']['PAIR_gpt-4o-2024-11-20'].values[0]
        new_row = pd.DataFrame([{
            "defence_method": 'None(Baseline)',
            "asr": asr_none,
            "win_rate": 50.0,
        }])
        asr_alpaca_df = pd.concat([asr_alpaca_df, new_row], ignore_index=True)
    print(asr_alpaca_df)

    plt.figure(figsize=(9, 6))
    sns.set(style="whitegrid")

    # 绘制散点图
    palette = sns.color_palette("GnBu", len(asr_alpaca_df))
    scatter = plt.scatter(
        asr_alpaca_df["asr"], asr_alpaca_df["win_rate"],
        s=100, c=palette, edgecolors='black'
    )

    # 添加标签
    texts = []
    for i in range(len(asr_alpaca_df)):
        texts.append(plt.text(
            asr_alpaca_df["asr"][i],
            asr_alpaca_df["win_rate"][i] + 1,
            asr_alpaca_df["defence_method"][i],
            fontsize=11
        ))
    adjust_text(texts,
                # arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),  # 启用连接线
                # force_text=(0.7, 0.7),  # 增强文本间排斥力
                # force_points=(0.9, 0.9),  # 增强点到文本吸引力
                expand_points=(1.5, 1.5),  # 扩大禁区范围
                # expand_text=(1.2, 1.2),  # 新增：文本边框扩展
                only_move={'points': 'y', 'text': 'xy'},  # 限制文本移动方向
                avoid_points=True,  # 新增：强制避开数据点
                lim=50  # 新增：最大迭代次数
                )

    # 设置标题和轴
    plt.title("LLM Jailbreak Defence Performance", fontsize=14, fontweight='bold')
    plt.xlabel("ASR (↓ Attack Success Rate)", fontsize=12)
    plt.ylabel("Win Rate (↑ AlpacaEval)", fontsize=12)
    plt.xlim(-0.2, max(asr_alpaca_df["asr"]) + 1)
    plt.ylim(0, 60)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 显示图形
    plt.savefig(os.path.join(ROOT_DIR, "results", f"{model}_asr_winrate.png"))
    # plt.show()


if __name__ == '__main__':
    asr_df, usage_df = get_data_df()
    # plot_asr(asr_df)
    # plot_usage(usage_df)
    # stage_df = tom_stage_analysis()
    # plot_stage(stage_df)
    alpaca_df = alpaca_analysis()
    plot_alpaca_asr(asr_df, alpaca_df)
