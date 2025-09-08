import os
import re
import sys
import json
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from adjustText import adjust_text

ROOT_DIR = "/mnt/home/tonghaibo/panda-guard"
RESULTS_DIR = os.path.join(ROOT_DIR, "benchmarks")
ALPACA_DIR = os.path.join(ROOT_DIR, "benchmarks", "alpaca_eval_judged")
JBB_DIR = os.path.join(ROOT_DIR, "benchmarks", "jbb_judged")
HM_DIR = os.path.join(ROOT_DIR, "benchmarks", "hm_judged")
OUTPUT_DIR = os.path.join(ROOT_DIR, "aaai_figs")


def get_data_df():
    judge_cls = {
        "GCG": 0,
        "PAIR_gpt-4o-2024-11-20": 0,
        "PAIR_Qwen_Qwen2.5-72B-Instruct": 0,
    }

    usage_cls = {"prompt_tokens": 0, "completion_tokens": 0}

    asr_df = pd.DataFrame(
        columns=[
            "dataset",
            "model_name",
            "attack_method",
            "defence_method",
            *judge_cls.keys(),
        ]
    )

    usage_df = pd.DataFrame(
        columns=[
            "dataset",
            "model_name",
            "attack_method",
            "defence_method",
            "prompt_tokens",
            "completion_tokens",
        ]
    )

    datasets = ['jbb', 'hm']
    for dataset in datasets:
        dataset_dir = os.path.join(RESULTS_DIR, f'{dataset}_judged')
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith(".json"):
                    with open(os.path.join(root, file), "r") as f:
                        results = json.load(f)["results"]
                    model_name = root.split("/")[-3]
                    attack_method = root.split("/")[-2].split("_")[-1]
                    if attack_method == 'ica':
                        continue
                    defence_method = root.split("/")[-1].removesuffix('Defender')

                    asr = deepcopy(judge_cls)
                    usage = deepcopy(usage_cls)

                    total_num = len(results)
                    for res in results:
                        ratio = res["jailbroken"]
                        for k, v in ratio.items():
                            if k in judge_cls:
                                asr[k] += (v == 10) * 100.0 / total_num

                        tokens = res["data"][0]["usage"]["defender"]
                        for k, v in tokens.items():
                            if k in usage_cls:
                                usage[k] += v * 1.0 / total_num

                    asr_new_row = pd.DataFrame([{
                        "dataset": dataset,
                        "model_name": model_name,
                        "attack_method": attack_method,
                        "defence_method": defence_method,
                        **asr
                    }])
                    asr_df = pd.concat([asr_df, asr_new_row], ignore_index=True)

                    usage_new_row = pd.DataFrame([{
                        "dataset": dataset,
                        "model_name": model_name,
                        "attack_method": attack_method,
                        "defence_method": defence_method,
                        **usage
                    }])
                    usage_df = pd.concat([usage_df, usage_new_row], ignore_index=True)
    return asr_df, usage_df


def plot_asr(asr_df):
    for dataset in asr_df.dataset.unique():
        out_dir = os.path.join(OUTPUT_DIR, dataset)
        asr_dataset_df = asr_df[asr_df["dataset"] == dataset]

        numeric_columns = asr_dataset_df.select_dtypes(include=["number"]).columns
        asr_total = asr_dataset_df.groupby(["model_name", "defence_method", "attack_method", ])[numeric_columns].mean()
        # print(asr_total)
        asr_avg = asr_total.groupby(["model_name", "defence_method"]).mean()
        # print(asr_avg)

        for model in asr_dataset_df["model_name"].unique():
            df_pivoted = asr_total.loc[model].pivot_table(
                index='defence_method', columns='attack_method',
                values=['GCG', 'PAIR_Qwen_Qwen2.5-72B-Instruct', 'PAIR_gpt-4o-2024-11-20'])

            attack_rename = {
                'Goal': 'None',
                'AIM': 'AIM',
                'DAN': 'BetterDAN',
                'gcg': 'GCG',
                # 'ica': 'ICA',
                'future': 'Future',
                'tense': 'Past',
                'pair': 'PAIR',
                'search': 'RanSearch',
                'Crescendo': 'Crescendo',
                'Actor': 'Actor',
                # 'XTeaming': 'X-Teaming'
            }
            defense_rename = {
                'Icl': 'ICD',
                'Tom': 'BIID(Ours)',
                'SemanticSmoothLLM': 'SemanticSmooth',
            }
            df_pivoted = df_pivoted.rename(columns=attack_rename, index=defense_rename)

            col_order = ['None', 'AIM', 'BetterDAN', 'GCG', 'ICA', 'Future', 'Past', 'PAIR', 'RanSearch', 'Crescendo', 'Actor']
            row_order = ['None', 'RPO', 'ICD', 'Paraphrase', 'SelfReminder', 'SelfDefense', 'SmoothLLM', 'SemanticSmooth', 'BIID(Ours)']

            df_heatmap = df_pivoted['GCG'].reindex(columns=col_order, index=row_order)
            plt.figure(figsize=(8, 8))
            sns.heatmap(
                data=df_heatmap,
                cmap='Blues',
                annot=True,
                fmt=".1f",
                linewidths=0.5,
            )
            plt.ylabel("Defence Method")
            plt.xlabel("Attack Method")
            plt.title(f'{model} ASR by GCG Judge')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{model}_asr_by_gcg_judge.png"))
            plt.close()
            # plt.show()

            df_heatmap = df_pivoted['PAIR_Qwen_Qwen2.5-72B-Instruct'].reindex(columns=col_order, index=row_order)
            plt.figure(figsize=(8, 8))
            sns.heatmap(
                data=df_heatmap,
                cmap='Blues',
                annot=True,
                fmt=".1f",
                linewidths=0.5,
            )
            plt.ylabel("Defence Method")
            plt.xlabel("Attack Method")
            plt.title(f'{model} ASR by Pair Judge Qwen2.5-72B')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{model}_asr_by_pair_qwen_judge.png"))
            plt.close()
            # plt.show()

            df_heatmap = df_pivoted['PAIR_gpt-4o-2024-11-20'].reindex(columns=col_order, index=row_order)
            plt.figure(figsize=(8, 8))
            sns.heatmap(
                data=df_heatmap,
                cmap='Blues',
                annot=True,
                fmt=".1f",
                linewidths=0.5,
            )
            plt.ylabel("Defence Method")
            plt.xlabel("Attack Method")
            plt.title(f'{model} ASR by Pair Judge GPT-4o-2024-11-20')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{model}_asr_by_pair_gpt4o_judge.png"))
            plt.close()

            df_print = df_heatmap.drop(columns=['None'])
            print('----------------')
            print(dataset, model)
            print('----------------')
            for row in df_print.itertuples(index=True):
                print('& ' + ' & '.join(str(cell) for cell in row) + r' \\')
            print('=================')
            print()
            print()

            df_melted = asr_df[asr_df["model_name"] == model].melt(
                id_vars=['defence_method'],
                value_vars=['PAIR_Qwen_Qwen2.5-72B-Instruct', 'PAIR_gpt-4o-2024-11-20'],
                var_name='Metric',
                value_name='Value').sort_values('defence_method')
            # print(df_melted)
            plt.figure(figsize=(8, 6))
            sns.barplot(data=df_melted, x='defence_method', y='Value', hue='Metric', palette="Blues")
            plt.ylabel("Value")
            plt.xlabel("Defence Method")
            plt.title(f'{model} average ASR')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{model}_average_asr.png"))
            plt.close()
            # plt.show()


def plot_usage(usage_df):
    for dataset in usage_df.dataset.unique():
        out_dir = os.path.join(OUTPUT_DIR, dataset)
        usage_df = usage_df[usage_df["dataset"] == dataset]

        usage_df = usage_df[usage_df["attack_method"] != 'Actor']
        usage_df = usage_df[usage_df["attack_method"] != 'Crescendo']

        numeric_columns = usage_df.select_dtypes(include=["number"]).columns
        usage_avg = usage_df.groupby(["model_name", "defence_method", "attack_method", ])[numeric_columns].mean()
        # print(usage_avg)
        usage_avg_across_attack = usage_avg.groupby(["model_name", "defence_method"]).mean()
        # print(usage_avg_across_attack)

        for model in usage_df["model_name"].unique():
            df_pivoted = usage_avg.loc[model].pivot_table(
                index='defence_method', columns='attack_method',
                values=['prompt_tokens', 'completion_tokens'])
            # none_row = df_pivoted['prompt_tokens'].loc['None'].copy()
            # for _, row in df_pivoted['prompt_tokens'].iterrows():
            #     row -= none_row

            plt.figure(figsize=(12, 6))
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
            plt.savefig(os.path.join(out_dir, f"{model}_prompt_tokens.png"))
            plt.close()
            # plt.show()

            plt.figure(figsize=(12, 6))
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
            plt.savefig(os.path.join(out_dir, f"{model}_completion_tokens.png"))
            plt.close()
            # plt.show()

            df_melted = usage_df[usage_df["model_name"] == model].melt(
                id_vars=['defence_method'],
                value_vars=['prompt_tokens', 'completion_tokens'],
                var_name='Metric',
                value_name='Value').sort_values('defence_method')
            # print(df_melted)
            plt.figure(figsize=(8, 6))
            sns.barplot(data=df_melted, x='defence_method', y='Value', hue='Metric', palette="Blues")
            plt.ylabel("Value")
            plt.xlabel("Defence Method")
            plt.title(f'{model} average usage')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{model}_average_usage.png"))
            plt.close()
            # plt.show()


def tom_stage_analysis(dataset_dir):
    stage_df = pd.DataFrame(
        columns=[
            "model_name",
            "attack_method",
            "defence_method",
            "prompt",
            "response",
            "undetected_safe",
            "undetected_danger"
        ]
    )
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".json"):
                model_name = root.split("/")[-3]
                attack_method = root.split("/")[-2]
                attack_method = attack_method.split("_")[-1]
                defence_method = root.split("/")[-1].removesuffix('Defender')
                if (
                        # model_name == 'meta-llama_Llama-3.1-8B-Instruct'
                        # and attack_method != 'Crescendo'
                        # and defence_method == 'Tom'
                        defence_method == 'Tom'
                ):
                    with open(os.path.join(root, file), "r") as f:
                        results = json.load(f)["results"]

                    stage = {'prompt': 0, 'response': 0, 'undetected_safe': 0, 'undetected_danger': 0}

                    for res in results:
                        goal = res["goal"]
                        try:
                            response = res['data'][0]['messages'][-1]['content']
                            jailbroken = res["jailbroken"]
                        except Exception:
                            s = 'prompt'
                        try:
                            s = re.search(r'unsafe request inferred from (.*?)\.', response).group(1)
                            assert s in ('prompt', 'response')
                        except AttributeError:
                            if jailbroken['PAIR_gpt-4o-2024-11-20'] == 10:
                                s = "undetected_danger"
                            else:
                                s = "undetected_safe"
                        stage[s] += 1

                    stage_new_row = pd.DataFrame([{
                        "model_name": model_name,
                        "attack_method": attack_method,
                        "defence_method": defence_method,
                        **stage
                    }])
                    stage_df = pd.concat([stage_df, stage_new_row], ignore_index=True)
    return stage_df


# def plot_stage(stage_df, dataset_dir):
#     dataset = dataset_dir.split("/")[-1].removesuffix('_judged')
#     out_dir = os.path.join(OUTPUT_DIR, dataset)
#
#     usage_total = stage_df.groupby(["model_name", "defence_method", "attack_method", ])[
#         ['prompt', 'response', 'undetected_safe', 'undetected_danger']].sum()
#     # print(usage_total)
#     for model in stage_df["model_name"].unique():
#         for defence_method in stage_df["defence_method"].unique():
#             usage_model = usage_total.loc[model, defence_method].reset_index()
#             # print(usage_model)
#             pie_num = usage_model.shape[0]
#             colors = sns.color_palette("Blues", 4)
#             plt.figure(figsize=(40, 6))
#             # plt.title(f"{model} Tom Stage Analysis")
#             for i, row in usage_model.iterrows():
#                 plt.subplot(1, pie_num, i + 1)
#                 plt.pie([row['prompt'], row['response'], row['undetected_safe'], row['undetected_danger']],
#                         colors=colors, autopct='%1.1f%%', pctdistance=1.2, )
#                 plt.title(row['attack_method'], y=-0.2)
#             plt.legend(labels=['prompt', 'response', 'undetected_safe', 'undetected_danger'], loc='lower right')
#             plt.suptitle(f"{model} {defence_method} Stage Analysis")
#             plt.tight_layout()
#             plt.savefig(os.path.join(out_dir, f"{model}_{defence_method}_stage.png"))
#             plt.close()
#             # plt.show()
#     # for model in stage_df["model_name"].unique():
#     #     model_df = stage_df[stage_df["model_name"] == model]
#     #     pie_num = len(stage_df["attack_method"].unique())
#     #     plt.figure(figsize=(30, 4))
#     #     for i, attack in enumerate(stage_df["attack_method"].unique()):
#     #         attack_df = model_df[model_df["attack_method"] == attack]
#     #         plt.subplot(1, pie_num, i + 1)
#     #         df_melted = attack_df.melt(
#     #             id_vars='jailbreak_type',
#     #             value_vars=['prompt', 'response', 'undetected'],
#     #             var_name='Stage',
#     #             value_name='Value')
#     #         sns.barplot(x='Value', y='jailbreak_type', hue='Stage', data=df_melted, orient='h', ci=None)
#     #     plt.tight_layout
#     #     plt.show()

def plot_stage(stage_df, dataset_dir):
    dataset = dataset_dir.split("/")[-1].removesuffix('_judged')
    out_dir = os.path.join(OUTPUT_DIR, dataset)
    os.makedirs(out_dir, exist_ok=True)

    # 数据处理
    usage_total = stage_df.groupby(["model_name", "defence_method", "attack_method"])[
        ['prompt', 'response', 'undetected_safe', 'undetected_danger']].sum()

    # 创建标题映射器和排序序列
    caster = {
        "AIM": "AIM", "DAN": "BetterDAN", "gcg": "GCG", "future": "Future",
        "tense": "Past", "pair": "PAIR", "search": "RanSearch", "crescendo": "Crescendo",
        "Actor": "Actor", "XTeaming": "X-Teaming", "Goal": "None"
    }
    sequence = ["None", "AIM", "BetterDAN", "GCG", "ICA", "Future", "Past", "PAIR", "RanSearch", "Crescendo", "Actor"]
    sequence = reversed(sequence)

    # 创建排序索引
    sequence_order = {key: idx for idx, key in enumerate(sequence)}

    for model in stage_df["model_name"].unique():
        for defence_method in stage_df["defence_method"].unique():
            if (model, defence_method) not in usage_total.index:
                continue
            if model == "meta-llama_Meta-Llama-3-8B-Instruct":
                continue

            usage_model = usage_total.loc[model, defence_method].reset_index()

            # 应用标题映射并添加排序依据
            attack_col = usage_model.columns[0]  # 获取攻击方法所在的列名
            usage_model['display_name'] = usage_model[attack_col].apply(lambda x: caster.get(x, x))
            usage_model['sort_idx'] = usage_model['display_name'].apply(
                lambda x: sequence_order.get(x, len(sequence_order))
            )
            usage_model = usage_model.sort_values('sort_idx')
            usage_model = usage_model[usage_model['attack_method'] != 'ica']

            usage_model = usage_model[~usage_model['display_name'].str.lower().str.contains('x')]

            # 准备堆叠条形图数据
            categories = usage_model['display_name'].to_list()
            prompt = usage_model['prompt'].to_list()
            response = usage_model['response'].to_list()
            undetected_safe = usage_model['undetected_safe'].to_list()
            undetected_danger = usage_model['undetected_danger'].to_list()

            # 创建图形和坐标轴
            fig, ax = plt.subplots(figsize=(12, 7.5))

            # 设置颜色方案和标签
            colors = sns.color_palette("PuOr_r", 4)
            labels = ['Forward', 'Backward', 'Undetected Safe', 'Undetected Harmful']

            # 绘制堆叠条形图（水平方向）
            bottom = np.zeros(len(categories))

            # 第一层：Prompt
            ax.barh(categories, prompt, height=0.6, color=colors[0],
                    label=labels[0], edgecolor='white', linewidth=0.5)

            # 第二层：Response（堆叠在Prompt上）
            bottom += prompt
            ax.barh(categories, response, left=bottom, height=0.6, color=colors[1],
                    label=labels[1], edgecolor='white', linewidth=0.5)

            # 第三层：Undetected safe
            bottom += response
            ax.barh(categories, undetected_safe, left=bottom, height=0.6, color=colors[2],
                    label=labels[2], edgecolor='white', linewidth=0.5)

            # 第四层：Undetected danger
            bottom += undetected_safe
            ax.barh(categories, undetected_danger, left=bottom, height=0.6, color=colors[3],
                    label=labels[3], edgecolor='white', linewidth=0.5)

            # 添加标签和标题
            # ax.set_title(f"{model} - {defence_method}", fontsize=14)
            # ax.set_xlabel('Percentage')
            # ax.set_ylabel('Attack Methods')
            ax.tick_params(
                axis='both',  # 同时修改x和y轴
                which='both',  # 修改主刻度和次刻度
                top=False,  # 不显示上侧的刻度
                right=False,  # 不显示右侧的刻度
                labeltop=False,  # 不显示上侧的刻度标签
                labelright=False,  # 不显示右侧的刻度标签
                # bottom=False,          # 不显示上侧的刻度
                # left=False,        # 不显示右侧的刻度
                # labelbottom=False,     # 不显示上侧的刻度标签
                # labelleft=False    # 不显示右侧的刻度标签
                labelsize=20
            )

            # 添加数值标签（可选）
            # for i, total in enumerate(bottom + undetected_danger):
            #     ax.text(total * 1.01, i, f"{int(total)}",
            #             va='center', fontsize=8)

            model_title = {
                "meta-llama_Llama-3.1-8B-Instruct": "Llama-3.1-8B",
                "meta-llama_Llama-3.3-70B-Instruct": "Llama-3.3-70B",
                "Qwen_Qwen3-8B": "Qwen3-8B"
            }
            dataset_title = {
                "jbb": "JailbreakBench",
                "hm": "HarmBench"
            }

            plt.title(f"{model_title[model]} on {dataset_title[dataset]}", fontsize=18)


            # 添加图例
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # ax.tick_params(font_size=40)
            # ax.spines['bottom'].set_visible(False)
            # ax.spines['left'].set_visible(False)
            if dataset == "jbb":
                plt.xlim(0, 105)
            elif dataset == "hm":
                plt.xlim(0, 210)
            # ax.legend(loc='lower right', ncol=4, fontsize=11, )
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=17)
            # 调整布局
            plt.tight_layout()

            # 保存图形
            plt.savefig(
                os.path.join(out_dir, f"{dataset}_{model}_stage.png"),
                dpi=300,
                bbox_inches='tight',
            )
            plt.close()


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
                # print(result)
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
    # asr_df = asr_df[asr_df['attack_method'] != 'XTeaming']
    for dataset in asr_df.dataset.unique():
        out_dir = os.path.join(OUTPUT_DIR, dataset)
        asr_dataset_df = asr_df[asr_df["dataset"] == dataset]

        # print(asr_df)
        asr_avg = asr_dataset_df.groupby(['model_name', 'defence_method'])['PAIR_gpt-4o-2024-11-20'].mean().reset_index()
        # print(asr_avg)
        # print(alpaca_df)
        for model in alpaca_df["model_name"].unique():
            asr_alpaca_df = pd.DataFrame(columns=["defence_method", "asr", "win_rate"])

            asrs = asr_avg[asr_avg['model_name'] == model]
            win_rates = alpaca_df[alpaca_df['model_name'] == model]
            for defence_method in alpaca_df["defence_method"].unique():
                try:
                    defence_asr = asrs[asrs['defence_method'] == defence_method]['PAIR_gpt-4o-2024-11-20'].values[0]
                    defence_win_rate = win_rates[win_rates['defence_method'] == defence_method]['win_rate'].values[0]
                    new_row = pd.DataFrame([{
                        "defence_method": defence_method,
                        "asr": defence_asr,
                        "win_rate": defence_win_rate,
                    }])
                    asr_alpaca_df = pd.concat([asr_alpaca_df, new_row], ignore_index=True)
                except Exception as e:
                    print(e)
                    continue
            asr_none = asrs[asrs['defence_method'] == 'None']['PAIR_gpt-4o-2024-11-20'].values[0]
            new_row = pd.DataFrame([{
                "defence_method": 'None(Baseline)',
                "asr": asr_none,
                "win_rate": 50.0,
            }])
            asr_alpaca_df = pd.concat([asr_alpaca_df, new_row], ignore_index=True)
            # print(asr_alpaca_df)

            asr_alpaca_df.loc[asr_alpaca_df['defence_method'] == 'Icl', 'defence_method'] = 'ICD'
            asr_alpaca_df.loc[asr_alpaca_df['defence_method'] == 'Tom', 'defence_method'] = 'BIID(Ours)'

            plt.figure(figsize=(6, 4))
            sns.set(style="whitegrid")

            # 绘制散点图
            palette = sns.color_palette("Accent", len(asr_alpaca_df) - 1)
            wo_ours = asr_alpaca_df[asr_alpaca_df['defence_method'] != 'BIID(Ours)']
            plt.scatter(
                wo_ours["asr"], wo_ours["win_rate"],
                s=100, c=palette, edgecolors='black',
            )
            ours = asr_alpaca_df[asr_alpaca_df['defence_method'] == 'BIID(Ours)']
            plt.scatter(
                ours["asr"], ours["win_rate"],
                s=180, c='red', edgecolors='black', marker='*'
            )

            # 添加标签
            texts = []
            for i in range(len(asr_alpaca_df)):
                texts.append(plt.text(
                    asr_alpaca_df["asr"][i],
                    asr_alpaca_df["win_rate"][i] + 1,
                    asr_alpaca_df["defence_method"][i],
                    fontsize=12
                ))
            adjust_text(texts,
                        # arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),  # 启用连接线
                        force_text=(0.9, 0.9),  # 增强文本间排斥力
                        # force_points=(0.9, 0.9),  # 增强点到文本吸引力
                        expand_points=(3.0, 3.0),  # 扩大禁区范围
                        # expand_text=(1.2, 1.2),  # 新增：文本边框扩展
                        only_move={'points': 'y', 'text': 'xy'},  # 限制文本移动方向
                        avoid_points=True,  # 新增：强制避开数据点
                        lim=200  # 新增：最大迭代次数
                        )

            # 设置标题和轴

            model_title = {
                "meta-llama_Llama-3.1-8B-Instruct": "Llama-3.1-8B",
                "meta-llama_Llama-3.3-70B-Instruct": "Llama-3.3-70B",
                "Qwen_Qwen3-8B": "Qwen3-8B"
            }
            dataset_title = {
                "jbb": "JailbreakBench",
                "hm": "HarmBench"
            }

            plt.title(f"{model_title[model]} on {dataset_title[dataset]}", fontsize=14)
            plt.xlabel("ASR (↓ Attack Success Rate)", fontsize=12)
            plt.ylabel("Win Rate (↑ AlpacaEval)", fontsize=12)
            plt.xlim(0., max(asr_alpaca_df["asr"]) + 1)
            plt.ylim(0, 60)
            plt.grid(True, linestyle='--', alpha=0.6)

            plt.axvspan(0, 10, alpha=0.1, color=(0, 0.6, 1), label='Ideal Region')
            plt.axhspan(40, 60, alpha=0.1, color=(0, 0.6, 1))

            plt.tight_layout()

            # 显示图形
            plt.savefig(os.path.join(out_dir, f"{dataset}_{model}_asr_winrate.png"), dpi=300, bbox_inches='tight')
            # plt.show()


def plot_qwen_asr_heatmap(asr_df):
    llama_jbb = asr_df[(asr_df['dataset'] == 'jbb') & (asr_df['model_name'] == 'meta-llama_Llama-3.1-8B-Instruct')]
    llama_hm = asr_df[(asr_df['dataset'] == 'hm') & (asr_df['model_name'] == 'meta-llama_Llama-3.1-8B-Instruct')]
    qwen_jbb = asr_df[(asr_df['dataset'] == 'jbb') & (asr_df['model_name'] == 'Qwen_Qwen3-8B')]
    qwen_hm = asr_df[(asr_df['dataset'] == 'hm') & (asr_df['model_name'] == 'Qwen_Qwen3-8B')]

    dfs = {
        'llama_jbb': llama_jbb,
        'llama_hm': llama_hm,
        'qwen_jbb': qwen_jbb,
        'qwen_hm': qwen_hm,
    }

    def precess_df(df):
        df = df.pivot_table(index='defence_method', columns='attack_method', values='PAIR_gpt-4o-2024-11-20')

        attack_rename = {
            'Goal': 'None',
            'AIM': 'AIM',
            'DAN': 'BetterDAN',
            'gcg': 'GCG',
            # 'ica': 'ICA',
            'future': 'Future',
            'tense': 'Past',
            'pair': 'PAIR',
            'search': 'RanSearch',
            'Crescendo': 'Crescendo',
            'Actor': 'Actor',
            # 'XTeaming': 'X-Teaming'
        }
        defense_rename = {
            'Icl': 'ICD',
            'Tom': 'BIID(Ours)',
            'SemanticSmoothLLM': 'SemanticSmooth',
        }
        df = df.rename(columns=attack_rename, index=defense_rename)

        col_order = ['None', 'AIM', 'BetterDAN', 'GCG', 'ICA', 'Future', 'Past', 'PAIR', 'RanSearch', 'Crescendo', 'Actor']
        row_order = ['None', 'RPO', 'ICD', 'Paraphrase', 'SelfReminder', 'SelfDefense', 'SmoothLLM', 'SemanticSmooth', 'BIID(Ours)']

        df = df.reindex(columns=col_order, index=row_order)

        return df

    dfs = {k: precess_df(v) for k, v in dfs.items()}

    plt.figure(figsize=(12, 9))

    # 使用GridSpec创建复杂布局
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.08], height_ratios=[1, 1])

    # 定义子图位置
    axes = [
        plt.subplot(gs[0, 0]),  # llama_jbb (左上)
        plt.subplot(gs[0, 1]),  # llama_hm (右上)
        plt.subplot(gs[1, 0]),  # qwen_jbb (左下)
        plt.subplot(gs[1, 1]),  # qwen_hm (右下)
    ]

    # 创建共享颜色条位置
    cbar_ax = plt.subplot(gs[:, 2])  # 最右侧一列整个高度

    # 计算统一颜色范围
    vmin = 0
    vmax = 100

    # 设置共享标签
    shared_xlabel = "Attack Methods"
    shared_ylabel = "Defense Methods"

    idx2key = ['llama_jbb', 'llama_hm', 'qwen_jbb', 'qwen_hm']
    titles = ['Llama-3.1-8B on JailBreakBench', 'Llama-3.1-8B on HarmBench', 'Qwen3-8B on JailBreakBench', 'Qwen3-8B on HarmBench']

    for i, ax in enumerate(axes):
        key = idx2key[i]
        sns.heatmap(
            dfs[key],
            ax=ax,
            cmap='Blues',
            annot=True,
            annot_kws={"size": 16},
            cbar=i == 0,  # 只在第一个热力图创建颜色条
            cbar_ax=cbar_ax if i == 0 else None,  # 指定颜色条位置
            vmin=vmin,
            vmax=vmax
        )

        # 设置子图标题
        ax.set_title(titles[i], fontsize=14)

        ax.tick_params(axis='x', labelsize=14)  # 设置x轴刻度标签大小
        ax.tick_params(axis='y', labelsize=14)  # 设置y轴刻度标签大小

        # 控制标签显示
        if i < 2:  # 上面两个子图
            ax.set_xlabel('')
            ax.set_xticklabels([])
            ax.tick_params(axis='x', length=0)  # 隐藏x轴刻度线
        else:  # 下面两个子图
            ax.set_xlabel(shared_xlabel, fontsize=12)

        if i % 2 == 1:  # 右侧两个子图
            ax.set_ylabel('')
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)  # 隐藏y轴刻度线
        else:  # 左侧两个子图
            ax.set_ylabel(shared_ylabel, fontsize=12)

    # 旋转x轴标签（仅底部子图）
    for ax in axes[2:]:
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # 添加颜色条标签
    cbar = cbar_ax.collections[0].colorbar
    # cbar.set_label('Attack Success Rate (ASR)', fontsize=10)

    plt.tight_layout()  # 为总标题留出空间

    # 保存图像（可选）
    plt.savefig(os.path.join(OUTPUT_DIR, 'qwen_llama_heatmap.png'), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    # asr_df, usage_df = get_data_df()
    # plot_asr(asr_df)
    # plot_usage(usage_df)
    #
    # stage_df = tom_stage_analysis(JBB_DIR)
    # plot_stage(stage_df, JBB_DIR)

    stage_df = tom_stage_analysis(HM_DIR)
    plot_stage(stage_df, HM_DIR)
    #
    # alpaca_df = alpaca_analysis()
    # plot_alpaca_asr(asr_df, alpaca_df)
    # plot_qwen_asr_heatmap(asr_df)
