#!/bin/bash

CONFIG_BASE=../../configs
ATTACK_DIR=$CONFIG_BASE/attacks/test
DEFENSE_DIR=$CONFIG_BASE/defenses/test
LLM_DIR=$CONFIG_BASE/defenses/llms/extend
TASK_CONFIG=$CONFIG_BASE/tasks/jbb.yaml

llms=(
  "$LLM_DIR/vllm_zephyr_7b_r2d2.yaml"
)

attacks=(
  "$ATTACK_DIR/new_art_prompt.yaml"
  "$ATTACK_DIR/new_scav.yaml"
  "$ATTACK_DIR/original.yaml"
  "$ATTACK_DIR/new_gcg.yaml"
  "$ATTACK_DIR/new_ica.yaml"
  "$ATTACK_DIR/new_cold.yaml"
  "$ATTACK_DIR/new_gpt4_cipher.yaml"
  "$ATTACK_DIR/new_pair.yaml"
  "$ATTACK_DIR/new_deepinception.yaml"
  "$ATTACK_DIR/new_tap.yaml"
  "$ATTACK_DIR/dev_mode_v2.yaml"
  "$ATTACK_DIR/future.yaml"
  "$ATTACK_DIR/new_gptfuzzer.yaml"
  "$ATTACK_DIR/dev_mode_ranti.yaml"
  "$ATTACK_DIR/better_dan.yaml"
  "$ATTACK_DIR/new_renellm.yaml"
  "$ATTACK_DIR/anti_gpt_v2.yaml"
  "$ATTACK_DIR/new_autodan.yaml"
  "$ATTACK_DIR/past.yaml"
  "$ATTACK_DIR/aim.yaml"
  "$ATTACK_DIR/new_random_search.yaml"
)

defenses=(
  "$DEFENSE_DIR/paraphrase.yaml"
  "$DEFENSE_DIR/semantic_smoothllm.yaml"
  "$DEFENSE_DIR/self_defense.yaml"
  "$DEFENSE_DIR/goal_priority.yaml"
  "$DEFENSE_DIR/self_reminder.yaml"
  "$DEFENSE_DIR/smoothllm.yaml"
  "$DEFENSE_DIR/icl.yaml"
  "$DEFENSE_DIR/rpo.yaml"
  "$DEFENSE_DIR/perplexity_filter.yaml"
  "$DEFENSE_DIR/none.yaml"
)

# 执行循环
for llm in "${llms[@]}"
do
  for attack in "${attacks[@]}"
  do
    for defense in "${defenses[@]}"
    do
      echo "tested llm: $llm"
      echo "Running attack: $attack"
      echo "With defense:  $defense"
      echo "--------------------------------------"

      python jbb_inference.py --config "$TASK_CONFIG" --attack "$attack" --defense "$defense" --llm "$llm" --visible >> infer_logs.txt

      if [ $? -eq 0 ]; then
        echo "Finished: attack=$attack, defense=$defense"
      else
        echo "Failed: attack=$attack, defense=$defense"
      fi

      echo "======================================"
      echo
    done
  done
done
