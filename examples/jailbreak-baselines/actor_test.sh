#!/bin/bash
echo "==== 任务开始: $(date) ===="
python jbb_inference.py --output-dir ../../benchmarks/jbb --config ../../configs/tasks/jbb.yaml --attack ../../configs/attacks/actor.yaml --defense ../../configs/defenses/none.yaml --llm ../../configs/defenses/llms/vllm_llama-3.1-8b-it-on12.yaml --visible > log1.txt &
python jbb_inference.py --output-dir ../../benchmarks/jbb --config ../../configs/tasks/jbb.yaml --attack ../../configs/attacks/actor.yaml --defense ../../configs/defenses/tom.yaml --llm ../../configs/defenses/llms/vllm_llama-3.1-8b-it-on12.yaml --visible > log2.txt &
python jbb_inference.py --output-dir ../../benchmarks/hm --config ../../configs/tasks/harmbench.yaml --attack ../../configs/attacks/actor_hm.yaml --defense ../../configs/defenses/none.yaml --llm ../../configs/defenses/llms/vllm_llama-3.1-8b-it-on12.yaml --visible > log3.txt &
python jbb_inference.py --output-dir ../../benchmarks/hm --config ../../configs/tasks/harmbench.yaml --attack ../../configs/attacks/actor_hm.yaml --defense ../../configs/defenses/tom.yaml --llm ../../configs/defenses/llms/vllm_llama-3.1-8b-it-on12.yaml --visible > log4.txt &

wait
echo "==== 任务完成: $(date) ===="
echo "finish"
