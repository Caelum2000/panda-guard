python run_all_inference.py \
--output-dir ../../benchmarks/sosbench \
--config ../../configs/tasks/sosbench.yaml \
--attack ../../configs/attacks/transfer/original.yaml \
--defense ../../configs/defenses/none.yaml \
--llm ../../configs/defenses/sosbench_llms/unfinished_3 \
--max-parallel 16 \
--log-level INFO
