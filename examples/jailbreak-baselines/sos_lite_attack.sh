python run_all_inference.py \
--output-dir ../../benchmarks/sos_lite \
--config ../../configs/tasks/sos_lite.yaml \
--attack ../../configs/attacks/sos_lite/unfinished \
--defense ../../configs/defenses/none.yaml \
--llm ../../configs/defenses/sosbench_llms/finished/s1-base-lite.yaml \
--max-parallel 16 \
--log-level INFO

python jbb_inference.py \
--output-dir ../../benchmarks/sos_lite \
--config ../../configs/tasks/sos_lite.yaml \
--attack ../../configs/attacks/sos_lite/unfinished/gcg.yaml \
--defense ../../configs/defenses/none.yaml \
--llm ../../configs/defenses/sosbench_llms/finished/s1-base-lite.yaml \
--log-level INFO \
--visible

python jbb_inference.py \
--output-dir ../../benchmarks/sos_lite \
--config ../../configs/tasks/sos_lite.yaml \
--attack ../../configs/attacks/sos_lite/unfinished/pair.yaml \
--defense ../../configs/defenses/none.yaml \
--llm ../../configs/defenses/sosbench_llms/finished/s1-base-lite.yaml \
--log-level INFO \
--visible
