### exp 1_base
1. test base model (configs/defenses/ai4sci_llm/base_models)
2. dataset: data/ai4sci_paper/20260423.csv 


### exp 1_base_judge 
1. judge for 1_base
2. need re-write cls to include risk type

### exp 1_sci
1. test sci model (configs/defenses/ai4sci_llm/sci_llm)
2. dataset: data/ai4sci_paper/20260423.csv

### exp 1_isolation (waiting)
1. test isolation model (configs/defenses/ai4sci_llm/ isolation)
2. dataset: data/ai4sci_paper/20260423.csv 

### exp 2_base
1. same as exp 1_base, but use template attack method
2. use data: data/ai4sci_paper/20260423_expanded.csv

### exp 2_sci
1. same as exp 1_sci, but use template attack method
2. use data: data/ai4sci_paper/20260423_expanded.csv
