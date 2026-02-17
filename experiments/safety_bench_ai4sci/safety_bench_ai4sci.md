## 前瞻安全框架AI4SCI测试

### exp 1
1. 初步测试，2026年1月4号
2. 包含8个模型，无攻防
3. 测试数据为 data/safetybench_ai4sci/safetybench_ai4sci.csv

### exp 2
1. 2026.1.9
2. 扩充原始数据集到每个子维度15条
3. 测试数据为 data/safetybench_ai4sci/safetybench_ai4sci_20250109.csv

### exp 3
1. 2026.1.13
2. 加上了模板攻击方法+无攻防，新增了一些模型


### exp 4
1. 2026.1.23
2. 使用sos_lite进行测评，加了模板攻击，文件为 data/safetybench_ai4sci/sos_lite_20260123_expanded.csv
3. 增加了模型数量 (先不测gpt+grok)，在exp 4_1测试
4. 为了与自己造的数据集进行合并

### exp 4_1
1. 在exp 4的基础上，测试grok+gpt

### exp 5
1. 2026.1.23
2. 使用自己构建的初版ai4sci_safebench进行测评，加了模板攻击，文件为data/safetybench_ai4sci/ai4sci_safebench_20260123_expanded.csv
3. 增加了模型数量 (先不测gpt+grok)，在exp 5_1测试
4. 为了与sos_lite进行合并
