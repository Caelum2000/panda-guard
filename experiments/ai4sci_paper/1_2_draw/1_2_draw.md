# Background
The `others/` directory contains the Attack Success Rate (ASR) results of different LLMs evaluated under jailbreak attack and defense frameworks. Among them, `1_base_eval` corresponds to the base models, while `1_sci_eval` contains the results of LLMs fine-tuned for scientific tasks. The evaluation dataset covers multiple scientific domains, where each domain further consists of several sub-disciplines and sub-dimensions. I now need you to analyze these ASR evaluation results and generate corresponding visualizations and figures that conform to the plotting and presentation standards commonly used in top-tier academic journals and conferences.

## Requirement
You should implement all data analysis and visualization code in `experiments/ai4sci_paper/1_draw`. The YAML files are used to store detailed configuration settings, while the shell (`.sh`) scripts are used to launch and execute the corresponding code.

### Task-1
You need to plot the ASR variation with respect to model size for each model. The model size information is stored in `others/model_size.csv`. The figure should be a scatter plot, where the y-axis represents ASR and the x-axis represents the number of model parameters. Each data point corresponds to an LLM listed in `others/model_size.csv`.

### Task-2
You need to plot the ASR variation with respect to the model release date for each model. The release date information of the models is stored in `others/Models_Merged.csv`. The figure should be a scatter plot, where the y-axis represents ASR and the x-axis represents the model release date. Each data point corresponds to an LLM listed in `others/Models_Merged.csv`.

### Task-3
You need to generate heatmaps and radar charts for the combinations of different sub-disciplines and different LLMs.

### Task-4
You need to generate heatmaps and radar charts for the combinations of different risk dimensions and different LLMs.



