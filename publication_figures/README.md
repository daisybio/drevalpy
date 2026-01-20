## How to reproduce which Figure/Table?

### Main text

- Figure 2 (dumbbell plot model performances): [main_results_dumbbellplots.ipynb](main_results_dumbbellplots.ipynb)
- Table 1 (LPO R^2 for naive predictors): [main_results_dumbbellplots.ipynb](main_results_dumbbellplots.ipynb)
- Figure 3 (Simpson's paradox DIPK): [simpsons.ipynb](simpsons.ipynb)
- Figure 4 (critical difference diagram LDO): [cd_diagrams.ipynb](cd_diagrams.ipynb)
- Figure 5 (cross-study generalization): [cross_study_visualization.ipynb](cross_study_visualization.ipynb)
- Figure 6 (ablation): [ablation_difference.R](ablation_difference.R)

### Appendix

Figures

- Figure S1 (Simpson's paradox example): [simpsons_paradox_simple.ipynb](simpsons_paradox_simple.ipynb)
- Figure S3 (UMAP drug embeddings): Code for UMAP generation in [make_umap_embedding.py](make_umap_embedding.py). To just render the plot, run [make_umap_embedding_plot_highlight.py](make_umap_embedding_plot_highlight.py)
- Figure S4 (comparison response measures original vs. CurveCurator ): [curvecurator_comparison.R](curvecurator_comparison.R)
- Figure S5 (critical difference diagrams LPO, LCO, LTO): [cd_diagrams.ipynb](cd_diagrams.ipynb)
- Figure S6 (Simpson's paradox DIPK): [simpsons.ipynb](simpsons.ipynb); simply add LDO
- Figure S7 (expression distributions): [distribution_plots.R](distribution_plots.R)
- Figure S8 (distribution of measured concentrations): [distribution_plots.R](distribution_plots.R)
- Figure S9 (distribution of response values): [distribution_plots.R](distribution_plots.R)

Tables

- Table S1 (LDO input ablation): [sup_tables_performance_results.R](sup_tables_performance_results.R)
- Table S2 (basic statistics drug screens): [distribution_plots.R](distribution_plots.R)
- Table S3 (coverage omic screens): [omic_overlaps.R](omic_overlaps.R)
- Table S5, S6 (lnIC50 prediction results LPO, LCO, LTO, LDO; all models): [sup_tables_performance_results.R](sup_tables_performance_results.R)
- Table S7 (model runtime): can be inferred from the nextflow execution reports
- Table S8 (MSEs all cross-study predictions): [sup_tables_performance_results.R](sup_tables_performance_results.R)
- Table S9 (cross-study predictions with other metrics): [sup_tables_performance_results.R](sup_tables_performance_results.R)
- Table S10 (ablation study results): [sup_tables_performance_results.R](sup_tables_performance_results.R)
- Table S11 (proteomics random forest): [sup_tables_performance_results.R](sup_tables_performance_results.R); lines are part of the output of the big latex table (S6, S7)
- Table S13 (statistical details Friedman-Chi2 test): [cd_diagrams.ipynb](cd_diagrams.ipynb)
- Table S14-S17 (statistical details pairwise post-hoc conover tests): [cd_diagrams.ipynb](cd_diagrams.ipynb)
