library(data.table)
library(ggplot2)
library(RColorBrewer)
library(patchwork)
library(xtable)


# Load data
path_to_data <- 'results/'
zenodo_data <- 'data/'
path_to_invariant_results <- "results/invariant_run/"
path_to_permutation_results <- "results/permutation_run/"
# example here: LDO, same for LPO and LCO run
lpo_id <- 'lpo_run/'
lco_id <- 'lco_run/'
ldo_id <- 'ldo_run/'
# Load evaluation results
evaluation_results_lpo <- fread(paste0(path_to_data, lpo_id, 'evaluation_results.csv'))
evaluation_results_lco <- fread(paste0(path_to_data, lco_id, 'evaluation_results.csv'))
evaluation_results_ldo <- fread(paste0(path_to_data, ldo_id, 'evaluation_results.csv'))
evaluation_results <- rbind(evaluation_results_lpo, evaluation_results_lco, evaluation_results_ldo)
true_vs_pred_lpo <- fread(paste0(path_to_data, lpo_id, 'true_vs_pred.csv'))
true_vs_pred_lco <- fread(paste0(path_to_data, lco_id, 'true_vs_pred.csv'))
true_vs_pred_ldo <- fread(paste0(path_to_data, ldo_id, 'true_vs_pred.csv'))
true_vs_pred <- rbind(true_vs_pred_lpo, true_vs_pred_lco, true_vs_pred_ldo)

####### Code for Table 1 / S7: All results ####### 
all_results <- evaluation_results[rand_setting == "predictions"]
all_results <- melt(all_results, 
                    id.vars = c("algorithm", "LPO_LCO_LDO", "CV_split"),
                    measure.vars = c("MSE", "R^2", "Pearson", "R^2: normalized", "Pearson: normalized"),
                    variable.name = "metric")
# calculate mean/sd per algorithm, LPO_LCO_LDO, metric
all_results[, mean := round(mean(value), 2), by = c("algorithm", "LPO_LCO_LDO", "metric")]
all_results[, sd := round(sd(value), 2), by = c("algorithm", "LPO_LCO_LDO", "metric")]
all_results[, value := paste0(mean, "±", sd)]
all_results[, c("CV_split", "mean", "sd") := NULL]
all_results <- unique(all_results)
all_results <- dcast(all_results, algorithm + LPO_LCO_LDO ~ metric, value.var = "value")

# calculate pearson correlation coefficients per algorithm, LPO_LCO_LDO, drug: y_true vs y_pred
tp <- true_vs_pred[rand_setting == "predictions"]
pearson_per_drug <- tp[, .(Pearson = cor(y_true, y_pred), 2), by = c("algorithm", "LPO_LCO_LDO", "drug")]
pearson_per_cl <- tp[, .(Pearson = cor(y_true, y_pred), 2), by = c("algorithm", "LPO_LCO_LDO", "cell_line")]
# calculate mean and standard deviation of pearsons per algorithm and LPO_LCO_LDO
all_results_per_drug <- pearson_per_drug[, .(Pearson = round(mean(Pearson, na.rm=T), 2), sd = round(sd(Pearson, na.rm=T), 2)), by = c("algorithm", "LPO_LCO_LDO")]
all_results_per_cl <- pearson_per_cl[, .(Pearson = round(mean(Pearson, na.rm=T), 2), sd = round(sd(Pearson, na.rm=T), 2)), by = c("algorithm", "LPO_LCO_LDO")]
all_results_per_drug[, Pearson := paste0(Pearson, "±", sd)]
setnames(all_results_per_drug, "Pearson", "Pearson per drug")
all_results_per_cl[, Pearson := paste0(Pearson, "±", sd)]
setnames(all_results_per_cl, "Pearson", "Pearson per cell line")

all_results <- merge(all_results, all_results_per_drug[, c("algorithm", "LPO_LCO_LDO", "Pearson per drug")], by = c("algorithm", "LPO_LCO_LDO"), all = TRUE)
all_results <- merge(all_results, all_results_per_cl[, c("algorithm", "LPO_LCO_LDO", "Pearson per cell line")], by = c("algorithm", "LPO_LCO_LDO"), all = TRUE)
all_results[, LPO_LCO_LDO := factor(LPO_LCO_LDO, levels = c("LPO", "LCO", "LDO"))]
# order by MSE and by LPO_LCO_LDO
all_results <- all_results[order(LPO_LCO_LDO, MSE)]
setnames(all_results, "algorithm", "Model")
setnames(all_results, "R^2", "$R^2$")
setnames(all_results, "R^2: normalized", "$R^2$: normalized")
# reorder columns:
all_results <- all_results[, c("Model", "LPO_LCO_LDO", "MSE", "$R^2$", "$R^2$: normalized", "Pearson", "Pearson: normalized", "Pearson per drug", "Pearson per cell line")]
# replace all NA with ""
all_results[is.na(all_results)] <- ""
# to LaTeX table
# underline everything in the lines of the NaiveMeanEffectsPredictor
all_results[Model == "NaiveMeanEffectsPredictor", 
            c("MSE", "$R^2$", "$R^2$: normalized", "Pearson", 
              "Pearson: normalized", "Pearson per drug", "Pearson per cell line") := 
              lapply(.SD, function(x) paste0("\\underline{", x, "}")), 
            .SDcols = c("MSE", "$R^2$", "$R^2$: normalized", "Pearson", 
                        "Pearson: normalized", "Pearson per drug", "Pearson per cell line")]
print(xtable(all_results), include.rownames=FALSE, sanitize.text.function = function(x){x})


####### Code for Table 2 / S4 / S5: Cross-study results ####### 
all_results <- evaluation_results
metric_for_visualization <- "MSE"
all_results <- melt(all_results, 
                    id.vars = c("algorithm", "LPO_LCO_LDO", "CV_split", "rand_setting"), 
                    measure.vars = c("MSE", "R^2", "Pearson"),
                    variable.name = "metric")
# calculate mean/sd per algorithm, LPO_LCO_LDO, metric
all_results[, mean := round(mean(value), 2), by = c("algorithm", "LPO_LCO_LDO", "rand_setting", "metric")]
all_results[, sd := round(sd(value), 2), by = c("algorithm", "LPO_LCO_LDO", "rand_setting", "metric")]
all_results[, value := paste0(mean, "±", sd)]
all_results[, c("CV_split", "mean", "sd") := NULL]
all_results <- unique(all_results)
all_results <- all_results[metric == metric_for_visualization]
all_results <- dcast(all_results, algorithm + LPO_LCO_LDO ~ rand_setting, value.var = "value")
all_results[, LPO_LCO_LDO := factor(LPO_LCO_LDO, levels = c("LPO", "LCO", "LDO"))]
all_results <- all_results[order(LPO_LCO_LDO, algorithm)]
all_results <- all_results[, c("algorithm", "LPO_LCO_LDO", "predictions", "cross-study-CTRPv1", "cross-study-CCLE", "cross-study-GDSC1", "cross-study-GDSC2")]
colnames(all_results) <- c("Model", "LPO_LCO_LDO", "CTRPv2", "Cross-study: CTRPv1", "Cross-study: CCLE", "Cross-study: GDSC1", "Cross-study: GDSC2")
print(xtable(all_results), include.rownames=FALSE, sanitize.text.function = function(x){x})

####### Code for Table 3 / S6: Ablation study results ####### 
invariant <- fread(paste0(path_to_invariant_results, 'evaluation_results.csv'))
invariant$rand_mode <- "invariant"
permutation <- fread(paste0(path_to_permutation_results, 'evaluation_results.csv'))
permutation$rand_mode <- "permutation"
all_results <- rbind(invariant, permutation)
all_results <- all_results[algorithm %in% c("MultiOmicsRandomForest", "DIPK")]
all_results <- melt(all_results, 
                    id.vars = c("algorithm", "LPO_LCO_LDO", "CV_split", "rand_setting", "rand_mode"), 
                    measure.vars = c("MSE", "R^2", "Pearson"),
                    variable.name = "metric")
prediction_values <- all_results[rand_setting == "predictions"]
prediction_values[, c("rand_setting", "rand_mode") := NULL]
prediction_values <- unique(prediction_values)
all_results <- all_results[!rand_setting == "predictions"]
all_results <- merge(all_results, prediction_values, by = c("algorithm", "LPO_LCO_LDO", "CV_split", "metric"), suffixes = c("", "_predictions"), allow.cartesian=TRUE)
all_results[, delta := value - value_predictions]

all_results[, mean := round(mean(delta), 2), by = c("algorithm", "LPO_LCO_LDO", "rand_setting", "rand_mode", "metric")]
all_results[, sd := round(sd(delta), 2), by = c("algorithm", "LPO_LCO_LDO", "rand_setting", "rand_mode","metric")]
all_results[, value := paste0(mean, "±", sd)]
all_results[, c("CV_split", "mean", "sd", "value_predictions", "delta") := NULL]
all_results <- unique(all_results)
all_results <- dcast(all_results, algorithm + LPO_LCO_LDO + rand_setting + rand_mode ~ metric, value.var = "value")
all_results[, LPO_LCO_LDO := factor(LPO_LCO_LDO, levels = c("LPO", "LCO", "LDO"))]
all_results[, rand_mode := factor(rand_mode, levels = c("permutation", "invariant"))]
all_results <- all_results[order(LPO_LCO_LDO, algorithm, rand_setting, rand_mode)]
print(xtable(all_results), include.rownames=FALSE, sanitize.text.function = function(x){x})

prediction_values[, mean := round(mean(value), 2), by = c("algorithm", "LPO_LCO_LDO", "metric")]
prediction_values[, sd := round(sd(value), 2), by = c("algorithm", "LPO_LCO_LDO", "metric")]
prediction_values[, value := paste0(mean, "±", sd)]
prediction_values[, c("CV_split", "mean", "sd", "rand_mode") := NULL]
prediction_values <- unique(prediction_values)
print(xtable(dcast(prediction_values, algorithm + LPO_LCO_LDO ~ metric, value.var = "value")), include.rownames=FALSE, sanitize.text.function = function(x){x})

####### Code for Figure S1: comparison curvecurator #######

ccle <- fread(paste0(zenodo_data, '/CCLE/CCLE.csv'))
ccle$dataset <- "CCLE"
ccle[, group := paste(cellosaurus_id, pubchem_id, sep = "_")]
ctrpv1 <- fread(paste0(zenodo_data, 'CTRPv1/CTRPv1.csv'))
ctrpv1$dataset <- "CTRPv1"
ctrpv1[, group := paste(cellosaurus_id, pubchem_id, sep = "_")]
ctrpv2 <- fread(paste0(zenodo_data, 'CTRPv2/CTRPv2.csv'))
ctrpv2$dataset <- "CTRPv2"
ctrpv2[, group := paste(cellosaurus_id, pubchem_id, sep = "_")]
gdsc1 <- fread(paste0(zenodo_data, 'GDSC1/GDSC1.csv'))
gdsc1$dataset <- "GDSC1"
gdsc1[, group := paste(cellosaurus_id, pubchem_id, sep = "_")]
gdsc2 <- fread(paste0(zenodo_data, 'GDSC2/GDSC2.csv'))
gdsc2$dataset <- "GDSC2"
gdsc2[, group := paste(cellosaurus_id, pubchem_id, sep = "_")]

group_intersection <- Reduce(intersect, list(ccle$group, ctrpv1$group, ctrpv2$group, gdsc1$group, gdsc2$group))
ccle <- ccle[group %in% group_intersection]
ccle[, LN_IC50 := log(`IC50 (µM)`)]
ccle[, pEC50 := -log10(`EC50 (µM)`)]
ctrpv1 <- ctrpv1[group %in% group_intersection]
ctrpv1$LN_IC50 <- NA
ctrpv1$pEC50 <- NA
ctrpv2 <- ctrpv2[group %in% group_intersection]
ctrpv2$LN_IC50 <- NA
ctrpv2[, pEC50 := -log10(EC50)]
gdsc1 <- gdsc1[group %in% group_intersection]
gdsc1$pEC50 <- NA
gdsc2 <- gdsc2[group %in% group_intersection]
gdsc2$pEC50 <- NA

cols_to_keep <- c("LN_IC50", "LN_IC50_curvecurator", "AUC", "AUC_curvecurator", "pEC50", "pEC50_curvecurator", "dataset", "group")
all_data <- rbind(
  ccle[, ..cols_to_keep], 
  ctrpv1[, ..cols_to_keep], 
  ctrpv2[, ..cols_to_keep],
  gdsc1[, ..cols_to_keep], 
  gdsc2[, ..cols_to_keep]
)

all_data <- melt(all_data, id.vars = c("dataset", "group"), variable.name = "metric", value.name = "value")
all_data[, Processing := ifelse(grepl("curvecurator", metric), "Curve Curator", "Original")]
all_data[, metric := gsub("_curvecurator", "", metric)]
all_data[, dataset := factor(dataset, levels = c("CTRPv1", "CTRPv2", "CCLE", "GDSC1", "GDSC2"))]

ggplot(all_data, aes(x = dataset, y = value, color=Processing)) +
  geom_point()+
  geom_line(aes(group = group), alpha=0.1) +
  facet_wrap(metric~Processing, scales = "free", ncol=2)+
  theme_minimal()+
  theme(text=element_text(size=18))+
  labs(x="Dataset", y="Value") 
ggsave("~/Downloads/Comparison_Curvecurator.pdf", width = 9, height = 6)

all_data <- dcast(all_data, dataset + group + metric ~ Processing, value.var = "value", fun.aggregate = mean)
# remove rows with NA values
all_data <- all_data[!is.na(`Curve Curator`) & !is.na(Original)]
sds <- all_data[, .("SD Curve Curator" = sd(`Curve Curator`), "SD Original" = sd(Original)), by = c("group", "metric")]
sds <- melt(sds, id.vars = c("group", "metric"), variable.name = "Processing", value.name = "Standard Deviation")
sds[metric == "AUC", metric := paste("AUC \nMean SD CurveCurator:", round(mean(sds[metric == "AUC" & Processing == "SD Curve Curator", `Standard Deviation`]), 3), 
                                     "\nSD Original:", round(mean(sds[metric == "AUC" & Processing == "SD Original", `Standard Deviation`]), 3))]
sds[metric == "LN_IC50", metric := paste("LN IC50 \nMean SD CurveCurator:", round(mean(sds[metric == "LN_IC50" & Processing == "SD Curve Curator", `Standard Deviation`], na.rm=T), 3),
                                         "\nSD Original:", round(mean(sds[metric == "LN_IC50" & Processing == "SD Original", `Standard Deviation`], na.rm=T), 3))]
sds[metric == "pEC50", metric := paste("pEC50 \nMean SD CurveCurator:", round(mean(sds[metric == "pEC50" & Processing == "SD Curve Curator", `Standard Deviation`], na.rm=T), 3),
                                       "\nSD Original:", round(mean(sds[metric == "pEC50" & Processing == "SD Original", `Standard Deviation`], na.rm=T), 3))]
ggplot(sds, aes(x = `Standard Deviation`, color = Processing)) +
  # ecdf
  stat_ecdf(geom = "step") +
  facet_wrap(~metric, scales = "free") +
  theme_minimal() +
  labs(x="Standard Deviation", y="Empirical Cumulative Distribution Function") 
ggsave("~/Downloads/Comparison_Curvecurator_SD.pdf", width = 9, height = 4)

####### Code for Figure S2: prediction metrics normalized vs unnormalized ####### 

all_results <- evaluation_results
all_results <- all_results[rand_setting == "predictions"]
all_results <- melt(all_results, 
                    id.vars = c("algorithm", "LPO_LCO_LDO", "CV_split"), 
                    measure.vars = c('R^2', 'R^2: normalized', 'Pearson', 'Pearson: normalized'),
                    variable.name = "metric")
all_results[, c("CV_split") := NULL]
all_results <- unique(all_results)
# all_results <- all_results[!algorithm %in% c("SingleDrugProteomicsElasticNet", "MultiOmicsRandomForest")]
all_results[, algorithm := factor(algorithm, 
                                  levels = c("DIPK", "SimpleNeuralNetwork", "RandomForest",
                                             "NaiveMeanEffectsPredictor", "GradientBoosting", 
                                             "NaiveDrugMeanPredictor", "SRMF", "MultiOmicsNeuralNetwork",
                                             "ElasticNet", "NaiveCellLineMeanPredictor", "NaivePredictor",
                                             "SuperFELTR"))]
all_results[metric == "Pearson: normalized" & LPO_LCO_LDO == "LPO", LPO_LCO_LDO := "LPO: normalized"]
all_results[metric == "Pearson: normalized" & LPO_LCO_LDO == "LCO", LPO_LCO_LDO := "LCO: normalized"]
all_results[metric == "Pearson: normalized" & LPO_LCO_LDO == "LDO", LPO_LCO_LDO := "LDO: normalized"]
all_results[metric == "Pearson: normalized", metric := "Pearson"]
all_results[metric == "R^2: normalized" & LPO_LCO_LDO == "LPO", LPO_LCO_LDO := "LPO: normalized"]
all_results[metric == "R^2: normalized" & LPO_LCO_LDO == "LCO", LPO_LCO_LDO := "LCO: normalized"]
all_results[metric == "R^2: normalized" & LPO_LCO_LDO == "LDO", LPO_LCO_LDO := "LDO: normalized"]
all_results[metric == "R^2: normalized", metric := "R^2"]
colnames(all_results) <- c("Model", "Setting", "Metric", "Value")
all_results[, Setting := factor(Setting, 
                                levels = c("LPO", "LPO: normalized", 
                                           "LCO", "LCO: normalized", 
                                           "LDO", "LDO: normalized"))]

ggplot(all_results, aes(x = Model, y = Value, color=Setting))+
  geom_boxplot() +
  facet_wrap(~ Metric, ncol = 3, scales="free_y") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 330, hjust = 0, size=10), text = element_text(size=15)) +
  scale_color_manual(values = brewer.pal(8, "Paired")[c(1,2,5,6,7,8)])+
  # remove x, y labs
  labs(x = "", y = "") 
ggsave("~/Downloads/overview_metrics.pdf", device=pdf, width=9, height=4)

####### Code for Figure S5: expression TPMs vs microarray ####### 
gex_ccle <- fread(paste0(zenodo_data, '/CCLE/gene_expression.csv'))
gex_gdsc <- fread(paste0(zenodo_data, 'GDSC1/gene_expression.csv'))
landmark_genes <- fread(paste0(zenodo_data, 'CCLE/gene_lists/landmark_genes.csv'))
subset_cols <- c("cellosaurus_id", "cell_line_name", landmark_genes$Symbol)
gex_ccle <- gex_ccle[, ..subset_cols]
gex_gdsc <- gex_gdsc[, ..subset_cols]

big_df <- melt(gex_ccle, id.vars = c("cellosaurus_id", "cell_line_name"), variable.name = "gene", value.name = "expression")
big_df$dataset <- "TPM values\n(CTRPv1, CTRPv2, CCLE)"
big_df2 <- melt(gex_gdsc, id.vars = c("cellosaurus_id", "cell_line_name"), variable.name = "gene", value.name = "expression")
big_df2$dataset <- "Microarray RMA-norm.\n(GDSC1, GDSC2)"
big_df <- rbind(big_df, big_df2)

ggplot(big_df, aes(x=expression, y = dataset, fill = dataset))+
  geom_boxplot()+
  scale_x_log10()+
  labs(x="Expression (log10)", y="Dataset")+
  theme_minimal()
ggsave("~/Downloads/boxplot_expression.pdf", device=pdf, width=8, height=3)

####### Code for Figure S6: genes with highest_tpm_values ####### 
big_df <- melt(gex_ccle, id.vars = c("cellosaurus_id", "cell_line_name"), 
               variable.name = "gene", value.name = "expression")
highly_expressed_genes <- big_df[, mean(expression), by = gene][order(-V1)][1:10, gene]
highly_expressed <- big_df[gene %in% highly_expressed_genes]
highly_expressed$gene <- factor(highly_expressed$gene, levels = highly_expressed_genes)

ggplot(highly_expressed, aes(x = gene, y = expression)) +
  geom_boxplot()+
  theme_minimal() +
  labs(x="Gene", y="Expression (TPM)", title="Top 10 most highly expressed genes") +
  theme(axis.text.x = element_text(angle=45, hjust=1))
ggsave("~/Downloads/boxplot_expression_top10.pdf", device=pdf, width=8, height=3)
