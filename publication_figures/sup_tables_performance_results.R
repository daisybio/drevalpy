library(data.table)
library(ggplot2)
library(RColorBrewer)
library(patchwork)
library(xtable)
library(readxl)


# Load data
path_to_source_data_dir <- '../results/SourceData'

# Load evaluation results
evaluation_results <- fread(paste0(path_to_source_data_dir, '/main_results/evaluation_results.csv'))
true_vs_pred <- fread(paste0(path_to_source_data_dir, '/main_results/true_vs_pred.csv'))

# function for S2, S6, S7, S12
overview_table <- function(eval_results, t_vs_p) {
  eval_results <- eval_results[rand_setting == "predictions"]
  eval_results <- melt(eval_results,
                      id.vars = c("algorithm", "test_mode", "CV_split"),
                      measure.vars = c("MSE", "R^2", "Pearson", "R^2: normalized", "Pearson: normalized"),
                      variable.name = "metric")
  # calculate mean/standard error per algorithm, test_mode, metric
  eval_results[, mean := round(mean(value), 2), by = c("algorithm", "test_mode", "metric")]
  eval_results[, se := round(sd(value)/sqrt(.N), 2), by = c("algorithm", "test_mode", "metric")]
  eval_results[, value := paste0(mean, "±", se)]
  eval_results[, c("CV_split", "mean", "se") := NULL]
  eval_results <- unique(eval_results)
  eval_results <- dcast(eval_results, algorithm + test_mode ~ metric, value.var = "value")

  # calculate pearson correlation coefficients per algorithm, test_mode, drug: y_true vs y_pred
  tp <- t_vs_p[rand_setting == "predictions"]
  pearson_per_drug <- tp[, .(Pearson = cor(y_true, y_pred), 2), by = c("algorithm", "test_mode", "drug_name")]
  pearson_per_cl <- tp[, .(Pearson = cor(y_true, y_pred), 2), by = c("algorithm", "test_mode", "cell_line_name")]
  # calculate mean and standard deviation of pearsons per algorithm and test_mode
  all_results_per_drug <- pearson_per_drug[, .(Pearson = round(mean(Pearson, na.rm=T), 2), se = round(sd(Pearson, na.rm=T)/sqrt(.N), 2)), by = c("algorithm", "test_mode")]
  all_results_per_cl <- pearson_per_cl[, .(Pearson = round(mean(Pearson, na.rm=T), 2), se = round(sd(Pearson, na.rm=T)/sqrt(.N), 2)), by = c("algorithm", "test_mode")]
  all_results_per_drug[, Pearson := paste0(Pearson, "±", se)]
  setnames(all_results_per_drug, "Pearson", "Pearson per drug")
  all_results_per_cl[, Pearson := paste0(Pearson, "±", se)]
  setnames(all_results_per_cl, "Pearson", "Pearson per cell line")

  all_results <- merge(eval_results, all_results_per_drug[, c("algorithm", "test_mode", "Pearson per drug")], by = c("algorithm", "test_mode"), all = TRUE)
  all_results <- merge(all_results, all_results_per_cl[, c("algorithm", "test_mode", "Pearson per cell line")], by = c("algorithm", "test_mode"), all = TRUE)
  all_results[, test_mode := factor(test_mode, levels = c("LPO", "LCO", "LTO", "LDO"))]
  # order by MSE and by test_mode
  all_results <- all_results[order(test_mode, MSE)]
  setnames(all_results, "algorithm", "Model")
  setnames(all_results, "R^2", "$R^2$")
  setnames(all_results, "R^2: normalized", "$R^2$: normalized")
  # reorder columns:
  all_results <- all_results[, c("Model", "test_mode", "MSE", "$R^2$", "$R^2$: normalized", "Pearson", "Pearson: normalized", "Pearson per drug", "Pearson per cell line")]
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
  return(all_results)
}

####### Code for Table S2: LDO input ablation study ######
drug_results <- fread(paste0(path_to_source_data_dir, '/drug_embeddings/evaluation_results.csv'))
drug_true_vs_pred <- fread(paste0(path_to_source_data_dir, '/drug_embeddings/true_vs_pred.csv'))
drug_results <- overview_table(drug_results, drug_true_vs_pred)
print(xtable(drug_results), include.rownames=FALSE, sanitize.text.function = function(x){x})

####### Code for Tables S6+S7+S12: All results #######
all_results <- overview_table(evaluation_results, true_vs_pred)
print(xtable(all_results), include.rownames=FALSE, sanitize.text.function = function(x){x})

####### Code for Table S9: Cross-study results #######
all_results <- evaluation_results
external_cs_results <- fread(paste0(path_to_source_data_dir, '/infer_pdx_beat/evaluation_results.csv'))
external_cs_results <- external_cs_results[rand_setting != "predictions"]
external_dipk_results <- fread(paste0(path_to_source_data_dir, '/dipk_pdx_beat/evaluation_results.csv'))
external_dipk_results <- external_dipk_results[rand_setting != "predictions"]
all_results <- rbind(all_results, external_cs_results)
all_results <- rbind(all_results, external_dipk_results)
metric_for_visualization <- "MSE"
all_results <- melt(all_results,
                    id.vars = c("algorithm", "test_mode", "CV_split", "rand_setting"),
                    measure.vars = c("MSE", "R^2", "Pearson"),
                    variable.name = "metric")
# calculate mean/sd per algorithm, test_mode, metric
all_results[, mean := round(mean(value), 2), by = c("algorithm", "test_mode", "rand_setting", "metric")]
all_results[, se := round(sd(value)/sqrt(.N), 2), by = c("algorithm", "test_mode", "rand_setting", "metric")]
all_results[, value := paste0(mean, "±", se)]
all_results[, c("CV_split", "mean", "se") := NULL]
all_results <- unique(all_results)
all_results <- all_results[metric == metric_for_visualization]
all_results <- dcast(all_results, algorithm + test_mode ~ rand_setting, value.var = "value")
all_results[, test_mode := factor(test_mode, levels = c("LPO", "LCO", "LTO", "LDO"))]
all_results <- all_results[order(test_mode, algorithm)]
all_results <- all_results[, c("algorithm", "test_mode", "predictions",
                               "cross-study-CTRPv1", "cross-study-CCLE",
                               "cross-study-GDSC1", "cross-study-GDSC2",
                               "cross-study-BeatAML2", "cross-study-PDX")]
colnames(all_results) <- c("Model", "test_mode", "CTRPv2", "CS: CTRPv1",
                           "CS: CCLE", "CS: GDSC1", "CS: GDSC2",
                           "CS: BeatAML2", "CS: PDX_Bruna")
print(xtable(all_results), include.rownames=FALSE, sanitize.text.function = function(x){x})

# Table S10: Cross-study results other measures
# LN IC50 data tables
all_results <- evaluation_results
external_cs_results <- fread(paste0(path_to_source_data_dir, '/infer_pdx_beat/evaluation_results.csv'))
external_cs_results <- external_cs_results[rand_setting != "predictions"]
all_results <- all_results[algorithm %in% c("RandomForest", "NaiveMeanEffectsPredictor")]
external_cs_results <- external_cs_results[algorithm %in% c("RandomForest", "NaiveMeanEffectsPredictor")]
external_cs_results <- external_cs_results[rand_setting != "predictions"]
all_results <- rbind(all_results, external_cs_results)
# AUC data table
all_results <- fread(paste0(path_to_source_data_dir, '/auc_run/evaluation_results.csv'))
# pEC50 data table
all_results <- fread(paste0(path_to_source_data_dir, '/ec50_run/evaluation_results.csv'))

all_results <- all_results[test_mode == 'LCO']
all_results <- melt(all_results,
                    id.vars = c("algorithm", "test_mode", "CV_split", "rand_setting"),
                    measure.vars = c("MSE", "R^2", "Pearson"),
                    variable.name = "metric")
all_results[, mean := round(mean(value), 2), by = c("algorithm", "test_mode", "rand_setting", "metric")]
all_results[, se := round(sd(value)/sqrt(.N), 2), by = c("algorithm", "test_mode", "rand_setting", "metric")]
all_results[, value := paste0(mean, "±", se)]
all_results[, c("CV_split", "mean", "se") := NULL]
all_results <- unique(all_results)
for(metric_for_visualization in c("MSE", "Pearson", "R^2")){
  print(metric_for_visualization)
  metric_results <- all_results[metric == metric_for_visualization]
  metric_results <- dcast(metric_results, algorithm + test_mode ~ rand_setting, value.var = "value")
  metric_results <- metric_results[order(test_mode, -algorithm)]
  metric_results <- metric_results[, c("algorithm", "test_mode", "predictions",
                                       "cross-study-CTRPv1", "cross-study-CCLE",
                                       "cross-study-GDSC1", "cross-study-GDSC2",
                                       "cross-study-BeatAML2", "cross-study-PDX")]
  colnames(metric_results) <- c("Model", "test_mode", "CTRPv2", "CS: CTRPv1",
                                "CS: CCLE", "CS: GDSC1", "CS: GDSC2",
                                "CS: BeatAML2", "CS: PDX_Bruna")
  print(xtable(metric_results), include.rownames=FALSE, sanitize.text.function = function(x){x})
}


####### Code for Table S11: Ablation study results #######
invariant <- fread(paste0(path_to_source_data_dir, '/invariant-rf/evaluation_results.csv'))
invariant_dipk <- fread(paste0(path_to_source_data_dir, '/invariant-dipk/evaluation_results.csv'))
invariant_dipk <- invariant_dipk[algorithm == "DIPK"]
invariant <- rbind(invariant, invariant_dipk)
invariant <- invariant[!startsWith(rand_setting, 'cross')]
invariant$rand_mode <- "invariant"
permutation <- fread(paste0(path_to_source_data_dir, '/main_results/evaluation_results.csv'))
permutation <- permutation[!startsWith(rand_setting, 'cross')]
permutation$rand_mode <- "permutation"
all_results <- rbind(invariant, permutation)
all_results <- all_results[algorithm %in% c("MultiOmicsRandomForest", "DIPK")]
all_results <- melt(all_results,
                    id.vars = c("algorithm", "test_mode", "CV_split", "rand_setting", "rand_mode"),
                    measure.vars = c("MSE", "R^2", "Pearson"),
                    variable.name = "metric")
prediction_values <- all_results[rand_setting == "predictions"]
prediction_values[, c("rand_setting", "rand_mode") := NULL]
prediction_values <- unique(prediction_values)
all_results <- all_results[!rand_setting == "predictions"]
all_results <- merge(all_results, prediction_values, by = c("algorithm", "test_mode", "CV_split", "metric"), suffixes = c("", "_predictions"), allow.cartesian=TRUE)
all_results[, delta := value - value_predictions]

all_results[, mean := round(mean(delta), 2), by = c("algorithm", "test_mode", "rand_setting", "rand_mode", "metric")]
all_results[, se := round(sd(delta)/sqrt(.N), 2), by = c("algorithm", "test_mode", "rand_setting", "rand_mode","metric")]
all_results[, value := paste0(mean, "±", se)]
all_results[, c("CV_split", "mean", "se", "value_predictions", "delta") := NULL]
all_results <- unique(all_results)
all_results <- dcast(all_results, algorithm + test_mode + rand_setting + rand_mode ~ metric, value.var = "value")
all_results[, test_mode := factor(test_mode, levels = c("LPO", "LCO", "LTO", "LDO"))]
all_results[, rand_mode := factor(rand_mode, levels = c("permutation", "invariant"))]
all_results <- all_results[order(test_mode, -algorithm, rand_setting, rand_mode)]
all_results <- all_results[, c("test_mode", "algorithm", "rand_mode", "rand_setting", "MSE", "R^2", "Pearson")]
print(xtable(all_results), include.rownames=FALSE, sanitize.text.function = function(x){x})
