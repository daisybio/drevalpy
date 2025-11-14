library(data.table)
library(ggplot2)
library(RColorBrewer)
library(patchwork)
library(xtable)
library(readxl)


# Load data
path_to_data <- '../results/all_results/'
zenodo_data <- 'path/to/data/'
path_to_preprocess <- 'path/to/preprocess_drp_data/'

# Load evaluation results
evaluation_results <- fread(paste0(path_to_data, 'evaluation_results.csv'))
true_vs_pred <- fread(paste0(path_to_data, 'true_vs_pred.csv'))

####### Code for Table 1 / S6+S7: All results #######
all_results <- evaluation_results[rand_setting == "predictions"]
all_results <- melt(all_results,
                    id.vars = c("algorithm", "test_mode", "CV_split"),
                    measure.vars = c("MSE", "R^2", "Pearson", "R^2: normalized", "Pearson: normalized"),
                    variable.name = "metric")
# calculate mean/standard error per algorithm, test_mode, metric
all_results[, mean := round(mean(value), 2), by = c("algorithm", "test_mode", "metric")]
all_results[, se := round(sd(value)/sqrt(.N), 2), by = c("algorithm", "test_mode", "metric")]
all_results[, value := paste0(mean, "±", se)]
all_results[, c("CV_split", "mean", "se") := NULL]
all_results <- unique(all_results)
all_results <- dcast(all_results, algorithm + test_mode ~ metric, value.var = "value")

# calculate pearson correlation coefficients per algorithm, test_mode, drug: y_true vs y_pred
tp <- true_vs_pred[rand_setting == "predictions"]
pearson_per_drug <- tp[, .(Pearson = cor(y_true, y_pred), 2), by = c("algorithm", "test_mode", "drug_name")]
pearson_per_cl <- tp[, .(Pearson = cor(y_true, y_pred), 2), by = c("algorithm", "test_mode", "cell_line_name")]
# calculate mean and standard deviation of pearsons per algorithm and test_mode
all_results_per_drug <- pearson_per_drug[, .(Pearson = round(mean(Pearson, na.rm=T), 2), se = round(sd(Pearson, na.rm=T)/sqrt(.N), 2)), by = c("algorithm", "test_mode")]
all_results_per_cl <- pearson_per_cl[, .(Pearson = round(mean(Pearson, na.rm=T), 2), se = round(sd(Pearson, na.rm=T)/sqrt(.N), 2)), by = c("algorithm", "test_mode")]
all_results_per_drug[, Pearson := paste0(Pearson, "±", se)]
setnames(all_results_per_drug, "Pearson", "Pearson per drug")
all_results_per_cl[, Pearson := paste0(Pearson, "±", se)]
setnames(all_results_per_cl, "Pearson", "Pearson per cell line")

all_results <- merge(all_results, all_results_per_drug[, c("algorithm", "test_mode", "Pearson per drug")], by = c("algorithm", "test_mode"), all = TRUE)
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
print(xtable(all_results), include.rownames=FALSE, sanitize.text.function = function(x){x})


####### Code for Table 2 / S9+S10: Cross-study results #######
all_results <- evaluation_results
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
all_results <- all_results[, c("algorithm", "test_mode", "predictions", "cross-study-CTRPv1", "cross-study-CCLE", "cross-study-GDSC1", "cross-study-GDSC2")]
colnames(all_results) <- c("Model", "test_mode", "CTRPv2", "Cross-study: CTRPv1", "Cross-study: CCLE", "Cross-study: GDSC1", "Cross-study: GDSC2")
print(xtable(all_results), include.rownames=FALSE, sanitize.text.function = function(x){x})

####### Code for Table 3 / S11: Ablation study results #######
# TODO: UPDATE
invariant <- fread(paste0(path_to_invariant_results, 'evaluation_results.csv'))
invariant$rand_mode <- "invariant"
permutation <- fread(paste0(path_to_permutation_results, 'evaluation_results.csv'))
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
all_results[, value := paste0(mean, "±", s2)]
all_results[, c("CV_split", "mean", "s2", "value_predictions", "delta") := NULL]
all_results <- unique(all_results)
all_results <- dcast(all_results, algorithm + test_mode + rand_setting + rand_mode ~ metric, value.var = "value")
all_results[, test_mode := factor(test_mode, levels = c("LPO", "LCO", "LTO", "LDO"))]
all_results[, rand_mode := factor(rand_mode, levels = c("permutation", "invariant"))]
all_results <- all_results[order(test_mode, algorithm, rand_setting, rand_mode)]
print(xtable(all_results), include.rownames=FALSE, sanitize.text.function = function(x){x})

prediction_values[, mean := round(mean(value), 2), by = c("algorithm", "test_mode", "metric")]
prediction_values[, se := round(sd(value)/sqrt(.N), 2), by = c("algorithm", "test_mode", "metric")]
prediction_values[, value := paste0(mean, "±", se)]
prediction_values[, c("CV_split", "mean", "se", "rand_mode") := NULL]
prediction_values <- unique(prediction_values)
print(xtable(dcast(prediction_values, algorithm + test_mode ~ metric, value.var = "value")), include.rownames=FALSE, sanitize.text.function = function(x){x})

####### Code for Figure S4: comparison curvecurator #######

ccle <- fread(paste0(zenodo_data, 'CCLE/CCLE.csv'))
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
ggsave("figures/supplementary/Comparison_Curvecurator.pdf", width = 12, height = 8)

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
ggsave("figures/supplementary/Comparison_Curvecurator_SD.pdf", width = 9, height = 4)

####### Code for Figure S7: expression TPMs vs microarray #######
gex_ccle <- fread(paste0(zenodo_data, 'CCLE/gene_expression.csv'))
gex_gdsc <- fread(paste0(zenodo_data, 'GDSC1/gene_expression.csv'))
landmark_genes <- fread(paste0(zenodo_data, 'meta/gene_lists/landmark_genes.csv'))
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
ggsave("figures/supplementary/boxplot_expression.pdf", device=pdf, width=8, height=3)


####### Code for Figure S8: doses per drugs, Table S3 #######

make_doses_per_drug <- function(df, colname_drug, colname_dose) {
  df <- unique(df[!is.na(get(colname_dose))])
  dpd <- df[, .N, by=get(colname_drug)]
  colnames(dpd) <- c('drug', 'nconcs')
  return(dpd)
}

raw_ccle <- as.data.table(read_excel(paste0(path_to_preprocess, 'CCLE/response/NIHMS361223-supplement-4.xlsx'), sheet=12, skip=2))
# every row contains 8 measurements
print(paste0('Nr of experiments CCLE: ', nrow(raw_ccle)*8))
raw_ccle <- raw_ccle[, c('Compound', 'Doses (uM)')]
raw_ccle <- raw_ccle[, c('dose_1', 'dose_2', 'dose_3', 'dose_4', 'dose_5', 'dose_6', 'dose_7', 'dose_8') := tstrsplit(`Doses (uM)`, split = ",")]
raw_ccle[, `Doses (uM)` := NULL]
raw_ccle <- melt(raw_ccle, id.vars = c('Compound'), variable.name = 'dose_number', value.name = 'dose_uM')
raw_ccle[, dose_number := NULL]
doses_per_drug <- make_doses_per_drug(raw_ccle, 'Compound', 'dose_uM')
doses_per_drug$dataset <- "CCLE"

raw_ctrpv1 <- fread(paste0(path_to_preprocess, 'CTRP/response/CTRPv1.0_2013_pub_Cell_154_1151/v10.D1.raw_viability_data.txt'), sep='\t')
print(paste0('Nr of experiments CTRPv1: ', nrow(raw_ctrpv1)))
raw_ctrpv1 <- raw_ctrpv1[, c('cpd_name', 'cpd_conc_umol')]
doses_per_drug2 <- make_doses_per_drug(raw_ctrpv1, 'cpd_name', 'cpd_conc_umol')
doses_per_drug2$dataset <- "CTRPv1"
doses_per_drug <- rbind(doses_per_drug, doses_per_drug2)

raw_ctrpv2 <- fread(paste0(path_to_preprocess, 'CTRP/response/CTRPv2.0_2015_ctd2_ExpandedDataset/v20.data.per_cpd_well.txt'), sep='\t')
print(paste0('Nr of experiments CTRPv2: ', nrow(raw_ctrpv2)))
raw_ctrpv2 <- raw_ctrpv2[, c('master_cpd_id', 'cpd_conc_umol')]
doses_per_drug2 <- make_doses_per_drug(raw_ctrpv2, 'master_cpd_id', 'cpd_conc_umol')
doses_per_drug2$dataset <- "CTRPv2"
doses_per_drug <- rbind(doses_per_drug, doses_per_drug2)

raw_gdsc1 <- fread(paste0(path_to_preprocess, 'GDSC/response/GDSC1_public_raw_data_27Oct23.csv'))
print(paste0('Nr of experiments GDSC1: ', nrow(raw_gdsc1)))
raw_gdsc1 <- raw_gdsc1[, c('DRUG_ID', 'CONC')]
doses_per_drug2 <- make_doses_per_drug(raw_gdsc1, 'DRUG_ID', 'CONC')
doses_per_drug2$dataset <- "GDSC1"
doses_per_drug <- rbind(doses_per_drug, doses_per_drug2)

raw_gdsc2 <- fread(paste0(path_to_preprocess, 'GDSC/response/GDSC2_public_raw_data_27Oct23.csv'))
print(paste0('Nr of experiments GDSC2: ', nrow(raw_gdsc2)))
raw_gdsc2 <- raw_gdsc2[, c('DRUG_ID', 'CONC')]
doses_per_drug2 <- make_doses_per_drug(raw_gdsc2, 'DRUG_ID', 'CONC')
doses_per_drug2$dataset <- "GDSC2"
doses_per_drug <- rbind(doses_per_drug, doses_per_drug2)

raw_beataml2 <- fread(paste0(path_to_preprocess, 'BeatAML2/response/beataml_wv1to4_raw_inhibitor_v4_dbgap.txt'), sep='\t')
print(paste0('Nr of experiments BeatAML2: ', nrow(raw_beataml2)))
raw_beataml2 <- raw_beataml2[, c('inhibitor', 'well_concentration')]
doses_per_drug2 <- make_doses_per_drug(raw_beataml2, 'inhibitor', 'well_concentration')
doses_per_drug2$dataset <- "BeatAML2"
doses_per_drug <- rbind(doses_per_drug, doses_per_drug2)

raw_pdxbruna <- fread(paste0(path_to_preprocess, 'PDX_Bruna/response/RawDataDrugsSingleAgents.txt'), sep='\t')
# every row contains 5 measurements
print(paste0('Nr of experiments PDX_Bruna: ', nrow(raw_pdxbruna)*5))
raw_pdxbruna <- raw_pdxbruna[, c('DRUG_ID', 'D1_CONC', 'D2_CONC', 'D3_CONC', 'D4_CONC', 'D5_CONC')]
raw_pdxbruna <- melt(raw_pdxbruna, id.vars = c('DRUG_ID'), variable.name = 'dose_number', value.name = 'dose_uM')
raw_pdxbruna[, dose_number := NULL]
doses_per_drug2 <- make_doses_per_drug(raw_pdxbruna, 'DRUG_ID', 'dose_uM')
doses_per_drug2$dataset <- "PDX_Bruna"
doses_per_drug <- rbind(doses_per_drug, doses_per_drug2)

most_common_dosis_per_screen <- doses_per_drug[, .N, by=.(nconcs, dataset)][order(dataset, -N), .SD[1], by=dataset]
most_common_dosis_per_screen[, label := paste0(dataset, ': most common nr. of conc=', nconcs)]
label_lookup <- most_common_dosis_per_screen$label
names(label_lookup) <- most_common_dosis_per_screen$dataset
doses_per_drug[, label := label_lookup[dataset]]

print("Minimum and maximum number of tested concentrations per drug per dataset:")
print(doses_per_drug[, .(min_nconcs=min(nconcs), max_nconcs=max(nconcs)), by=dataset])

ggplot(doses_per_drug, aes(x=nconcs, fill=dataset))+
  geom_histogram(binwidth=1)+
  facet_wrap(~label, scales='free')+
  scale_fill_manual(values = c('CTRPv1'='#A6CEE3',
                               'CTRPv2'='#1F78B4',
                               'CCLE'='#B2DF8A',
                               'GDSC1'='#FB9A99',
                               'GDSC2'='#E31A1C',
                               'BeatAML2'='#FDBF6F',
                               'PDX_Bruna'='#CAB2D6')
                    )+
  theme_minimal()+
  labs(x='Number of tested concentrations per drug', y='Number of drugs')+
  theme(legend.position='none')
ggsave("figures/supplementary/n_conc_per_dataset.png", width=8, height=6)


####### Code for Figure S9: distribution of response value #######
ccle <- fread(paste0(zenodo_data, 'CCLE/CCLE.csv'))
ccle[, LN_IC50 := log(`IC50 (µM)`)]
ccle[, pEC50 := -log10(`EC50 (µM)`)]
ccle$dataset <- "CCLE"
ctrpv1 <- fread(paste0(zenodo_data, 'CTRPv1/CTRPv1.csv'))
ctrpv1$LN_IC50 <- NA
ctrpv1$pEC50 <- NA
ctrpv1$dataset <- "CTRPv1"
ctrpv2 <- fread(paste0(zenodo_data, 'CTRPv2/CTRPv2.csv'))
ctrpv2$LN_IC50 <- NA
ctrpv2[, pEC50 := -log10(EC50)]
ctrpv2$dataset <- "CTRPv2"
gdsc1 <- fread(paste0(zenodo_data, 'GDSC1/GDSC1.csv'))
gdsc1$pEC50 <- NA
gdsc1$dataset <- "GDSC1"
gdsc2 <- fread(paste0(zenodo_data, 'GDSC2/GDSC2.csv'))
gdsc2$pEC50 <- NA
gdsc2$dataset <- "GDSC2"
beataml2 <- fread(paste0(zenodo_data, 'BeatAML2/BeatAML2.csv'))
beataml2[, LN_IC50 := log(IC50)]
#beataml2[, pEC50_curvecurator := pEC50_curvecurator + 6]
beataml2$pEC50 <- NA
beataml2$dataset <- "BeatAML2"
pdxbruna <- fread(paste0(zenodo_data, 'PDX_Bruna/PDX_Bruna.csv'))
pdxbruna[, IC50 := as.numeric(IC50)]
pdxbruna[, LN_IC50 := log(IC50)]
#pdxbruna[, pEC50_curvecurator := pEC50_curvecurator + 6]
pdxbruna$pEC50 <- NA
pdxbruna$dataset <- "PDX_Bruna"

cols_to_keep <- c("cell_line_name", "drug_name", "LN_IC50", "LN_IC50_curvecurator", "AUC", "AUC_curvecurator", "pEC50", "pEC50_curvecurator", "dataset")
all_data <- rbind(
  ccle[, ..cols_to_keep],
  ctrpv1[, ..cols_to_keep],
  ctrpv2[, ..cols_to_keep],
  gdsc1[, ..cols_to_keep],
  gdsc2[, ..cols_to_keep],
  beataml2[, ..cols_to_keep],
  pdxbruna[, ..cols_to_keep]
)

# limit LN_IC50 between -10 and 10
#all_data[LN_IC50 < -10 | LN_IC50 > 10, LN_IC50 := NA]
#all_data[LN_IC50_curvecurator < -10 | LN_IC50_curvecurator > 10, LN_IC50_curvecurator := NA]
# limit AUC between 0 and 3
#all_data[AUC < 0 | AUC > 3, AUC := NA]
#all_data[AUC_curvecurator < 0 | AUC_curvecurator > 3, AUC_curvecurator := NA]
# limit pEC50 between -10 and 10
#all_data[pEC50 < -10 | pEC50 > 10, pEC50 := NA]
#all_data[pEC50_curvecurator < -10 | pEC50_curvecurator > 10, pEC50_curvecurator := NA]

all_data <- melt(all_data, id.vars = c("cell_line_name", "drug_name", "dataset"),
                 variable.name = 'response_type', value.name = 'response_value')

ggplot(all_data, aes(x=response_value, color=dataset))+
  geom_density()+
  facet_wrap(~response_type, scales='free', ncol=2)+
  scale_color_manual(values = c('CTRPv1'='#A6CEE3',
                               'CTRPv2'='#1F78B4',
                               'CCLE'='#B2DF8A',
                               'GDSC1'='#FB9A99',
                               'GDSC2'='#E31A1C',
                               'BeatAML2'='#FDBF6F',
                               'PDX_Bruna'='#CAB2D6')
                    )+
  theme_minimal()+
  labs(x='Distribution of response values', y='Density')
ggsave("figures/supplementary/dataset_distributions.png", width=8, height=4)

########## Basic Statistics of the final datasets: Table S3 ################

num_drugs_cell_lines <- all_data[, .(num_drugs = uniqueN(drug_name), num_cell_lines = uniqueN(cell_line_name)), by = dataset]
print(num_drugs_cell_lines)
