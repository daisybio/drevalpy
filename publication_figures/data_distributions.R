library(ggplot2)
library(data.table)

ctrpv2 <- fread('../data/CTRPv2/CTRPv2.csv')
ctrpv2$LN_IC50 <- NA
ctrpv2[, pEC50 := -log10(EC50)]
ctrpv1 <- fread('../data/CTRPv1/CTRPv1.csv')
ctrpv1$LN_IC50 <- NA
ctrpv1$pEC50 <- NA
ccle <- fread('../data/CCLE/CCLE.csv')
ccle[, LN_IC50 := log(`IC50 (µM)`)]
ccle[, pEC50 := -log10(`EC50 (µM)`)]
gdsc1 <- fread('../data/GDSC1/GDSC1.csv')
gdsc1$pEC50 <- NA
gdsc2 <- fread('../data/GDSC2/GDSC2.csv')
gdsc2$pEC50 <- NA

subset_df <- function(df, dataset_name) {
  df <- df[, c("cell_line_name", "drug_name", "LN_IC50", "LN_IC50_curvecurator", "pEC50", "pEC50_curvecurator", "AUC", "AUC_curvecurator")]
  colnames(df) <- c("cell_line_name", "drug_name", "LN_IC50 original", "LN_IC50 CurveCurator", "pEC50 original", "pEC50 CurveCurator", "AUC original", "AUC CurveCurator")
  df <- melt(df, id.vars = c("cell_line_name", "drug_name"),
             measure.vars = c("LN_IC50 original", "LN_IC50 CurveCurator", "pEC50 original", "pEC50 CurveCurator", "AUC original", "AUC CurveCurator"),
             variable.name = "metric", value.name = "value")
  df$dataset <- dataset_name
  # remove rows with NA values in the 'value' column
  df <- df[!is.na(df$value), ]
  return(df)
}

ctrpv2 <- subset_df(ctrpv2, "CTRPv2")
ctrpv1 <- subset_df(ctrpv1, "CTRPv1")
ccle <- subset_df(ccle, "CCLE")
gdsc1 <- subset_df(gdsc1, "GDSC1")
gdsc2 <- subset_df(gdsc2, "GDSC2")

joined_dataset <- rbind(ctrpv2, ctrpv1, ccle, gdsc1, gdsc2)

ggplot(joined_dataset,
       aes(x=value, fill=dataset, color=dataset)) +
  geom_density(alpha=0.05, size=1) +
  facet_wrap(~metric, scales="free", ncol=2) +
  scale_color_manual(values=c("CTRPv2"="#1f78b4", "CTRPv1"="#a6cee3", "CCLE"="#b2df8a", "GDSC1"="#fb9a99", "GDSC2"="#e31a1c")) +
  xlab("Distribution of response values")+
  ylab("Density") +
  theme_minimal() +
  theme(legend.title=element_blank(), text = element_text(size=20))
ggsave("figures/supplementary/dataset_distributions.png", width=14, height=6)

################ Basic Statistics of the final datasets ################

# get a table of unique number of drugs and cell lines per dataset
num_drugs_cell_lines <- joined_dataset[, .(num_drugs = uniqueN(drug_name), num_cell_lines = uniqueN(cell_line_name)), by = dataset]
print(num_drugs_cell_lines)

################ Basic Statistics of the raw datasets ################

## Load raw datasets
# CTRPv2
path_to_raw_datasets <- '/Users/judithbernett/Downloads/raw_responses/'
ctrpv2_raw <- fread(paste0(path_to_raw_datasets, 'CTRPv2.0_2015_ctd2_ExpandedDataset', '/' , 'v20.data.per_cpd_well.txt'))
# number of unique drugs
length(unique(ctrpv2_raw$master_cpd_id))
# number of unique cell lines
length(unique(ctrpv2_raw$experiment_id))
# number of experiments
nrow(ctrpv2_raw)
# per drug: get the distinct cpd_conc_umol values
n_conc <- ctrpv2_raw[, .(n_concentrations = uniqueN(cpd_conc_umol)), by = master_cpd_id]
n_conc$dataset <- 'CTRPv2'
colnames(n_conc) <- c("drug", "n_concentrations", "dataset")

# CTRPv1
ctrpv1_raw <- fread(paste0(path_to_raw_datasets, 'CTRPv1.0_2013_pub_Cell_154_1151', '/' , 'v10.D2.avg_pct_viability_data.txt'))
# number of unique drugs
length(unique(ctrpv1_raw$cpd_name))
# number of unique cell lines
length(unique(ctrpv1_raw$ccl_name))
# number of experiments
nrow(ctrpv1_raw)
# per drug: get the distinct cpd_conc_umol values
n_conc2 <- ctrpv1_raw[, .(n_concentrations = uniqueN(cpd_conc_umol)), by = cpd_name]
n_conc2$dataset <- 'CTRPv1'
colnames(n_conc2) <- c("drug", "n_concentrations", "dataset")
n_conc <- rbind(n_conc, n_conc2)

# CCLE
library(readxl)
ccle_raw <- read_excel(paste0(path_to_raw_datasets, 'NIHMS361223-supplement-4.xlsx'), sheet = 12, skip = 2)
ccle_raw <- as.data.table(ccle_raw)
# number of unique drugs
length(unique(ccle_raw$Compound))
# number of unique cell lines
length(unique(ccle_raw$`CCLE Cell Line Name`))
# number of experiments
nrow(ccle_raw)
# get the number of doses
n_conc2 <- ccle_raw[, tstrsplit(`Doses (uM)`, ",", fixed=T)]
# count the number of non-NA columns per row
n_conc2 <- n_conc2[, .(n_concentrations = rowSums(!is.na(.SD))), .SDcols = names(n_conc2)]
n_conc2$drug <- ccle_raw$Compound
n_conc2$dataset <- 'CCLE'
colnames(n_conc2) <- c("n_concentrations", "drug", "dataset")
n_conc <- rbind(n_conc, n_conc2)

# GDSC1
gdsc1_raw <- fread(paste0(path_to_raw_datasets, 'GDSC1_public_raw_data_27Oct23.csv'))
# remove missing drugs
gdsc1_raw <- gdsc1_raw[!is.na(gdsc1_raw$DRUG_ID), ]
# number of unique drugs
length(unique(gdsc1_raw$CELL_LINE_NAME))
# number of unique cell lines
length(unique(gdsc1_raw$DRUG_ID))
# number of experiments
nrow(gdsc1_raw)
# get the number of doses
n_conc2 <- gdsc1_raw[, .(n_concentrations = uniqueN(CONC)), by = DRUG_ID]
n_conc2$dataset <- 'GDSC1'
colnames(n_conc2) <- c("drug", "n_concentrations", "dataset")
n_conc <- rbind(n_conc, n_conc2)

# GDSC2
gdsc2_raw <- fread(paste0(path_to_raw_datasets, 'GDSC2_public_raw_data_27Oct23.csv'))
# remove missing drugs
gdsc2_raw <- gdsc2_raw[!is.na(gdsc2_raw$DRUG_ID), ]
# number of unique drugs
length(unique(gdsc2_raw$CELL_LINE_NAME))
# number of unique cell lines
length(unique(gdsc2_raw$DRUG_ID))
# number of experiments
nrow(gdsc2_raw)
# get the number of doses
n_conc2 <- gdsc2_raw[, .(n_concentrations = uniqueN(CONC)), by = DRUG_ID]
n_conc2$dataset <- 'GDSC2'
colnames(n_conc2) <- c("drug", "n_concentrations", "dataset")
n_conc <- rbind(n_conc, n_conc2)

# calculate most common number of concentrations per dataset
most_common_concs <- n_conc[, .(most_common_n_concentrations = as.integer(names(sort(table(n_concentrations), decreasing=TRUE)[1]))), by = dataset]
most_common_concs[, title := paste0(dataset, ": most common nr. of conc. = ", most_common_n_concentrations)]
naming_vector <- setNames(most_common_concs$title, most_common_concs$dataset)

# rename datasets for plotting
n_conc[, dataset_title := naming_vector[dataset]]

# plot distribution of number of concentrations per drug, per dataset
ggplot(n_conc, aes(x=n_concentrations, fill=dataset_title, color=dataset_title)) +
  geom_histogram(binwidth=1) +
  facet_wrap(~dataset_title, scales="free") +
  theme_minimal() +
  xlab("Number of concentrations per drug") +
  ylab("Number of drugs") +
  scale_color_manual(values=c("#b2df8a", "#a6cee3", "#1f78b4", "#fb9a99", "#e31a1c")) +
  scale_fill_manual(values=c("#b2df8a", "#a6cee3", "#1f78b4", "#fb9a99", "#e31a1c")) +
  # custom x-axis breaks
  scale_x_continuous(breaks = scales::breaks_pretty(n=10)) +
  theme(legend.position="none", text = element_text(size=16))
ggsave('figures/supplementary/n_conc_per_dataset.png', width=12, height=6)

################ Investigate cross-study predictions ################
all_results <- fread('../results/all_runs/true_vs_pred.csv')
all_results <- all_results[!startsWith(rand_setting, "randomize")]

subset <- all_results[algorithm %in% c("DIPK", "GradientBoosting", "SimpleNeuralNetwork", "RandomForest", "ElasticNet", "NaiveMeanEffectsPredictor")]
subset_lco <- subset[test_mode == "LCO"]
# randomly subsample 100 points per algorithm & rand_setting combination
set.seed(42)
subset_lco2 <- subset_lco[, .SD[sample(.N, min(1000, .N))], by=.(algorithm, rand_setting)]

library(ggpubr)
library(patchwork)
plot1 <- ggplot(subset_lco2[rand_setting %in% c("predictions", "cross-study-CCLE")], aes(x=y_true, y=y_pred, color=rand_setting))+
  geom_point(alpha=0.2)+
  facet_wrap(~algorithm)+
  xlim(c(-10, 8))+
  ylim(c(-10, 8))+
  geom_abline(slope=1, intercept=0, linetype="dashed", color="black")+
  theme_minimal()+
  theme(legend.title=element_blank())

dens1 <- ggplot(subset_lco2[rand_setting %in% c("predictions", "cross-study-CCLE")], aes(x = y_true, fill = rand_setting)) +
  geom_density(alpha = 0.4) +
  facet_wrap(~algorithm)+
  theme_minimal() +
  theme(legend.position = "none")

dens2 <- ggplot(subset_lco2[rand_setting %in% c("predictions", "cross-study-CCLE")], aes(x = y_pred, fill = rand_setting)) +
  geom_density(alpha = 0.4) +
  facet_wrap(~algorithm)+
  theme_minimal() +
  theme(legend.position = "none") +
  coord_flip()

dens1 + plot_spacer() + plot1 + dens2 +
  plot_layout(ncol = 2, nrow = 2, widths = c(4, 1), heights = c(1, 4))
