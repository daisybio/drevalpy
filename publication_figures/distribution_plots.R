library(data.table)
library(ggplot2)
library(readxl)

path_to_source_data_dir <- '../results/SourceData'

####### Code for Figure S7: expression TPMs vs microarray #######
gex_ccle <- fread(paste0(path_to_source_data_dir, '/zenodo_data/gene_expression_CCLE.csv'))
gex_gdsc <- fread(paste0(path_to_source_data_dir, '/zenodo_data/gene_expression_GDSC1.csv'))
gex_beat <- fread(paste0(path_to_source_data_dir, '/zenodo_data/gene_expression_BeatAML2.csv'))
gex_bruna <- fread(paste0(path_to_source_data_dir, '/zenodo_data/gene_expression_PDX_Bruna.csv'))
landmark_genes <- fread(paste0(path_to_source_data_dir, '/zenodo_data/landmark_genes_reduced.csv'))
subset_cols <- c("cell_line_name", landmark_genes$Symbol)
gex_ccle <- gex_ccle[, ..subset_cols]
gex_gdsc <- gex_gdsc[, ..subset_cols]
gex_beat <- gex_beat[, ..subset_cols]
gex_bruna <- gex_bruna[, ..subset_cols]

big_df <- melt(gex_ccle, id.vars = c("cell_line_name"), variable.name = "gene", value.name = "expression")
big_df$dataset <- "TPM values\n(CTRPv1, CTRPv2, CCLE)"
big_df2 <- melt(gex_gdsc, id.vars = c("cell_line_name"), variable.name = "gene", value.name = "expression")
big_df2$dataset <- "Microarray RMA-norm.\n(GDSC1, GDSC2)"
big_df <- rbind(big_df, big_df2)
big_df3 <- melt(gex_beat, id.vars = c("cell_line_name"), variable.name = "gene", value.name = "expression")
big_df3$dataset <- "Norm. RNAseq values\n(BeatAML2)"
big_df <- rbind(big_df, big_df3)
big_df4 <- melt(gex_bruna, id.vars = c("cell_line_name"), variable.name = "gene", value.name = "expression")
big_df4$dataset <- "Microarray values\n(PDX_Bruna)"
big_df <- rbind(big_df, big_df4)

ggplot(big_df, aes(x=expression, y = dataset, fill = dataset))+
  geom_boxplot()+
  scale_x_log10()+
  labs(x="Expression (log10)", y="Dataset")+
  theme_minimal()+
  theme(legend.position="none")
ggsave("figures/supplementary/boxplot_expression.pdf", device=pdf, width=9, height=5)


####### Code for Figure S8: doses per drugs, Table S3 #######

make_doses_per_drug <- function(df, colname_drug, colname_dose) {
  df <- unique(df[!is.na(get(colname_dose))])
  dpd <- df[, .N, by=get(colname_drug)]
  colnames(dpd) <- c('drug', 'nconcs')
  return(dpd)
}

raw_ccle <- as.data.table(read_excel(paste0(path_to_source_data_dir, '/raw_data/NIHMS361223-supplement-4.xlsx'), sheet=12, skip=2))
# every row contains 8 measurements
print(paste0('Nr of experiments CCLE: ', nrow(raw_ccle)*8))
raw_measures <- data.table(dataset=c('CCLE'), raw_experiments=nrow(raw_ccle)*8)
raw_ccle <- raw_ccle[, c('Compound', 'Doses (uM)')]
raw_ccle <- raw_ccle[, c('dose_1', 'dose_2', 'dose_3', 'dose_4', 'dose_5', 'dose_6', 'dose_7', 'dose_8') := tstrsplit(`Doses (uM)`, split = ",")]
raw_ccle[, `Doses (uM)` := NULL]
raw_ccle <- melt(raw_ccle, id.vars = c('Compound'), variable.name = 'dose_number', value.name = 'dose_uM')
raw_ccle[, dose_number := NULL]
doses_per_drug <- make_doses_per_drug(raw_ccle, 'Compound', 'dose_uM')
doses_per_drug$dataset <- "CCLE"

raw_ctrpv1 <- fread(paste0(path_to_source_data_dir, '/raw_data/v10.D1.raw_viability_data.txt'), sep='\t')
print(paste0('Nr of experiments CTRPv1: ', nrow(raw_ctrpv1)))
raw_measures <- rbind(raw_measures, data.table(dataset=c('CTRPv1'), raw_experiments=nrow(raw_ctrpv1)))
raw_ctrpv1 <- raw_ctrpv1[, c('cpd_name', 'cpd_conc_umol')]
doses_per_drug2 <- make_doses_per_drug(raw_ctrpv1, 'cpd_name', 'cpd_conc_umol')
doses_per_drug2$dataset <- "CTRPv1"
doses_per_drug <- rbind(doses_per_drug, doses_per_drug2)

raw_ctrpv2 <- fread(paste0(path_to_source_data_dir, '/raw_data/v20.data.per_cpd_well.txt'), sep='\t')
print(paste0('Nr of experiments CTRPv2: ', nrow(raw_ctrpv2)))
raw_measures <- rbind(raw_measures, data.table(dataset=c('CTRPv2'), raw_experiments=nrow(raw_ctrpv2)))
raw_ctrpv2 <- raw_ctrpv2[, c('master_cpd_id', 'cpd_conc_umol')]
doses_per_drug2 <- make_doses_per_drug(raw_ctrpv2, 'master_cpd_id', 'cpd_conc_umol')
doses_per_drug2$dataset <- "CTRPv2"
doses_per_drug <- rbind(doses_per_drug, doses_per_drug2)

raw_gdsc1 <- fread(paste0(path_to_source_data_dir, '/raw_data/GDSC1_public_raw_data_27Oct23.csv'))
print(paste0('Nr of experiments GDSC1: ', nrow(raw_gdsc1)))
raw_measures <- rbind(raw_measures, data.table(dataset=c('GDSC1'), raw_experiments=nrow(raw_gdsc1)))
raw_gdsc1 <- raw_gdsc1[, c('DRUG_ID', 'CONC')]
doses_per_drug2 <- make_doses_per_drug(raw_gdsc1, 'DRUG_ID', 'CONC')
doses_per_drug2$dataset <- "GDSC1"
doses_per_drug <- rbind(doses_per_drug, doses_per_drug2)

raw_gdsc2 <- fread(paste0(path_to_source_data_dir, '/raw_data/GDSC2_public_raw_data_27Oct23.csv'))
print(paste0('Nr of experiments GDSC2: ', nrow(raw_gdsc2)))
raw_measures <- rbind(raw_measures, data.table(dataset=c('GDSC2'), raw_experiments=nrow(raw_gdsc2)))
raw_gdsc2 <- raw_gdsc2[, c('DRUG_ID', 'CONC')]
doses_per_drug2 <- make_doses_per_drug(raw_gdsc2, 'DRUG_ID', 'CONC')
doses_per_drug2$dataset <- "GDSC2"
doses_per_drug <- rbind(doses_per_drug, doses_per_drug2)

raw_beataml2 <- fread(paste0(path_to_source_data_dir, '/raw_data/beataml_wv1to4_raw_inhibitor_v4_dbgap.txt'), sep='\t')
print(paste0('Nr of experiments BeatAML2: ', nrow(raw_beataml2)))
raw_measures <- rbind(raw_measures, data.table(dataset=c('BeatAML2'), raw_experiments=nrow(raw_beataml2)))
raw_beataml2 <- raw_beataml2[, c('inhibitor', 'well_concentration')]
doses_per_drug2 <- make_doses_per_drug(raw_beataml2, 'inhibitor', 'well_concentration')
doses_per_drug2$dataset <- "BeatAML2"
doses_per_drug <- rbind(doses_per_drug, doses_per_drug2)

raw_pdxbruna <- fread(paste0(path_to_source_data_dir, '/raw_data/RawDataDrugsSingleAgents.txt'), sep='\t')
# every row contains 5 measurements
print(paste0('Nr of experiments PDX_Bruna: ', nrow(raw_pdxbruna)*5))
raw_measures <- rbind(raw_measures, data.table(dataset=c('PDX_Bruna'), raw_experiments=nrow(raw_pdxbruna)*5))
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
doses_per_drug[, nconcs := as.integer(nconcs)]

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
  scale_x_continuous(
    breaks = function(x) {
      br <- scales::pretty_breaks()(x)
      br[br %% 1 == 0]  # keep only integer labels
    }
  ) +
  theme_minimal()+
  labs(x='Number of tested concentrations per drug', y='Number of drugs')+
  theme(legend.position='none')
ggsave("figures/supplementary/n_conc_per_dataset.png", width=8, height=6)


####### Code for Figure S9: distribution of response value #######
ccle <- fread(paste0(path_to_source_data_dir, '/zenodo_data/CCLE.csv'))
num_curves <- data.table(dataset=c('CCLE'), num_curves=nrow(ccle))
ccle[, LN_IC50 := log(`IC50 (µM)`)]
ccle[, pEC50 := -log10(`EC50 (µM)`)]
ccle$dataset <- "CCLE"
ctrpv1 <- fread(paste0(path_to_source_data_dir, '/zenodo_data/CTRPv1.csv'))
num_curves <- rbind(num_curves, data.table(dataset=c('CTRPv1'), num_curves=nrow(ctrpv1)))
ctrpv1$LN_IC50 <- NA
ctrpv1$pEC50 <- NA
ctrpv1$dataset <- "CTRPv1"
ctrpv2 <- fread(paste0(path_to_source_data_dir, '/zenodo_data/CTRPv2.csv'))
num_curves <- rbind(num_curves, data.table(dataset=c('CTRPv2'), num_curves=nrow(ctrpv2)))
ctrpv2$LN_IC50 <- NA
ctrpv2[, pEC50 := -log10(EC50)]
ctrpv2$dataset <- "CTRPv2"
gdsc1 <- fread(paste0(path_to_source_data_dir, '/zenodo_data/GDSC1.csv'))
num_curves <- rbind(num_curves, data.table(dataset=c('GDSC1'), num_curves=nrow(gdsc1)))
gdsc1$pEC50 <- NA
gdsc1$dataset <- "GDSC1"
gdsc2 <- fread(paste0(path_to_source_data_dir, '/zenodo_data/GDSC2.csv'))
num_curves <- rbind(num_curves, data.table(dataset=c('GDSC2'), num_curves=nrow(gdsc2)))
gdsc2$pEC50 <- NA
gdsc2$dataset <- "GDSC2"
beataml2 <- fread(paste0(path_to_source_data_dir, '/zenodo_data/BeatAML2.csv'))
num_curves <- rbind(num_curves, data.table(dataset=c('BeatAML2'), num_curves=nrow(beataml2)))
beataml2[, LN_IC50 := log(IC50)]
beataml2$pEC50 <- NA
beataml2$dataset <- "BeatAML2"
pdxbruna <- fread(paste0(path_to_source_data_dir, '/zenodo_data/PDX_Bruna.csv'))
num_curves <- rbind(num_curves, data.table(dataset=c('PDX_Bruna'), num_curves=nrow(pdxbruna)))
pdxbruna[, IC50 := as.numeric(IC50)]
pdxbruna[, LN_IC50 := log(IC50)]
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
plot_data <- all_data
plot_data[LN_IC50 < -10 | LN_IC50 > 10, LN_IC50 := NA]
plot_data[LN_IC50_curvecurator < -10 | LN_IC50_curvecurator > 10, LN_IC50_curvecurator := NA]
# limit AUC between 0 and 3
plot_data[AUC < 0 | AUC > 3, AUC := NA]
plot_data[AUC_curvecurator < 0 | AUC_curvecurator > 3, AUC_curvecurator := NA]
# limit pEC50 between -10 and 10
plot_data[pEC50 < -10 | pEC50 > 10, pEC50 := NA]
plot_data[pEC50_curvecurator < -10 | pEC50_curvecurator > 10, pEC50_curvecurator := NA]

plot_data <- melt(plot_data, id.vars = c("cell_line_name", "drug_name", "dataset"),
                 variable.name = 'response_type', value.name = 'response_value')

ggplot(plot_data, aes(x=response_value, color=dataset))+
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
all_data <- melt(all_data, id.vars = c("cell_line_name", "drug_name", "dataset"),
                  variable.name = 'response_type', value.name = 'response_value')
num_drugs_cell_lines <- all_data[, .(num_drugs = uniqueN(drug_name), num_cell_lines = uniqueN(cell_line_name)), by = dataset]
num_drugs_cell_lines <- merge(num_drugs_cell_lines, num_curves)
num_drugs_cell_lines <- merge(num_drugs_cell_lines, raw_measures)
min_max_doses <- doses_per_drug[, .(min_nconcs=min(nconcs), max_nconcs=max(nconcs)), by=dataset]
num_drugs_cell_lines <- merge(num_drugs_cell_lines, min_max_doses)
num_drugs_cell_lines[, dataset := factor(dataset, levels=c('CTRPv2', 'CTRPv1', 'CCLE', 'GDSC1', 'GDSC2', 'BeatAML2', 'PDX_Bruna'))]
num_drugs_cell_lines <- num_drugs_cell_lines[order(dataset)]
print(num_drugs_cell_lines)
