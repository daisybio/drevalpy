library(data.table)
library(xtable)
library(readxl)

# Load data
path_to_source_data_dir <- '../results/SourceData'

# Read in drug response data
ccle <- fread(paste0(path_to_source_data_dir, '/zenodo_data/CCLE.csv'))
ccle <- unique(ccle$cell_line_name)
ctrpv1 <- fread(paste0(path_to_source_data_dir, '/zenodo_data/CTRPv1.csv'))
ctrpv1 <- unique(ctrpv1$cell_line_name)
ctrpv2 <- fread(paste0(path_to_source_data_dir, '/zenodo_data/CTRPv2.csv'))
ctrpv2 <- unique(ctrpv2$cell_line_name)
gdsc1 <- fread(paste0(path_to_source_data_dir, '/zenodo_data/GDSC1.csv'))
gdsc1 <- unique(gdsc1$cell_line_name)
gdsc2 <- fread(paste0(path_to_source_data_dir, '/zenodo_data/GDSC2.csv'))
gdsc2 <- unique(gdsc2$cell_line_name)
beataml2 <- fread(paste0(path_to_source_data_dir, '/zenodo_data/BeatAML2.csv'))
beataml2 <- unique(beataml2$cell_line_name)
pdxbruna <- fread(paste0(path_to_source_data_dir, '/zenodo_data/PDX_Bruna.csv'))
pdxbruna <- unique(pdxbruna$cell_line_name)

out_df <- data.table(Name=c('No. CLs in drug screen'), CCLE=length(ccle), CTRPv1=length(ctrpv1), CTRPv2=length(ctrpv2), GDSC1=length(gdsc1), GDSC2=length(gdsc2), BeatAML2=length(beataml2), PDX_Bruna=length(pdxbruna), Other=c('No. CLs in omics screen'))

# Read in omics screens
rnaseq_ccle <- fread(paste0(path_to_source_data_dir, '/zenodo_data/gene_expression_CCLE.csv'))
rnaseq_ccle <- unique(rnaseq_ccle$cell_line_name)
microarray_gdsc <- fread(paste0(path_to_source_data_dir, '/zenodo_data/gene_expression_GDSC1.csv'))
microarray_gdsc <- unique(microarray_gdsc$cell_line_name)
rnaseq_beataml2 <- fread(paste0(path_to_source_data_dir, '/zenodo_data/gene_expression_BeatAML2.csv'))
rnaseq_beataml2 <- unique(rnaseq_beataml2$cell_line_name)
microarray_bruna <- fread(paste0(path_to_source_data_dir, '/zenodo_data/gene_expression_PDX_Bruna.csv'))
microarray_bruna <- unique(microarray_bruna$cell_line_name)
methylation_ccle <- fread(paste0(path_to_source_data_dir, '/zenodo_data/methylation_CCLE.csv'))
methylation_ccle <- unique(methylation_ccle$cell_line_name)
methylation_gdsc <- fread(paste0(path_to_source_data_dir, '/zenodo_data/methylation_GDSC1.csv'))
methylation_gdsc <- unique(methylation_gdsc$cell_line_name)
cnv_cls <- fread(paste0(path_to_source_data_dir, '/zenodo_data/copy_number_variation_gistic_cellmodelpassports.csv'))
cnv_cls <- unique(cnv_cls$cell_line_name)
cnv_bruna <- fread(paste0(path_to_source_data_dir, '/zenodo_data/copy_number_variation_gistic_pdxbruna.csv'))
cnv_bruna <- unique(cnv_bruna$cell_line_name)
mutation <- fread(paste0(path_to_source_data_dir, '/zenodo_data/mutations_cellmodelpassports.csv'))
mutation <- unique(mutation$cell_line_name)
proteomics <- fread(paste0(path_to_source_data_dir, '/zenodo_data/proteomics_dia.csv'))
proteomics <- unique(proteomics$cell_line_name)


overlap_screen <- function(omic, screen) {
  overlap <- length(intersect(omic, screen))
  percentage <- round(100*overlap/length(screen))
  return(paste0(overlap, ' (', percentage, '%)'))
}

calculate_overlaps <- function(omic, title) {
  no_in_omics_screen <- length(omic)
  row_df <- data.table(
    Name=title,
    CCLE=overlap_screen(omic=omic, screen=ccle),
    CTRPv1=overlap_screen(omic=omic, screen=ctrpv1),
    CTRPv2=overlap_screen(omic=omic, screen=ctrpv2),
    GDSC1=overlap_screen(omic=omic, screen=gdsc1),
    GDSC2=overlap_screen(omic=omic, screen=gdsc2),
    BeatAML2=overlap_screen(omic=omic, screen=beataml2),
    PDX_Bruna=overlap_screen(omic=omic, screen=pdxbruna),
    Other=no_in_omics_screen
    )
  return(row_df)
}

row_rnaseq_CCLE <- calculate_overlaps(omic=rnaseq_ccle, title='RNAseq gene expr. CCLE')
row_micro_GDSC <- calculate_overlaps(omic=microarray_gdsc, title='Microarray gene expr. GDSC')
row_met_CCLE <- calculate_overlaps(omic=methylation_ccle, title='RRBS methylation CCLE')
row_met_GDSC <- calculate_overlaps(omic=methylation_gdsc, title='BeadChip methylation GDSC')
row_cnv_cmp <- calculate_overlaps(omic=cnv_cls, title='CNV Cell Model Passports')
row_mut_cmp <- calculate_overlaps(omic=mutation, title='Mutation Cell Model Passports')
row_prot <- calculate_overlaps(omic=proteomics, title='DIA Proteomics')
row_rnaseq_beat <- calculate_overlaps(omic=rnaseq_beataml2, title='RNAseq gene expr. BeatAML2')
row_micro_bruna <- calculate_overlaps(omic=microarray_bruna, title='Microarray gene expr. PDX_Bruna')
row_cnv_bruna <- calculate_overlaps(omic=cnv_bruna, title='CNV PDX_Bruna')

# for beataml2 and bruna, there were more samples measured but we excluded them from zenodo
out_df <- rbind(
  out_df,
  row_rnaseq_CCLE,
  row_micro_GDSC,
  row_met_CCLE,
  row_met_GDSC,
  row_cnv_cmp,
  row_mut_cmp,
  row_prot,
  row_rnaseq_beat,
  row_micro_bruna,
  row_cnv_bruna
)

print(xtable(out_df[!Name %in% c('Microarray gene expr. PDX_Bruna', 'CNV PDX_Bruna'), -c('BeatAML2', 'PDX_Bruna')]), include.rownames=FALSE, sanitize.text.function = function(x){x})
print(xtable(out_df[Name %in% c('No. CLs in drug screen', 'Microarray gene expr. PDX_Bruna', 'CNV PDX_Bruna'), c('Name', 'BeatAML2', 'PDX_Bruna', 'Other')]), include.rownames=FALSE, sanitize.text.function = function(x){x})
