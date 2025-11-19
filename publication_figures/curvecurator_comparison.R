library(data.table)
library(ggplot2)

path_to_source_data_dir <- '../results/SourceData'

####### Code for Figure S4: comparison curvecurator #######
ccle <- fread(paste0(path_to_source_data_dir, '/zenodo_data/CCLE.csv'))
ccle$dataset <- "CCLE"
ccle[, group := paste(cellosaurus_id, pubchem_id, sep = "_")]
ctrpv1 <- fread(paste0(path_to_source_data_dir, '/zenodo_data/CTRPv1.csv'))
ctrpv1$dataset <- "CTRPv1"
ctrpv1[, group := paste(cellosaurus_id, pubchem_id, sep = "_")]
ctrpv2 <- fread(paste0(path_to_source_data_dir, '/zenodo_data/CTRPv2.csv'))
ctrpv2$dataset <- "CTRPv2"
ctrpv2[, group := paste(cellosaurus_id, pubchem_id, sep = "_")]
gdsc1 <- fread(paste0(path_to_source_data_dir, '/zenodo_data/GDSC1.csv'))
gdsc1$dataset <- "GDSC1"
gdsc1[, group := paste(cellosaurus_id, pubchem_id, sep = "_")]
gdsc2 <- fread(paste0(path_to_source_data_dir, '/zenodo_data/GDSC2.csv'))
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
