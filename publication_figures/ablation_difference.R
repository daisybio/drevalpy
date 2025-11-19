library(data.table)
library(ggplot2)
library(latex2exp)

path_to_source_data_dir <- '../results/SourceData'
colors <- c(
  "#F7DC05",
  "#3d98d3",
  "#EC0B88",
  "#5e35b1",
  "#f9791e",
  "#3dd378",
  "#c6c6c6",
  "#444444"
)

# Load data
all_results <- fread(paste0(path_to_source_data_dir, '/main_results/evaluation_results.csv'))

all_results <- all_results[algorithm %in% c("MultiOmicsRandomForest", "DIPK") & rand_setting %in% c("predictions", "randomize-SVRD-fingerprints", "randomize-SVRC-mutations", "randomize-SVRC-gene-expression", "randomize-SVRC-copy-number-variation-gistic", "randomize-SVRC-methylation", "randomize-SVRD-molgnet-features", "randomize-SVRC-bionic-features")]
all_results <- all_results[, c("algorithm", "rand_setting", "test_mode", "CV_split", "R^2")]
all_results <- melt(all_results, id.vars = c("algorithm", "test_mode", "CV_split", "rand_setting"), variable.name = "metric")

prediction_values <- all_results[rand_setting == "predictions"]
prediction_values[, c("rand_setting") := NULL]
prediction_values <- unique(prediction_values)
all_results <- all_results[!rand_setting == "predictions"]
all_results <- merge(all_results, prediction_values, by = c("algorithm", "test_mode", "CV_split", "metric"), suffixes = c("", "_predictions"))
all_results[, delta := value - value_predictions]
all_results[, rand_setting := gsub("randomize-", "", rand_setting)]
all_results[, rand_setting := gsub("SVRD-", "", rand_setting)]
all_results[, rand_setting := gsub("SVRC-", "", rand_setting)]
all_results[, rand_setting := gsub("-", " ", rand_setting)]
all_results[, rand_setting := gsub("copy number variation gistic", "CNV", rand_setting)]

all_results[, mean := mean(delta), by = c("algorithm", "test_mode", "rand_setting", "metric")]
all_results[, se := sd(delta)/sqrt(.N), by = c("algorithm", "test_mode", "rand_setting", "metric")]
all_results <- all_results[, c("algorithm", "test_mode", "rand_setting", "metric", "mean", "se")]
all_results <- all_results[test_mode != "LTO"]
all_results <- unique(all_results)
all_results[, test_mode := factor(test_mode, levels = c("LPO", "LCO", "LDO"))]
# capitalize rand_setting
all_results[, rand_setting := tools::toTitleCase(rand_setting)]
all_results[, rand_setting := gsub("Molgnet Features", "MoLGNet Features", rand_setting)]
all_results[, rand_setting := factor(rand_setting,
                                     levels = c("MoLGNet Features", "Fingerprints", "Gene Expression", "Methylation", "CNV", "Mutations", "Bionic Features"))]

ggplot(all_results, aes(y=rand_setting, x=mean, fill=test_mode)) +
  geom_col(position='dodge') +
  # add standard errors
  geom_errorbar(aes(xmin=mean-1.96*se, xmax=mean+1.96*se), width=.2, position=position_dodge(.9)) +
  geom_vline(xintercept=0, linetype="dotted", color = "black") +
  facet_wrap(~ algorithm, scales='free_y') +
  scale_fill_manual(values = colors)+
  theme_minimal() +
  theme(
    axis.title.y = element_blank(),
    text = element_text(size=18, family="Helvetica"),
    # turn off legend title
    legend.title = element_blank(),
  ) +
  labs(x = TeX("Mean difference in $R^2$ to unperturbed input"))
ggsave('figures/delta_ablation.pdf', device='pdf', width=9, height=5)
