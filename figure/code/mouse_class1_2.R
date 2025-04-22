# Load differential IG score data (after epoch 9)
ig <- read.csv("ig_score_changes_epoch_9.csv", stringsAsFactors = FALSE)
ig <- ig[abs(ig$Difference - 1) > 0.0001, ]  # Remove entries where difference is ~1 (no change)

# Load class I and class II peptide datasets
classI <- read.csv("class_I.csv", stringsAsFactors = FALSE)
classII <- read.csv("class_II.csv", stringsAsFactors = FALSE)

# Keep only peptides of typical MHC-binding lengths
classI <- classI[classI$Peptide.Length >= 8 & classI$Peptide.Length <= 12, ]
classII <- classII[classII$Peptide.Length >= 12 & classII$Peptide.Length <= 18, ]

# Read FASTA file and extract protein sequences mapped by gene name
fasta <- readLines("fasta_stagei.fasta")
protein_seq <- list()

for (i in 1:length(fasta)) {
  if (grepl(">", fasta[i], perl = TRUE)) {
    # Extract gene name after "GN=" in the header
    gene_info <- strsplit(fasta[i], "\\|")[[1]][3]
    gene_name <- sub("\\s.*", "", sub(".*GN=", "", gene_info, perl = TRUE), perl = TRUE)
    protein_seq[[gene_name]] <- ""
  } else {
    protein_seq[[gene_name]] <- paste0(protein_seq[[gene_name]], fasta[i])
  }
}

# Check how many IG genes have matched protein sequences
table(ig$Gene %in% names(protein_seq))  # TRUE / FALSE counts

# Filter IG data to only those with available protein sequences
ig <- ig[ig$Gene %in% names(protein_seq), ]
ig$protein_seq <- sapply(ig$Gene, function(g) protein_seq[[g]])
ig$len <- nchar(ig$protein_seq)

# Initialize counters for class I and II matches
ig$count_1 <- 0
ig$count_2 <- 0

# For each peptide, count how many genes contain it in their protein sequence
for (i in 1:nrow(classI)) {
  ig$count_1 <- ig$count_1 + grepl(classI$Peptide.Sequence[i], ig$protein_seq, fixed = TRUE)
}
for (i in 1:nrow(classII)) {
  ig$count_2 <- ig$count_2 + grepl(classII$Peptide.Sequence[i], ig$protein_seq, fixed = TRUE)
}

# Save processed data
save(ig, file = "ig.RData")

# Sort by IG difference
ig <- ig[order(-ig$Difference), ]

# Plot log(epitope density) binned by 100 rows
boxplot(log((ig$count_1 + 1) / ig$len) ~ factor(round(1:nrow(ig) / 100)), 
        main = "Class I peptide density")
boxplot(log((ig$count_2 + 1) / ig$len) ~ factor(round(1:nrow(ig) / 100)), 
        main = "Class II peptide density")

# Define "ratio" metrics for peptide density per residue (normalized differently for I/II)
ig$ratio_1 <- ig$count_1 / (ig$len * 5)
ig$ratio_2 <- ig$count_2 / (ig$len * 7)

# Filter to remove genes with high peptide density
ig_filter <- ig[ig$ratio_1 < 0.005 & ig$ratio_2 < 0.005, ]

# Plot again after filtering
boxplot(log((ig_filter$count_1 + 1) / ig_filter$len) ~ factor(round(1:nrow(ig_filter) / 100)),
        main = "Filtered Class I peptide density")
boxplot(log((ig_filter$count_2 + 1) / ig_filter$len) ~ factor(round(1:nrow(ig_filter) / 100)),
        main = "Filtered Class II peptide density")




# Load Libraries ---------------------------------------------------------------
library(ggplot2)
library(viridis)
library(scales)
library(patchwork)
library(dplyr)
library(tidyr)
library(clinfun)

# Define Custom Theme ----------------------------------------------------------
theme_journal <- theme_classic(base_size = 19) +
  theme(
    plot.title    = element_text(face = "bold", hjust = 0.5, size = 22),
    axis.title    = element_text(face = "plain", size = 19),
    axis.text     = element_text(color = "black", size = 17),
    axis.line     = element_line(color = "grey50", linewidth = 0.6),
    axis.ticks    = element_line(color = "grey50"),
    legend.title  = element_text(face = "bold", size = 14),
    legend.text   = element_text(size = 10),
    legend.key.size = unit(0.1, "cm"),
    legend.position = c(0.05, 0.20),
    legend.justification = c("left", "top"),
    panel.spacing = unit(0.3, "lines"),
    plot.margin   = margin(3, 3, 3, 3)
  )

# 1. Prepare Group IDs ----------------------------------------------------------
ig$GroupID <- ceiling(seq_len(nrow(ig)) / 100)

# 2. Calculate log densities and trend test -------------------------------------
ig <- ig %>%
  mutate(
    log_density_I = log((count_1 + 1) / len),
    log_density_II = log((count_2 + 1) / len)
  )

pval_I  <- jonckheere.test(ig$log_density_I, ig$GroupID, alternative = "decreasing")$p.value
pval_II <- jonckheere.test(ig$log_density_II, ig$GroupID, alternative = "decreasing")$p.value

# 3. Summary: median & standard error -------------------------------------------
summary_I <- ig %>%
  group_by(GroupID) %>%
  summarise(
    y = median(log_density_I),
    se = sd(log_density_I) / sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(Class = sprintf("Class I (p = %.1e)", pval_I))

summary_II <- ig %>%
  group_by(GroupID) %>%
  summarise(
    y = median(log_density_II),
    se = sd(log_density_II) / sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(Class = sprintf("Class II (p = %.1e)", pval_II))

# 4. Normalize secondary axis ---------------------------------------------------
scale_factor <- max(summary_I$y) / max(summary_II$y)
summary_II$y <- summary_II$y * scale_factor
summary_II$se <- summary_II$se * scale_factor

# 5. Combine and plot -----------------------------------------------------------
summary_combined <- bind_rows(summary_I, summary_II)

p_dual <- ggplot(summary_combined, aes(x = GroupID, y = y, color = Class, fill = Class)) +
  geom_line(aes(group = Class), linewidth = 1.4) +
  geom_point(shape = 21, size = 3, stroke = 0.3, color = "grey30") +
  geom_errorbar(aes(ymin = y - se, ymax = y + se), width = 0.15, linewidth = 1.2, alpha = 0.75) +
  scale_x_continuous(breaks = 1:15) +
  scale_y_continuous(
    name = "Class I",
    sec.axis = sec_axis(~ . / scale_factor, name = "Class II")
  ) +
  scale_color_manual(values = setNames(c("#36BA98", "#FFB200"), unique(summary_combined$Class))) +
  scale_fill_manual(values  = setNames(c("#36BA98", "#FFB200"), unique(summary_combined$Class))) +
  labs(
    x = "Group rank by SPACER score",
    color = NULL,
    fill = NULL
  ) +
  theme_journal +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

# 6. Save & Show ----------------------------------------------------------------
ggsave("classI_classII_joint.pdf",
       plot = p_dual,
       width = 7.8, height = 6.3, units = "in", dpi = 300, useDingbats = FALSE)

print(p_dual)
