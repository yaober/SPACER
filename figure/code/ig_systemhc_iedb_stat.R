  # Load Required Libraries -------------------------------------------------------
library(ggplot2)
library(viridis)
library(clinfun)   # for Jonckheere-Terpstra test
library(scales)
library(patchwork)

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
    legend.position = c(0.25, 0.30),
    legend.justification = c("left", "top"),
    panel.spacing = unit(0.3, "lines"),
    plot.margin   = margin(3, 3, 3, 3)
  )

# 1. Load and Preprocess IEDB Data --------------------------------------------
ig1 <- read.csv("../archive_data/ig_score_count_iedb_classi_human_epoch9.csv", stringsAsFactors = FALSE)
ig1 <- ig1[ig1$Difference != 1 & !is.na(ig1$count), ]
ig1$y <- log((ig1$count + 1) / ig1$Length)
ig1$GroupID <- ceiling(seq_len(nrow(ig1)) / 150)

summary_iedb <- aggregate(y ~ GroupID, data = ig1, FUN = function(x) c(med = median(x), se = sd(x)/sqrt(length(x))))
summary_iedb <- do.call(data.frame, summary_iedb)
colnames(summary_iedb) <- c("GroupID", "y", "se")

# 2. Load and Preprocess SysteMHC Data -----------------------------------------
ig2 <- read.csv("../archive_data/ig_score_count_systemhc.csv", stringsAsFactors = FALSE)
ig2 <- ig2[ig2$Difference != 1 & !is.na(ig2$count), ]
ig2$y <- log((ig2$count + 1) / ig2$Length)
ig2$GroupID <- ceiling(seq_len(nrow(ig2)) / 150)

summary_sysmhc <- aggregate(y ~ GroupID, data = ig2, FUN = function(x) c(med = median(x), se = sd(x)/sqrt(length(x))))
summary_sysmhc <- do.call(data.frame, summary_sysmhc)
colnames(summary_sysmhc) <- c("GroupID", "y", "se")

# Normalize SysteMHC to IEDB scale
scale_factor <- max(summary_iedb$y) / max(summary_sysmhc$y)
summary_sysmhc$y <- summary_sysmhc$y * scale_factor
summary_sysmhc$se <- summary_sysmhc$se * scale_factor

# 3. Jonckheere-Terpstra Trend Test --------------------------------------------
pval_iedb <- jonckheere.test(ig1$y, ig1$GroupID, alternative = "decreasing")$p.value
pval_sys  <- jonckheere.test(ig2$y, ig2$GroupID, alternative = "decreasing")$p.value

# 4. Update Labels to Include p-values -----------------------------------------
label_iedb <- sprintf("IEDB (p = %.1e)", pval_iedb)
label_sys  <- sprintf("SysteMHC (p = %.1e)", pval_sys)
summary_iedb$Source <- label_iedb
summary_sysmhc$Source <- label_sys

summary_combined <- rbind(summary_iedb, summary_sysmhc)

# 5. Plotting -------------------------------------------------------------------
p_dual_legend <- ggplot(summary_combined, aes(x = GroupID, y = y, color = Source, fill = Source)) +
  geom_line(aes(group = Source), linewidth = 1.4) +
  geom_point(shape = 21, size = 3, stroke = 0.3, color = "grey30") +
  geom_errorbar(aes(ymin = y - se, ymax = y + se), width = 0.15, linewidth = 1.2, alpha = 0.75) +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(
    name = "IEDB",
    sec.axis = sec_axis(~ . / scale_factor, name = "SysteMHC")
  ) +
  scale_color_manual(values = setNames(c("#f4ae6f", "#2878b4"), c(label_iedb, label_sys))) +
  scale_fill_manual(values  = setNames(c("#f4ae6f", "#2878b4"), c(label_iedb, label_sys))) +
  labs(
    x = "Group rank by SPACER score",
    color = NULL,
    fill  = NULL
  ) +
  theme_journal +  # your original background
  theme(
    axis.text.x     = element_text(angle = 0, hjust = 0.5),
    legend.position = c(0.05, 0.20)
  )

# 6. Save and Display ----------------------------------------------------------
ggsave("iedb_systemhc_joint.pdf",
       plot = p_dual_legend,
       width = 7.8, height = 6.3, units = "in", dpi = 300, useDingbats = FALSE)

print(p_dual_legend)
