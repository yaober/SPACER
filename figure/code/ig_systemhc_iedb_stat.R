## ------------------------------------------------------------------
##  SysteMHC epitope‑density trend, plotted in the same style as Fig 1
## ------------------------------------------------------------------

library(ggplot2)
library(viridis)        # continuous colour palette
library(clinfun)        # Jonckheere‑Terpstra trend test

# 1. Load and clean input ------------------------------------------------------
ig <- read.csv("../archive_data/ig_score_count_systemhc.csv",
               stringsAsFactors = FALSE)

# Remove control / irrelevant rows and rows with missing epitope counts
ig <- ig[ig$Difference != 1 & !is.na(ig$count), ]

# Compute log‑normalised epitope density
ig$y <- log((ig$count + 1) / ig$Length)

# 2. Generate a sequential group ID every 150 rows -----------------------------
#    ‘ceiling()’ avoids a group 0 (which could appear with ‘round()’)
ig$GroupID <- ceiling(seq_len(nrow(ig)) / 150)

# 3. Get the median y within each group (mirrors “median‑by‑cutoff” idea) ------
summary_df <- aggregate(y ~ GroupID, data = ig, median)
# (Alternatively: dplyr::summarise)

# 4. Plot: thin line + gradient points ----------------------------------------
ggplot(summary_df, aes(x = GroupID, y = y, colour = GroupID)) +
  geom_line(size = 0.8) +          # thin line connecting medians
  geom_point(size = 2) +           # gradient‑coloured points
  scale_colour_viridis_c(option = "D") +
  labs(
    title = "IEDB Epitope Density Trend (Median per 150 rows)",
    x     = "Sequential Group Index",
    y     = "Log(Normalized HLA Binder Counts)",
    colour = "Group ID"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.text  = element_text(colour = "black")
  )

# 5. Jonckheere–Terpstra test for a monotonic decrease -------------------------
jt_result <- jonckheere.test(ig$y, ig$GroupID, alternative = "decreasing")
print(jt_result)




library(ggplot2)
library(scales)      # for muted() if you want extra‑soft colors
library(patchwork)

## ───────── 1. Color palette ─────────
main_col   <- "#4A6FA5"                                 # deep gray‑blue for lines/points
accent_pal <- c("#E0E0E0", "#B0B8C4", "#708090")        # gray → blue‑gray gradient

## ───────── 2. Common theme (tight panel, large fonts) ─────────
theme_journal <- theme_classic(base_size = 19) +        # slightly larger base font
  theme(
    plot.title      = element_text(face = "bold", hjust = 0.5, size = 22),
    axis.title      = element_text(face = "bold", size = 19),
    axis.text       = element_text(color = "black", size = 17),
    axis.line       = element_line(color = "grey50", linewidth = 0.6),
    axis.ticks      = element_line(color = "grey50"),
    legend.title    = element_text(face = "bold", size = 17),
    legend.text     = element_text(size = 15),
    panel.spacing   = unit(0.3, "lines"),               # minimal gap between panels
    plot.margin     = margin(3, 3, 3, 3),               # 3‑pt white space around the plot
    panel.grid.major = element_blank(),                 # remove major grid lines
    panel.grid.minor = element_blank()
  )


## ───────── 4. Panel B: SysteMHC density trend ─────────
summary_df$GroupID <- factor(
  summary_df$GroupID,
  levels = sort(unique(summary_df$GroupID))
)

n_groups <- length(levels(summary_df$GroupID))

# Expand the 3‑color seed to n_groups colors
fill_pal <- colorRampPalette(accent_pal)(n_groups)

pB <- ggplot(summary_df, aes(GroupID, y, group = 1)) +
  geom_line(color = main_col, linewidth = 1.4) +
  geom_point(aes(fill = GroupID),
             shape = 21, size = 3, stroke = 0.3, color = "grey30") +
  scale_fill_manual(values = fill_pal) +          # now exactly n colors
  labs(
    x = "Group",
    y = expression(log[10]~"(Normalized HLA Binder Counts)"),
    title = "SysteMHC Epitope Density Trend "
  ) +
  theme_journal +
  theme(
    axis.text.x     = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )

print(pB)

ggsave(
  "iedb.pdf",
  pB,
  width  = 6, height = 6, units = "in",
  dpi    = 300,
  useDingbats = FALSE                                   # avoid font‑embedding issues
)
