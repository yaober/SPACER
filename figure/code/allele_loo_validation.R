# Leave-One-Allele-Out (LOOA) validation figure
#
# Reads the CSV files produced by loo_allele_bootstrap.py and creates
# a 4-panel figure (IEDB Class I/II, SysteMHC Class I/II).
#
# Each panel shows:
#   • Thin semi-transparent grey lines   – one per LOO allele exclusion
#   • Shaded 5%-95% ribbon               – LOO distribution band
#   • Bold coloured line                 – reference curve (all alleles)
#   • Spearman ρ and JT p-value          – annotated on reference curve
#
# Run from the project root:
#   Rscript figure/code/allele_loo_validation.R

library(ggplot2)
library(patchwork)
library(clinfun)   # jonckheere.test

theme_journal <- theme_classic(base_size = 16) +
  theme(
    plot.title    = element_text(face = "bold", hjust = 0.5, size = 17),
    plot.subtitle = element_text(hjust = 0.5, size = 12, color = "grey35"),
    axis.text     = element_text(color = "black", size = 13),
    axis.title    = element_text(size = 14)
  )

# Colours for the four panels
PANEL_COLORS <- c(
  "iedb_classI"     = "#f4ae6f",
  "iedb_classII"    = "#e07b39",
  "systemhc_classI" = "#2878b4",
  "systemhc_classII"= "#1a5276"
)

PANEL_TITLES <- c(
  "iedb_classI"     = "IEDB – MHC Class I",
  "iedb_classII"    = "IEDB – MHC Class II",
  "systemhc_classI" = "SysteMHC – MHC Class I",
  "systemhc_classII"= "SysteMHC – MHC Class II"
)

# ── helper: build one panel ────────────────────────────────────────────────────
make_panel <- function(csv_path, panel_key) {
  if (!file.exists(csv_path)) {
    message("File not found: ", csv_path)
    return(NULL)
  }

  df <- read.csv(csv_path, stringsAsFactors = FALSE)

  # Reference curve (all alleles)
  ref <- df[df$allele == "all", ]

  # LOO curves (all permutations)
  loo <- df[df$allele != "all", ]

  # Per-GroupID quantile ribbon across LOO curves
  ribbon <- do.call(rbind, lapply(
    split(loo, loo$GroupID),
    function(g) {
      data.frame(
        GroupID  = g$GroupID[1],
        lo       = quantile(g$median_y, 0.05,  na.rm = TRUE),
        hi       = quantile(g$median_y, 0.95,  na.rm = TRUE),
        mid      = quantile(g$median_y, 0.50,  na.rm = TRUE)
      )
    }
  ))

  # JT trend test on reference curve  (decreasing = higher rank has higher y)
  jt <- jonckheere.test(ref$median_y, ref$GroupID, alternative = "decreasing")
  pval_jt <- jt$p.value
  jt_label <- if (pval_jt < 0.001) "JT p < 0.001" else sprintf("JT p = %.3f", pval_jt)

  # Spearman rho on full-data curve
  rho_val <- cor(ref$GroupID, ref$median_y, method = "spearman")
  rho_label <- sprintf("rho = %.2f\n%s", rho_val, jt_label)

  col <- PANEL_COLORS[panel_key]

  ggplot() +
    # LOO individual lines
    geom_line(
      data = loo,
      aes(x = GroupID, y = median_y, group = allele),
      color = "grey70", linewidth = 0.35, alpha = 0.45
    ) +
    # 5%-95% ribbon
    geom_ribbon(
      data = ribbon,
      aes(x = GroupID, ymin = lo, ymax = hi),
      fill = col, alpha = 0.18
    ) +
    # LOO median line (50th pct)
    geom_line(
      data = ribbon,
      aes(x = GroupID, y = mid),
      color = col, linewidth = 0.8, linetype = "dashed", alpha = 0.7
    ) +
    # Reference curve (all alleles)
    geom_line(
      data = ref,
      aes(x = GroupID, y = median_y),
      color = col, linewidth = 1.6
    ) +
    geom_point(
      data = ref,
      aes(x = GroupID, y = median_y),
      color = col, size = 3, shape = 21,
      fill = "white", stroke = 1.3
    ) +
    # Annotation
    annotate(
      "text",
      x = max(ref$GroupID) * 0.7, y = Inf,
      label = rho_label,
      vjust = 1.4, hjust = 0, size = 4.2, color = col
    ) +
    scale_x_continuous(
      breaks = function(x) pretty(x, n = 6),
      name   = "Gene rank group by SPACER score (1 = highest)"
    ) +
    labs(
      title    = PANEL_TITLES[panel_key],
      subtitle = sprintf(
        "Bold = all alleles combined; grey lines = leave-one-allele-out (M = %d)",
        length(unique(loo$allele))
      ),
      y = "Median log[(peptide count + 1) / protein length]"
    ) +
    theme_journal
}

# ── build all four panels ──────────────────────────────────────────────────────
base_dir <- "validation_tumor/pmhc_allele/per_allele"

panels <- list(
  iedb_classI      = make_panel(file.path(base_dir, "loo_curves_iedb_classI.csv"),      "iedb_classI"),
  iedb_classII     = make_panel(file.path(base_dir, "loo_curves_iedb_classII.csv"),     "iedb_classII"),
  systemhc_classI  = make_panel(file.path(base_dir, "loo_curves_systemhc_classI.csv"),  "systemhc_classI"),
  systemhc_classII = make_panel(file.path(base_dir, "loo_curves_systemhc_classII.csv"), "systemhc_classII")
)

# Drop any NULL panels (file not found)
panels <- panels[!sapply(panels, is.null)]

if (length(panels) == 0) stop("No panel data found – run loo_allele_bootstrap.py first.")

combined <- wrap_plots(panels, ncol = 2) +
  plot_annotation(
    title   = "Leave-One-Allele-Out validation of SPACER immunogenicity scores",
    theme   = theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 19))
  )

ggsave(
  "figure/allele_loo_validation.pdf",
  plot   = combined,
  width  = 16, height = 13, units = "in",
  dpi    = 300, useDingbats = FALSE
)
cat("Saved: figure/allele_loo_validation.pdf\n")
