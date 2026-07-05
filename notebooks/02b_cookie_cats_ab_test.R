# =============================================================================
# Notebook 02b — Cookie Cats A/B Test, in R
#
# Same dataset and question as notebook 02 (Python): does moving the progress
# gate from level 30 to level 40 change D1 / D7 retention? This script is the
# R-native version of that read — two-proportion z-test, chi-square test of
# independence, a retention plot, and a short written interpretation.
#
# Run from the repo root (or from notebooks/ — the path check below handles
# both):
#   Rscript notebooks/02b_cookie_cats_ab_test.R
#
# Dependency: ggplot2 (for the plot only). Everything else is base R.
# =============================================================================

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop(
    "Package 'ggplot2' is required for the plot in this script.\n",
    "Install it first with: install.packages('ggplot2')"
  )
}
library(ggplot2)

# -----------------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------------

# Both the input path and the plot's output path are built off the same
# repo-root guess, so this script works whether it's launched from the repo
# root or from inside notebooks/.
repo_root <- if (file.exists(file.path("data", "cookie_cats.csv"))) "." else ".."
data_path <- file.path(repo_root, "data", "cookie_cats.csv")

cookie_cats <- read.csv(data_path, stringsAsFactors = FALSE)

# retention_1 / retention_7 arrive as "True"/"False" strings from the CSV.
# as.logical() understands that spelling directly, but coercing explicitly
# (rather than relying on read.csv's type guessing) keeps the column types
# unambiguous regardless of the R version reading the file.
cookie_cats$retention_1 <- as.logical(cookie_cats$retention_1)
cookie_cats$retention_7 <- as.logical(cookie_cats$retention_7)

# Fix factor level order so gate_30 (control) always prints/plots first.
cookie_cats$version <- factor(cookie_cats$version, levels = c("gate_30", "gate_40"))

cat("Rows:", nrow(cookie_cats), "\n")
cat("Arm sizes:\n")
print(table(cookie_cats$version))

# -----------------------------------------------------------------------------
# 2. Two-proportion z-test (manual, pooled standard error under H0: p1 = p2)
# -----------------------------------------------------------------------------
# Same construction as the Python notebook: under the null that both arms
# share one true retention rate, the pooled proportion is the right input to
# the standard error, not each arm's own variance.

two_proportion_ztest <- function(x1, n1, x2, n2) {
  p1 <- x1 / n1
  p2 <- x2 / n2
  p_pool <- (x1 + x2) / (n1 + n2)
  se_pooled <- sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
  z <- (p1 - p2) / se_pooled
  p_value <- 2 * (1 - pnorm(abs(z)))
  list(p1 = p1, p2 = p2, diff = p1 - p2, z = z, p_value = p_value)
}

n1 <- sum(cookie_cats$version == "gate_30")
n2 <- sum(cookie_cats$version == "gate_40")

ztest_results <- list()
for (metric in c("retention_1", "retention_7")) {
  x1 <- sum(cookie_cats[cookie_cats$version == "gate_30", metric])
  x2 <- sum(cookie_cats[cookie_cats$version == "gate_40", metric])
  ztest_results[[metric]] <- two_proportion_ztest(x1, n1, x2, n2)
}

cat("\n--- Two-proportion z-test (pooled SE) ---\n")
for (metric in names(ztest_results)) {
  r <- ztest_results[[metric]]
  cat(sprintf(
    "%-12s gate_30=%.4f  gate_40=%.4f  diff=%+.4f  z=%.3f  p=%.5f\n",
    metric, r$p1, r$p2, r$diff, r$z, r$p_value
  ))
}

# -----------------------------------------------------------------------------
# 3. Chi-square test of independence (2x2 contingency table)
# -----------------------------------------------------------------------------
# For a 2x2 table, the chi-square test of independence is mathematically
# equivalent to the two-proportion z-test above: chi_sq = z^2, and the
# p-values match exactly *when the continuity correction is turned off*
# (correct = FALSE). chisq.test() applies Yates' continuity correction by
# default (correct = TRUE), which nudges the p-value up slightly (more
# conservative) — a deliberate adjustment for the discreteness of counts that
# matters most in small samples. Both are reported here so the equivalence
# (and the effect of the correction) is visible rather than assumed.

cat("\n--- Chi-square test of independence ---\n")
for (metric in c("retention_1", "retention_7")) {
  tbl <- table(cookie_cats$version, cookie_cats[[metric]])
  # column order: FALSE, TRUE (R's default alphabetical/logical ordering)

  chisq_uncorrected <- chisq.test(tbl, correct = FALSE)
  chisq_corrected <- chisq.test(tbl, correct = TRUE)

  cat(sprintf("\n%s contingency table:\n", metric))
  print(tbl)
  cat(sprintf(
    "  No continuity correction: chi-sq=%.3f  df=%d  p=%.5f  (matches z^2 = %.3f)\n",
    chisq_uncorrected$statistic, chisq_uncorrected$parameter, chisq_uncorrected$p.value,
    ztest_results[[metric]]$z^2
  ))
  cat(sprintf(
    "  Yates' continuity correction: chi-sq=%.3f  df=%d  p=%.5f\n",
    chisq_corrected$statistic, chisq_corrected$parameter, chisq_corrected$p.value
  ))
}

# -----------------------------------------------------------------------------
# 4. Plot — retention rate by version, D1 and D7, with 95% CI whiskers
# -----------------------------------------------------------------------------
# Wilson score interval, not the Wald/normal approximation — Wilson keeps
# accurate coverage away from p = 0.5, which matters here since D7 retention
# sits around 18-19%, not near 50%.

wilson_ci <- function(successes, n, z = 1.96) {
  p <- successes / n
  denom <- 1 + z^2 / n
  center <- (p + z^2 / (2 * n)) / denom
  half_width <- (z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2))) / denom
  c(lower = center - half_width, upper = center + half_width)
}

plot_data <- do.call(rbind, lapply(c("gate_30", "gate_40"), function(ver) {
  sub <- cookie_cats[cookie_cats$version == ver, ]
  n <- nrow(sub)
  do.call(rbind, lapply(c("retention_1", "retention_7"), function(metric) {
    x <- sum(sub[[metric]])
    ci <- wilson_ci(x, n)
    data.frame(
      version = ver,
      day = ifelse(metric == "retention_1", "Day 1", "Day 7"),
      rate = x / n,
      ci_lower = ci[["lower"]],
      ci_upper = ci[["upper"]]
    )
  }))
}))
plot_data$day <- factor(plot_data$day, levels = c("Day 1", "Day 7"))
# gate_30 sits just above its point, gate_40 just below — the two rates are
# close enough (D1 especially) that a shared vjust makes the labels collide.
plot_data$label_vjust <- ifelse(plot_data$version == "gate_30", -1.2, 2.2)

# Same two colors used for the Python notebook's charts, so the two artifacts
# read as one consistent analysis if viewed side by side.
arm_colors <- c(gate_30 = "#2a78d6", gate_40 = "#1baf7a")

retention_plot <- ggplot(plot_data, aes(x = day, y = rate, color = version, group = version)) +
  geom_line(linewidth = 1, position = position_dodge(width = 0.08)) +
  geom_point(size = 3, position = position_dodge(width = 0.08)) +
  geom_errorbar(
    aes(ymin = ci_lower, ymax = ci_upper),
    width = 0.08, linewidth = 0.7, position = position_dodge(width = 0.08)
  ) +
  geom_text(
    aes(label = scales::percent(rate, accuracy = 0.1), vjust = label_vjust),
    position = position_dodge(width = 0.08), size = 3.5, fontface = "bold",
    show.legend = FALSE
  ) +
  scale_color_manual(values = arm_colors, name = "Arm") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Cookie Cats retention by gate placement",
    subtitle = "Points = observed rate, whiskers = 95% Wilson CI",
    x = NULL, y = "Retention rate"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank())

plot_path <- file.path(repo_root, "notebooks", "02b_retention_plot.png")
ggsave(plot_path, retention_plot, width = 7, height = 5, dpi = 150)
cat(sprintf("\nPlot saved to: %s\n", plot_path))

# -----------------------------------------------------------------------------
# 5. Interpretation
# -----------------------------------------------------------------------------
cat("\n--- Interpretation ---\n")
cat("
Day 1 retention is not statistically significant (p ~ 0.07): the 95%
interval for the gate_30 - gate_40 difference spans zero, so we cannot
rule out 'no effect' on next-day return from this data alone.

Day 7 retention IS statistically significant (p ~ 0.0016 unadjusted,
still significant with Yates' correction): gate_30 retains meaningfully
more players a week out (~19.0% vs ~18.2%, a ~4.5% relative lift). The
chi-square and z-test agree exactly once the continuity correction is
removed, as expected for a 2x2 table.

Recommendation: do not move the progress gate from level 30 to level 40.
The D7 result is the one to weight, since it's both significant and
practically meaningful, and there is no compensating gain in this dataset
to offset it. See notebook 02 for the fuller treatment (Bayesian
expected-loss framing, experiment-validity checks, and the go/no-go
write-up this script's numbers feed into).
")
