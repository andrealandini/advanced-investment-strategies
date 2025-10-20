library(dplyr)
library(readr)
library(ggplot2)
library(purrr)

# ======================================================
# === Load Data ========================================
# ======================================================
preds <- read_csv("data/vol-output/vasa_predictions_v2.csv")
preds$date <- as.Date(preds$date)
preds <- preds %>% rename(sigma_hat = vasa_pred)

returns <- read_tsv("data/crsp_daily_top500.txt")
names(returns) <- tolower(names(returns))
returns$date <- as.Date(returns$date)

# Merge volatility forecasts with realized returns
df <- returns %>% 
  inner_join(preds, by = c("permno", "date")) %>%
  arrange(permno, date)

# ======================================================
# === Auto-detect return scaling =======================
# ======================================================
mean_ret <- mean(abs(df$ret), na.rm = TRUE)
if (mean_ret > 0.5) {  # likely in percentages
  cat(sprintf("Detected returns in percentage units (mean=%.4f). Scaling down by 100.\n", mean_ret))
  df$ret <- df$ret / 100.0
} else {
  cat(sprintf("Detected returns already in decimal form (mean=%.4f).\n", mean_ret))
}

# ======================================================
# === Helper: Performance metrics ======================
# ======================================================
perf <- function(r) {
  r <- na.omit(r)
  if (length(r) == 0) return(data.frame(mean_daily = NA, vol_daily = NA, sharpe_native = NA, 
                                       hit_ratio = NA, max_drawdown = NA))
  
  cum_ret <- cumprod(1 + r) - 1
  running_max <- cummax(cum_ret)
  drawdown <- cum_ret - running_max
  
  data.frame(
    mean_daily = mean(r),
    vol_daily = sd(r),
    sharpe_native = ifelse(sd(r) > 0, mean(r) / sd(r), NA),
    hit_ratio = mean(r > 0),
    max_drawdown = min(drawdown)
  )
}

# ======================================================
# === Strategy 1: Inverse-volatility (long-only) =======
# ======================================================
inverse_vol_portfolio <- function(df, w_max = 0.02, tc_bps = 1.0) {
  df <- df %>%
    group_by(date) %>%
    mutate(
      inv_vol = 1 / replace(sigma_hat, sigma_hat == 0, NA),
      inv_vol = inv_vol / sum(inv_vol, na.rm = TRUE),
      inv_vol = pmin(inv_vol, w_max),
      inv_vol = inv_vol / sum(inv_vol, na.rm = TRUE)
    ) %>%
    ungroup()
  
  port_ret <- df %>%
    group_by(date) %>%
    summarise(gross = sum(inv_vol * ret, na.rm = TRUE)) %>%
    ungroup()
  
  fee <- tc_bps / 10000
  port_ret <- port_ret %>%
    mutate(
      net = gross - fee,
      cumret = cumprod(1 + net) - 1
    )
  
  return(port_ret)
}

# ======================================================
# === Strategy 2: Low–High volatility (L/S) ============
# ======================================================
low_minus_high <- function(df, q = 0.2, tc_bps = 1.0) {
  daily_lmh <- function(g) {
    n <- nrow(g)
    if (n < 20) return(0.0)
    
    g_sorted <- g %>% arrange(sigma_hat)
    k <- max(1, floor(q * n))
    
    long_ret <- g_sorted %>% slice(1:k) %>% pull(ret) %>% mean(na.rm = TRUE)
    short_ret <- g_sorted %>% slice((n - k + 1):n) %>% pull(ret) %>% mean(na.rm = TRUE)
    
    return(long_ret - short_ret)
  }
  
  daily <- df %>%
    group_by(date) %>%
    group_modify(~ data.frame(gross = daily_lmh(.x))) %>%
    ungroup()
  
  fee <- tc_bps / 10000
  daily <- daily %>%
    mutate(
      net = gross - fee,
      cumret = cumprod(1 + net) - 1
    )
  
  return(daily)
}

# ======================================================
# === Strategy 3: Target-volatility overlay ============
# ======================================================
target_volatility_strategy <- function(df, sigma_target = 0.01) {
  mkt_data <- df %>%
    group_by(date) %>%
    summarise(
      sigma_hat = mean(sigma_hat, na.rm = TRUE),
      ret_mkt = mean(ret, na.rm = TRUE)
    ) %>%
    ungroup() %>%
    arrange(date)
  
  mkt_data <- mkt_data %>%
    mutate(
      w = pmin(sigma_target / lag(sigma_hat), 2),
      w = replace_na(w, 1),
      ret_strat = lag(w) * ret_mkt,
      cumret = cumprod(1 + ret_strat) - 1
    )
  
  return(mkt_data)
}

# ======================================================
# === Run all strategies ===============================
# ======================================================
iv <- inverse_vol_portfolio(df)
lmh <- low_minus_high(df)
tv <- target_volatility_strategy(df)

# ======================================================
# === Evaluate and print ===============================
# ======================================================
cat("\n=== Strategy Performance ===\n")
cat("Inverse-volatility:\n")
print(perf(iv$net))
cat("\nLow–High volatility:\n")
print(perf(lmh$net))
cat("\nTarget-volatility overlay:\n")
print(perf(tv$ret_strat))

# ======================================================
# === Plot cumulative returns ==========================
# ======================================================
ggplot() +
  geom_line(data = iv, aes(x = date, y = cumret, color = "Inverse-volatility (long-only)")) +
  geom_line(data = lmh, aes(x = date, y = cumret, color = "Low–High volatility (L/S)")) +
  geom_line(data = tv, aes(x = date, y = cumret, color = "Target-volatility overlay")) +
  labs(title = "VASA-based Investment Strategies",
       x = "Date", y = "Cumulative Return", color = "Strategy") +
  theme_minimal() +
  theme(legend.position = "bottom")

# ======================================================
# === Save results =====================================
# ======================================================
out_dir <- "data/vol-output/results_v2"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_csv(iv, file.path(out_dir, "inverse_vol_v2.csv"))
write_csv(lmh, file.path(out_dir, "low_high_v2.csv"))
write_csv(tv, file.path(out_dir, "target_vol_v2.csv"))

cat(sprintf("\n✅ Saved results in %s/\n", out_dir))