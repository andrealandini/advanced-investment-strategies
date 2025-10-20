library(dplyr)
library(readr)
library(ggplot2)
library(lubridate)

# ======================================================
# === Parameters =======================================
# ======================================================
N_STOCKS <- 10
TC_BPS <- 1.0
YEARS_BACK <- 10
RESULTS_DIR <- "data/vol-output/results_v2"
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

# ======================================================
# === Load Data ========================================
# ======================================================
preds <- read_csv("data/vol-output/vasa_predictions_v2.csv")
preds$date <- as.Date(preds$date)
preds <- preds %>% rename(sigma_hat = vasa_pred)

returns <- read_tsv("data/crsp_daily_top500.txt")
names(returns) <- tolower(names(returns))
returns$date <- as.Date(returns$date)

# Merge volatility forecasts with returns
df <- returns %>% 
  inner_join(preds, by = c("permno", "date")) %>%
  arrange(permno, date)

# ======================================================
# === Detect and fix return scaling ====================
# ======================================================
if (mean(abs(df$ret), na.rm = TRUE) > 0.5) {
  cat("Detected returns in percentage units — scaling down by 100.\n")
  df$ret <- df$ret / 100.0
}

# ======================================================
# === Restrict to last 10 years ========================
# ======================================================
end_date <- max(df$date)
start_date <- end_date - years(YEARS_BACK)
df <- df %>% filter(date >= start_date & date <= end_date)
cat(sprintf("Using data from %s to %s\n", start_date, end_date))

# ======================================================
# === Assign month period ==============================
# ======================================================
df <- df %>% mutate(month = floor_date(date, "month"))
months <- unique(df$month) %>% sort()

# ======================================================
# === Storage for results ==============================
# ======================================================
portfolio_records <- list()
portfolio_returns <- list()

# ======================================================
# === Monthly Rebalancing Loop =========================
# ======================================================
for (i in 1:(length(months) - 1)) {
  current_month <- months[i]
  next_month <- months[i + 1]
  
  subset <- df %>% filter(month == current_month)
  if (nrow(subset) == 0) next
  
  # Last observation per stock in current month
  last_obs <- subset %>%
    group_by(permno) %>%
    filter(date == max(date)) %>%
    ungroup()
  
  # Select lowest-volatility stocks
  selected <- last_obs %>%
    arrange(sigma_hat) %>%
    slice_head(n = N_STOCKS) %>%
    select(permno, sigma_hat) %>%
    mutate(month = current_month)
  
  portfolio_records[[i]] <- selected
  
  # Next month's returns for those stocks
  next_df <- df %>% 
    filter(month == next_month, permno %in% selected$permno)
  
  if (nrow(next_df) > 0) {
    daily_returns <- next_df %>%
      group_by(date) %>%
      summarise(daily_ret = mean(ret, na.rm = TRUE)) %>%
      pull(daily_ret)
    
    gross_return <- prod(1 + daily_returns, na.rm = TRUE) - 1
    net_return <- gross_return - (TC_BPS / 10000)
    
    portfolio_returns[[i]] <- data.frame(
      month = next_month,
      gross = gross_return,
      net = net_return
    )
  }
}

# ======================================================
# === Combine results ==================================
# ======================================================
portfolios_df <- bind_rows(portfolio_records) %>% distinct()
returns_df <- bind_rows(portfolio_returns) %>%
  mutate(
    wealth = cumprod(1 + net),
    cumret = wealth - 1
  )

# ======================================================
# === Save results =====================================
# ======================================================
write_csv(portfolios_df, file.path(RESULTS_DIR, "monthly_portfolios_vasa_10y.csv"))
write_csv(returns_df, file.path(RESULTS_DIR, "monthly_returns_vasa_10y.csv"))

# ======================================================
# === Display summary ==================================
# ======================================================
cat("\n✅ Saved 10-year monthly portfolio compositions and returns in:\n")
cat(sprintf("  %s/monthly_portfolios_vasa_10y.csv\n", RESULTS_DIR))
cat(sprintf("  %s/monthly_returns_vasa_10y.csv\n", RESULTS_DIR))

cat("\nSample of last-month portfolios:\n")
print(tail(portfolios_df, 15))

cat("\nPerformance summary:\n")
print(tail(returns_df))

