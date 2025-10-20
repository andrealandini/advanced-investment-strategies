library(dplyr)
library(readr)
library(purrr)
library(Rcpp)

# === Load your existing features file (with returns etc.) ===
df <- read_tsv("../data/crsp_features_full_v.txt")
df <- df %>% arrange(permno, date)

# --- Rolling Shannon entropy helper ---
# Using Rcpp for efficient rolling entropy calculation
Rcpp::cppFunction('
double rolling_entropy(const NumericVector& x, int bins = 20) {
    if (x.size() < 2) return NA_REAL;
    
    double min_val = min(x);
    double max_val = max(x);
    if (min_val == max_val) return 0.0; // All values same
    
    NumericVector hist(bins, 0.0);
    double bin_width = (max_val - min_val) / bins;
    int count = 0;
    
    // Build histogram
    for (int i = 0; i < x.size(); i++) {
        if (!NumericVector::is_na(x[i])) {
            int bin_idx = (x[i] - min_val) / bin_width;
            if (bin_idx == bins) bin_idx = bins - 1; // Handle edge case
            hist[bin_idx] += 1.0;
            count++;
        }
    }
    
    if (count == 0) return NA_REAL;
    
    // Calculate entropy
    double entropy = 0.0;
    for (int i = 0; i < bins; i++) {
        if (hist[i] > 0) {
            double p = hist[i] / count;
            entropy -= p * log2(p);
        }
    }
    return entropy;
}
')

# Alternative R version (slower but no compilation required)
rolling_entropy_r <- function(x, bins = 20) {
    if (length(x) < 2 || sd(x, na.rm = TRUE) == 0) return(0)
    
    # Remove NA values
    x_clean <- na.omit(x)
    if (length(x_clean) < 2) return(NA)
    
    hist_data <- hist(x_clean, breaks = bins, plot = FALSE)
    probs <- hist_data$counts / sum(hist_data$counts)
    probs <- probs[probs > 0]  # remove zero probabilities
    
    -sum(probs * log2(probs))
}

# --- Compute target for each firm ---
add_entropy <- function(group) {
    group <- group %>% arrange(date)
    
    # Create rolling entropy using slider package for efficient rolling operations
    if (!requireNamespace("slider", quietly = TRUE)) {
        install.packages("slider")
    }
    library(slider)
    
    group <- group %>%
        mutate(
            # Shift returns forward by 19 days to align t with t+1:t+20
            future_ret = lead(ret, 19),
            # Calculate rolling entropy on the future returns
            entropy_20d_forward = slide_dbl(
                future_ret, 
                ~ rolling_entropy(.x, bins = 20), 
                .before = 19, 
                .complete = TRUE
            )
        ) %>%
        select(permno, date, entropy_20d_forward) %>%
        filter(!is.na(entropy_20d_forward))
    
    return(group)
}

# Apply to all groups
entropy_df <- df %>%
    group_by(permno) %>%
    group_modify(~ add_entropy(.x)) %>%
    ungroup()

# --- Save result ---
write_tsv(entropy_df, "../data/entropy_target.txt")
cat("✅ Saved ../data/entropy_target.txt with columns: permno, date, entropy_20d_forward\n")

# Print summary
cat("\nSummary of entropy target:\n")
print(summary(entropy_df$entropy_20d_forward))