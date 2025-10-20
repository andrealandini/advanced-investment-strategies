library(dplyr)
library(readr)
library(purrr)
library(glmnet)
library(Metrics)
library(progress)

# === Parameters ===
B <- 100
k <- 3
alphas <- c(0.1, 1.0, 10.0)
set.seed(42)

# === Helper: Evaluation metrics ===
eval_metrics <- function(y_true, y_pred) {
  if (length(y_true) != length(y_pred)) {
    stop("y_true and y_pred must have the same length")
  }
  
  # Remove NA values for correlation calculations
  complete_cases <- complete.cases(y_true, y_pred)
  y_true_clean <- y_true[complete_cases]
  y_pred_clean <- y_pred[complete_cases]
  
  # Directional accuracy (sign of differences)
  dir_acc <- if (length(y_true_clean) > 1) {
    mean(sign(diff(y_true_clean)) == sign(diff(y_pred_clean)), na.rm = TRUE)
  } else {
    NA_real_
  }
  
  list(
    R2 = cor(y_true_clean, y_pred_clean)^2,
    RMSE = rmse(y_true_clean, y_pred_clean),
    MAE = mae(y_true_clean, y_pred_clean),
    Bias = mean(y_pred_clean - y_true_clean, na.rm = TRUE),
    Spearman = cor(y_true_clean, y_pred_clean, method = "spearman"),
    DirAcc = dir_acc
  )
}

# === Load base model predictions ===
df <- read_csv('../data/entropy-output/ml_base_predictions_e.csv')
base_models <- c('ridge_pred', 'lasso_pred', 'enet_pred', 'rf_pred', 'gbrt_pred', 'nn_pred')

permnos <- unique(df$permno)
all_y_true <- list()
all_y_pred <- list()
all_permno <- list()

cat(sprintf("Running cross-sectional VASA on %d firms...\n", length(permnos)))

# Initialize progress bar
pb <- progress_bar$new(total = length(permnos), format = "[:bar] :percent :eta")

for (permno in permnos) {
  pb$tick()
  
  sub <- df %>% 
    filter(permno == !!permno) %>%
    drop_na(all_of(c(base_models, "true")))
  
  if (nrow(sub) < 60) next
  
  n_train <- floor(0.8 * nrow(sub))
  
  X_train <- as.matrix(sub[1:n_train, base_models])
  y_train <- sub$true[1:n_train]
  X_test <- as.matrix(sub[(n_train + 1):nrow(sub), base_models])
  y_test <- sub$true[(n_train + 1):nrow(sub)]
  
  vasa_preds_test_all <- matrix(NA, nrow = B, ncol = length(y_test))
  
  for (b in 1:B) {
    subset_idx <- sample(length(base_models), size = k, replace = FALSE)
    Xb_train <- X_train[, subset_idx, drop = FALSE]
    Xb_test <- X_test[, subset_idx, drop = FALSE]
    
    # Ridge regression with cross-validation
    cv_fit <- cv.glmnet(Xb_train, y_train, alpha = 0, lambda = alphas, 
                        standardize = TRUE, intercept = TRUE)
    
    # Predict with best lambda
    preds <- predict(cv_fit, newx = Xb_test, s = "lambda.min")
    vasa_preds_test_all[b, ] <- as.numeric(preds)
  }
  
  vasa_pred <- colMeans(vasa_preds_test_all, na.rm = TRUE)
  
  all_y_true[[length(all_y_true) + 1]] <- y_test
  all_y_pred[[length(all_y_pred) + 1]] <- vasa_pred
  all_permno[[length(all_permno) + 1]] <- rep(permno, length(y_test))
}

# === Combine and evaluate globally ===
y_true_all <- unlist(all_y_true)
y_pred_all <- unlist(all_y_pred)
permno_all <- unlist(all_permno)

metrics <- eval_metrics(y_true_all, y_pred_all)
metrics_df <- as.data.frame(metrics)
write_csv(metrics_df, '../data/entropy-output/vasa_cross_metrics_e.csv')
cat("✅ Saved ../data/entropy-output/vasa_cross_metrics_e.csv\n")

r2_df <- data.frame(
  Train_R2 = NA,
  Test_R2 = metrics$R2,
  row.names = "VASA_Cross"
)
write_csv(r2_df, '../data/entropy-output/vasa_cross_r2_e.csv')
cat("✅ Saved ../data/entropy-output/vasa_cross_r2_e.csv\n")

cat("\nCross-sectional R²:\n")
print(r2_df)
cat("\nFull metrics:\n")
print(metrics_df)