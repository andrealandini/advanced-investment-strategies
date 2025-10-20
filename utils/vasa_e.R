library(dplyr)
library(readr)
library(purrr)
library(glmnet)
library(Metrics)
library(progress)

# === Parameters ===
B <- 100
k <- 3
alphas <- 10^seq(-2, 2, length = 10)
set.seed(42)

# === Helper: Evaluation metrics ===
eval_metrics <- function(y_true, y_pred, vasa_preds_all = NULL, best_base_r2 = NULL) {
  complete_cases <- complete.cases(y_true, y_pred)
  y_true_clean <- y_true[complete_cases]
  y_pred_clean <- y_pred[complete_cases]
  
  r2 <- cor(y_true_clean, y_pred_clean)^2
  rmse <- rmse(y_true_clean, y_pred_clean)
  mae <- mae(y_true_clean, y_pred_clean)
  bias <- mean(y_pred_clean - y_true_clean)
  spearman <- cor(y_true_clean, y_pred_clean, method = "spearman")
  
  dir_acc <- if (length(y_true_clean) > 1) {
    mean(sign(diff(y_true_clean)) == sign(diff(y_pred_clean)), na.rm = TRUE)
  } else {
    NA_real_
  }
  
  # QLIKE calculation
  ratio <- y_true_clean / y_pred_clean
  qlike <- mean(ratio - log(ratio) - 1, na.rm = TRUE)
  
  # Tail RMSE
  tail_mask <- y_true_clean > quantile(y_true_clean, 0.95, na.rm = TRUE)
  tail_rmse <- if (sum(tail_mask, na.rm = TRUE) > 0) {
    sqrt(mean((y_true_clean[tail_mask] - y_pred_clean[tail_mask])^2, na.rm = TRUE))
  } else {
    NA_real_
  }
  
  # Prediction variance
  pred_var <- if (!is.null(vasa_preds_all)) {
    mean(apply(vasa_preds_all, 2, var, na.rm = TRUE), na.rm = TRUE)
  } else {
    NA_real_
  }
  
  # Improvement ratio
  improvement <- if (!is.null(best_base_r2)) {
    (r2 - best_base_r2) / best_base_r2
  } else {
    NA_real_
  }
  
  mean_ratio <- mean(y_pred_clean, na.rm = TRUE) / mean(y_true_clean, na.rm = TRUE)
  
  list(
    R2 = r2, RMSE = rmse, MAE = mae, Bias = bias, Spearman = spearman,
    DirAcc = dir_acc, QLIKE = qlike, TailRMSE = tail_rmse,
    PredVar = pred_var, Improvement = improvement, MeanRatio = mean_ratio
  )
}

# === Load predictions ===
df <- read_csv('../data/entropy-output/ml_base_predictions_e.csv')
base_models <- c('ridge_pred', 'lasso_pred', 'enet_pred', 'rf_pred', 'gbrt_pred', 'nn_pred')

# === Split train/test chronologically ===
n_train <- floor(0.8 * nrow(df))
X_train <- as.matrix(df[1:n_train, base_models])
y_train <- df$true[1:n_train]
X_test <- as.matrix(df[(n_train + 1):nrow(df), base_models])
y_test <- df$true[(n_train + 1):nrow(df)]

# === Best base model R² for improvement ratio ===
r2_base <- map_dbl(base_models, ~ {
  test_pred <- df[[.x]][(n_train + 1):nrow(df)]
  cor(y_test, test_pred, use = "complete.obs")^2
})
names(r2_base) <- base_models
best_base_r2 <- max(r2_base, na.rm = TRUE)
cat(sprintf("Best base model test R²: %.4f\n", best_base_r2))

# === Run VASA ===
vasa_preds_train_all <- matrix(NA, nrow = B, ncol = length(y_train))
vasa_preds_test_all <- matrix(NA, nrow = B, ncol = length(y_test))

cat("Running VASA with", B, "subsamples...\n")
pb <- progress_bar$new(total = B, format = "[:bar] :percent :eta")

for (b in 1:B) {
  pb$tick()
  
  subset <- sample(length(base_models), size = k, replace = FALSE)
  Xb_train <- X_train[, subset, drop = FALSE]
  Xb_test <- X_test[, subset, drop = FALSE]
  
  cv_fit <- cv.glmnet(Xb_train, y_train, alpha = 0, lambda = alphas, 
                      standardize = TRUE, intercept = TRUE)
  
  vasa_preds_train_all[b, ] <- as.numeric(predict(cv_fit, newx = Xb_train, s = "lambda.min"))
  vasa_preds_test_all[b, ] <- as.numeric(predict(cv_fit, newx = Xb_test, s = "lambda.min"))
}

vasa_train_pred <- colMeans(vasa_preds_train_all, na.rm = TRUE)
vasa_test_pred <- colMeans(vasa_preds_test_all, na.rm = TRUE)

# === Evaluate ===
metrics_train <- eval_metrics(y_train, vasa_train_pred, vasa_preds_train_all, best_base_r2)
metrics_test <- eval_metrics(y_test, vasa_test_pred, vasa_preds_test_all, best_base_r2)

# === Save outputs ===
r2_df <- data.frame(
  Train_R2 = metrics_train$R2,
  Test_R2 = metrics_test$R2,
  row.names = "VASA"
)
write_csv(r2_df, '../data/entropy-output/vasa_r2_e.csv')
cat("✅ Saved ../data/entropy-output/vasa_r2_e.csv\n")

metrics_df <- bind_rows(
  as.data.frame(metrics_train) %>% mutate(Set = "Train"),
  as.data.frame(metrics_test) %>% mutate(Set = "Test")
) %>% select(Set, everything())
write_csv(metrics_df, '../data/entropy-output/vasa_metrics_e.csv')
cat("✅ Saved ../data/entropy-output/vasa_metrics_e.csv\n")

cat("\nR² summary:\n")
print(r2_df)
cat("\nFull metrics:\n")
print(metrics_df)