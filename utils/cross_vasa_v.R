library(tidyverse)
library(Metrics)
library(psych)
library(progress)

set.seed(42)
B <- 100
k <- 3
alphas <- c(0.1, 1.0, 10.0)

eval_metrics <- function(y_true, y_pred) {
  dir_acc <- mean(sign(diff(y_true)) == sign(diff(y_pred)))
  spearman <- suppressWarnings(cor(y_true, y_pred, method = "spearman"))
  list(
    R2 = summary(lm(y_pred ~ y_true))$r.squared,
    RMSE = rmse(y_true, y_pred),
    MAE = mae(y_true, y_pred),
    Bias = mean(y_pred - y_true),
    Spearman = spearman,
    DirAcc = dir_acc
  )
}

df <- read_csv("../data/ml_base_predictions_v.csv")
base_models <- c("ridge_pred", "lasso_pred", "enet_pred", "rf_pred", "gbrt_pred", "nn_pred")

permnos <- unique(df$permno)
all_y_true <- list()
all_y_pred <- list()
all_permno <- c()

cat(sprintf("Running cross-sectional VASA on %d firms...\n", length(permnos)))
pb <- progress_bar$new(total = length(permnos))

for (permno in permnos) {
  pb$tick()
  sub <- df %>% filter(permno == !!permno) %>%
    drop_na(all_of(c(base_models, "true")))
  
  if (nrow(sub) < 60) next
  
  n_train <- floor(0.8 * nrow(sub))
  X_train <- as.matrix(sub[1:n_train, base_models])
  y_train <- sub$true[1:n_train]
  X_test  <- as.matrix(sub[(n_train + 1):nrow(sub), base_models])
  y_test  <- sub$true[(n_train + 1):nrow(sub)]
  
  vasa_preds_test_all <- matrix(NA, nrow = B, ncol = length(y_test))
  
  for (b in 1:B) {
    subset_idx <- sample(1:length(base_models), k, replace = FALSE)
    Xb_train <- X_train[, subset_idx, drop = FALSE]
    Xb_test  <- X_test[, subset_idx, drop = FALSE]
    
    # Ridge regression with CV across alphas
    best_r2 <- -Inf
    best_alpha <- alphas[1]
    best_model <- NULL
    
    for (alpha in alphas) {
      model <- lm.ridge(y_train ~ Xb_train, lambda = alpha)
      preds <- scale(Xb_test, center = TRUE, scale = FALSE) %*% coef(model)[-1] + coef(model)[1]
      r2_val <- 1 - sum((y_train - preds)^2) / sum((y_train - mean(y_train))^2)
      if (r2_val > best_r2) {
        best_r2 <- r2_val
        best_alpha <- alpha
        best_model <- model
      }
    }
    
    preds_final <- scale(Xb_test, center = TRUE, scale = FALSE) %*% coef(best_model)[-1] + coef(best_model)[1]
    vasa_preds_test_all[b, ] <- preds_final
  }
  
  vasa_pred <- colMeans(vasa_preds_test_all, na.rm = TRUE)
  
  all_y_true[[length(all_y_true) + 1]] <- y_test
  all_y_pred[[length(all_y_pred) + 1]] <- vasa_pred
  all_permno <- c(all_permno, rep(permno, length(y_test)))
}

# === Combine and evaluate globally ===
y_true_all <- unlist(all_y_true)
y_pred_all <- unlist(all_y_pred)
permno_all <- unlist(all_permno)

metrics <- eval_metrics(y_true_all, y_pred_all)
metrics_df <- as_tibble(metrics)
write_csv(metrics_df, "../data/vasa_cross_metrics_v.csv")
cat("✅ Saved ../data/vasa_cross_metrics_v.csv\n")

r2_df <- tibble(Train_R2 = NA, Test_R2 = metrics$R2)
rownames(r2_df) <- "VASA_Cross"
write_csv(r2_df, "../data/vasa_cross_r2_v.csv")
cat("✅ Saved ../data/vasa_cross_r2_v.csv\n")

cat("\nCross-sectional R²:\n")
print(r2_df)
cat("\nFull metrics:\n")
print(metrics_df)
