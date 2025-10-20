library(tidyverse)
library(Metrics)
library(caret)
library(glmnet)
library(randomForest)
library(gbm)
library(keras)
library(tensorflow)

set.seed(42)

# === Load dataset ===
features_df <- read_delim("../data/crsp_features_full_v.txt", delim = "\t")
entropy_df <- read_delim("../data/entropy_target.txt", delim = "\t")

# Merge on firm and date
df <- inner_join(features_df, entropy_df, by = c("permno", "date"))
cat("✅ Loaded merged dataset:", dim(df), "\n")

# === Feature and target selection ===
features <- c(
  "ret_lag1", "ret_excess_lag1", "ret_lag2", "ret_lag3",
  "ret_lag4", "ret_lag5", "skew_20d", "mean_20d", "sd_20d"
)
target <- "entropy_20d_forward"

df <- df %>% drop_na(all_of(c(features, target)))

X <- df %>% select(all_of(features)) %>% as.matrix()
y <- df[[target]]

# === Standardize predictors ===
scaler <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(scaler, X)

# === Chronological split (80/20) ===
n_train <- floor(0.8 * nrow(df))
X_train <- X_scaled[1:n_train, ]
X_test  <- X_scaled[(n_train + 1):nrow(df), ]
y_train <- y[1:n_train]
y_test  <- y[(n_train + 1):nrow(df)]

# === Evaluation function ===
evaluate <- function(y_true, y_pred) {
  list(
    R2 = cor(y_true, y_pred)^2,
    RMSE = rmse(y_true, y_pred),
    MAE = mae(y_true, y_pred),
    Bias = mean(y_pred - y_true),
    Spearman = suppressWarnings(cor(y_true, y_pred, method = "spearman")),
    DirAcc = mean(sign(diff(y_true)) == sign(diff(y_pred)))
  )
}

results <- list()
preds_train <- list()
preds_test <- list()

# === Ridge ===
ridge <- cv.glmnet(X_train, y_train, alpha = 0)
preds_train$ridge_pred <- as.vector(predict(ridge, X_train, s = "lambda.min"))
preds_test$ridge_pred  <- as.vector(predict(ridge, X_test, s = "lambda.min"))
results$ridge <- evaluate(y_test, preds_test$ridge_pred)
cat("ridge done\n")

# === LASSO ===
lasso <- cv.glmnet(X_train, y_train, alpha = 1)
preds_train$lasso_pred <- as.vector(predict(lasso, X_train, s = "lambda.min"))
preds_test$lasso_pred  <- as.vector(predict(lasso, X_test, s = "lambda.min"))
results$lasso <- evaluate(y_test, preds_test$lasso_pred)
cat("lasso done\n")

# === Elastic Net ===
enet <- cv.glmnet(X_train, y_train, alpha = 0.5)
preds_train$enet_pred <- as.vector(predict(enet, X_train, s = "lambda.min"))
preds_test$enet_pred  <- as.vector(predict(enet, X_test, s = "lambda.min"))
results$enet <- evaluate(y_test, preds_test$enet_pred)
cat("enet done\n")

# === Random Forest ===
rf <- randomForest(X_train, y_train, ntree = 200, mtry = floor(sqrt(ncol(X_train))), maxnodes = 2^6)
preds_train$rf_pred <- predict(rf, X_train)
preds_test$rf_pred  <- predict(rf, X_test)
results$rf <- evaluate(y_test, preds_test$rf_pred)
cat("rf done\n")

# === Gradient Boosting ===
gbrt <- gbm(y_train ~ ., data = as.data.frame(X_train), n.trees = 200, interaction.depth = 3, shrinkage = 0.05, distribution = "gaussian", verbose = FALSE)
preds_train$gbrt_pred <- predict(gbrt, as.data.frame(X_train), n.trees = 200)
preds_test$gbrt_pred  <- predict(gbrt, as.data.frame(X_test), n.trees = 200)
results$gbrt <- evaluate(y_test, preds_test$gbrt_pred)
cat("gbrt done\n")

# === Neural Network ===
keras::use_backend("tensorflow")
tensorflow::tf$random$set_seed(42)

model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu", input_shape = ncol(X_train)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "mse"
)

early_stop <- callback_early_stopping(patience = 5, restore_best_weights = TRUE)

history <- model %>% fit(
  X_train, y_train,
  epochs = 30,
  batch_size = 128,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(early_stop)
)

preds_train$nn_pred <- as.vector(model %>% predict(X_train))
preds_test$nn_pred  <- as.vector(model %>% predict(X_test))
results$nn <- evaluate(y_test, preds_test$nn_pred)
cat("nn done\n")

# === Combine predictions ===
pred_df <- tibble(
  permno = df$permno,
  date = df$date,
  !!!setNames(lapply(preds_train, function(x) c(x, rep(NA, length(y_test)))), names(preds_train)),
  true = y
)

for (nm in names(preds_test)) {
  pred_df[(n_train + 1):nrow(df), nm] <- preds_test[[nm]]
}

write_csv(pred_df, "../data/entropy-output/ml_base_predictions_e.csv")
cat("✅ Saved ../data/entropy-output/ml_base_predictions_e.csv\n")

# === Save metrics ===
metrics_df <- map_dfr(results, as_tibble, .id = "Model")
write_csv(metrics_df, "../data/entropy-output/ml_model_metrics_e.csv")
cat("✅ Saved ../data/entropy-output/ml_model_metrics_e.csv\n")

r2_df <- metrics_df %>% select(Model, R2) %>% rename(Test_R2 = R2)
write_csv(r2_df, "../data/entropy-output/ml_r2_e.csv")
cat("✅ Saved ../data/entropy-output/ml_r2_e.csv\n")

cat("\nR² summary:\n")
print(r2_df)
cat("\nFull metrics:\n")
print(metrics_df)
