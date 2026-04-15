library(keras)
library(tensorflow)

genoTrn <- scale(genoTrn)
genoTst <- scale(genoTst, center = attr(geno1, "scaled:center"), scale = attr(genoTrn, "scaled:scale"))
phenoTrn$Phenotype = scale(phenoTrn$Phenotype)

cor_metric <- custom_metric("cor", function(y_true, y_pred) {
K <- backend()
y_true_centered <- y_true - K$mean(y_true)
y_pred_centered <- y_pred - K$mean(y_pred)
numerator <- K$sum(y_true_centered * y_pred_centered)
denominator <- K$sqrt(K$sum(K$square(y_true_centered)) * K$sum(K$square(y_pred_centered)))
numerator / (denominator + K$epsilon())
})

combo_loss <- function(alpha = 0.3, beta = 0.7) {
 function(y_true, y_pred) {
 K <- backend()
 
 y_true <- K$cast(y_true, dtype = "float32")
 y_pred <- K$cast(y_pred, dtype = "float32")
 
 mse <- K$mean(K$square(y_true - y_pred))
  
  y_true_centered <- y_true - K$mean(y_true)
  y_pred_centered <- y_pred - K$mean(y_pred)
  
  numerator <- K$sum(y_true_centered * y_pred_centered)
  denominator <- K$sqrt(K$sum(K$square(y_true_centered)) * K$sum(K$square(y_pred_centered)))
  
  cor <- numerator / (denominator + K$epsilon())
  
  loss <- alpha * mse + beta * (1 - cor)
  return(loss)
 }
}


input_dim <- ncol(genoTrn)
l2_lambda <- 1e-4

model <- keras_model_sequential() %>%
 layer_dense(units = 256 , activation = "elu", 
       input_shape = c(input_dim), 
       kernel_regularizer = regularizer_l2(l2_lambda)) %>%
 layer_batch_normalization() %>%
 layer_gaussian_noise(stddev = 0.01) %>%
 layer_dropout(0.3) %>%
 layer_dense(units = 128, activation = "elu", 
       kernel_regularizer = regularizer_l2(l2_lambda)) %>%
 layer_batch_normalization() %>%
 layer_dropout(0.3) %>%
 layer_dense(units = 64, activation = "elu", 
       kernel_regularizer = regularizer_l2(l2_lambda)) %>%
 layer_batch_normalization() %>%
 layer_dropout(0.2) %>%
 layer_dense(units = 1)

model %>% compile(
 loss = combo_loss(alpha = 0.2, beta = 0.8),
 optimizer = optimizer_adam(learning_rate = 0.001),
 metrics = list(cor_metric)
)

callbacks_list <- list(
 callback_early_stopping(
  monitor = "val_cor",
  patience = 30,
  mode = "max",
  restore_best_weights = TRUE
 ),
 callback_model_checkpoint(
  filepath = "best_model.h5",
  monitor = "val_cor",
  save_best_only = TRUE,
  save_weights_only = TRUE,
  mode = "max"
 )
)

history <- model %>% fit(
 x = genoTrn,
 y = phenoTrn$Phenotype,
 epochs = 200,
 batch_size = 1024,
 validation_split = 0.2,
 callbacks = callbacks_list,
 verbose = 1
)
model %>% load_model_weights_hdf5("best_model.h5")

ebv_test <- model %>% predict(genoTst)
cor_test <- cor(as.vector(ebv_test), phenoTst$Phenotype)
cor_test

plot(history$metric$val_cor, type = "l", col = "blue", ylim = c(0, 1), 
  xlab = "Epochs", ylab = "Validation Correlation", main = "Validation Correlation Over Epochs")

k_clear_session()
