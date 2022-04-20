library(keras)
cifar100 <- dataset_cifar100()
smaller <- sample(seq(dim(cifar100$train$x)[1]), 10000)
class(smaller)
more_smaller <- sample(seq(dim(cifar100$test$x)[1]), 5000)
#dim(cifar100$train$y)
x_train <- cifar100$train$x #[smaller,,,]
#dim(x_train)
g_train <- cifar100$train$y #[smaller,]
class(g_train)
x_test <- cifar100$test$x #[more_smaller,,,]
g_test <- cifar100$test$y #[more_smaller,]
length(g_test)
#g_test is a vector

#rescale to the unit interval
x_train <- x_train / 255
x_test <- x_test / 255
y_train <- to_categorical(g_train, 100) #100 classes to choose from
dim(y_train)
#y_train is a matrix array
class(y_train)
#so y_test must be a matrix array
y_test <- to_categorical(g_test, 100)
#class(y_test)

#look at some training images
set.seed(1)
library(jpeg)
par(mar = c(0,0,0,0), mfrow = c(5,5))
#get a sequence of s25 amples from 1 to 1000
index <- sample(seq(1000), 25)
for (i in index) plot(as.raster(x_train[i,,, ]))

#build the CNN

model <- keras_model_sequential() %>% #%>% is the pip operator
  layer_conv_2d(filters = 32, kernel_size = c(3,3), padding = "same", activation = "relu", input_shape = c(32,32,3)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  #16x16
  layer_conv_2d(filters = 64, kernel_size = c(3,3), padding = "same", activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  #8x8
  layer_conv_2d(filters = 128, kernel_size = c(3,3), padding = "same", activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  #4x4
  layer_flatten() %>%
  layer_dropout(rate = 0.75) %>% #best so far is 0.65
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 100, activation = "softmax")
summary(model)

#compile and run/assess the model
model %>% compile(loss = "categorical_crossentropy", optimizer = optimizer_rmsprop(), metrics = c("accuracy"))
history <- model %>% fit(x_train, y_train, epochs = 30, batch_size = 128, validation_split = 0.2)
results <- model %>% predict(x_test) %>% k_argmax()

#test accuracy
evaluate(model, x_test, y_test, verbose = 0)

#accuracy function I stole that I haven't used
compute_accuracy <- function(model_result, v_xs, v_ys){
  y_pre <- sess$run(model_result, feed_dict = dict(xs = v_xs, keep_prob_s= 1))
  correct_prediction <- tf$equal(tf$argmax(y_pre, 1L), tf$argmax(v_ys, 1L))
  accuracy <- tf$cast(correct_prediction, tf$float32) %>% tf$reduce_mean(.)
  result <- sess$run(accuracy, feed_dict = dict(xs = v_xs, ys = v_ys, keep_prob_s= 1))
  return(result)
}

#compute_accuracy(results, x_test, y_test)