
# Helper function to compute non-conformity scores
# for binary classification problems. The non-conformity
# score is a special case of the one proposed by
# Romano (2020) for multiclass classification.
# The function takes as input a binary vector Y
# and a vector of predicted probabilities predict_prob.
# The output is a vector of non-conformity scores.

find_conformal_scores_logit = function(Y, predict_prob){
  #unif_vector = runif(n)
  n = length(Y)
  tau_vec = rep(0,n)
  for(ii in 1:n) {
    if((Y[ii]==0) && (predict_prob[ii] >0.5 )){
      tau_vec[ii] = predict_prob[ii]
    }
    if((Y[ii]==1) && (predict_prob[ii] <0.5 )){
      tau_vec[ii] = 1-predict_prob[ii]
    }
  }
  return(tau_vec)
}


# Helper function that outputs a list related to the performance of both the classification
# algorithm and information related to conformal prediction sets.  The function takes as
# input a vector of estimated probabilities, a binary vector of test values, and a critical
# value for conformal prediction.

check_coverage_logit = function(prob_vec, Y_vec, crit_val){
  n = length(prob_vec)
  conformal_set = list()
  coverage_vec  = rep(0,n)
  error_vec = rep(0,n)
  for(ii in 1:n){
    sorted_probs_temp = sort(c((1-prob_vec[ii]),prob_vec[ii]) ,index.return=TRUE,decreasing = TRUE)
    total_prob_temp = cumsum(sorted_probs_temp$x)
    output_temp = min(which(total_prob_temp>=crit_val))
    label_out = sorted_probs_temp$ix[1:output_temp]-1
    conformal_set[[ii]] = label_out
    if(label_out[1] != Y_vec[ii]){
      error_vec[ii] = 1
    }
    if(is.element(Y_vec[ii], label_out )){
      coverage_vec[ii] = 1
    }
    
  }
  error_out = mean(error_vec)
  coverage_out = mean(coverage_vec)
  return(list(conformal_set,coverage_out,error_out))
}

require(lda)
require(nnet)
require(RSpectra)
require(igraph)
require(randomForest)

set.seed(2)
# Load dataset
cora_content = read.table("cora/cora.content.txt",quote="",stringsAsFactors=TRUE,row.names=1,skip=0,comment.char = "")

# Extract common words matrix
cora_x = cora_content[,-ncol(cora_content)]

# Load citation information
cora_adj_list = read.delim("cora/cora.cites")

# Create igraph object for citation graph
cora_obj = graph_from_data_frame(cora_adj_list, directed = FALSE,vertices=as.numeric(rownames(cora_content)))
cora_adj_mat = as.matrix(as_adjacency_matrix(cora_obj))

### Fit embeddings from latent space model of Ma et al (2020)
# M is adjacency matrix, n is the number of observations,
# num_iter_coef relates to the max number of iterations for the
# fitting procedure, k is the dimension of the model, seed_list,
# Z_0, alpha_0 are related to initialization of the model
# and tau_z and tau_alpha are related to step size for stochastic
# gradient descent.  The RCpp code is due to Weijing Tang.
M = matrix(1,nrow(cora_adj_mat),nrow(cora_adj_mat))
n = nrow(cora_adj_mat)
num_iter_coef = 100
rel_tol = 0.01
k=3
Z_0 = matrix(rnorm(n*k, 0, 1), ncol = k)
Z_0 = scale(Z_0, center = TRUE, scale = FALSE)
alpha_0 = runif(n, -5, 5)
sourceCpp("function_PGD_fast.cpp")
logit_latent = Z_estimation_cpp(A_true=cora_adj_mat,M, Z_0, alpha_0, tau_z=0.005, tau_alpha=0.005, num_iter_coef, rel_tol)

# Perform PCA on common word matrix to construct word embeddings
cora_pca = princomp(x=cora_x,scores=TRUE)
cora_top = cora_pca$scores[,1:20]

# Construct RDPG embeddings from adjacency matrix
cora_latent = extract_latent_positions(cora_adj_mat,3,eig_method_extract = "LM")
cora_latent_bind = cbind(cora_latent[[1]],cora_latent[[2]])

# Construct subgraph counts from citation matrix
cora_degrees = rowSums(cora_adj_mat)
cora_triangles = count_triangles(cora_obj)

# Randomly permute indices to construct train, calibration, and test indices
permuted_indices = sample(1:2708)

# Construct neighbor-weighted response variables.  The statistics below are split
# network statistics, which retain finite-sample validity.

# Construct data frames for each set of covariate sets introduced in the paper.
cora_data_0 = data.frame(cora_top)
cora_data_0$Y = factor(ifelse(as.character(cora_content[,ncol(cora_content)]) == "Neural_Networks",1,0),levels=c(0,1))
cora_data_1 = data.frame(Y=cora_data_0$Y,cora_degrees,cora_top,cora_latent_bind)
cora_data_2 = data.frame(Y=cora_data_0$Y,cora_top,logit_latent$Z_0[,1],logit_latent$Z_0[,2],logit_latent$Z_0[,3],as.numeric(logit_latent$alpha_0))

# Construct neighbor-weighted averages of response
cora_neighbor_y = apply(cora_adj_mat[,permuted_indices[1:903]],1, function(x){ if(sum(x) >0) {
  output = 1/sum(x)
  } else {
    output = 0
    }
  return(output)}) *cora_adj_mat[,permuted_indices[1:903]] %*% (as.numeric(cora_data_1$Y[permuted_indices[1:903]])-1)

cora_data_3 = data.frame(Y=cora_data_0$Y,cora_top, cora_degrees,as.numeric(cora_neighbor_y))
cora_data_4 = data.frame(Y=cora_data_0$Y,cora_top, cora_degrees,as.numeric(cora_neighbor_y),cora_latent_bind)
cora_data_5 = data.frame(Y=cora_data_0$Y,cora_top, as.numeric(cora_neighbor_y),logit_latent$Z_0[,1],logit_latent$Z_0[,2],logit_latent$Z_0[,3],as.numeric(logit_latent$alpha_0))

# Initialize training, calibration, and test sets for each set of covariates
cora_data_0_train =cora_data_0[permuted_indices[1:903],]
cora_data_0_cal =cora_data_0[permuted_indices[904:1805],]
cora_data_0_pred = cora_data_0[permuted_indices[1806:2708],]

cora_data_1_train =cora_data_1[permuted_indices[1:903],]
cora_data_1_cal =cora_data_1[permuted_indices[904:1805],]
cora_data_1_pred = cora_data_1[permuted_indices[1806:2708],]

cora_data_2_train =cora_data_2[permuted_indices[1:903],]
cora_data_2_cal =cora_data_2[permuted_indices[904:1805],]
cora_data_2_pred = cora_data_2[permuted_indices[1806:2708],]

cora_data_3_train =cora_data_3[permuted_indices[1:903],]
cora_data_3_cal =cora_data_3[permuted_indices[904:1805],]
cora_data_3_pred = cora_data_3[permuted_indices[1806:2708],]

cora_data_4_train =cora_data_4[permuted_indices[1:903],]
cora_data_4_cal =cora_data_4[permuted_indices[904:1805],]
cora_data_4_pred = cora_data_4[permuted_indices[1806:2708],]

cora_data_5_train =cora_data_5[permuted_indices[1:903],]
cora_data_5_cal =cora_data_5[permuted_indices[904:1805],]
cora_data_5_pred = cora_data_5[permuted_indices[1806:2708],]

# Fit logistic regression on each covariate set
cora_logit_0 = glm(Y~.,data=cora_data_0_train,family=binomial(link='logit'))
cora_logit_1 = glm(Y~.,data=cora_data_1_train,family=binomial(link='logit'))
cora_logit_2 = glm(Y~.,data=cora_data_2_train,family=binomial(link='logit'))
cora_logit_3 = glm(Y~.,data=cora_data_3_train,family=binomial(link='logit'))
cora_logit_4 = glm(Y~.,data=cora_data_4_train,family=binomial(link='logit'))
cora_logit_5 = glm(Y~.,data=cora_data_5_train,family=binomial(link='logit'))

# Fit random forests on each covariate set
cora_rf_0 = randomForest(Y~.,data=cora_data_0_train)
cora_rf_1 = randomForest(Y~.,data=cora_data_1_train)
cora_rf_2 = randomForest(Y~.,data=cora_data_2_train)
cora_rf_3 = randomForest(Y~.,data=cora_data_3_train)
cora_rf_4 = randomForest(Y~.,data=cora_data_4_train)
cora_rf_5 = randomForest(Y~.,data=cora_data_5_train)

# Output predicted probability of class 1 for logistic regression models
cora_pred_cal_0_logit = predict(cora_logit_0, newdata=cora_data_0_cal,"response")
cora_pred_cal_1_logit = predict(cora_logit_1, newdata=cora_data_1_cal,"response")
cora_pred_cal_2_logit = predict(cora_logit_2, newdata=cora_data_2_cal,"response")
cora_pred_cal_3_logit = predict(cora_logit_3, newdata=cora_data_3_cal,"response")
cora_pred_cal_4_logit = predict(cora_logit_4, newdata=cora_data_4_cal,"response")
cora_pred_cal_5_logit = predict(cora_logit_5, newdata=cora_data_5_cal,"response")

# Output predicted probability of class 1 for random forest models
cora_pred_cal_0_rf = predict(cora_rf_0, newdata=cora_data_0_cal,"prob")[,2]
cora_pred_cal_1_rf = predict(cora_rf_1, newdata=cora_data_1_cal,"prob")[,2]
cora_pred_cal_2_rf = predict(cora_rf_2, newdata=cora_data_2_cal,"prob")[,2]
cora_pred_cal_3_rf = predict(cora_rf_3, newdata=cora_data_3_cal,"prob")[,2]
cora_pred_cal_4_rf = predict(cora_rf_4, newdata=cora_data_4_cal,"prob")[,2]
cora_pred_cal_5_rf = predict(cora_rf_5, newdata=cora_data_5_cal,"prob")[,2]

# Create binary vector for Y values in calibration set
Y_cal_numeric = as.numeric(cora_data_3_cal$Y)-1

# Compute non-conformity scores for logistic regression models
cora_scores_0_logit = find_conformal_scores_logit(Y_cal_numeric, cora_pred_cal_0_logit)
cora_scores_1_logit = find_conformal_scores_logit(Y_cal_numeric, cora_pred_cal_1_logit)
cora_scores_2_logit = find_conformal_scores_logit(Y_cal_numeric, cora_pred_cal_2_logit)
cora_scores_3_logit = find_conformal_scores_logit(Y_cal_numeric, cora_pred_cal_3_logit)
cora_scores_4_logit = find_conformal_scores_logit(Y_cal_numeric, cora_pred_cal_4_logit)
cora_scores_5_logit = find_conformal_scores_logit(Y_cal_numeric, cora_pred_cal_5_logit)

# Compute non-conformity scores for random forest models
cora_scores_0_rf = find_conformal_scores_logit(Y_cal_numeric, cora_pred_cal_0_rf)
cora_scores_1_rf = find_conformal_scores_logit(Y_cal_numeric, cora_pred_cal_1_rf)
cora_scores_2_rf = find_conformal_scores_logit(Y_cal_numeric, cora_pred_cal_2_rf)
cora_scores_3_rf = find_conformal_scores_logit(Y_cal_numeric, cora_pred_cal_3_rf)
cora_scores_4_rf = find_conformal_scores_logit(Y_cal_numeric, cora_pred_cal_4_rf)
cora_scores_5_rf = find_conformal_scores_logit(Y_cal_numeric, cora_pred_cal_5_rf)

# Find critical value for logistic regression models
quantile_out_0_logit = quantile(cora_scores_0_logit,0.9*(1+1/nrow(cora_data_1_cal)))
quantile_out_1_logit = quantile(cora_scores_1_logit,0.9*(1+1/nrow(cora_data_1_cal)))
quantile_out_2_logit = quantile(cora_scores_2_logit,0.9*(1+1/nrow(cora_data_1_cal)))
quantile_out_3_logit = quantile(cora_scores_3_logit,0.9*(1+1/nrow(cora_data_1_cal)))
quantile_out_4_logit = quantile(cora_scores_4_logit,0.9*(1+1/nrow(cora_data_1_cal)))
quantile_out_5_logit = quantile(cora_scores_5_logit,0.9*(1+1/nrow(cora_data_1_cal)))

# Find critical value for random forest models
quantile_out_0_rf = quantile(cora_scores_0_rf,0.9*(1+1/nrow(cora_data_1_cal)))
quantile_out_1_rf = quantile(cora_scores_1_rf,0.9*(1+1/nrow(cora_data_1_cal)))
quantile_out_2_rf = quantile(cora_scores_2_rf,0.9*(1+1/nrow(cora_data_1_cal)))
quantile_out_3_rf = quantile(cora_scores_3_rf,0.9*(1+1/nrow(cora_data_1_cal)))
quantile_out_4_rf = quantile(cora_scores_4_rf,0.9*(1+1/nrow(cora_data_1_cal)))
quantile_out_5_rf = quantile(cora_scores_5_rf,0.9*(1+1/nrow(cora_data_1_cal)))

# Compute estimated probabilities on test set for logistic regression models
new_predictions_cora_0_logit = predict(cora_logit_0, newdata=cora_data_0_pred,"response")
new_predictions_cora_1_logit = predict(cora_logit_1, newdata=cora_data_1_pred,"response")
new_predictions_cora_2_logit = predict(cora_logit_2, newdata=cora_data_2_pred,"response")
new_predictions_cora_3_logit = predict(cora_logit_3, newdata=cora_data_3_pred,"response")
new_predictions_cora_4_logit = predict(cora_logit_4, newdata=cora_data_4_pred,"response")
new_predictions_cora_5_logit = predict(cora_logit_5, newdata=cora_data_5_pred,"response")

# Compute estimated probabilities on test set for random forest models
new_predictions_cora_0_rf = predict(cora_rf_0,newdata=cora_data_0_pred,"prob")[,2]
new_predictions_cora_1_rf = predict(cora_rf_1, newdata=cora_data_1_pred,"prob")[,2]
new_predictions_cora_2_rf = predict(cora_rf_2, newdata=cora_data_2_pred,"prob")[,2]
new_predictions_cora_3_rf = predict(cora_rf_3, newdata=cora_data_3_pred,"prob")[,2]
new_predictions_cora_4_rf = predict(cora_rf_4, newdata=cora_data_4_pred,"prob")[,2]
new_predictions_cora_5_rf = predict(cora_rf_5, newdata=cora_data_5_pred,"prob")[,2]

# Create binary vector for Y values in test set
Y_pred = as.numeric(cora_data_0_pred$Y)-1

# Output lists related to conformal prediction sets for logistic regression
cp_logit_list_0 = check_coverage_logit(prob_vec=new_predictions_cora_0_logit,Y_vec=Y_pred,crit_val = quantile_out_0_logit)
cp_logit_list_1 = check_coverage_logit(prob_vec=new_predictions_cora_1_logit,Y_vec=Y_pred,crit_val = quantile_out_1_logit)
cp_logit_list_2 = check_coverage_logit(prob_vec=new_predictions_cora_2_logit,Y_vec=Y_pred,crit_val = quantile_out_2_logit)
cp_logit_list_3 = check_coverage_logit(prob_vec=new_predictions_cora_3_logit,Y_vec=Y_pred,crit_val = quantile_out_3_logit)
cp_logit_list_4 = check_coverage_logit(prob_vec=new_predictions_cora_4_logit,Y_vec=Y_pred,crit_val = quantile_out_4_logit)
cp_logit_list_5 = check_coverage_logit(prob_vec=new_predictions_cora_5_logit,Y_vec=Y_pred,crit_val = quantile_out_5_logit)

# Compute average set sizes (widths) of prediction sets for logistic regression
size_0_logit= mean(sapply(cp_logit_list_0[[1]], function(x) {return(length(x))}))
size_1_logit = mean(sapply(cp_logit_list_1[[1]], function(x) {return(length(x))}))
size_2_logit = mean(sapply(cp_logit_list_2[[1]], function(x) {return(length(x))}))
size_3_logit = mean(sapply(cp_logit_list_3[[1]], function(x) {return(length(x))}))
size_4_logit = mean(sapply(cp_logit_list_4[[1]], function(x) {return(length(x))}))
size_5_logit = mean(sapply(cp_logit_list_5[[1]], function(x) {return(length(x))}))

# Output lists related to conformal prediction sets for random forests
cp_rf_list_0 = check_coverage_logit(prob_vec=new_predictions_cora_0_rf,Y_vec= Y_pred,crit_val = quantile_out_0_rf)
cp_rf_list_1 = check_coverage_logit(prob_vec=new_predictions_cora_1_rf,Y_vec= Y_pred,crit_val = quantile_out_1_rf)
cp_rf_list_2 = check_coverage_logit(prob_vec=new_predictions_cora_2_rf,Y_vec= Y_pred,crit_val = quantile_out_2_rf)
cp_rf_list_3 = check_coverage_logit(prob_vec=new_predictions_cora_3_rf,Y_vec= Y_pred,crit_val = quantile_out_3_rf)
cp_rf_list_4 = check_coverage_logit(prob_vec=new_predictions_cora_4_rf,Y_vec= Y_pred,crit_val = quantile_out_4_rf)
cp_rf_list_5 = check_coverage_logit(prob_vec=new_predictions_cora_5_rf,Y_vec= Y_pred,crit_val = quantile_out_5_rf)

# Compute average set sizes (widths) of prediction sets for random forests
size_0_rf= mean(sapply(cp_rf_list_0[[1]], function(x) {return(length(x))}))
size_1_rf = mean(sapply(cp_rf_list_1[[1]], function(x) {return(length(x))}))
size_2_rf = mean(sapply(cp_rf_list_2[[1]], function(x) {return(length(x))}))
size_3_rf = mean(sapply(cp_rf_list_3[[1]], function(x) {return(length(x))}))
size_4_rf = mean(sapply(cp_rf_list_4[[1]], function(x) {return(length(x))}))
size_5_rf = mean(sapply(cp_rf_list_5[[1]], function(x) {return(length(x))}))




