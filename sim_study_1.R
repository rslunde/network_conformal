require(RSpectra)
require(Rcpp)
source("check_ci_properties.R")
source("simulate_graph.R")

# Code for experiments in Section 5.1.1

# Sample size and sparsity settings considered in the paper
# are initialized in the following vectors
n_vec = c(500,1000,2000,3000)
sparsity_vec = c(-0.1,-0.25,-0.5,-0.75)

# Choose sample size, sparsity level, and number of simulations 
n=n_vec[2]
sparsity_level = sparsity_vec[4]
B = 500

# Initialize matrices that store prediction intervals constructed
# from Gaussian linear model approach and conformal prediction
lm_pred_int_mat = matrix(NA,nrow=B,ncol=2)
cp_pred_int_mat = matrix(NA,nrow=B,ncol=2)

# Initialize a vector that stores test value for Y on 500 runs  
Y_true= rep(NA,B)

# Set sample sizes for training set and calibration sets 
n_train = n/2
n_calib = n/2 -1

# Loop over B iterations to assess coverage and average width 
for(ii in 1:B){
  print(paste("ii=",ii))

# Simulate uniform latent variables
latent_unifs = runif(n)

# Initialize matrix of latent variables for the RDPG model
latent_mat = matrix(0,n,ncol=3)
K=3
# Generate latent positions from RDPG model
for(jj in 1:K){
  latent_mat[,jj] = (rdpg_eigenvalues(jj))^0.5* rdpg_eigenfunction(latent_unifs,jj)  
}

# Generate Y and X
X_vec = runif(n,1,2)*4*latent_mat[,1] +rnorm(n)
Y_vec = 3 + 2*X_vec +10*latent_mat[,1] +15*latent_mat[,2]-17*latent_mat[,3]+rnorm(n)
Y_true[ii] = Y_vec[n]
# Set Sparsity level
rho = n^sparsity_level

# Construct the matrix P, where P_ij gives the probability of an edge
P_mat_new = rho*latent_mat %*% t(latent_mat)

# Construct adjacency matrix from P
Adj_mat_new=  simulate_from_P_mat(P_mat_new)

# Estimate latent positions using adjacency spectral embedding
latent_vars_est = extract_latent_positions(Adj_mat_new,latent_dim = 3)
latent_vars_est = cbind(latent_vars_est[[1]],latent_vars_est[[2]])

# Fit linear model on training data
data_all = data.frame(Y_vec, cbind(X_vec,latent_vars_est)) 
colnames(data_all) = c("Y","X1","X2","X3","X4")
data_train = data_all[1:n_train,]
lm_train = lm(Y~.,data=data_train)
my_data  = data_all[n,]

# Predict for test point 
lm_predict = predict(lm_train, newdata=my_data,interval="predict",level=0.9)

# Store prediction interval generated from Gaussian linear model approach
lm_pred_int_mat[ii,] = lm_predict[c(2,3)]

# Implement split conformal prediction
calib_indices = (n_train+1):(n-1)
calib_data = data_all[calib_indices,]

# Compute non-conformity score on calibration set
abs_err = abs(Y_vec[calib_indices] - predict(lm_train,newdata =calib_data)) 
cp_radius = quantile(abs_err, 0.9*(1+1/n_calib))

# Store prediction interval
cp_pred_int_mat[ii,] =  c(lm_predict[1]-cp_radius,lm_predict[1]+ cp_radius)
}

# Compute coverage of prediction intervals
check_coverage(cp_pred_int_mat,Y_true)
check_coverage(lm_pred_int_mat,Y_true)

# Compute average width of prediction intervals
check_width(cp_pred_int_mat)
check_width(lm_pred_int_mat)



