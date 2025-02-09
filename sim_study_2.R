require(Rcpp)
require(mvtnorm)

source("simulate_graph.R")
source("check_ci_properties.R")

## Code for experiments in Section 5.1.2 related to spatial autoregressive models

# Sample size and sparsity settings considered in the paper
# are initialized in the following vectors

n_vec = c(1000,3000)
sparsity_vec = c(-0.1,-0.25,-0.5,-0.75)

# Size of superpopulation is initialized below
N=3000

# Choose sample size, sparsity level, and number of simulations
B=500

which_n = 1
n = n_vec[which_n]
sparsity_level = sparsity_vec[3]

# Initialize sparsity level
rho = N^sparsity_level

# Initialize vector to store test point
Y_vec = rep(NA,B)

# Initialize matrices to store prediction intervals 
lm_1_normal_pi_mat = matrix(NA,nrow=B,ncol=2)
lm_2_normal_pi_mat = matrix(NA,nrow=B,ncol=2)
lm_3_normal_pi_mat = matrix(NA,nrow=B,ncol=2)
lm_1_cp_mat = matrix(NA,nrow=B,ncol=2)
lm_2_cp_mat = matrix(NA,nrow=B,ncol=2)
lm_3_cp_mat = matrix(NA,nrow=B,ncol=2)

for(ii in 1:B) {
  print(paste("ii = ",ii))
  
  # Generate covariates
  X_vec = rmvnorm(N,mean=c(1,3,0), sigma = matrix(c(1,0.6 ,0.3, 0.6,4,-0.4,0.3,-0.4,1 ),nrow=3,ncol=3,byrow=T))
  
  # Generate probability matrix according to a Gaussian latent space model
  P_mat_new = generate_P_mat_glsm(X_vec[,3])
  
  # Generate adjacency matrix from probability matrix
  Adj_mat_new = simulate_from_P_mat(rho*P_mat_new)
  
  # Compute a nxn matrix with inverse of degrees on the diagonal.  If degree is 0, inverse is also set to 0.
  D_mat_inv = diag(ifelse( colSums(Adj_mat_new) > 0, 1/colSums(Adj_mat_new),0),nrow=N)
  
  # Generate Y from spatial autoregressive model using degree matrix constructed above as normalization
  Y_new = solve(diag(1,nrow=N) - 0.7*D_mat_inv %*%Adj_mat_new)%*%(D_mat_inv %*%Adj_mat_new%*%X_vec[,1:2]%*%c(2,3) +2+ X_vec[,1:2] %*%c(4,5) + rnorm(N))
  my_data = data.frame(Y=Y_new,X1=X_vec[,1],X2=X_vec[,2], X3=X_vec[,3])
    
  # The following inverse degree matrix excludes the test point and observations 1000-2999 in the case
  # where n=1000.  This inverse degree matrix is used to construct network covariates.
  if(which_n==1){
      Adj_mat_new_2 = Adj_mat_new[1:(n-1), 1:(n-1)]
      D_mat_inv_2 = diag(ifelse( colSums(Adj_mat_new_2) > 0, 1/colSums(Adj_mat_new_2),0),nrow=(n-1))
    } else {
      D_mat_inv_2 = D_mat_inv[-n,-n]    
    }
    
  # Construct network covariates for different models
  X_neighb_new = D_mat_inv_2 %*%Adj_mat_new[-(n:N),-(n:N)]%*%X_vec[-(n:N),1:2]
  Y_neighb_new = D_mat_inv_2 %*%Adj_mat_new[-(n:N),-(n:N)]%*%Y_new[-(n:N)]
  my_data_new = my_data[-(n:N),]
  my_data_new$X4 = X_neighb_new[,1]
  my_data_new$X5 = X_neighb_new[,2]
  my_data_new$X6 = as.numeric(Y_neighb_new)
    
  # Set size of training data
  n_train = n/2
    
  # Construct data frames for training and calibration sets
  my_data_train = my_data_new[1:n_train,]
  my_data_calib = my_data_new[(n_train+1):(n-1),]
    
  # Construct data frame for test point
  my_data_test = my_data[n,]
    
  # Construct network covariates for test point
  X_neighb_test= ifelse( rep(sum(Adj_mat_new[1:n,n]),2) > 0,  as.numeric(Adj_mat_new[n,1:n] %*% X_vec[1:n,1:2]/sum(Adj_mat_new[1:n,n])),0)
  Y_neighb_test = ifelse( sum(Adj_mat_new[1:n,n]) > 0,  Adj_mat_new[n,1:n] %*% Y_new[1:n]/sum(Adj_mat_new[1:n,n]),0)
  my_data_test$X4 = X_neighb_test[1]
  my_data_test$X5 = X_neighb_test[2]
  my_data_test$X6 = Y_neighb_test
    
  # Store true value of the response for test point
  Y_vec[ii] = my_data_test$Y
    
  # Fit linear models with different covariates
  my_model_1 = lm(Y~X1+X2,data=my_data_train)
  my_model_2 = lm(Y~X1+X2+X4+X5,data=my_data_train)
  my_model_3 = lm(Y~X1+X2+X4+X5+X6,data=my_data_train)

  # Construct standard Gaussian linear model prediction intervals
  pred_par_1 = predict(my_model_1,newdata=my_data_test,interval="prediction", alpha =0.9)
  pred_par_2 = predict(my_model_2,newdata=my_data_test,interval="prediction",alpha =0.9)
  pred_par_3 = predict(my_model_3,newdata=my_data_test,interval="prediction",alpha =0.9)
  
  # Store Gaussian linear model prediction intervals
  lm_1_normal_pi_mat[ii,] = pred_par_1[2:3]
  lm_2_normal_pi_mat[ii,] = pred_par_2[2:3]
  lm_3_normal_pi_mat[ii,] = pred_par_3[2:3]
  
  # Run conformal prediction
  cp_radius_1 = quantile(abs(my_data_calib$Y - predict(my_model_1,newdata=my_data_calib)),0.9*(1+1/(n_train-1)))
  cp_radius_2 =  quantile(abs(my_data_calib$Y - predict(my_model_2,newdata=my_data_calib)),0.9*(1+1/(n_train-1)))
  cp_radius_3 =  quantile(abs(my_data_calib$Y - predict(my_model_3,newdata=my_data_calib)),0.9*(1+1/(n_train-1)))
  
  # Construct and store conformal prediction intervals
  lm_1_cp_mat[ii,] = c(pred_par_1[1]-cp_radius_1, pred_par_1[1]+cp_radius_1)
  lm_2_cp_mat[ii,] = c(pred_par_2[1]-cp_radius_2, pred_par_2[1]+cp_radius_2)
  lm_3_cp_mat[ii,] = c(pred_par_3[1]-cp_radius_3, pred_par_3[1]+cp_radius_3)
  }
