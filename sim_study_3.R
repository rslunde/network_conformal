require(np)
require(Rcpp)
require(RSpectra)
require(RcppArmadillo)
source("simulate_graph.R")
source("check_ci_properties.R")

#Code for experiments in Section 5.2

# Sample size and sparsity settings considered in the paper
# are initialized in the following vectors
n_vec = c(1000,2000,3000)
sparsity_vec = c(-0.1,-0.2,-0.5,-0.75)

# Number of iterations is initialized below
B=500
alpha = 0.1

sparsity_level = 3
n = n_vec[3]

# Initialize graphon
my_graphon = graphon_1

# Initialize degree function associated with graphon
my_degree_fun = function(x){
  return(x^2-x+0.5)
}

# Initialize vector to store degree of test point 
degree_vec = rep(NA,B)

# Initialize matrices that store prediction intervals
cp_mat_cond = matrix(NA,nrow=B,ncol=2)
cp_mat_lm = matrix(NA,nrow=B,ncol=2)
cp_mat_spline = matrix(NA,nrow=B,ncol=2)

# Create new graphon object
graphon_obj = new("graphon",w_function=my_graphon,nu_seq=function(x)return(x^sparsity_vec[sparsity_level]))

# Create function to simulate from (sparse) graphon
graphon_sim_fun = create_simulation_function.graphon(graphon_obj,return_P_mat=TRUE)

n_train = n/2
n_calib = n_train-1
  
for(ii in 1:B){
    # Simulate from graphon function
    output_new = graphon_sim_fun(n)
    
    # Compute degree covariate associated with each node. output_new[[3]] contains
    # latent positions of the graphon model
    degree_covariate = my_degree_fun(output_new[[3]])
    
    # Simulate from the data generating process
    Y_vec = 4+5*sin(3*pi*degree_covariate) + exp(15*degree_covariate)/250 *rnorm(n,0,1)
    
    # Compute degrees, which will be used as a regressor
    degrees = colMeans(output_new[[1]])
    
    # Store value of the response for the test point
    Y_test = Y_vec[n]
   
    # Construct data frame, which will be used to fit models
    my_data = data.frame(degrees,Y_vec)
    my_data_train = my_data[1:n_train,]
    my_data_calib = my_data[(n_train+1):(n_train+n_calib),]
    
    # Estimate CDF using kernel regression
    cdf_obj_train = with(my_data_train,npcdistbw(formula=Y_vec~degrees))
    
    # Evaluate CDF object on calibration set
    cdf_obj_calib = npcdist(cdf_obj_train,newdata=my_data_calib)
    
    # Construct non-conformity score for estimated CDF
    nonconformity_scores_cond = abs(1/2- cdf_obj_calib$condist)
    
    # Construct non-conformity score for linear model
    lm_train = lm(Y_vec~degrees,data=my_data_train)
    nonconformity_scores_lm = abs(my_data_calib$Y_vec - predict(lm_train,newdata=my_data_calib))
    
    # Construct non-conformity score for smoothing plines
    spline_train = smooth.spline(x=my_data_train$degrees,y=my_data_train$Y_vec)
    nonconformity_scores_spline = abs(my_data_calib$Y_vec - predict(spline_train,x=my_data_calib$degrees)$y)
    
    # Find critical value for non-conformity scores
    d_cond = quantile(nonconformity_scores_cond, (1-alpha)*(1+1/n_calib))
    d_lm = quantile(nonconformity_scores_lm, (1-alpha)*(1+1/n_calib))
    d_spline = quantile(nonconformity_scores_spline, (1-alpha)*(1+1/n_calib))
    
    # For non-conformity score based on CDF, conduct a grid search over y to find left and right
    # end points of the interval.
    
    # Note that the y input is the point at which the CDF is evaluated. 
    # Since the estimated CDF value changes only at y in the training set, it suffices to consider
    # a grid of y consisting only of these points.
    grid_test = data.frame(degrees=my_data$degrees[n],Y_vec=my_data_train$Y_vec)
    cdf_obj_train_eval = npcdist(cdf_obj_train,newdata=grid_test)
    l.end = which.min( ifelse( ( (1/2-  cdf_obj_train_eval$condist - d_cond) > 0), (1/2-  cdf_obj_train_eval$condist - d_cond),2))
    r.end = which.min( ifelse( (1/2+d_cond -cdf_obj_train_eval$condist) > 0 ,(1/2+d_cond - cdf_obj_train_eval$condist),2))
    cp_mat_cond[ii,] = c(Y_vec[l.end], Y_vec[r.end])
    
    # Store prediction interval for linear model
    lm_pred = predict(lm_train,newdata=data.frame(degrees=degrees[n]))
    lm_cp = c(lm_pred-d_lm, lm_pred+d_lm)
    cp_mat_lm[ii,] = lm_cp
    
    # Store prediction interval for smoothing spline
    spline_pred = predict(spline_train,degrees[n])
    spline_cp = c(spline_pred$y-d_spline, spline_pred$y+d_spline)
    cp_mat_spline[ii,] = spline_cp
  
  }

# The following code produces conditional coverage plots
require(ggplot2)

# Compute coverage for conformal prediction intervals
cp_cond_cov = check_coverage(cp_mat_cond,Y_test_vec)
cp_lm_cov = check_coverage(cp_mat_lm,Y_test_vec)
cp_spline_cov = check_coverage(cp_mat_spline,Y_test_vec)

# Fit smoothing splines to estimate coverage probability as a function of z
spline_cond = smooth.spline(x=degree_vec,y=cp_cond_cov)
spline_lm = smooth.spline(x=degree_vec,y=cp_lm_cov)
spline_spline = smooth.spline(x=degree_vec,y=cp_spline_cov)

# Create data frame for ggpplot
y_vals = c(spline_cond$y, spline_lm$y, spline_spline$y)
x_vals = rep(spline_cond$x,3)
method= c(rep("Conditional CDF",B), rep("Linear Model",B),rep("Spline",B))
plot_data_frame = data.frame(y_vals,method,x_vals)

# Create ggplot object
p = ggplot(data = plot_data_frame, aes(x=x_vals, y = y_vals, group=method)) +
  geom_line(aes(linetype=method,color=method))+
  geom_hline(yintercept=0.9,linetype="dotted") +
  labs(x="z", y="coverage") + ylim(0.15, 1.1) + xlim(0.25,0.5)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))
