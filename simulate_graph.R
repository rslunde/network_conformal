require(methods)

# Eigenfunctions for data generating process in sim_study_1.R  
rdpg_eigenfunction = function(x,k) {
  return(sin((2*k-1)*(pi*x)/2))
}

# Eigenvalues for data generating process in sim_study_1.R  
rdpg_eigenvalues= function(k){
  return((2/((2*k-1)*pi))^2)
}

# Function to generate an adjacency matrix from a nxn matrix
# of probability of edge formation. 

simulate_from_P_mat = function(P_mat){
  cppFunction( 'arma::mat my_function( int n, NumericMatrix P_mat
  ) {
               arma::mat A_mat(n,n,arma::fill::zeros);
               
               for (int ii = 0; ii < n; ii++)
               {
               for(int jj = ii+1; jj < n; jj++) 
               {
               
               double P_temp = P_mat(ii,jj);
               if( P_temp < 0) {
               P_temp = 0;
               }
               
               A_mat(ii,jj) = R::rbinom(1,P_temp);
               }
               }
               A_mat = arma::symmatu(A_mat);
               return(A_mat);
}
',depends = "RcppArmadillo")
  
  output = my_function(n= nrow(P_mat),P_mat=P_mat)
  return(output)
}

#   Function that computes RDPG embeddings from an adjacency matrix.  The function takes as input
#   nxn adjacency matrix, the dimension of the embedding, and a criteria for what constitutes the top
#   eigenvalue (e.g. largest magnitude, which is the default setting).  The method_extract is an
#   argument for the eigs_sym function in the RSpectra package.

extract_latent_positions = function(Adj_mat,latent_dim,eig_method_extract="LM") {
    k_vec = 1:latent_dim
    eigen_decomp = eigs_sym(Adj_mat,k=latent_dim,which=eig_method_extract)
    pos_eigs_index = k_vec[which(eigen_decomp$values > 0)]
    neg_eigs_index = k_vec[-pos_eigs_index]
    
    pos_component = eigen_decomp$vector[,pos_eigs_index]%*% diag(x=eigen_decomp$values[pos_eigs_index]^0.5,nrow= length(pos_eigs_index))
    neg_component = numeric(0)
    
    if(length(neg_eigs_index) != 0) {
        neg_component =eigen_decomp$vector[,neg_eigs_index]%*% diag(x=(-1*eigen_decomp$values[neg_eigs_index])^0.5,nrow= length(neg_eigs_index))
    }
    latent_positions = list(pos_component,neg_component)
    return(latent_positions)
    }

# Function to generate probability matrix from Gaussian latent space
# model considered in sim_study_2.R
generate_P_mat_glsm = function(latent_vec){
  n = length(latent_vec)
  P_mat_out = matrix(0,nrow=n,ncol=n)
  for(ii in 1:(n-1)){
    for(jj in 1:(n-ii)) {
      P_mat_out[ii,ii+jj] = exp(-(latent_vec[ii]-latent_vec[ii+jj])^2/4)
    }
  }
  P_mat_out[lower.tri(P_mat_out)] = t(P_mat_out)[lower.tri(P_mat_out)]
  return(P_mat_out)
}

# Function to generate adjacency matrix from graphon 
# model considered in sim_study_3.R

# This function takes as input a graphon object and outputs a function that simulate
# from the graphon.  The output function takes as input a sample size n and ouputs
# an nxn adjacency matrix

# If P_mat= TRUE, the resulting function will take as input a sample size n
# and ouput an adjacency matrix, the probability matrix, and the uniformly
#distributed latent positions.

# The (sparse) graphon object takes as input the graphon function, which
# is passed as a string to RCpp, and a sparsity function.


setClass("graphon",representation(w_function="character",nu_seq = "function"))

create_simulation_function.graphon = function(graphon,return_P_mat =FALSE) {
  
  cppFunction( 'arma::mat my_function_1( int n, double nu) {
               arma::mat A_mat(n,n,arma::fill::zeros);
               NumericVector latent_positions = runif(n,0,1);
               for (int ii = 0; ii < n; ii++)
               {
               for(int jj = ii+1; jj< n; jj++) 
               {
               double P_temp = nu * w_function(latent_positions[ii],latent_positions[jj]);
               A_mat(ii,jj) =  R::rbinom(1,P_temp);
               }
               }
               A_mat = arma::symmatu(A_mat);
               return(A_mat);
}
',depends = "RcppArmadillo", includes = graphon@w_function)
  
  cppFunction( 'List my_function_2( int n, double nu) {
               arma::mat A_mat(n,n,arma::fill::zeros);
               arma::mat P_mat(n,n,arma::fill::zeros);
               NumericVector latent_positions = runif(n,0,1);
               for (int ii = 0; ii < n; ii++)
               {
               for(int jj = ii+1; jj< n; jj++) 
               {
               double P_temp = nu * w_function(latent_positions[ii],latent_positions[jj]);
               A_mat(ii,jj) =  R::rbinom(1,P_temp);
               P_mat(ii,jj) = P_temp;
               }
               }
               A_mat = arma::symmatu(A_mat);
               P_mat = arma::symmatu(P_mat);
               List ret;
               ret["A_mat"] = A_mat;
               ret["P_mat"] = P_mat;
               ret["latent_positions"] = latent_positions;
               return ret;
}
',depends = "RcppArmadillo", includes = graphon@w_function) 
  
  
  if(return_P_mat){  
    output = function(n) {
      return( my_function_2(n=n,nu=graphon@nu_seq(n)))
    }
    return(output)
  }
  else{
    output = function(n) {
      return( my_function_1(n=n,nu=graphon@nu_seq(n)))
      
      return(output)
    }
  }
}

# The graphon function considered in the simulation study is given below. 

graphon_1 = 'double w_function( double u, double v) {
return(std::abs(u-v));
}'


