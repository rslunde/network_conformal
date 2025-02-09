// Code to simulate from latent space model of Ma et al (2020) provided
// by Weijing Tang. 


# include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
Rcpp::List Z_estimation_cpp(
    arma::mat A_true,
    arma::mat M,
    arma::mat Z_0,
    arma::colvec alpha_0,
    double tau_z,
    double tau_alpha,
    int num_iter,
    double rel_tol
){
  int n = Z_0.n_rows;
  int k = Z_0.n_cols;
  arma::colvec one_col = arma::ones<arma::colvec>(n);
  arma::mat A_abs = abs(A_true);
  bool stopflag = FALSE;
  int iter = 0;
  arma::mat Theta = alpha_0 * one_col.t() + one_col * alpha_0.t() + Z_0 * Z_0.t();
  arma::colvec  nll_collect = arma::zeros<arma::colvec>(num_iter); // negative log-likelihood
  arma::mat tsr = (A_abs % Theta + log(1 - 1/(1+exp(-Theta)))) % M;
  nll_collect(0) = - accu(tsr) / pow((double)n,2);
  
  arma::mat Z_1 = arma::zeros<arma::mat>(n,k);
  arma::colvec alpha_1 = arma::zeros<arma::colvec>(n);
  arma::rowvec mvec = arma::zeros<arma::rowvec>(k);
  while (!stopflag) {
    Z_1 = Z_0 + 2 * tau_z * ((M % (A_abs - 1/(1+exp(-Theta)))) * Z_0);  // Gradient descent step
    alpha_1 = alpha_0 + 2 * tau_alpha * ((M % (A_abs - 1/(1+exp(-Theta)))) * one_col);
    mvec = arma::mean(Z_1,0);
    Z_1.each_row() -= mvec;  // Centering Z

    double grad_norm_z = pow(norm(Z_1 - Z_0, "fro"),2);
    double grad_norm_alpha = pow(norm(alpha_1 - alpha_0, "fro"),2);
    Z_0 = Z_1; // Prepare for the next iteration
    alpha_0 = alpha_1;
    Theta = alpha_0 * one_col.t() + one_col * alpha_0.t() + Z_0 * Z_0.t();

    tsr = (A_abs % Theta + log(1 - 1/(1+exp(-Theta)))) % M;
    double nll_1 = - accu(tsr) / pow((double)n,2);
    if (iter - floor(iter / (int)50) * (int)50 == 0){
      Rprintf("%i iteration loss: %f\n", iter, nll_1);
    }
    double rel_nll = (nll_collect(iter) - nll_1) / nll_collect(iter);
    if (rel_nll < 0) {rel_nll *= -1;}
    

    iter++; // Iteration number controller
    if (grad_norm_z <= rel_tol && grad_norm_alpha <= rel_tol && rel_nll <= rel_tol){
      stopflag = TRUE;
      Rprintf("Z: separate algorithm converged at iteration %i ! \n", iter);
    } else {
      if (iter >= num_iter){
        stopflag = TRUE;
        Rprintf("Z: separate algorithm achieved maximum iteration number!");
      } else {
        nll_collect(iter) = nll_1;
      }
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("Z_0")=Z_0, 
                            Rcpp::Named("alpha_0")=alpha_0, 
                            Rcpp::Named("nll")=nll_collect);
}

