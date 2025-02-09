# A function that takes as input a nx2 matrix and a length n scalar
# and outputs the proportion of times the scalar is sandwiches between
# the entries of the matrix.
check_coverage = function(my_mat,my_vec){
  n= length(my_vec)
  binary_vec = rep(0,length(my_vec))
  
  for(ii in 1:n){
    if( (my_mat[ii,1] <= my_vec[ii]) && (my_mat[ii,2] >= my_vec[ii])){
      binary_vec[ii]=1
    } 
  }
  output = mean(binary_vec)
  return(output)
}

# A function that takes as input a nx2 matrix and ouputs the average
# difference between the left and right points.  
check_width = function(my_mat){
  output = mean((my_mat[,2] - my_mat[,1]))
  return(output)
}