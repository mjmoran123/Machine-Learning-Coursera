function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%X = [ones(size(X, 1), 1) X]
X_T = transpose(X);
theta_T = transpose(theta);
h = theta_T * X_T;
h_T = transpose(h);
diff = h_T .- y;
diff_squared = diff .^ 2;
sub = (0.5 * (1 / m)) * sum(diff_squared);


reg = (0.5) * (lambda) * (1 / m) * sum(theta(2:end) .^ 2);

J = sub + reg;


acc = X * theta;
grad = ((X_T * (acc - y)) / m) + ((lambda / m) * [0; theta(2:end)]);











% =========================================================================

grad = grad(:);

end
