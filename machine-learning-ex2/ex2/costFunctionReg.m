function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sum = 0;
for i = 1:m
	sum += (-y(i) * log(sigmoid(X(i,:) * theta))) - ((1 - y(i)) * log(1 - (sigmoid(X(i,:) * theta))));
endfor

reg_sum = 0;
for j = 2:size(theta)
	reg_sum += theta(j) * theta(j);
endfor
J = (sum / m) + ((reg_sum * lambda) / (2 * m));



diffy = sigmoid(X * theta) - y;




grad(1) = ((diffy .* X(:,1))' * ones(m,1)) / m;




for j = 2:columns(X)
	grad(j) = ((diffy .* X(:,j))' * ones(m,1)) / m + (lambda / m) * theta(j);
endfor





% =============================================================

end
