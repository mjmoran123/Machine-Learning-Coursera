function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

y_matrix = eye(num_labels)(y,:);
X = [ones(m, 1) X];
z2 = Theta1 * X';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2'];
z3 = Theta2 * a2';
a3 = sigmoid(z3);
a3 = a3';

thing = -1 * y_matrix .* log(a3) - (1 .- y_matrix) .* log(1 .- a3);
temp = sum(thing(:)) / m;

T1_squared = Theta1 .^ 2;
T2_squared = Theta2 .^ 2;
T1_squared = T1_squared(:,2:end);
T2_squared = T2_squared(:,2:end);
reg_thing = sum((T1_squared)(:)) + sum((T2_squared)(:));
J = temp + (lambda / (2 * m)) * reg_thing;


%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

%for t = 1:m
%	a1_t = X(t,:);
%	z2_t = Theta1 * a1_t';
%	a2_t = sigmoid(z2_t);
%	a2_t = [1; a2_t];
%	z3_t = Theta2 * a2_t;
%	a3_t = sigmoid(z3_t); 
%	new_y_matrix = transpose(y_matrix);
%	d3 = a3_t - new_y_matrix(:,t);
%	d2 = Theta2' * d3 .* sigmoidGradient([1; z2_t]);
%	d2 = d2(2:end);
%	Delta1 = Delta1 + d2 * a2_t';
%	Delta2 = Delta2 + d3 * a2_t';
	
%endfor

d3 = a3 - y_matrix;
d2 = sigmoidGradient(z2)' .* (d3 * Theta2(:,2:end));
Delta1 = d2' * X;
Delta2 = d3' * a2;
Theta1_grad = Delta1 * (1 / m);
Theta2_grad = Delta2 * (1 / m);


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1 = (lambda / m) * Theta1;
Theta2 = (lambda / m) * Theta2;

Theta1_grad += Theta1;
Theta2_grad += Theta2;

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
