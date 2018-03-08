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



H = sigmoid(X * theta);
logH = log(H);
logHmin1 = log(1 - H);


logRegH = log(H(2,:));
logRegHmin1 = log(1 - H(2,:));


%J = transpose(yReg) * logRegH + transpose(1 - yReg) * logRegHmin1;
%J
%J = -J/m + ( transpose(thetaReg) * thetaReg )* lambda/ (2 * m);

%J = J - (y(1) * log(H(1)) + (1 - y(1)) * log(1 - H(1)) / m);

%fprintf("J (1) = %f",J);

grad = 1/m .* transpose(X) *(H - y) + theta * lambda /m;

grad(1) = grad(1) - theta(1) * (lambda / m);

grad;

%J =  ((transpose(y) * logH )+ (transpose(1 - y) * logHmin1))/(-1 * m) + ((transpose(theta) * theta )* lambda / (2 * m) )- (theta(1) * theta(1) * lambda/(2 * m));
J =  ((transpose(y) * logH )+ (transpose(1 - y) * logHmin1))/(-1 * m) ;
J = J + (transpose(theta(2:end,:)) * theta(2:end,:) ) * lambda/(2 * m) ; 
J
%J = J + (transpose(theta )* theta ) * lambda/(2 * m) ; 
%J
%fprintf("J (2) = %f",J);

%xJ =  -1 * ( y(1) * logH(1) + ( 1 - y(1)) * logHmin1(1))/m;
%for i = 2 : m

%J = J -  ( y(i) * logH(i) + ( 1 - y(i)) * logHmin1(i))/m ;

%end

%for i = 2:length(theta)
%J = J + lambda * theta(i) * theta(i)/(2 * m) ;
%end

%for i = 1:length(theta)

%  for j = 1 : m
%    grad(i) = grad(i) + (H(j) - y(j)) * X(j,i) /m ;
%  end

%   if (i > 1)
%    grad(i) = grad(i) + theta(i) * lambda / m ;
%    end
%end



% =============================================================

end
