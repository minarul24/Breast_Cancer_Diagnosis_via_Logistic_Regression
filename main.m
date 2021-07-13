%
%Author:     Minarul Shawon
%Date:       July 1, 2021
%
%   Breast Cancer Diagnosis via Logistic Regression
%

load D_wdbc.mat

%some functions
%
%confusion matrix
function [c,accuracy] = confusion(predicted,actual)
c = zeros(2,2);
N = size(predicted,2);

for i = 1:N
    if predicted(i) == 1 && actual(i) == 1
        c(1,1) = c(1,1) + 1;
    end
    if predicted(i) == 1 && actual(i) == -1
        c(1,2) = c(1,2) + 1;
    end
    if predicted(i) == -1 && actual(i) == 1
        c(2,1) = c(2,1) + 1;
    end
    if predicted(i) == -1 && actual(i) == -1
        c(2,2) = c(2,2) + 1;
    end
accuracy = (trace(c)./sum(c,'all'))*100;
%end of confusion matrix


%main
D_tr = D_wdbc(:,1:285);
D_te = D_wbdc(:,286:569);

y_tr = D_tr(31,:);
y_te = D_te(31,:);

X_tr = zeros(30,285);
%for mean
m = zeros(1,30);
%for variance
v = zeros(1,30);

for i = 1:30
    x_i = D_tr(i,:);
    m(i) = mean(x_i);
    v(i) = sqrt(var(x_i));
    X_tr(i,:) = (x_i - m(i))/v(i);
end

D_tr = [X_tr;ones(1,285);y_tr];
D_te = [X_te;ones(1,285);y_te];

w = zeros(1,31)';
class_Dte = D_te(1:31,:);

%Case 1
mu = 0;
K = 10;
[x_s1, f_s1,k_s1] = grad_desc('f)wdbc','g_wdbc',w,K,D_tr,mu);

sign_Dte_1 = x_s1'*class_Dte;
D_te_1 = ones(1,284);

for j = 1:284
    if sign_Dte_1(j) < 0
        D_te_1(j) = -1;
    end
end

[c_1,acc_1] = confusion(D_te_1,y_te);
