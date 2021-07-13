%
%Author:     Minarul Shawon
%Date:       July 1, 2021
%
%   Breast Cancer Diagnosis via Logistic Regression
%

load D_wdbc.mat

%main
D_tr = D_wdbc(:,1:285);
D_te = D_wdbc(:,286:569);

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

X_te = zeros(30,284);
for k = 1:30
    xk = D_te(k,:);
    X_te(k,:) = (xk - m(k))/v(k);
end

D_tr = [X_tr;ones(1,285);y_tr];
D_te = [X_te;ones(1,284);y_te];

w = zeros(1,31)';
class_Dte = D_te(1:31,:);

%Case 1, mu = 0 and K = 10
mu = 0;
K = 10;
[x_s1, f_s1,k_s1] = grad_desc('f_wdbc','g_wdbc',w,K,D_tr,mu);

sign_Dte_1 = x_s1'*class_Dte;
D_te_1 = ones(1,284);

for j = 1:284
    if sign_Dte_1(j) < 0
        D_te_1(j) = -1;
    end
end

[c_1,acc_1] = confusion(D_te_1,y_te);

%Case 2, mu = 0.1 and K = 10
mu = 0.1;
K = 10;
[x_s2, f_s2,k_s2] = grad_desc('f_wdbc','g_wdbc',w,K,D_tr,mu);

sign_Dte_2 = x_s2'*class_Dte;
D_te_2 = ones(1,284);

for j = 1:284
    if sign_Dte_2(j) < 0
        D_te_2(j) = -1;
    end
end

[c_2,acc_2] = confusion(D_te_2,y_te);

%Case 3, mu = 0 and K = 30
mu = 0;
K = 30;
[x_s3, f_s3,k_s3] = grad_desc('f_wdbc','g_wdbc',w,K,D_tr,mu);

sign_Dte_3 = x_s3'*class_Dte;
D_te_3 = ones(1,284);

for j = 1:284
    if sign_Dte_3(j) < 0
        D_te_3(j) = -1;
    end
end

[c_3,acc_3] = confusion(D_te_3,y_te);

%Case 4, mu = 0.075 and K = 30
mu = 0.075;
K = 30;
[x_s4, f_s4,k_s4] = grad_desc('f_wdbc','g_wdbc',w,K,D_tr,mu);

sign_Dte_4 = x_s4'*class_Dte;
D_te_4 = ones(1,284);

for j = 1:284
    if sign_Dte_4(j) < 0
        D_te_4(j) = -1;
    end
end

[c_4,acc_4] = confusion(D_te_4,y_te);

%Displaying Results - Confusion Matrix and Accuracy

fprintf('\n\n *************************************************\n');
fprintf('Case 1 - mu = 0 & K = 10 \n Confusion Matrix \n')
disp(c_1);
fprintf('Accuracy = ');
disp(acc_1);
fprintf('\n\n *************************************************\n');
fprintf('Case 2 - mu = 0.1 & K = 10 \n Confusion Matrix \n')
disp(c_2);
fprintf('Accuracy = ');
disp(acc_2);
fprintf('\n\n *************************************************\n');
fprintf('Case 3 - mu = 0 & K = 30 \n Confusion Matrix \n')
disp(c_3);
fprintf('Accuracy = ');
disp(acc_3);
fprintf('\n\n *************************************************\n');
fprintf('Case 4 - mu = 0.075 & K = 30 \n Confusion Matrix \n')
disp(c_4);
fprintf('Accuracy = ');
disp(acc_4);
fprintf('\n\n *************************************************\n');

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
end
accuracy = (trace(c)./sum(c,'all'))*100;
end
%end of confusion matrix