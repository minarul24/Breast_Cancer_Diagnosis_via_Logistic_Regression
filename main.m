%
%Author:     Minarul Shawon
%Date:       July 1, 2021
%
%   Breast Cancer Diagnosis via Logistic Regression
%

load D_wdbc.mat

D_tr = D_wdbc(:,1:285);
D_te = D_wbdc(:,286:569);

y_tr = D_tr(31,:);
y_te = D_te(31,:);

X_tr = zeros(30,285);
%for mean
m = zeros(1,30);
v = zeros(1,30);

for i = 1:30
    x_i = D_tr(i,:);
    m(i) = mean(x_i);
    v(i) = sqrt(var(x_i));
    X_tr(i,:) = (x_i - m(i))/v(i);
end
