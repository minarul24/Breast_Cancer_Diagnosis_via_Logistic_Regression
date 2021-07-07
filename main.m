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

