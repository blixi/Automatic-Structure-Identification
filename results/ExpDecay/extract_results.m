% Sept. 2025 - Eva Memmel
% ------------------------------------------
D = 9;
J = 10;

als_train = zeros(D,J);
als_val = zeros(D,J);
als_test = zeros(D,J);

sota_train = zeros(D,J);
sota_val = zeros(D,J);
sota_test = zeros(D,J);

svd_train = zeros(D,1);
svd_val = zeros(D,1);
svd_test = zeros(D,1);

for d = 1:D
    for j = 1:J
        als_train(d,j) = results.increaseD.etrain{j}.D{d}.RMSE_train(end);
        sota_train(d,j) = results.increaseD.etrain{j+J+1}.D{d}.RMSE_train(end);

        als_val(d,j) = results.increaseD.val{j}.D{d}.RMSE;
        sota_val(d,j) = results.increaseD.val{j+J+1}.D{d}.RMSE;

        als_test(d,j) = results.increaseD.test{j}.D{d}.RMSE;
        sota_test(d,j) = results.increaseD.test{j+J+1}.D{d}.RMSE;
    end
    svd_train(d,1) = results.increaseD.etrain{J+1}.D{d}.RMSE_train(end);
    svd_val(d,1) = results.increaseD.val{J+1}.D{d}.RMSE;
    svd_test(d,1) = results.increaseD.test{J+1}.D{d}.RMSE;
end

% disp(['mean train als: ', num2str(mean(als_train,2)') ])
% disp(['mean train svd: ', num2str(mean(svd_train,2)') ])
% disp(['mean train vtn: ', num2str(mean(sota_train,2)') ])
% 
% disp(['mean val als: ', num2str(mean(als_val,2)') ])
% disp(['mean val svd: ', num2str(mean(svd_val,2)') ])
% disp(['mean val vtn: ', num2str(mean(sota_val,2)') ])
% 
% disp(['mean test als: ', num2str(mean(als_test,2)') ])
% disp(['mean test svd: ', num2str(mean(svd_test,2)') ])
% disp(['mean test vtn: ', num2str(mean(sota_test,2)') ])


als_train_time = zeros(D,J);
als_val_time = zeros(D,J);
als_test_time = zeros(D,J);



sota_train_time = zeros(D,J);
sota_val_time = zeros(D,J);
sota_test_time = zeros(D,J);


svd_train_time = zeros(D,1);
svd_val_time = zeros(D,1);
svd_test_time = zeros(D,1);

for d = 1:D
    for j = 1:J
        als_train_time(d,j) = results.increaseD.timetrain{j}.D{d};
        sota_train_time(d,j) = results.increaseD.timetrain{j+J+1}.D{d};

        als_val_time(d,j) = results.increaseD.timeval{j}.D{d};
        sota_val_time(d,j) = results.increaseD.timeval{j+J+1}.D{d};

        als_test_time(d,j) = results.increaseD.timetest{j}.D{d};
        sota_test_time(d,j) = results.increaseD.timetest{j+J+1}.D{d};
    end
    svd_train_time(d,1) = results.increaseD.timetrain{J+1}.D{d};
    svd_val_time(d,1) = results.increaseD.timeval{J+1}.D{d};
    svd_test_time(d,1) = results.increaseD.timetest{J+1}.D{d};
end

disp(['time train als: ', num2str(sum(sum(als_train_time,2))') ])
disp(['time train svd: ', num2str(sum(sum(svd_train_time,2))') ])
disp(['time train vtn: ', num2str(sum(sum(sota_train_time,2))') ])

disp(['time val als: ', num2str(sum(sum(als_val_time,2))') ])
disp(['time val svd: ', num2str(sum(sum(svd_val_time,2))') ])
disp(['time val vtn: ', num2str(sum(sum(sota_val_time,2))') ])

disp(['time test als: ', num2str(sum(sum(als_test_time,2))') ])
disp(['time test svd: ', num2str(sum(sum(svd_test_time,2))') ])
disp(['time test vtn: ', num2str(sum(sum(sota_test_time,2))') ])


