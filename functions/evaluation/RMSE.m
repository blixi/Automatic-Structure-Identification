function err_RMSE = RMSE(yhat,y)
    err_RMSE = sqrt(mean((yhat-y).^2));
end