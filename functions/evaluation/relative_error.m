function err_rel = relative_error(yhat,y)
    err_rel = norm(yhat-y)/norm(y);
end