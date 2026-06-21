function [results, yhat_val] = evaluateResults(TN, u_val, y_val, M)

% Sept. 2025 - Eva Memmel
% ------------------------------------------

    results = struct();
    L = size(y_val,2);
    
    % Measure simulation time for sim_VolterraTN
    sim = tic;
    yhat_val = sim_volterraTN(u_val,TN);
    results.timesim = toc(sim);
    
    % Compute validation metrics
    results.relError = relative_error(yhat_val(L*M+1:end), y_val(L*M+1:end));
    results.RMSE = RMSE(yhat_val(L*M+1:end), y_val(L*M+1:end));
end