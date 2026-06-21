function VAF= vaf(y,yhat)
    VAF = (1 - var(y-yhat)/var(y))*100;
end