function R = calculateRisk(Y,Y_hat)
    L = max(size(Y));
    R = sum(abs(Y - Y_hat))./(2*L);
%     R = (norm(Y-Y_hat).^2)./(2*L);
end