function P = GMM(Matrix_u,setCVf,vector_pi,vector_x)
    %Gaussian mixture model

    [M N Q] = size(setCVf);
    P = zeros(M,N);
    for ii=1:Q
        CVmat = setCVf(:,:,ii);
        Y = normpdf(vector_x,Matrix_u(:,ii),CVmat);%I might need to implement this function as well
        P = P + (vector_pi(ii,)*Y);
    end
end