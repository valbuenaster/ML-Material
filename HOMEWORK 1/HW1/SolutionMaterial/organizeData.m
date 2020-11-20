function [X, Y] = organizeData(x1,x2)
    X = [];
    Y = [];
    meanSelect = [];%1 for x1, 0 for x2
    ii = 0;
    flagI = 0;
    jj = 0;
    flagJ = 0;

    while(size(X,1) < (size(x1,1)+size(x2,1)))
        if(rand(1)>=0.5)
            flagI = 1;
            if(ii >= (size(x1,1)))
                jj = jj + 1;
                flagJ = 1;
                flagI = 0;
            else
                ii = ii + 1;
            end
        else
            flagJ = 1;
            if(jj >= (size(x2,1)))
                ii = ii + 1;
                flagI = 1;
                flagJ = 0;
            else
                jj = jj + 1;
            end
        end

        if(flagI==1)
            X = [X; x1(ii,:)];
            meanSelect = [meanSelect;1];
            Y = [Y; 1];
        end
        if(flagJ==1)
            X = [X; x2(jj,:)];
            meanSelect = [meanSelect;-1];
            Y = [Y; -1];
        end
        flagI = 0;
        flagJ = 0;
    end
end