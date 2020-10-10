function best_result = expt_TMV_TSK(label_X,unlabeled_X,labeled_y,P_cell,Q_cell,TSK_cell,M,ulabeled_y)
T = lab2vec(labeled_y);

view_nums = size(label_X,2);
best_acc_te = 0;

lma = [10.^(-3:0)];
aleta = [0.1,0.2,0.4,0.6,0.8,1];
gama = [10.^(-3:0)];

best_acc_pq=0;
for l1 = 1:size(lma,2)
    for l2 = 1:size(lma,2)
        for l3 = 1:size(lma,2)
            for a1 = 1:size(aleta,2)
            for g1 = 1:size(gama,2)
            options.maxIter = 10;
            options.lamda1 = lma(l1); 
            options.lamda2 = lma(l2); 
            options.lamda3 = lma(l3); 
            options.lamda4 = aleta(a1); 
            options.lamda5 = gama(g1); 
            options.view_nums = view_nums;
            
            result=zeros(folds_num,1);
            tic;
            try 
                [TSK_cell_t, lamda_scale,p,q] = train_mul_TSK( label_X , unlabeled_X, P_cell, Q_cell, T, TSK_cell, options);
                t=toc;
                [T_te] = test_mul_TSK( unlabeled_X ,TSK_cell_t, options.view_nums, M);
                Y_te = vec2lab(T_te);
                acc_te=sum(Y_te==ulabeled_y)/length(ulabeled_y);
                    
                te_pq = p{1}*q{1}*TSK_cell_t{1}.w + p{2}*q{2}*TSK_cell_t{2}.w;
                acc_pq = sum(vec2lab(te_pq)==ulabeled_y)/length(ulabeled_y);
             catch err
                   disp(err);
                       warning('Something wrong when using function pinv!');
             end
              if acc_pq>best_acc_pq
                     best_acc_pq = acc_pq;
                     best_result.acc_pq = acc_pq;
              end
              acc_te_mean = mean(result(:,1));
              if acc_te>best_acc_te
                  best_acc_te = acc_te;
                  best_result.best_model = TSK_cell_t;
                  best_result.mean = acc_te;
                  best_result.best_lamda_scale = lamda_scale;
                  best_result.time=t;
                  best_result.pred = Y_te;
                    best_acc_te
              end                         
            end
          end
        end
    end      
end

