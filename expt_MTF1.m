function best_result = expt_MTF1(label_X,unlabeled_X,labeled_y,P_cell,Q_cell,TSK_cell,M,ulabeled_y,Y_te)
T = lab2vec(labeled_y);
folds_num = 1;
masks_te=cv_fold(folds_num,T);
view_nums = size(label_X,2);
best_acc_te = 0;

lma = [10.^(-3:0)];
% lma2 = 10.^(-3:3);
aleta = [0.1,0.2,0.4,0.6,0.8,1];
% aleta = [10.^(-3:0)];
% beta = [10.^(-3:0)];
gama = [10.^(-3:0)];
i=1;v=1;u=1;r=1;w=1;c=1;
best_acc_pq=0;
% for l1 = 1:size(lma,2)
%     for l2 = 1:size(lma,2)
%         for l3 = 1:size(lma,2)
%             for a1 = 1:size(aleta,2)
%             for g1 = 1:size(gama,2)
            options.maxIter = 10;
            options.lamda1 = 0.001; %正则化系数 lma(l1)
            options.lamda2 = 0.01; %协同项系数 lma(l2)
            options.lamda3 = 0.001; %权值项系数 lma(l3)
            options.lamda4 = 0.1; %control the influences of unlabeled data aleta(a1)
            options.lamda5 = 0; %weighting parameter for regularization of p and q
            options.lamda6 = 0.001; %control the influences of view consistency gama(g1)
            options.view_nums = view_nums;
            
            result=zeros(folds_num,1);
            tic;
             try
                 for fold_num=1:folds_num
                      
                    [TSK_cell_t, lamda_scale,p,q] = train_mul_TSK( label_X , unlabeled_X, P_cell, Q_cell, T, TSK_cell, options,Y_te);
                    t=toc;
                    [T_te] = test_mul_TSK( unlabeled_X ,TSK_cell_t, options.view_nums, M);
                    Y_te = vec2lab(T_te);
%                     te = vec2lab(clusters_te);
                    acc_te=sum(Y_te==ulabeled_y)/length(ulabeled_y);
                    result(fold_num,1)=acc_te;
                    
                    te_pq = p{1}*q{1}*TSK_cell_t{1}.w + p{2}*q{2}*TSK_cell_t{2}.w;
                    acc_pq = sum(vec2lab(te_pq)==ulabeled_y)/length(ulabeled_y);
                    result_pq(fold_num,1) = acc_pq;
                 end
              catch err
                   disp(err);
                       warning('Something wrong when using function pinv!');
%                    break;
             end
%              reg(r,1) = aleta(a1);
%              reg(r,2) = mean(result(:,1));
%              reg(r,3) = mean(result_pq(:,1));
%              best_result.ul=reg;
%              r=r+1;
             if acc_pq>best_acc_pq
                     best_acc_pq = acc_pq;
                     best_result.acc_pq = acc_pq;
              end
              acc_te_mean = mean(result(:,1));
              acc_te_std = std(result(:,1));
              acc_te_min = min(result(:,1));
              acc_te_max = max(result(:,1));
              if acc_te_mean>best_acc_te
                  best_acc_te = acc_te_mean;
                  best_result.best_model = TSK_cell_t;
                  best_result.mean = acc_te_mean;
                  best_result.std = acc_te_std;
                  best_result.min = acc_te_min;
                  best_result.max = acc_te_max;
                  best_result.best_lamda_scale = lamda_scale;
                  best_result.time=t;
                  best_result.pred = Y_te;
                    best_acc_te
              end                         
%             end
%           end
%         end
%     end      
% end

