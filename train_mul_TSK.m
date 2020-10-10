function [TSK_cell, lamda_scale,P_cell,Q_cell] = train_mul_TSK( mulview_data_cell, ulabelX, P_cell, Q_cell, T, TSK_cell, options)
%输入多视角数据元祖及类标，输出每个视角的TSK模型,TSK_cell每一行为一个视角的模型，第一列为pg，第二列为v，第三列为b
view_nums = options.view_nums;
lamda_scale = zeros(6,1);
%T c*n
c = size(T,2);  %类别数
M = size( mulview_data_cell{1,1}, 1);   %标记样本数
N = size(ulabelX{1,1},1);   %伪标记样本数 
lamda1 = options.lamda1;    
lamda2 = options.lamda2;    
lamda3 = options.lamda3;    
lamda4 = options.lamda4;    
lamda5 = options.lamda5;    
maxIter = options.maxIter;

for i = 1:maxIter
    %计算权值更新中的分母
    sum_weight = 0;
    for view_num = 1:view_nums
        p = P_cell{view_num};
        q = Q_cell{view_num};
        model = TSK_cell{ 1,view_num };
        temp_pg = model.pg;
        temp_v = model.v;
        temp_b = model.b;
        x = mulview_data_cell{1,view_num};
        ux = ulabelX{1,view_num};
        temp_x = fromXtoZ(x,temp_v,temp_b);
        temp_ux = fromXtoZ(ux,temp_v,temp_b);
        sum_variance = (sum(temp_x * temp_pg - T ))*sum(temp_x * temp_pg - T)'+(lamda4)*(sum(temp_ux * temp_pg-p*q)*sum(temp_ux * temp_pg - p*q)');%11
        sum_variance = exp(-lamda3*sum_variance);
        sum_weight = sum_weight+sum_variance;
    end
    %更新权值及后件输出
    for view_num = 1:view_nums
        p = P_cell{view_num};
        q = Q_cell{view_num};
        model = TSK_cell{ 1,view_num };
        acc_pg = model.pg;
        acc_v = model.v;
        acc_b = model.b;
        acc_x = mulview_data_cell{1,view_num};
        acc_ux = ulabelX{1,view_num};
        x = fromXtoZ(acc_x,acc_v,acc_b);
        ux = fromXtoZ(acc_ux,acc_v,acc_b);
        
        variance = ((sum(x * acc_pg - T))*sum(x * acc_pg - T)')+(lamda4)*(sum(ux * acc_pg - p*q)*sum(ux * acc_pg - p*q)');
        acc_w = exp(-lamda3*variance)/sum_weight;
        
        sum_y = zeros(M,c);
        sum_uy = zeros(N,c);
        sum_pq = zeros(N,c);
        for j = 1:view_nums     %计算y_cooperate
            if j ~= view_num
                model = TSK_cell{1,j};
                temp_pg = model.pg;
                temp_v = model.v;
                temp_b = model.b;
                temp_x = mulview_data_cell{1,j};
                temp_ux = ulabelX{1,j};
                temp_x = fromXtoZ(temp_x,temp_v,temp_b);
                temp_ux = fromXtoZ(temp_ux,temp_v,temp_b);
                sum_y = sum_y + temp_x*temp_pg;
                sum_uy = sum_uy + temp_ux*temp_pg;
                
                lamda_scale(1,1) = lamda1;
                lamda_scale(2,1) = lamda2;
                lamda_scale(3,1) = lamda3;
                lamda_scale(4,1) = lamda4;
                lamda_scale(6,1) = lamda6;
                sum_pq = sum_pq + P_cell{j}*Q_cell{j};
            end
        end

        y_cooperate = sum_y/(view_nums - 1);
        uy_cooperate = sum_uy/(view_nums - 1);
        z = acc_w*(x)'*x+lamda4*acc_w*(ux)'*ux;
        acc_pg = pinv( z + lamda1 * eye( size( z)) +lamda2*((x)'*x + ux'*ux))*(acc_w*(x'*T+lamda4*ux'*p*q)+lamda2*(x'*y_cooperate + ux'*uy_cooperate));%10
        
        %更新P,Q
        pq_cooperate = sum_pq;

        q = pinv( (lamda4*acc_w+2*lamda6)*p'*p)*( lamda4*acc_w*p'*((ux*acc_pg))+2*lamda6*p'*pq_cooperate);%k*k*k*n=k*n

        p = pinv( (lamda4*acc_w+2*lamda6)*q*q' )*( lamda4*acc_w*(ux*acc_pg)*q'+2*lamda6*pq_cooperate*q')';
        p=p';
        
        model1.pg = acc_pg;
        model1.w = acc_w;
        model1.v = acc_v;
        model1.b = acc_b;
        TSK_cell{1,view_num} = model1;
        P_cell{view_num} = p;
        Q_cell{view_num} = q;
        

    end %end view_num
        
    
end
