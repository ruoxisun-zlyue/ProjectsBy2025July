warning("off","all")

animal_lst = ["JS13", "JS14", "JS15", "JS17", "JS21","ZT2","ER1"];

for i = 5
    
disp("!!!!!!!!!!!!!!!!")
disp("The "+i+" Animal")
disp("!!!!!!!!!!!!!!!!")

animal = animal_lst(i);

[Option, Spk] = analysis.setup_idphi_reward(animal);


%%%%%%%%%%%%%%%% SETUP PARTITIONS AND RESULT STRUCTURES %%%%%%%%%%%%%%%%%%
disp("------------------------")
disp(" Subsampling partitions ")
disp("------------------------")
Option.numPartition = 50;
[Patterns, ~] = trialSpikes.partitionAndInitialize(Spk, Option);

a1 = size(Spk.hpc.T{1,1});
a2 = size(Spk.hpc.T{1,2});
a3 = size(Spk.hpc.T{1,3});
a4 = size(Spk.hpc.T{1,4});
trial = [a1(3),a2(3),a3(3),a4(3)];


disp("------------------------")
%% 
disp("        jpecc           ")
disp("------------------------")

Option.analysis.JPECC = 1;


%J_Patterns.(animal) = analysis.JPECC(Patterns, Option, 1, trial);
%J_Patterns2.(animal) = analysis.JPECC(Patterns, Option, 2, trial);
disp("------------------------")
disp("3 round")
disp("------------------------")
%J_Patterns3.(animal) = analysis.JPECC(Patterns, Option, 3, trial);

J_Patterns4.(animal) = analysis.JPECC(Patterns, Option, 4, trial);

end




%%%%%%Gaussian Filter%%%%%%
sigma = 0.85;
filter_size = 3;
[x, y] = meshgrid(-filter_size:filter_size, -filter_size:filter_size);
h = exp(-(x.^2 + y.^2) / (2 * sigma^2));
h = h / sum(h(:));  % normalize the filter

%%%%%%Set for Plotting%%%%%%%%
%{
stackedMatrices = cell(4, 40);
meanMatrices = cell(1,4);
for j = 1:4
    P = J_Patterns.(animal_lst(j));
for i = 1:40
    x = nd.fieldGet(P(i).jpecc, "val1");
    stackedMatrices{j,i} = x;
end
stackedMatrice = cat(40, stackedMatrices{j,:});
meanMatrices{j} = nanmean(stackedMatrice, 40);
end

stackedMatrices2 = cell(4, 40);
meanMatrices2 = cell(1,4);
for j = 1:4
    P = J_Patterns2.(animal_lst(j));
for i = 1:40
    x = nd.fieldGet(P(i).jpecc, "val1");
    stackedMatrices2{j,i} = x;
end
stackedMatrice2 = cat(40, stackedMatrices2{j,:});
meanMatrices2{j} = nanmean(stackedMatrice2, 40);
end
%}
%animal_lst = ["JS17","ER1","JS21","ZT2","JS14"];
animal_lst = ["JS17"];
stackedMatrices3 = cell(1, 50);
meanMatrices3 = cell(1,1);
for j = 1
    P = J_Patterns3.(animal_lst(j));
for i = 1:50
    x = nd.fieldGet(P(i).jpecc, "val1");
    stackedMatrices3{j,i} = x;
end
stackedMatrice3 = cat(50, stackedMatrices3{j,:});
meanMatrices3{j} = nanmean(stackedMatrice3, 50);
end


stackedMatrices3p = cell(1, 50);
combined_significance = cell(1,1); % 用于存储组合后的显著性情况

threshold = 0.05;
significant_ratio_threshold = 0.9;

for j = 1
    P = J_Patterns3.(animal_lst(j));
    for i = 1:50
        x = nd.fieldGet(P(i).jpecc, "p1");
        stackedMatrices3p{j,i} = x;
    end
    stackedMatrice3p = cat(50, stackedMatrices3p{j,:});
    
    % 计算每个单元格中p值小于阈值的比例
    significance_ratio = sum(stackedMatrice3p < threshold, 50) / 50;
    
    % 根据比例确定哪些单元格是显著的
    combined_significance{j} = significance_ratio > significant_ratio_threshold;
end


stackedMatrices4 = cell(1, 50);
meanMatrices4 = cell(1,1);
for j = 1
    P = J_Patterns4.(animal_lst(j));
for i = 1:50
    x = nd.fieldGet(P(i).jpecc, "val1");
    stackedMatrices4{j,i} = x;
end
stackedMatrice4 = cat(50, stackedMatrices4{j,:});
meanMatrices4{j} = nanmean(stackedMatrice4, 50);
end

stackedMatrices4p = cell(1, 50);
combined_significance2 = cell(1,1); % 用于存储组合后的显著性情况

threshold = 0.05;
significant_ratio_threshold = 0.9;

for j = 1
    P = J_Patterns4.(animal_lst(j));
    for i = 1:50
        x = nd.fieldGet(P(i).jpecc, "p1");
        stackedMatrices4p{j,i} = x;
    end
    stackedMatrice4p = cat(50, stackedMatrices4p{j,:});
    
    % 计算每个单元格中p值小于阈值的比例
    significance_ratio2 = sum(stackedMatrice4p < threshold, 50) / 50;
    
    % 根据比例确定哪些单元格是显著的
    combined_significance2{j} = significance_ratio2 > significant_ratio_threshold;
end
%%%%%%%%%%Plot IdPhi_High%%%%%%%%
for i = 1
    Matrix = meanMatrices3{i};
    pvalueMatrix = combined_significance{i}; % 假设您的pvalue矩阵名为pvalueMatrices3

    % 将pvalue大于0.05的部分设置为NaN
    Matrix = conv2(Matrix, h, 'same');
    Matrix(pvalueMatrix == 0) = nan;
    

    % heatmap
    figure;
    t = pcolor(Matrix);

    % 设置colormap
    %caxis([0 0.15]);
    colormap(cmocean('haline'));
    colorbar;

    %{ 
    为NaN值分配灰色
    [row, col] = find(isnan(Matrix));
    for k = 1:length(row)
        rectangle('Position', [col(k)-0.5, row(k)-0.5, 1.01, 1.01], 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'none');
    end
    %}
    

    set(t, 'EdgeColor', 'none');
    line([1 100], [1 100], 'Color', [0,0,0], 'LineWidth', 0.8,"LineStyle", "--");
    xline(50, 'LineWidth', 1)
    yline(50, 'LineWidth', 1)

    xlabel('PFC');
    ylabel('HPC');
    title('IdPhi-High-' + animal_lst(i));
    %saveas(gcf, "F:\ComSub\Figures\IdPhi2\top5\IdPhi-High-" + animal_lst(i) + ".fig");
    %saveas(gcf, "F:\ComSub\Figures\IdPhi2\top5\IdPhi-High-" + animal_lst(i) + ".pdf");
end


%%%%%%%%%%Plot IdPhi_Low%%%%%%%%
for i = 1
Matrix = meanMatrices4{i};
pvalueMatrix = combined_significance2{i}; % 假设您的pvalue矩阵名为pvalueMatrices3

% 将pvalue大于0.05的部分设置为NaN
Matrix = conv2(Matrix, h, 'same');
Matrix(pvalueMatrix == 0) = nan;

% heatmap
figure;
t = pcolor(Matrix);
%shading interp;

%caxis([0 0.15]);
cmocean('haline')
colorbar

%{
%caxis([-0.4 0.4]);
colormap(cmap);
colorbar;
%}

set(t, 'EdgeColor', 'none');
line([1 100], [1 100], 'Color', [0,0,0], 'LineWidth', 0.8,"LineStyle", "--");
xline(50, 'LineWidth', 1)
yline(50, 'LineWidth', 1)

xlabel('PFC');
ylabel('HPC');
title('IdPhi-Low-' + animal_lst(i));
%saveas(gcf, "F:\ComSub\Figures\IdPhi2\top5\IdPhi-Low-" + animal_lst(i) + ".fig");
%saveas(gcf, "F:\ComSub\Figures\IdPhi2\top5\IdPhi-Low-" + animal_lst(i) + ".pdf");
end

%%%%%%%%%%Plot Reward_On%%%%%%%%
for i = 1:4
Matrix = meanMatrices{i};
%Matrix = conv2(Matrix, h, 'same');

% heatmap
figure;
t = pcolor(Matrix);
%shading interp;

caxis([-0.1 0.3]);
cmocean('haline')
colorbar

%{
%caxis([-0.4 0.4]);
colormap(cmap);
colorbar;
%}

set(t, 'EdgeColor', 'none');
line([1 60], [1 60], 'Color', [1,1,1], 'LineWidth', 0.8,"LineStyle", "--");

xlabel('PFC');
ylabel('HPC');
title('Reward-On-' + animal_lst(i));
saveas(gcf, "F:\ComSub\Figures\multi_animals\idphi and reward\Reward-On-" + animal_lst(i) + ".png");
end

%%%%%%%%%%Plot Reward_Off%%%%%%%%
for i = 1:4
Matrix = meanMatrices2{i};
%Matrix = conv2(Matrix, h, 'same');

% heatmap
figure;
t = pcolor(Matrix);
%shading interp;

caxis([-0.1 0.3]);
cmocean('haline')
colorbar

%{
%caxis([-0.4 0.4]);
colormap(cmap);
colorbar;
%}

set(t, 'EdgeColor', 'none');
line([1 60], [1 60], 'Color', [1,1,1], 'LineWidth', 0.8,"LineStyle", "--");

xlabel('PFC');
ylabel('HPC');
title('Reward-Off-' + animal_lst(i));
saveas(gcf, "F:\ComSub\Figures\multi_animals\idphi and reward\Reward-Off-" + animal_lst(i) + ".png");
end