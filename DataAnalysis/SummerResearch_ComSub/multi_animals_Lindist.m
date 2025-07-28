warning("off","all")

animal_lst = ["JS13", "JS14", "JS15", "JS17", "JS21", "ZT2", "ER1"];

for i = 2:3
animal = animal_lst(i);
[Option, Spk, spike_counts_by_area] = analysis.setup_lindist(animal);

%Patterns_overall = generateProjectionVec(Spk, Option);
Patterns_overall = nan;

disp("------------------------")
disp("        jpecc           ")
disp("------------------------")

Option.analysis.JPECC = 1;

%epoches = ['f','i','r','s'];

%for e = 1:length(epoches)
    %epoch = epoches(e);
    Option.threshold = [0 180];
    Jpecc_Results.(animal) = analysis.JPECC_task2(spike_counts_by_area.CA1, spike_counts_by_area.PFC, 0, Patterns_overall, Option);
    %Jpecc_Results.(animal).(epoch) = analysis.JPECC_task2_bin(spike_counts_by_area.(epoch).CA1, spike_counts_by_area.(epoch).PFC, 0, Patterns_overall, Option, 30);
    Option.threshold = [-180 0];
    Jpecc_Results2.(animal) = analysis.JPECC_task2(spike_counts_by_area.CA1, spike_counts_by_area.PFC, 0, Patterns_overall, Option);
    %Jpecc_Results2.(animal).(epoch) = analysis.JPECC_task2_bin(spike_counts_by_area.(epoch).CA1, spike_counts_by_area.(epoch).PFC, 0, Patterns_overall, Option, 30);
%end
end
save('F:\ComSub\Lindist_1.mat','Jpecc_Results');
save('F:\ComSub\Lindist_2.mat','Jpecc_Results2');


%%%%%%Gaussian Filter%%%%%%
sigma = 0.85;
filter_size = 3;
[x, y] = meshgrid(-filter_size:filter_size, -filter_size:filter_size);
h = exp(-(x.^2 + y.^2) / (2 * sigma^2));
h = h / sum(h(:));  % normalize the filter

%%%%%%cmap%%%%%%%
%{
% 创建一个颜色矩阵
n = 256; % 颜色的数量

% 定义橙黄色和天蓝色
orange_yellow = [1, 0.7, 0];
sky_blue = [0.5, 0.75, 1];

% 创建颜色矩阵
r = [linspace(orange_yellow(1), 1, n/4)'; linspace(1, 0, 6*n/8)'; linspace(0, 0.4, 6*n/8)'; linspace(0.4, sky_blue(1), n/4)'];
g = [linspace(orange_yellow(2), 0.13, n/4)'; linspace(0, 0, 6*n/8)'; linspace(0, 0.5, 6*n/8)'; linspace(0.5, sky_blue(2), n/4)'];
b = [linspace(orange_yellow(3), 0, n/4)'; linspace(0, 0, 6*n/8)'; linspace(0, 1, 6*n/8)'; linspace(1, sky_blue(3), n/4)'];

cmap = [r, g, b];
cmap = flipud(cmap.^1.1);
%}
cmap = cmocean('thermal');
%%%%%%Plotting%%%%%%%

conditions = {'single', 'combined'};
%categories = {'overall', 'theta', 'delta', 'ripple'};
animals = {'JS21'};
%epochs = {'f','i','r','s'};
sessions = {'HPCafterPFC', 'PFCafterHPC'};
results = {Jpecc_Results, Jpecc_Results2};

Heatmap = struct();

for sessIdx = 1:length(sessions)
    session = sessions{sessIdx};
    result = results{sessIdx};
    
    for condIdx = 1:length(conditions)
        condition = conditions{condIdx};
        
        for ani = 1:length(animals)
            animal = animals{ani};
        %for epo = 1:length(epochs)
            %epoch = epochs{epo};
        %for catIdx = 1:length(categories)
            %category = categories{catIdx};
            
            %if strcmp(category, 'overall')
                %fieldName = ['r', condition]; % Special case for 'overall'
            %else
                %fieldName = [category, '_', condition];
            %end
            fieldName =  ['r', condition];
            [SM, Single] = stackAndAverageMatrices(result.(animal).jpecc, fieldName, 4, 30);
            Heatmap.(animal).(condition).(session).Mean = Single;
            Heatmap.(animal).(condition).(session).stackedmatrices = SM;

            %{
            [SM, Single] = stackAndAverageMatrices(result.(animal).(epoch).jpecc, fieldName, 4, 30);
            Heatmap.(animal).(epoch).(condition).(session).Mean = Single;
            Heatmap.(animal).(epoch).(condition).(session).stackedmatrices = SM;
            %}
        %end
        end
    end
end

conditions = {'single', 'combined'};
%categories = {'overall', 'theta', 'delta', 'ripple'};
animals =  {'JS21'};
sessions = {'HPCafterPFC', 'PFCafterHPC'};
%eponames = {'24','68','1012','1416'};
save_folder = 'C:\Users\BrainMaker\commsubspace\ruoxi\Figures\Lindist_Spectra';


for condIdx = 1:length(conditions)
    condition = conditions{condIdx};
    
    %for catIdx = 1:length(categories)
        %category = categories{catIdx};
    for ani = 1:length(animals)
        animal = animals{ani};
        
        for sessIdx = 1:length(sessions)
            session = sessions{sessIdx};
        %for epo = 1:length(epochs)
            %epoch = epochs{epo};
            %epochname = eponames{epo};
            
            Single = Heatmap.(animal).(condition).(session).Mean;
            plotAndSave(animal, Single, h, cmap, session, condition, save_folder);

            %{
            Single = Heatmap.(animal).(epoch).(condition).(session).Mean;
            plotAndSave_epoch(animal, Single, h, cmap, session, condition, save_folder, epochname);
        end
            %}
        end
    end
end





