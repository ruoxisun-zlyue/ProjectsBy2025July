if ~exist('Option','var')
    disp("Option struct does not exist, creating one")
    addpath(genpath(codedefine()));
    Option = option.defaults(); 
else
    disp("Option struct already exists, using that")
    Option = option.setdefaults(Option);
    disp("Option struct is: ")
    disp(Option)
end

% Windowing parameters
Option.spikeBinSize = 0.025;
Option.winSize = [-0.5 0.5];
Option.preProcess_zscore = false;


%%%%%%%%%%%%%%%% DISPLAY OUR OPTIONS TO USER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isequal(Option.loadifexists, false) && ...
    exist(store.gethash(Option) + ".mat", 'file')
    disp("Loading from file: " + store.gethash(Option) + ".mat")
    m = matfile(store.gethash(Option) + ".mat");
    % m = matfile("bef0923.mat", "Writable", true);
    disp("Loaded variables: ")
    Events             = util.matfile.getdefault(m, 'Events', []);
    Spk                = util.matfile.getdefault(m, 'Spk', []);
    Patterns           = util.matfile.getdefault(m, 'Patterns', []);
    Patterns_overall   = util.matfile.getdefault(m, 'Patterns_overall', []);
    Components         = util.matfile.getdefault(m, 'Components', []);
    Components_overall = util.matfile.getdefault(m, 'Components_overall', []);
    Option             = util.matfile.getdefault(m, 'Option', []);
    disp("...done")
else

    %%%%%%%%%%%%%%%% OBTAIN EVENT MATRICES    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Getting behavior
    [behavior, thrown_out_times] = table.behavior.lookup(Option.animal, ...
                                                         []);
    Events = events.initEvents();
    % Test case 1: TriggerOn when velocity (vel) exceeds a certain threshold
    Events = events.generateFromBehavior(behavior, {'rewardTimes'}, ...
        Events, 'triggerOnValues', {1}, ...
                'triggerOnVarnames', {'rewardTimes'}, ...
                'windowParameters',Option.winSize);
    Events = events.generateFromBehavior(behavior, {'rewardTimes'}, ...
        Events, 'triggerOffValues', {1}, ...
                'triggerOffVarnames', {'rewardTimes'}, ...
                'windowParameters',Option.winSize);
    qselect = @(x) x.idphi > quantile(x.idphi, 0.95);
    Events = events.generateFromBehavior(behavior, {'idphi'}, ...
        Events, 'triggerVarLambdas', {qselect}, ...
                'triggerVars', {'idphi'}, ...
                'maxWindows', 2000, ...
                'windowParameters',Option.winSize);
    % Characterize events
    figure; events.plot_events(Events, 'rewardTimes');
    % figure; events.plotIEI(Events, 'NumBins', 1000);
    figure; events.plotCrossIEI(Events, 'NumBins', 1000, 'layout', 'matrix');
    Option.nPatternAndControl = numel(Events.cellOfWindows);
    Option.patternNames = ["rewardOn", "rewardOff"];

    %%%%%%%%%%%%%%%% ACQUIRE SPIKES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Getting spikes
    disp("------------------------")
    disp("    Getting spikes      ")
    disp("------------------------")
    Spk = spikes.getSpikeTrain(Option.animal, Option.spikeBinSize, ...
                               Option.samplingRate);
    % filter the neurons whose firing rate is lower than specified threshold
    if Option.preProcess_FilterLowFR 
        disp("------------------------")
        disp("Filtering low FR neurons")
        disp("------------------------")
        Spk = trialSpikes.filterFR(Spk, 0.1);
        disp("Mean FR: " + sort(Spk.avgFR))
    end
    if Option.preProcess_gaussianFilter
        % Gaussian filter the spikeCountMatrix/spikeRateMatrix
        gauss = gausswin(Option.preProcess_gaussianFilter);
        for i = progress(1:size(Spk.spikeRateMatrix, 1), 'Title', 'Gaussian filtering')
            Spk.spikeRateMatrix(i, :)  = conv(Spk.spikeRateMatrix(i, :), gauss, 'same');
            Spk.spikeCountMatrix(i, :) = conv(Spk.spikeCountMatrix(i, :), gauss, 'same');
        end
    end

    if Option.preProcess_zscore
        % Z-score the spikeCountMatrix/spikeRateMatrix
        disp(" Z-scoring ")
        if ~isfield(Spk, 'muFR')
            Spk.muFR  = mean(Spk.spikeRateMatrix, 2);
            Spk.stdFR = std(Spk.spikeRateMatrix, 0, 2);
        end
        Spk.spikeRateMatrix  = zscore(Spk.spikeRateMatrix,  0, 2);
        Spk.spikeCountMatrix = zscore(Spk.spikeCountMatrix, 0, 2);
        Spk.avgFR = mean(Spk.spikeRateMatrix, 2);
    end
    prewindow_copy = Spk;

    % %%%%%%%%%%%%%% ACQUIRE TRIALS FROM WINDOWS + SPIKES %%%%%%%%%%%%%%%%%%%
    % RYAN bug here .. timeBinStartEnd instead of timeBinMidPoints
    disp("------------------------")
    disp("   Windowing spikes     ")
    disp("------------------------")
    Spk = trialSpikes.generate(Spk, Events, Option);

    %%%%%%%%%%%%%%%%% SETUP RAW DATA STRUCTURE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Structure for separated data
    %%%%%%%%%%%%%%%% SEPRATE BRAIN AREA DATASULT STRUCTURES %%%%%%%%%%%%%%%%%%
    % Separate spikesSampleMatrix/Tensor by area that neurons are in PFC and
    % neurons that in HPC
    %% Separate firing pattern into source and target
    [Spk.nSource,~,~] = size(Spk.hpc.X{1});
    [Spk.nTarget,~,~] = size(Spk.pfc.X{1});
    Spk.celllookup = cellInfo.getCellIdentities(Option.animal, Spk.cell_index,...
                                                Spk.areaPerNeuron);
    system("pushover-cli 'Finished munging data for analysis'");

    %%%%%%%%%%%%%%%% SETUP PARTITIONS AND RESULT STRUCTURES %%%%%%%%%%%%%%%%%%
    disp("------------------------")
    disp(" Subsampling partitions ")
    disp("------------------------")
    [Patterns, Patterns_overall] = trialSpikes.partitionAndInitialize(Spk, Option);
    Components = nd.initFrom(Patterns, ...
    {'index_source', 'index_target', 'directionality', 'name'});
    Components_overall = nd.initFrom(Patterns_overall, ...
    {'index_source', 'index_target', 'directionality', 'name'});
end
