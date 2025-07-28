if ~exist('Option','var')
    Option = option.defaults();
else
    Option = option.setdefaults(Option);
end

% Windowing parameters
Option.spikeBinSize = 0.025;
Option.timesPerTrial = 100;
Option.winSize = [-0.5 0.5];
Option.preProcess_zscore = false;
Option.numPartition = 50;
Option.animal = "JS21";

%%%%%%%%%%%%%%%% DISPLAY OUR OPTIONS TO USER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("Running with Option struct => ")
disp(Option);

%%%%%%%%%%%%%%%% OBTAIN EVENT MATRICES    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("------------------------")
disp("    Obtaining events    ")
disp("------------------------")
Events = events.ThetaDeltaRipple(Option);
% Documentation
% Events is a struct with fields:
% - .times : array of times of events
% - .H     : Event Matrix,    T x 3, and each column are theta, delta, ripple
% - .Hvals : Event Matrix,    T x 3, values without nans
% - .Hnanlocs : Event Matrix, T x 3, logicals of nans


%%%%%%%%%%%%%%%% CUT WINDOWS WITH EVENT MATRICES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("------------------------")
disp("    Cutting windows     ")
disp("------------------------")
Events = windows.ThetaDeltaRipple(Events, Option);
% -  cutoffs:       nPatterns x 1 vector of cutoffs
% TODO: modify to be able to include overall pattern and track patterns
% PRIORITY; overall: medium, track: very low, overall can be included in
% cellOfWindows, whereas, track can be included as a separate output
Events_Osc = Events;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

idphi_high_windows = Events.cellOfWindows{3, 1};
delta_windows = Events_Osc.cellOfWindows{1, 1};
theta_windows = Events_Osc.cellOfWindows{1, 2};
ripple_windows = Events_Osc.cellOfWindows{1, 3};

% 创建一个与"idphi-high"时间窗数量相同的逻辑数组来标记是否有交集
has_overlap_delta = false(size(idphi_high_windows, 1), 1);
has_overlap_theta = false(size(idphi_high_windows, 1), 1);
has_overlap_ripple = false(size(idphi_high_windows, 1), 1);

% 检查"idphi-high"时间窗是否与"delta"时间窗有交集
for i = 1:size(idphi_high_windows, 1)
    for j = 1:size(delta_windows, 1)
        % 检查时间窗是否有交集
        if (idphi_high_windows(i, 1) <= delta_windows(j, 2)) && (idphi_high_windows(i, 2) >= delta_windows(j, 1))
            has_overlap_delta(i) = true;
            break;  % 如果有交集，跳出内循环
        end
    end
end

% 检查"idphi-high"时间窗是否与"theta"时间窗有交集
for i = 1:size(idphi_high_windows, 1)
    for j = 1:size(theta_windows, 1)
        % 检查时间窗是否有交集
        if (idphi_high_windows(i, 1) <= theta_windows(j, 2)) && (idphi_high_windows(i, 2) >= theta_windows(j, 1))
            has_overlap_theta(i) = true;
            break;  % 如果有交集，跳出内循环
        end
    end
end

% 检查"idphi-high"时间窗是否与"ripple"时间窗有交集
for i = 1:size(idphi_high_windows, 1)
    for j = 1:size(ripple_windows, 1)
        % 检查时间窗是否有交集
        if (idphi_high_windows(i, 1) <= ripple_windows(j, 2)) && (idphi_high_windows(i, 2) >= ripple_windows(j, 1))
            has_overlap_ripple(i) = true;
            break;  % 如果有交集，跳出内循环
        end
    end
end

% 选择有交集的时间窗
overlapping_delta_windows = idphi_high_windows(has_overlap_delta, :);
overlapping_theta_windows = idphi_high_windows(has_overlap_theta, :);
overlapping_ripple_windows = idphi_high_windows(has_overlap_ripple, :);

Events.cellOfWindows{5,1} = overlapping_delta_windows;
Events.cellOfWindows{6,1} = overlapping_theta_windows;
Events.cellOfWindows{7,1} = overlapping_ripple_windows;

Events.cellOfWin_varnames{1,5} = "delta";
Events.cellOfWin_varnames{1,6} = "theta";
Events.cellOfWin_varnames{1,7} = "ripple";