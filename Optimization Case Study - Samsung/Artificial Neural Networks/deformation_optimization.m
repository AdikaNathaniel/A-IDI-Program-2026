%% FORMING PARAMETER OPTIMIZATION USING A NEURAL NETWORK SURROGATE
%
%   Goal: find the temperature, strain rate, and strain that MINIMIZE
%   predicted flow stress during hot forming.
%
%   Method:
%     1. Train a neural network to predict flow stress from
%        (temperature, strain rate, strain), using experimental data.
%     2. Treat that trained network as an objective function for the real
%        material behaviour.
%     3. Use Optimization techniques (fmincon) to predict the lowest stress.
%

%% SECTION 1: Prepare the data
% Neural network functions in MATLAB expect one COLUMN per sample,
% so we transpose the data (rows become columns).

if istable(Data1)
    Data1 = table2array(Data1);
end
if istable(Data2)
    Data2 = table2array(Data2);
end

% If Data2 has an extra empty leading column, keep only the stress column:
if size(Data2, 2) > 1
    Data2 = Data2(:, end);
end

inputs = Data1';   % 3 x 400
targets = Data2';  % 1 x 400


%% SECTION 2: Build and train the neural network
% A small feedforward network with one hidden layer is enough to learn
% the relationship between (temperature, strain rate, strain) and stress.

hiddenLayerSize = 12;
trainFcn = 'trainlm';   % Levenberg-Marquardt: fast, good for small networks

net = fitnet(hiddenLayerSize, trainFcn);

% Automatically split the 400 samples: 70% train, 15% validation, 15% test.
% Validation stops training before it overfits; test gives an honest,
% untouched check of accuracy afterwards.
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.15;

net.performFcn = 'mse';   % mean squared error: standard choice for regression

[net, trainRecord] = train(net, inputs, targets);


%% SECTION 3: Check how well the network learned
% This is the objective function's accuracy check -- if this is poor,
% the "optimum" found later is only optimal for a bad model of reality.

predictions = net(inputs);
performance = perform(net, targets, predictions);
fprintf('Overall training MSE: %.6f\n', performance);

view(net)   % opens a diagram of the trained network


%% SECTION 4: Define the objective function for optimization

scaleFactor = 348.6897;   % 1.0 (normalized) = 348.6897 MPa

function stress = predictStress(x, net, scaleFactor)
    T_celsius  = x(1);
    strainRate = x(2);
    strain     = x(3);

    T_kelvin = T_celsius + 273.15;
    features = [1/T_kelvin; log(strainRate); strain];

    normalizedStress = net(features);
    stress = normalizedStress * scaleFactor;   % convert back to MPa
end

objective = @(x) predictStress(x, net, scaleFactor);


%% SECTION 5: Search for the forming parameters that minimize stress
% fmincon finds a LOCAL minimum from a single starting point, so we run
% it many times from random starting points ("multi-start") and keep
% the best result -- this guards against getting stuck in a poor
% local minimum.

lowerBounds = [900,  0.1, 0.1];   % [deg C, per second, true strain]
upperBounds = [1050, 15.0, 0.7];

numStarts = 40;
bestStress = Inf;
bestParams = [];

options = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'off');

for i = 1:numStarts
    x0 = lowerBounds + (upperBounds - lowerBounds) .* rand(1, 3);
    [x, stress, exitFlag] = fmincon(objective, x0, [], [], [], [], ...
                                     lowerBounds, upperBounds, [], options);
    if exitFlag > 0 && stress < bestStress
        bestStress = stress;
        bestParams = x;
    end
end


%% SECTION 6: Report the result

fprintf('\n================ Optimization Result ================\n');
fprintf('Best of %d random starting points:\n', numStarts);
fprintf('  Temperature : %.2f C\n', bestParams(1));
fprintf('  Strain rate : %.4f /s\n', bestParams(2));
fprintf('  Strain      : %.4f\n', bestParams(3));
fprintf('Minimum predicted stress: %.4f MPa\n', bestStress);
fprintf('=======================================================\n');
