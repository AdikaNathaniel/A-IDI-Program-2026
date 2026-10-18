%% Case Study: Repairing a Scratch With a Laser and a Heating Plate
%
% This optimization case study is based on a real Samsung Display patent,
% US 12,428,339:
% Flip a scratched piece of glass so the scratch faces down. Press a metal
% plate against that scratched face. Fire a laser through the other
% side of the glass. The glass barely absorbs the laser, so it passes
% through and heats the area right around the scratch. That local
% area softens, and gravity pulls the softened glass down into the
% scratch, closing it.
%
% This only works if we get the settings right:
% Too little heat and the scratch never actually closes. Too much
% heat and you either damage whatever sits underneath the glass, or
% you soften a much bigger area than you needed to, which shows up
% as visible distortion around the repair.
%
% We are going to use two decision variables, x and y:
%
%   x = power delivered to the plate, in watts
%       how much energy the laser is putting into the repair
%
%   y = exposure time, in seconds
%       how long the laser stays on before it switches off
%
% Our objective is to minimize how far the heat spreads sideways from
% the scratch, since that spread is what shows up as distortion. Our
% constraints are that the scratch has to actually get hot enough to
% soften and flow, and that whatever is underneath the glass does not
% get too hot in the process.

clear
clc
close all

%% Step 1: Material properties and geometry
%
% These are fixed physical facts about the glass.

k_glass   = 1.0;       % how well the glass conducts heat, W/(m*K)
rho_glass = 2500;      % density of the glass, kg/m^3
cp_glass  = 800;       % how much energy it takes to heat the glass, J/(kg*K)
alpha     = k_glass / (rho_glass * cp_glass);

% alpha is called thermal diffusivity. It tells you how fast heat
% spreads through a material. Bigger alpha means heat spreads faster.

T_amb      = 25;       % room temperature, degrees C
T_soft     = 750;      % temperature the glass needs to reach to soften and flow
T_max_back = 150;      % the highest temperature we can allow on the far side of
                        % the glass, to protect whatever sits underneath it

Lz         = 0.7e-3;   % glass thickness, meters
spot_width = 0.15e-3;  % how wide the heated spot is, set by the plate size,
                        % this is fixed, not something we are optimizing

HAZ_fraction = 0.10;   % The edge of the affected zone as the
                        % point where the temperature rise drops to 10
                        % percent of what is needed to soften the glass.

%% Step 2: Bounds on our two variables
%
% Think of this as a real limit coming
% from the hardware or from how much heat the plate can take before it
% gets damaged. That cap is what makes this an interesting tradeoff.
% If you could use unlimited power, you would just blast it for a
% tiny fraction of a second and never have to think about the back
% surface at all.

lb = [0.01, 0.01];   % [minimum power, minimum exposure time]
ub = [0.40, 3.00];   % [maximum power, maximum exposure time]
x0 = [0.00, 0.00];   % a reasonable starting guess for the optimizer

%% Step 3: Collect the constants into one struct
%
% Here, we create one struct and store all our fixed parameters in it.

params.alpha = alpha;
params.k = k_glass;
params.T_amb = T_amb;
params.T_soft = T_soft;
params.T_max_back = T_max_back;
params.Lz = Lz;
params.spot_width = spot_width;
params.HAZ_fraction = HAZ_fraction;

%% Step 4: Run the optimization

options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
                        'OutputFcn', @(v, optimValues, state) trace_fun(v, optimValues, state, params));

fprintf('\n%-6s %-12s %-12s %-14s %-14s\n', 'Iter', 'x, power', 'y, time', 'T_front (C)', 'T_back (C)');
fprintf('%s\n', repmat('-', 1, 60));

[x_opt, J_opt, exitflag] = fmincon(@(v) objective_fun(v, params), x0, [], [], [], [], lb, ub, @(v) constraint_fun(v, params), options);

power_opt = x_opt(1);
time_opt  = x_opt(2);

fprintf('\n================ OPTIMIZATION RESULT ================\n');
fprintf('Best power, x        = %.4f W\n', power_opt);
fprintf('Best exposure time, y = %.4f s\n', time_opt);
fprintf('Smallest affected zone found = %.4f mm\n', J_opt*1e3);
fprintf('Exit flag, 1 means it converged normally = %d\n', exitflag);
fprintf('=======================================================\n\n');

%% Step 5: Build a grid so we can see the whole picture
%
% The optimizer gave us one answer but explored several others.
% Let's view a grid of power and exposure time value that was also possible

power_vals = linspace(lb(1), ub(1), 60);
time_vals  = linspace(lb(2), ub(2), 60);
[Xg, Yg] = meshgrid(power_vals, time_vals);

Tfront_grid = zeros(size(Xg));
Tback_grid  = zeros(size(Xg));
HAZ_grid    = zeros(size(Xg));

for i = 1:numel(Xg)
    xi = Xg(i);
    yi = Yg(i);

    q0 = flux_amplitude(xi, spot_width);

    % front temperature always peaks right when the laser switches off
    Tfront_grid(i) = T_amb + 2*q0*sqrt(alpha*yi/pi)/k_glass;

    % back temperature might peak a little after the laser switches off,
    % so we check a range of times instead of just one
    tt = linspace(max(yi/1000, 1e-6), 4*yi, 50);
    dT_back = surface_flux_temp_rise(q0, yi, Lz, tt, alpha, k_glass);
    Tback_grid(i) = T_amb + max(dT_back);

    HAZ_grid(i) = haz_radius(xi, yi, params);
end

feasible = (Tfront_grid >= T_soft) & (Tback_grid <= T_max_back);

%% Step 6: Plot the front temperature
%
% Where the scratch actually gets hot enough to soften. The red line
% marks the exact boundary, the softening threshold T_soft.

figure('Name', 'Front Temperature', 'Position', [80 80 750 560])
contourf(Xg, Yg, Tfront_grid, 20)
hold on
contour(Xg, Yg, Tfront_grid, [T_soft T_soft], 'r-', 'LineWidth', 2)
colorbar
clim([25 3800])
xlabel('x, power (W)')
ylabel('y, exposure time (s)')
title('Front (Scratch Side) Temperature Across the Search Grid, Red Line Marks the Softening Threshold T_{soft}')

%% Step 7: Plot the back temperature
%
% Where the far side of the glass stays safe. The red line marks the
% exact boundary, the maximum safe temperature T_max_back.

figure('Name', 'Back Temperature', 'Position', [150 80 750 560])
contourf(Xg, Yg, Tback_grid, 20)
hold on
contour(Xg, Yg, Tback_grid, [T_max_back T_max_back], 'r-', 'LineWidth', 2)
colorbar
clim([25 3800])
xlabel('x, power (W)')
ylabel('y, exposure time (s)')
title('Back (Far Side) Temperature Across the Search Grid, Red Line Marks the Safety Limit T_{max,back}')

%% Step 8: Plot the heat affected zone size
%
% Our actual objective, the size of the affected zone, with the
% feasible region traced in white and the optimizer's answer marked
% with a star.

figure('Name', 'Heat Affected Zone Size', 'Position', [220 80 750 560])
contourf(Xg, Yg, HAZ_grid*1e3, 20)
hold on
contour(Xg, Yg, double(feasible), [0.5 0.5], 'w-', 'LineWidth', 2.5)
plot(power_opt, time_opt, 'r*', 'MarkerSize', 16, 'LineWidth', 2)
colorbar
clim([25 3800])
xlabel('x, power (W)')
ylabel('y, exposure time (s)')
title('Heat Affected Zone Radius Across the Search Grid, White Outline Marks the Feasible Region, Star Marks the Optimizer''s Answer')

%% Step 9: Plot what actually happens over time at the answer we found

t_history = linspace(1e-4, 3*time_opt, 200);
q0_opt = flux_amplitude(power_opt, spot_width);

T_front_hist = T_amb + surface_flux_temp_rise(q0_opt, time_opt, 0, t_history, alpha, k_glass);
T_back_hist  = T_amb + surface_flux_temp_rise(q0_opt, time_opt, Lz, t_history, alpha, k_glass);

figure('Name', 'Temperature Over Time', 'Position', [300 80 750 560])
plot(t_history, T_front_hist, 'r-', 'LineWidth', 2)
hold on
plot(t_history, T_back_hist, 'b-', 'LineWidth', 2)
yline(T_soft, 'r--', 'must reach this to soften')
yline(T_max_back, 'b--', 'must not cross this line')
xline(time_opt, 'k:', 'laser switches off here')
xlabel('Time (s)')
ylabel('Temperature (degrees C)')
legend('Scratch side', 'Far side of the glass', 'Location', 'best')
title(sprintf('At x = %.3f W and y = %.3f s', power_opt, time_opt))
grid on

%% ================================================================
%  Local functions
% ================================================================

function y = ierfc(eta)
    % This shows up naturally when you solve the heat equation for a
    % surface heated at a constant rate.
    y = (1/sqrt(pi)) * exp(-eta.^2) - eta .* erfc(eta);
end

function q0 = flux_amplitude(power, spot_width)
    % Converts total power into how concentrated the heat is at the
    % center of the spot. A narrower spot with the same power gets
    % hotter at the center, which is exactly what you would expect.
    q0 = power / (2*pi*spot_width^2);
end

function dT = surface_flux_temp_rise(q0, exposure_time, depth, t, alpha, k)
    % This gives the temperature rise at a given depth and time, for a
    % surface that is heated at a constant rate starting at t = 0 and
    % switched off at t = exposure_time.
    dT = zeros(size(t));
    for i = 1:numel(t)
        ti = t(i);
        if ti <= exposure_time
            eta = depth / (2*sqrt(alpha*ti));
            dT(i) = 2*q0*sqrt(alpha*ti)/k * ierfc(eta);
        else
            % once the laser is off, we subtract an equal and opposite
            % heating that would have started at exposure_time, this
            % trick correctly captures the cooling that follows
            eta1 = depth / (2*sqrt(alpha*ti));
            eta2 = depth / (2*sqrt(alpha*(ti - exposure_time)));
            dT(i) = 2*q0*sqrt(alpha)/k * ...
                    (sqrt(ti)*ierfc(eta1) - sqrt(ti - exposure_time)*ierfc(eta2));
        end
    end
end

function R = haz_radius(power, exposure_time, params)
    % Works out how far sideways from the scratch the glass still
    % gets hot enough to count as affected. Longer exposure times let
    % heat spread further sideways, which is why exposure_time shows
    % up inside the square root here.
    q0 = flux_amplitude(power, params.spot_width);

    dT_center = 2*q0*sqrt(params.alpha*exposure_time/pi) / params.k;

    spread_width = sqrt(params.spot_width^2 + 2*params.alpha*exposure_time);

    dT_threshold = params.HAZ_fraction * (params.T_soft - params.T_amb);

    if dT_center <= dT_threshold
        R = 0;
    else
        R = spread_width * sqrt(2*log(dT_center/dT_threshold));
    end
end

function J = objective_fun(v, params)
    % v(1) is our x, the power. v(2) is our y, the exposure time.
    % We want to minimize the size of the affected zone.
    J = haz_radius(v(1), v(2), params);
end

function [c, ceq] = constraint_fun(v, params)
    % fmincon expects every constraint written as something that must
    % stay less than or equal to zero, so that is how these are set up.
    power = v(1);
    exposure_time = v(2);
    q0 = flux_amplitude(power, params.spot_width);

    % constraint 1, the front/scratch must reach the softening temperature
    T_front_peak = params.T_amb + 2*q0*sqrt(params.alpha*exposure_time/pi)/params.k;
    c1 = params.T_soft - T_front_peak;

    % constraint 2, the back must not exceed the safety limit
    tt = linspace(max(exposure_time/1000, 1e-6), 4*exposure_time, 50);
    dT_back = surface_flux_temp_rise(q0, exposure_time, params.Lz, tt, params.alpha, params.k);
    T_back_peak = params.T_amb + max(dT_back);
    c2 = T_back_peak - params.T_max_back;

    c = [c1; c2];
    ceq = [];
end

function stop = trace_fun(v, optimValues, state, params)
    % This shows all the mini experiments the optimizer ran to find the
    % best answer

    stop = false;

    if strcmp(state, 'iter')
        power = v(1);
        exposure_time = v(2);

        q0 = flux_amplitude(power, params.spot_width);
        T_front = params.T_amb + 2*q0*sqrt(params.alpha*exposure_time/pi)/params.k;

        tt = linspace(max(exposure_time/1000, 1e-6), 4*exposure_time, 50);
        dT_back = surface_flux_temp_rise(q0, exposure_time, params.Lz, tt, params.alpha, params.k);
        T_back = params.T_amb + max(dT_back);

        fprintf('%-6d %-12.4f %-12.4f %-14.1f %-14.1f\n', ...
                optimValues.iteration, power, exposure_time, T_front, T_back);
    end
end
