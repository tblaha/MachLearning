clear
close all

% This recreates the animation in the revolut.com app, just for fun
values = [0 2 3 5 5.2 5.4 5.6 5.8 6 6 4 2 1 0.5 3 3.25 3.5 3.6 3.7 3.75 3.775 3.8 3.5 4 4.25 4.75 5.75 6 6 6];

%%

x = linspace(0,1,101);
y = interp1(linspace(0,1,30),values,x);
%y = y + [zeros(1,21), ones(1,59), zeros(1,21)];
%rng(1)
%y = rand(size(x));

dt = 0.1;
ts = 0:dt:250;
outy = zeros(length(ts), length(y));
controly = zeros(length(ts), length(y));
%controly = y;
d2udx2 = zeros(length(ts), length(y));
olderror = 0;

% controller config
k_p = 0.5;
k_d = 3;
% k_p = 0;
% k_d = 0;
I = 10;

% wave physics
if true
    c = 0.1;
    zeta = 1;
    k_i = 0.002;
else
    c = 0;
    zeta = 0;
    k_i = 0.00;
end
%outy = y;

% inits
outy_dot = 0;
y_ext = zeros(1,length(x) + 2);
error_i = 0;

tic
i = 1;
ystep = 1;
for t = ts
   
   
   % two forces act on each osciallator: the control force to reach the
   % final shape and the one given by the linear wave equation. Instead of
   % forces, lets directly use accelerations
   
   % first apply the linear control accelaration to each oscillator:
   % sequentially add more authority to the controller:
   controly(i,:) = controly(max(i-1,1),:);
   if ~mod(t,1)
       controly(i, min(ystep,length(y))) = y( min(ystep,length(y)) );
       ystep = ystep + 1;
   end
   
   % evaluate errors
   error = controly(i,:) - outy(max(i-1,1),:);
   error_d = (error - olderror) / dt; % fwd 
   error_i = error_i + error;
   olderror = error;
   
   % linear control law:
   outy_dotdot_control = error_i * k_i + error * k_p + error_d * k_d;
   
   
   % linear wave equation, beun as fuck
   % central difference 2nd order
   h = [1 -2 1];
   % extend domain
   y_ext(2:end-1) = outy(max(i-1,1),:);
   y_ext(1) = controly(1);
   y_ext(end) = controly(end);
   % compute 2nd derivative using image filtering techniques (yes, beun)
   d2udx2_ext = imfilter(y_ext,h); % on extended domain
   d2udx2(i,:) = d2udx2_ext(2:end-1);
   % fwd derivative of curvature (model friction damping) u_xxt
   % this also acts as artificial dispersion
   speed = ( d2udx2(max(i-1,1),:) - d2udx2(max(i-2,1),:) ) / dt;
   outy_dotdot_wave = c^2 * d2udx2(i,:) + zeta * speed;
   
   % time stepping:
   outy_dotdot = 1/I * (outy_dotdot_control + outy_dotdot_wave);
   outy_dot    = outy_dot + outy_dotdot * dt; % fwd euler, should be good enough
   outy(i,:)   = outy(max(i-1,1),:) + outy_dot * dt; % fwd euler, should be good enough
   
   
   i = i + 1;
end
toc

%% oscillator response

if false
    fig1 = figure();
    plot(ts,outy(:,50))
    hold on
    plot(ts,controly(:,50))
end

%% animate

if true
    
    fig = figure();
    hold on;
    i = 1;
    for i = 1:2/dt:length(ts)
        clf
        plot(outy(i,:))
        ylim([-2.5,8]);
        xlim([0,120]);
        drawnow
        pause(0.001)
        if ~(mod(i-1,100))
            disp(num2str(round((i) / length(ts) * 100)))
        end
    end
    

end