clear
close all

% 1D string, fixed BCs at each end


%% Initial conditions
 
% init conditions

values = [0 2 3 5 5.2 5.4 5.6 5.8 6 6 4 2 1 0.5 3 3.25 3.5 3.6 3.7 3.75 3.775 3.8 3.5 4 4.25 4.75 5.75 6 6 0];


% interpolate init conditions
x = linspace(0,1,101);
y = [3*sin(pi*x(1:31) *100 / 31), zeros(1,101-31)];
%y = interp1(linspace(0,1,30),values,x);

%% simulation configuration

% time integration method (fwd or bwd). Central in space is fix
method = {'fwd', 'fwd'}; % [first integration, second integration]

% time step
dt = 1;

% time range
ts = 0:dt:1000; 
outy   = zeros(length(ts), length(y));
outy(1,:)   = y;
d2udx2 = zeros(length(ts), length(y));

% wave physics parameters
c = 0.5; % wave speed
zeta = 0; % friction damping (proportional to curvative changes in time)


%% Running the simulation

% initiating some arrays
outy_dot = 0;
y_ext = zeros(1, length(x));
error_i = 0;

i = 1;
tic
ystep = 1;
for t = ts  
   
   % linear wave equation
   % 0    = u_xxt * zeta + c^2*u_xx - u_tt
   % u_tt = c^2 * u_xx + zeta * u_xxt
   

   % extend domain
%    y_ext(2:end-1) = outy(max(i-1,1),2:end-1);
%    y_ext(1) = y(1);
%    y_ext(end) = y(end);
   % compute 2nd derivative using image filtering techniques (yes, this works)

   %%%% Time derivatives?%%%%
   % fwd derivative of curvature (model friction damping) u_xxt
   % this also acts as artificial dispersion. Not goint to change this to
   % backwards derivatives...
   speed = ( d2udx2(max(i-1,1),:) - d2udx2(max(i-2,1),:) ) / dt;
   outy_dotdot_term_two = zeta * speed;
   outy_dotdot_wave = outy_dotdot_term_one + outy_dotdot_term_two;
   
   %%%% Integration
   outy_dotdot = outy_dotdot_wave;
   if strcmp(method(1), 'fwd')
       %%%% Spacial derivatives %%%%
       % central difference 2nd order
       h = [1 -2 1];
       d2udx2(i,:) = imfilter(outy(max(i-1,1),:),h); 
       outy_dotdot_term_one = c^2 * d2udx2(i,:);
       outy_dot    = outy_dot + outy_dotdot * dt; 
   else
       % solve outy_dot_i+1 - out_dotdot * dt = outy_dot
   end
   if strcmp(method(2), 'fwd')
       outy(i,2:end-1)   = outy(max(i-1,1),2:end-1) + outy_dot(2:end-1) * dt; 
   else
       % do some implicit stuff here.
   end
   
   i = i + 1;
end
toc

%% oscillator response

if false
    fig1 = figure();
    plot(ts,outy(:,50))
end

%% animate

if true
    
    fig = figure();
    hold on;
    i = 1;
    for i = 1:2/dt:length(ts)
        clf
        plot(x,outy(i,:))
        ylim([-8,8]);
        xlim([0,1.05]);
        drawnow
        pause(0.001)
        if ~(mod(i-1,100))
            disp(num2str(round((i) / length(ts) * 100)))
        end
    end
    

end