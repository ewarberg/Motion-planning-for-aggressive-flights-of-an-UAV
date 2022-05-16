%% --- Get trajectory data ---
clear all
% close all
clear classes

clc

py.importlib.import_module('numpy');
py.importlib.import_module('polytope');
py.importlib.import_module('scipy');
py.importlib.import_module('collections');

mod = py.importlib.import_module('motion_planning');
py.importlib.reload(mod);

pyOut = py.motion_planning.main();
trajsize = double(pyOut{1});
p = double(pyOut{2}); 
v = double(pyOut{3}); 
a = double(pyOut{4}); 

R_array = double(pyOut{5});

node_data = double(pyOut{6});  
timevec = double(pyOut{7})';        

obs_extreme = double(pyOut{8});
goal = double(pyOut{9});
obs_pos = double(pyOut{10});
uav_width = double(pyOut{11});
uav_height = double(pyOut{12});
f =  double(pyOut{13})'; 
w =  double(pyOut{14});

Rvec = zeros(3,3,length(timevec));
for i = 1:length(timevec)
    Rvec(:,:,i) = [R_array(i,:,1)', R_array(i,:,2)', R_array(i,:,3)'];
 end

P2 = [-uav_width/2 -uav_width/2 -uav_height/2; -uav_width/2 uav_width/2 -uav_height/2;uav_width/2 uav_width/2 -uav_height/2;uav_width/2 -uav_width/2 -uav_height/2;uav_width/2 -uav_width/2 uav_height/2;-uav_width/2 -uav_width/2 uav_height/2;-uav_width/2 uav_width/2 uav_height/2;uav_width/2 uav_width/2 uav_height/2];

px = p(:,1);
py = p(:,2);
pz = p(:,3);

vx = v(:,1);
vy = v(:,2);
vz = v(:,3);

ax = a(:,1);
ay = a(:,2);
az = a(:,3);

%% --- Run Simulink simulation --- 
% structure input signal R_d for simulink
Rd_mat.signals.values = Rvec;
Rd_mat.signals.dimensions = [3 3];
Rd_mat.time = timevec;

vd_vec.signals.values = v;
vd_vec.signals.dimensions = 3;
vd_vec.time = timevec;

X0 = [px(1), py(1), pz(1), vx(1), vy(1), vz(1), 0, 0, 0]';

% Run the Simulink simulation:
start = num2str(timevec(1));
stop = num2str(timevec(end));
step = num2str(0.003);       
SimOut = sim('Attitude_velocity_controller','StartTime',start,'StopTime',stop,'FixedStep',step);
states = SimOut.states;
R_sim = SimOut.R_sim;
F_sim = SimOut.F_sim;
F_sim_disturbed = SimOut.F_sim_disturbed;
e_v_sim = SimOut.e_v_sim;
e_R_sim = SimOut.e_R_sim;

sim_x = states.data(:,1);
sim_y = states.data(:,2);
sim_z = states.data(:,3);
R_mat_sim = R_sim.data;
n_sim = zeros(length(sim_x), 3); n_sim(:,3) = 1;
f_sim_disturbed = F_sim_disturbed.data;
f_sim = F_sim.data;
e_v = e_v_sim.data;
e_R = e_R_sim.data;
timevec_sim = 0:str2double(step):str2double(stop);
%% --- Plot 3D trajectory ---

%close all

figure
plot3(px, py, pz, '.'); grid on; axis equal;
hold on
for i = 1:length(obs_pos(:,1))-6
    X = obs_extreme(i,:,1);
    Y = obs_extreme(i,:,2);
    Z = obs_extreme(i,:,3);
    [k1,av1] = convhull(X,Y,Z);
    h2 = trisurf(k1,X,Y,Z,'FaceColor','r','FaceAlpha',.4); % plot surfaces of obstacles
end
axis([0 5 0 5 0 5]);
xlabel('x [m]','FontSize',12);
ylabel('y [m]','FontSize',12)
zlabel('z [m]','FontSize',12)
title('UAV trajectory (algorithm)')
hold on
plot3(goal(1), goal(2), goal(3), '*');  % goal

pause
for i = 1:round(length(timevec)/30):length(timevec)  % animate orientation
    R = Rvec(:,:,i);
    P_rot = R*P2';
    P_rot(1,:) = P_rot(1,:) + p(i,1);
    P_rot(2,:) = P_rot(2,:) + p(i,2);
    P_rot(3,:) = P_rot(3,:) + p(i,3);
    X = P_rot(1,:);
    Y = P_rot(2,:);
    Z = P_rot(3,:);
    [k1,av1] = convhull(X,Y,Z);
    h2 = trisurf(k1,X,Y,Z,'FaceColor','c'); % plot surfaces of uav
    pause(0.01)
end

% --- Plot simulink trajectory ---

figure
plot3(sim_x, sim_y, sim_z, '.'); grid on; axis equal; %Plot all positions of the simulated trajectory
hold on
for i = 1:length(obs_pos(:,1))-6
    X = obs_extreme(i,:,1);
    Y = obs_extreme(i,:,2);
    Z = obs_extreme(i,:,3);
    [k1,av1] = convhull(X,Y,Z);
    h2 = trisurf(k1,X,Y,Z,'FaceColor','r','FaceAlpha',.4); % plot surfaces of obstacles
end
axis([0 5 0 5 0 5]);
xlabel('x [m]','FontSize',12);
ylabel('y [m]','FontSize',12)
zlabel('z [m]','FontSize',12)
title('UAV trajectory (simulation)')
hold on
pause
for i = 1:round(length(sim_x)/30):length(sim_x)  % animate orientation
    R = R_mat_sim(:,:,i);
    n_sim(i,:) = R*n_sim(i,:)';
    P_rot = R*P2';
    P_rot(1,:) = P_rot(1,:) + sim_x(i);
    P_rot(2,:) = P_rot(2,:) + sim_y(i);
    P_rot(3,:) = P_rot(3,:) + sim_z(i);
    X = P_rot(1,:);
    Y = P_rot(2,:);
    Z = P_rot(3,:);
    [k1,av1] = convhull(X,Y,Z);
    h2 = trisurf(k1,X,Y,Z,'FaceColor','c'); % plot surfaces of uav
    pause(0.01)
end

%%  --- Plot errors in Simulink simulation ---
% close all
% plot errors in velocity and position
% display attitude error as error in angle
hold on
e_alpha = zeros(1,length(e_R));
for i = 1:length(e_R)
    vec = (eye(3)+e_R(:,:,i))*[0;0;1];
    e_alpha(i) = acos(dot(vec,[0;0;1])/norm(vec));
end

figure
subplot(2,1,1)
plot(timevec_sim, e_v); grid on; title('Error in velocity [m/s]')
xlabel('t [s]')
ylabel('e_v [m/s]') 
subplot(2,1,2)
plot(timevec_sim, e_alpha); grid on; title('Error in attitude')
xlabel('t [s]')
ylabel('e_? [rad]') 

% compare thrust profiles
figure
subplot(2,1,1)
plot(timevec, f); grid on; title('Thrust profile (no disturbance)')
xlabel('t [s]')
ylabel('f [m/s]') 
subplot(2,1,2)
plot(timevec_sim, f_sim_disturbed); grid on; title('Thrust profile (disturbance)')
xlabel('t [s]')
ylabel('f [rad]') 


%%  --- Plot trajectory parameters ---
% 
% figure
% subplot(3,1,1)
% plot(timevec, px); grid on; title('Position in axis')
% hold on
% for i = 0:trajsize - 1  % plot nodes
%     plot(timevec(i*(npp-1) + 1), px(i*(npp-1) + 1), 'x', 'linewidth', 5)
% end
% xlabel('t [s]')
% ylabel('x-position [m]')
% subplot(3,1,2); grid on
% plot(timevec, py); grid on
% hold on
% for i = 0:trajsize - 1  % plot nodes
%     plot(timevec(i*(npp-1) + 1), py(i*(npp-1) + 1), 'x', 'linewidth', 5)
% end
% xlabel('t [s]')
% ylabel('y-position [m]')
% subplot(3,1,3); grid on
% plot(timevec, pz); grid on
% hold on
% for i = 0:trajsize - 1  % plot nodes
%     plot(timevec(i*(npp-1) + 1), pz(i*(npp-1) + 1), 'x', 'linewidth', 5)
% end
% xlabel('t [s]')
% ylabel('z-position [m]')
% 
% figure
% subplot(3,1,1); grid on;
% plot(timevec, vx); grid on; title('Velocity in axis')
% hold on
% for i = 0:trajsize - 1  % plot nodes
%     plot(timevec(i*(npp-1) + 1), vx(i*(npp-1) + 1), 'x', 'linewidth', 5)
% end
% xlabel('t [s]')
% ylabel('x-velocity [m/s]')
% subplot(3,1,2); grid on
% plot(timevec, vy); grid on
% hold on
% for i = 0:trajsize - 1  % plot nodes
%     plot(timevec(i*(npp-1) + 1), vy(i*(npp-1) + 1), 'x', 'linewidth', 5)
% end
% xlabel('t [s]')
% ylabel('y-velocity [m/s]')
% subplot(3,1,3); grid on
% plot(timevec, vz); grid on
% hold on
% for i = 0:trajsize - 1  % plot nodes
%     plot(timevec(i*(npp-1) + 1), vz(i*(npp-1) + 1), 'x', 'linewidth', 5)
% end
% xlabel('t [s]')
% ylabel('z-velocity [m/s]')
% 
% figure
% subplot(3,1,1)
% plot(timevec, ax); grid on; title('Acceleration in axis')
% hold on
% for i = 0:trajsize - 1  % plot nodes
%     plot(timevec(i*(npp-1) + 1), ax(i*(npp-1) + 1), 'x', 'linewidth', 5)
% end
% xlabel('t [s]')
% ylabel('x-acceleration [m/s^2]')
% subplot(3,1,2); grid on
% plot(timevec, ay); grid on
% hold on
% for i = 1:trajsize - 1  % plot nodes
%     plot(timevec(i*(npp-1) + 1), ay(i*(npp-1) + 1), 'x', 'linewidth', 5)
% end
% xlabel('t [s]')
% ylabel('y-acceleration [m/s^2]')
% subplot(3,1,3); grid on
% plot(timevec, az); grid on
% hold on
% for i = 1:trajsize - 1  % plot nodes
%     plot(timevec(i*(npp-1) + 1), az(i*(npp-1) + 1), 'x', 'linewidth', 5)
% end
% xlabel('t [s]')
% ylabel('z-acceleration [m/s^2]')
% 
% figure
% plot(timevec, f); grid on; title('Thrust')
% hold on
% for i = 0:trajsize - 1  % plot nodes
%     plot(timevec(i*(npp-1) + 1), f(i*(npp-1) + 1), 'x', 'linewidth', 5)
% end
% xlabel('t [s]')
% ylabel('Thrust [N]')
% 
% figure
% subplot(3,1,1)
% plot(timevec, w(:,1)); grid on; title('Body rates in axis')
% hold on
% for i = 0:trajsize - 1  % plot nodes
%     plot(timevec(i*(npp-1) + 1), w(i*(npp-1) + 1, 1), 'x', 'linewidth', 5)
% end
% xlabel('t [s]')
% ylabel('x-body rates [rad/s]')
% subplot(3,1,2); grid on
% plot(timevec, w(:,2)); grid on
% hold on
% for i = 1:trajsize - 1  % plot nodes
%     plot(timevec(i*(npp-1) + 1), w(i*(npp-1) + 1, 2), 'x', 'linewidth', 5)
% end
% xlabel('t [s]')
% ylabel('y-body rates [rad/s]')
% subplot(3,1,3); grid on
% plot(timevec, w(:,3)); grid on
% hold on
% for i = 1:trajsize - 1  % plot nodes
%     plot(timevec(i*(npp-1) + 1), w(i*(npp-1) + 1, 3), 'x', 'linewidth', 5)
% end
% xlabel('t [s]')
% ylabel('z-body rates [rad/s]')
