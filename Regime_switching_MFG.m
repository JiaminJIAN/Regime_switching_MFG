% Regime switching mean field games with quadratic costs
% Section 5: Numerical results

close all
clc
clear

T = 5;
pi_00 = 0.3;
pi_10 = 1-pi_00;
gamma_0 = 0.5;
gamma_1 = 0.6;
h_0 = 2;
h_1 = 5;
g_0 = 1;
g_1 = 3;
mu_0 = 0.5;
mu_1 = -0.5;
nu_0 = 4;
nu_1 = 1;

delta = 0.01;
N = T/delta+1;
t_all = 0:delta:T;
eps_b = 10^(-5);

% find a
a = zeros(2,N);
a(:,end) = [g_0;g_1];
v = [-h_0;-h_1];
Q = [-gamma_0, gamma_0;gamma_1,-gamma_1];

for i = 1:(N-1)
    a(:,N-i) = a(:,N-i+1) - delta*(v+2*a(:,N-i+1).^2 - Q*a(:,N-i+1));    
end

% hold on;
% plot(t_all,a(1,:));
% plot(t_all,a(2,:));
% hold off;
% xlabel('t')
% ylabel('a_0,a_1')
% legend({'a_0','a_1'})


% Broyden's method to find b, mu
pi_0 = 1/(gamma_0+gamma_1)*(gamma_1+(pi_00*gamma_0-pi_10*gamma_1)*exp(-(gamma_0+gamma_1)*t_all));
pi_1 = 1/(gamma_0+gamma_1)*(gamma_0+(pi_10*gamma_1-pi_00*gamma_0)*exp(-(gamma_0+gamma_1)*t_all));
dpi_0 = -(pi_00*gamma_0-pi_10*gamma_1)*exp(-(gamma_0+gamma_1)*t_all);
dpi_1 = -(pi_10*gamma_1-pi_00*gamma_0)*exp(-(gamma_0+gamma_1)*t_all);
phi_0 = dpi_0./pi_0;
phi_1 = dpi_1./pi_1;
phi_01 = pi_0./pi_1;
phi_10 = pi_1./pi_0;

% Create matrix of each time t to the ODE
M_BMU = zeros([4,4,N]);
M_BMU(1,1,:) = 2*a(1,:)+gamma_0;
M_BMU(1,2,:) = -gamma_0*ones([N,1]);
M_BMU(1,3,:) = 2*h_0*ones([N,1]);
M_BMU(2,1,:) = -gamma_1*ones([N,1]);
M_BMU(2,2,:) = 2*a(2,:)+gamma_1;
M_BMU(2,4,:) = 2*h_1*ones([N,1]);
M_BMU(3,1,:) = -1*ones([N,1]);
M_BMU(3,3,:) = -(2*a(1,:) + gamma_0 + phi_0);
M_BMU(3,4,:) = gamma_1*phi_10;
M_BMU(4,2,:) = -1*ones([N,1]);
M_BMU(4,3,:) = gamma_0*phi_01;
M_BMU(4,4,:) = -(2*a(2,:) + gamma_1 + phi_1);

% Check the matrix of err of b_beta(0) is invertible
H = [2*h_0,0; 0, 2*h_1];
M_T = (sum(M_BMU,3)-0.5*(M_BMU(:,:,1)+M_BMU(:,:,end)))*delta;
Check = [eye(2), H]*expm(M_T)*[eye(2);zeros([2,2])];
det(Check)

delta_b = 0.01;
b_0 = zeros([4,3,N]);
b_0(1,3,1) = -delta_b;
b_0(2,2,1) = delta_b;
b_0(3:4,:,1) = [mu_0, mu_0, mu_0; mu_1, mu_1, mu_1];

for i = 1:(N-1)
   b_0(:,:,i+1) = b_0(:,:,i)+M_BMU(:,:,i)*b_0(:,:,i)*delta; 
end

J_b = -1/delta_b*[ b_0(1,3,end)+2*g_0*b_0(3,3,end)-b_0(1,1,end)-2*g_0*b_0(3,1,end) , ...
    -b_0(1,2,end)-2*g_0*b_0(3,2,end)+b_0(1,1,end)+2*g_0*b_0(3,1,end) ;...
    b_0(2,3,end)+2*g_1*b_0(4,3,end)-b_0(2,1,end)-2*g_1*b_0(4,1,end) ,...
    -b_0(2,2,end)-2*g_1*b_0(4,2,end)+b_0(2,1,end)+2*g_1*b_0(4,1,end)];
J_b_inv = inv(J_b);

bmu = b_0(:,1,:);
bmu_temp = zeros([4,1,N]);
bmu_temp(:,1,1) = [0.5; 0.5; mu_0; mu_1];
for i = 1:(N-1)
    bmu_temp(:,1,i+1) = bmu_temp(:,1,i)+M_BMU(:,:,i)*bmu_temp(:,1,i)*delta;
end

while norm(bmu(1:2,end) + 2*[g_0;g_1].*bmu(3:4,end)) > eps_b 
    df = bmu(1:2,end)+2*[g_0;g_1].*bmu(3:4,end)-bmu_temp(1:2,end)-2*[g_0;g_1].*bmu_temp(3:4,end);
    dx = bmu(1:2,1)-bmu_temp(1:2,1);
    J_b_inv = J_b_inv + (dx - J_b_inv*df)/(dx'*J_b_inv*df)*dx'*J_b_inv; % Update the inverse of Jacobian
    bmu_temp = bmu;
    bmu(1:2,1) = bmu_temp(1:2,1) - J_b_inv*(bmu_temp(1:2,end)+2*[g_0;g_1].*bmu(3:4,end));
    for i = 1:(N-1)
        bmu(:,i+1) = bmu(:,i)+M_BMU(:,:,i)*bmu(:,i)*delta;
    end
end

bmu = reshape(bmu,[4,N]);

% Find nu
M_nu = zeros([2,2,N]);
M_nu(1,1,:) = -4*a(1,:)-(gamma_0 + phi_0);
M_nu(1,2,:) = gamma_1*phi_10;
M_nu(2,1,:) = gamma_0*phi_01;
M_nu(2,2,:) = -4*a(2,:)-(gamma_1 + phi_1);
V_nu = zeros([2,N]);
V_nu(1,:) = 1 + (gamma_0+phi_0).*bmu(3,:).^2 + gamma_1*phi_10.*(bmu(4,:).^2 - 2*bmu(3,:).*bmu(4,:));
V_nu(2,:) = 1 + (gamma_1+phi_1).*bmu(4,:).^2 + gamma_0*phi_01.*(bmu(3,:).^2 - 2*bmu(3,:).*bmu(4,:));

nu = zeros([2,N]);
nu(1,1) = nu_0;
nu(2,1) = nu_1;
for i = 1:(N-1)
    nu(:,i+1) = nu(:,i) + (V_nu(:,i)+M_nu(:,:,i)*nu(:,i))*delta;
end

% Find c
c = zeros([2,N]);
c(:,end) = [g_0;g_1].*(bmu(3:4,end).^2+nu(:,end));
M_c = [gamma_0 , -gamma_0; -gamma_1, gamma_1];
V_c = zeros([2,N]);
V_c(1,:) = -a(1,:)+1/2*bmu(1,:).^2-h_0*(bmu(3,:).^2+nu(1,:));
V_c(2,:) = -a(2,:)+1/2*bmu(2,:).^2-h_1*(bmu(4,:).^2+nu(2,:));

for i = 1:(N-1)
    c(:,N-i) = c(:,N-i+1) - (V_c(:,N-i+1) + M_c*c(:,N-i+1))*delta;
end

figure(1);
hold on;
plot(t_all,bmu(3,:));
plot(t_all,bmu(4,:));
hold off;
xlabel('t')
ylabel('Mean');
legend('\mu_0','\mu_1')

figure(2);
hold on;
plot(t_all,nu(1,:));
plot(t_all,nu(2,:));
hold off;
xlabel('t')
ylabel('Variance');
legend('\nu_0','\nu_1')

% hold on;
% plot(t_all,bmu(1,:));
% plot(t_all,bmu(2,:));
% hold off;
% xlabel('t')
% ylabel('b_0,b_1')
% legend({'b_0','b_1'})
% 
% hold on;
% plot(t_all,c(1,:));
% plot(t_all,c(2,:));
% hold off;
% xlabel('t')
% ylabel('c_0,c_1')
% legend({'c_0','c_1'})


N_0 = [10 , 70,  100, 400];
status = zeros([1,N]);
status(N_0(1):N_0(2)) = ones([1,N_0(2)-N_0(1)+1]);
status(N_0(3):N_0(4)) = ones([1,N_0(4)-N_0(4)+1]);
sample = 10;
rng(100)
W_t = sqrt(delta)*normrnd(0,1,[sample,N]);
X_t = zeros([sample,N]);
X_t(:,1) = 1/sqrt(delta)*W_t(:,1);
alpha = zeros([sample,N]);
for i = 1:(N-1)
    if status(i) == 0
        X_t(:,i+1) = X_t(:,i) -(2*a(1,i)*X_t(:,i)+bmu(1,i))*delta + W_t(:,i+1);
        alpha(:,i) = -(2*a(1,i)*X_t(:,i)+bmu(1,i));
    else
        X_t(:,i+1) = X_t(:,i) -(2*a(2,i)*X_t(:,i)+bmu(2,i))*delta + W_t(:,i+1);
        alpha(:,i) = -(2*a(2,i)*X_t(:,i)+bmu(2,i));
    end
end
if status(N) == 0
    alpha(:,N) = -(2*a(1,N)*X_t(:,N)+bmu(1,N));
else
    alpha(:,N) = -(2*a(2,N)*X_t(:,N)+bmu(2,N));
end
figure(3)
plot(t_all,X_t)
for i = 1:4
    xline(N_0(i)*delta);
end
xlabel('t')
ylabel('X_t')
title('Sample paths of X_t')
figure(4)
plot(t_all,alpha)
for i = 1:4
    xline(N_0(i)*delta);
end
xlabel('t')
ylabel('\alpha_t')
title('Sample paths of optimal control \alpha_t')



