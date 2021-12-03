% LQG Mean Field Games with a Markov chain as its common noise
% Section 5: Numerical results

close all
clc
clear

% parameters

T = 5;
gamma_0 = 0.5;
gamma_1 = 0.6;

sigma_0 = 2;
h_0 = 2;
h_1 = 5;
g_0 = 3;
g_1 = 1;

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

% find b

b = zeros(2,N);
b(:,end) = [g_0; g_1];
for i = 1:(N-1)
    b(:,N-i) = b(:,N-i+1) - delta*(v - Q*b(:,N-i+1) + 4*a(:,N-i+1).*b(:,N-i+1));
end

% find c

c = zeros(2,N);
for i = 1:(N-1)
    c(:,N-i) = c(:,N-i+1) - delta*(-b(:,N-i+1) - a(:,N-i+1) -Q*c(:,N-i+1));
end

% generate Y

rng(100);
Y_0 = (rand(1,N)>= 1-gamma_0*delta);
Y_1 = (rand(1,N)>= gamma_1*delta);
Y_temp = [Y_0;Y_1];
Y = false(1,N);
for i = 2:length(Y_0)
    if Y_temp(Y(i-1)+1,i) == Y(i-1)
        Y(i) = Y(i-1);
    else
        Y(i) = ~Y(i-1);
    end
end


% generate Z

Z = zeros(1,N);
Z(1) = sigma_0;

for i = 2:N
    Z(i) = Z(i-1) + delta*(1-4*a(Y(i-1)+1,i-1)*Z(i-1));
end


% generate alpha, X

alpha = zeros(1,N);
X = zeros(1,N);
rng(20);
W = normrnd(0,sqrt(delta),[1,N]);
for i = 2:N
    X(i) = X(i-1) + delta*alpha(i-1) + W(i-1);
    alpha(i) = -2*a(Y(i)+1,i)*X(i);
end

% generate V

V = zeros(1,N);
for i = 1:N
    V(i) = a(Y(i)+1,i)*X(i)^2 + b(Y(i)+1,i)*Z(i) + c(Y(i)+1,i);
end

% sigma_t

sigma = Z;

% plot the results

figure(1)
hold on;
plot(t_all,a(1,:));
plot(t_all,a(2,:));
hold off;
xlabel('t')
ylabel('a_0,a_1')
legend({'a_0','a_1'})

figure(2)
hold on;
plot(t_all,b(1,:));
plot(t_all,b(2,:));
hold off;
xlabel('t')
ylabel('b_0,b_1')
legend({'b_0','b_1'})

figure(3)
hold on;
plot(t_all,c(1,:));
plot(t_all,c(2,:));
hold off;
xlabel('t')
ylabel('c_0,c_1')
legend({'c_0','c_1'})

figure(4)
hold on;
plot(t_all,V);
plot(t_all,alpha);
plot(t_all,sigma);
hold off;
xlabel('t')
ylabel('Value')
legend({'V','\alpha','\nu_t'})


% N-player games

PlyNm_all = [10,20,50,100];
%PlyNm_all = 5;
Mean_0_all = zeros(length(PlyNm_all), N);
Sigma_N_all = zeros(length(PlyNm_all), N);
Value_1_all = zeros(length(PlyNm_all), N);
for num = 1:length(PlyNm_all)
    PlyNm = PlyNm_all(num);

    Lambda = zeros([PlyNm, PlyNm, PlyNm]); % terminal for A_i0 A_i1
    one_temp = ones([PlyNm,1]);
    for i = 1:PlyNm
        Lambda(:,:,i) = eye(PlyNm); 
        Lambda(i,:,i) = -one_temp';
        Lambda(:,i,i) = -one_temp;
        Lambda(i,i,i) = PlyNm-1;
    end

    A_i0 = zeros([PlyNm, PlyNm^2, N]); 
    A_i1 = zeros([PlyNm, PlyNm^2, N]);
    C_i0 = zeros([PlyNm, N]);
    C_i1 = zeros([PlyNm, N]);

    for i = 1:PlyNm
        A_i0(:,(((i-1)*PlyNm+1):i*PlyNm),end) = g_0/PlyNm*Lambda(:,:,i);
        A_i1(:,(((i-1)*PlyNm+1):i*PlyNm),end) = g_1/PlyNm*Lambda(:,:,i);
    end

    A_M = zeros([PlyNm^2, PlyNm, PlyNm]);
    for i = 1:PlyNm
        A_M( (0:(PlyNm-1))*PlyNm+(1:PlyNm),1:PlyNm,i) = 4*eye(PlyNm);
        A_M((i-1)*PlyNm+i,i,i) = 2;
    end

    for i = 1:(N-1)
        for ply = 1:PlyNm
            A_i0(:,(((ply-1)*PlyNm+1):ply*PlyNm),N-i) = A_i0(:,(((ply-1)*PlyNm+1):ply*PlyNm),N-i+1)- delta*( ...
                A_i0(:,:,N-i+1)*A_M(:,:,ply)*A_i0(:,(((ply-1)*PlyNm+1):ply*PlyNm),N-i+1)+gamma_0*...
                A_i0(:,(((ply-1)*PlyNm+1):ply*PlyNm),N-i+1)-gamma_0*A_i1(:,(((ply-1)*PlyNm+1):ply*PlyNm),N-i+1)...
                -h_0/PlyNm*Lambda(:,:,ply) );
            A_i1(:,(((ply-1)*PlyNm+1):ply*PlyNm),N-i) = A_i1(:,(((ply-1)*PlyNm+1):ply*PlyNm),N-i+1)- delta*( ...
                A_i1(:,:,N-i+1)*A_M(:,:,ply)*A_i1(:,(((ply-1)*PlyNm+1):ply*PlyNm),N-i+1)+gamma_1*...
                A_i1(:,(((ply-1)*PlyNm+1):ply*PlyNm),N-i+1)-gamma_1*A_i0(:,(((ply-1)*PlyNm+1):ply*PlyNm),N-i+1)...
                -h_1/PlyNm*Lambda(:,:,ply) );
            C_i0(ply,N-i) = C_i0(ply,N-i+1) - delta*(-trace(A_i0(:,(((ply-1)*PlyNm+1):ply*PlyNm),N-i+1)) +...
                gamma_0*C_i0(ply,N-i+1)-gamma_0*C_i1(ply,N-i+1));
            C_i1(ply,N-i) = C_i1(ply,N-i+1) - delta*(- trace(A_i1(:,(((ply-1)*PlyNm+1):ply*PlyNm),N-i+1)) +...
                gamma_1*C_i1(ply,N-i+1)-gamma_1*C_i0(ply,N-i+1));
        end  
    end
    
    sample = 100;
    rng(20);
    W_N = sqrt(delta)*normrnd(0,1,[N,PlyNm,sample]);
    X_0 = zeros([PlyNm,sample,N]);
    X_0(:,:,1) = 1/sqrt(delta)*reshape(W_N(1,:,:),[PlyNm,sample])*sqrt(sigma_0);
    for i = 1:(N-1)
        if Y(i) == 0
            A_0_temp = A_i0(:,(0:(PlyNm-1))*PlyNm+(1:PlyNm),i);
            X_0(:,:,i+1) = X_0(:,:,i)-delta*2*A_0_temp'*X_0(:,:,i)+reshape(W_N(i+1,:,:),[PlyNm,sample]); 
            Value_1_all(num,i) = X_0(:,1,i).'*A_i0(:,1:PlyNm,i)*X_0(:,1,i)+C_i0(1,i);
        else
            A_0_temp = A_i1(:,(0:(PlyNm-1))*PlyNm+(1:PlyNm),i);
            X_0(:,:,i+1) = X_0(:,:,i)-delta*2*A_0_temp'*X_0(:,:,i)+reshape(W_N(i+1,:,:),[PlyNm,sample]); 
            Value_1_all(num,i) = X_0(:,1,i).'*A_i1(:,1:PlyNm,i)*X_0(:,1,i)+C_i1(1,i);
        end
      
    end
    
    Mean_0 = sum(sum(X_0,1),2)/PlyNm/sample;
    Mean_0_all(num,:) = reshape(Mean_0,[1,N]);
    Sigma_N = sum(sum(X_0.^2,1),2)/PlyNm/sample;
    Sigma_N_all(num,:) = reshape(Sigma_N,[1,N]);

end


% plot the results

figure(5)
hold on;
plot(t_all,zeros(1,N),'DisplayName','Generic Player')
for i = 1:length(PlyNm_all)
    text = [num2str(PlyNm_all(i)),' Players'];
    plot(t_all, Mean_0_all(i,:),'DisplayName',text); 
end
hold off;
xlabel('t')
ylabel('Mean');
legend show

figure(6)
hold on;
plot(t_all,sigma,'DisplayName','Generic Player');
for i = 1:length(PlyNm_all)
    text = [num2str(PlyNm_all(i)),' Players'];
    plot(t_all, Sigma_N_all(i,:),'DisplayName',text); 
end
hold off;
xlabel('t')
ylabel('2nd Momoent');
legend show;

figure(7)
hold on;
plot(t_all,V,'DisplayName','Value function of the Generic Player');
for i = 1:length(PlyNm_all)
    text = ['Player 1 value function under ',num2str(PlyNm_all(i)),' Players'];
    plot(t_all, Value_1_all(i,:),'DisplayName',text); 
end
hold off;
xlabel('t')
ylabel('Value function');
legend show;


