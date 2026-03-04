clear; close all; clc;

%% 1. 参数设置
N = 512;                % 矩阵大小
numModes = 36;          % Zernike 模式数 (Noll 索引从 1 到 36)
Nsamples = 200;         % 生成样本数
lambda = 2*pi;          % 将相位以弧度表示，1个 lambda 对应 2pi

% 目标范围 [0.8*lambda, 15*lambda]
pv_min = 0.8 * lambda;
pv_max = 15 * lambda;

% 文件夹准备
save_folder1 = 'Zer_phi';       % 原始相位图
save_folder2 = 'slm_Zer_phi';   % SLM 灰度图 (mod 2pi)
folders = {save_folder1, save_folder2};
for i = 1:length(folders)
    if ~exist(folders{i}, 'dir'), mkdir(folders{i}); end
end

%% 2. 坐标系建立
x = linspace(-1, 1, N);
[X, Y] = meshgrid(x, x);
[theta, rho] = cart2pol(X, Y);
mask = rho <= 1; % 瞳径掩膜

%% 3. 预计算 Zernike 基底 (为了提高速度)
fprintf('正在预计算 Zernike 基底...\n');
Z_basis = zeros(N, N, numModes);
for j = 1:numModes
    Z_basis(:,:,j) = single_zer(j, rho, theta) .* mask;
end

%% 4. 循环生成随机相位
all_coeffs = zeros(Nsamples, numModes); % 存储系数 [100, 36]

fprintf('正在生成并保存相位图...\n');
for k = 1:Nsamples
    % a. 生成随机系数并添加物理衰减 (低阶能量高，高阶能量低)
    % 使用 1/n 规律模拟大气湍流或一般光学元件像差分布
    coeffs = randn(numModes, 1);
    for j = 1:numModes
        [n, ~] = noll_to_nm(j);
        coeffs(j) = coeffs(j) / (n + 1); % 阶数越高，权重越小
    end
    
    % b. 合成相位
    phi = zeros(N, N);
    for j = 1:numModes
        phi = phi + coeffs(j) * Z_basis(:,:,j);
    end
    
    % c. 缩放相位到指定 PV 范围 [0.8*lambda, 15*lambda]
    current_pv = max(phi(mask)) - min(phi(mask));
    target_pv = pv_min + (pv_max - pv_min) * rand();
    phi = phi * (target_pv / current_pv);
    
    % 记录系数 (对应缩放后的相位)
    all_coeffs(k, :) = coeffs' * (target_pv / current_pv);
    
    % d. 保存原始相位灰度图 (归一化到 0-1 用于查看)    
    phi_norm = (phi - min(phi(:))) / (max(phi(:)) - min(phi(:)));
    phi_norm(~mask) = 255;
    imwrite(phi_norm, fullfile(save_folder1, sprintf('phi_%d.png', k)));
    
    % e. 保存 SLM 灰度图 (phi mod 2pi)
    % SLM 通常只能显示 0-2pi，映射到 0-255 像素值
    phi_slm = mod(phi, 2*pi) / (2*pi); 
    imwrite(phi_slm, fullfile(save_folder2, sprintf('slm_%d.png', k)));
end

%% 5. 保存系数矩阵
save('zernike_coeffs.mat', 'all_coeffs');
fprintf('任务完成！共生成 %d 个样本。\n', Nsamples);

%% --- 函数定义 ---

function Z = single_zer(j, rho, theta)
    [n, m] = noll_to_nm(j);
    Z = zernike_nm(n, m, rho, theta);
end

function Z = zernike_nm(n, m, rho, theta)
    Z = zeros(size(rho));
    idx = rho <= 1;
    if m == 0
        R = radial_poly(n, 0, rho(idx));
        Z(idx) = R;
    else
        R = radial_poly(n, abs(m), rho(idx));
        if m > 0
            Z(idx) = R .* cos(m*theta(idx));
        else
            Z(idx) = R .* sin(abs(m)*theta(idx));
        end
    end
    % Noll 归一化系数
    if m == 0
        Z = Z * sqrt(n + 1);
    else
        Z = Z * sqrt(2 * (n + 1));
    end
end

function R = radial_poly(n, m, rho)
    R = zeros(size(rho));
    for s = 0:(n-m)/2
        num = (-1)^s * factorial(n-s);
        den = factorial(s) * factorial((n+m)/2 - s) * factorial((n-m)/2 - s);
        R = R + (num/den) * rho.^(n-2*s);
    end
end

function [n, m] = noll_to_nm(j)
    % 寻找径向阶数 n
    n = 0;
    while (n + 1) * (n + 2) / 2 < j
        n = n + 1;
    end
    
    % 寻找在该 n 阶下的序列位置
    m_limit = n;
    % 该阶数 n 之前的累计模式数
    j_prior = n * (n + 1) / 2;
    % 在当前 n 阶中的相对位置 (1, 2, ..., n+1)
    k = j - j_prior;
    
    % 构造当前 n 阶下所有可能的 m 值，并按 Noll 规则排序
    % Noll 规则：低频率优先，同频率下正值优先（或根据 j 的奇偶性）
    if mod(n, 2) == 0
        m_list = [0, reshape([-(2:2:n); (2:2:n)], 1, [])];
    else
        m_list = reshape([-(1:2:n); (1:2:n)], 1, []);
    end
    
    % Noll 排序的精髓：对于相同的 |m|，偶数 j 对应正 m (cos)，奇数 j 对应负 m (sin)
    % 我们先按 |m| 从小到大排序
    [~, idx] = sort(abs(m_list));
    m_list = m_list(idx);
    
    % 这里的 k 就是在当前 n 阶下的索引
    m = m_list(k);
end