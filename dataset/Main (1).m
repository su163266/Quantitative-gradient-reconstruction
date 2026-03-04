%完全由随机数种子，可以复现
clear; close all; clc;
seed = 2026; % 自定义种子值，建议记下来（后续复现必须用同一个）


rng(seed, 'twister'); % twister是MATLAB默认的随机数生成器，兼容性最好
% path for dataset saving
path1='.\train_in2\'; % path for the training dataset input
path2='.\train_gt\'; % path for the training dataset ground truth
path3='.\test_in\'; % path for the testing dataset input
path4='.\test_gt\'; % path for the testing dataset ground truth
path5='.\gt_phi\'; % path for the testing dataset ground truth
path6='.\slm_phi\'; % path for the testing dataset ground truth

% path generation
if ~exist(path1,'dir')
    mkdir(path1)
end
if ~exist(path2,'dir')
    mkdir(path2)
end
if ~exist(path3,'dir')
    mkdir(path3)
end
if ~exist(path4,'dir')
    mkdir(path4)
end
if ~exist(path5,'dir')
    mkdir(path5)
end
if ~exist(path6,'dir')
    mkdir(path6)
end

%% ===================== 1. 参数设置 =====================
phase_size = 256;      % 最终网络输入相位尺寸
train_num  = 9000;      % 数据样本数量
mu = 595;           % 均值
sigma = 270;        % 标准差
height_min = 6;
height_max = 1200;  % 最大高度限制
n_samples = train_num;   % 采样数量

h = mu + sigma * randn(n_samples, 1);
h_vals_clipped = h;
h_vals_clipped(h_vals_clipped < height_min) = height_min;
h_vals_clipped(h_vals_clipped > height_max) = height_max;
h_val = h_vals_clipped;


% --- 剪切参数 ---
pixel_shear = 4;       % 剪切位移（像素）
size_min = 5;          % 初始随机矩阵最小尺寸
size_max = 6;         % 初始随机矩阵最大尺寸

% --- 噪声参数 ---
noise_gaussian_std = 0.05;   % 读出噪声（高斯）
full_well = 2000;             % 满阱电子数（泊松噪声）

I0 = 1;   % 干涉背景强度
%% ===================== 2. 主循环生成数据 =====================
for jj = 1:train_num
    fprintf('正在生成第 %d / %d 个样本...\n', jj, train_num);

    size_xy = randi([size_min, size_max]);
    initial_matrix = rand(size_xy, size_xy);


    % 双三次插值生成平滑相位
    phase_pre = imresize(initial_matrix, ...
        [256,256], 'bicubic');

    %% ---------- 随机采样相位高度（分段分布） ----------
    % if jj <= 0.05 * train_num
    %     h_val = unifrnd(0.1*height_min, height_min);
    % elseif jj <= 0.2 * train_num
    %     h_val = unifrnd(height_min , height_min + 0.2*h_range);
    % elseif jj <= 0.4 * train_num
    %     h_val = unifrnd(height_min + 0.2*h_range, height_min + 0.5*h_range);
    % elseif jj <= 0.95 * train_num
    %     h_val = unifrnd(height_min + 0.9*h_range, height_max);
    % else
    %     h_val = unifrnd(height_max, 1.5*height_max);
    % end
    % % h_val = unifrnd(height_max , height_max+2);
   

    phase_pre = (phase_pre - min(phase_pre(:))) / ...
        (max(phase_pre(:)) - min(phase_pre(:))) * h_val(jj);


   %% ------------------ 前向差分梯度 as label grad ------------------
    [H, W] = size(phase_pre);
    gx = zeros(H,W);
    gy = zeros(H,W);

    % 前向差分
    gx(:,1:W-1) = phase_pre(:,2:W) - phase_pre(:,1:W-1);
    gy(1:H-1,:) = phase_pre(2:H,:) - phase_pre(1:H-1,:);

    % Neumann 边界 (natural extension)
    gx(:,W) = 0; gy(H,:) = 0;

   delta_phi_x = gx; delta_phi_y = gy;

    %% ---------- E. 剪切干涉强度（含噪声） ----------
    gamma = unifrnd(0.9, 0.95); % 模拟 70% 到 95% 的条纹对比度
    Ix_ideal = I0 * (1 + gamma * sin(delta_phi_x));
    Iy_ideal = I0 * (1 + gamma * sin(delta_phi_y));

    % 泊松散粒噪声（光子计数）
    Ix_poisson = poissrnd(Ix_ideal * full_well) / full_well;
    Iy_poisson = poissrnd(Iy_ideal * full_well) / full_well;

    % 读出噪声（高斯）
    Ix_final = Ix_poisson + noise_gaussian_std * randn(size(Ix_poisson));
    Iy_final = Iy_poisson + noise_gaussian_std * randn(size(Iy_poisson));

    % [gx,gy] = gradient(ap_x,1);


    %% ---------- F. 输出定义 ----------
    
    input_data = cat(3, single(Ix_final), single(Iy_final));
    
    % 标签数据：将 gx 和 gy 堆叠成 (H, W, 2)
    label_data = cat(3, single(gx), single(gy));
    
    % 绝对相位真值
     ap_truth = single(phase_pre);
     ap_512 = imresize(single(ap_truth), [512,512], 'bilinear');
     ap_truth_mod = mod(ap_512, 2*pi);

     gray_img = ap_truth_mod / (2*pi);

     %imshow(Ix_final,[]);
   

    %% ---------- G. 统一保存 ----------
    % 每个样本只存一个文件，包含所有必要信息
    % 格式：sample_0001.mat, sample_0002.mat ...
    save_name = fullfile(path1, sprintf('I_%d.mat', jj));
        save(save_name, 'input_data');
    % save_name2 = fullfile(path2, sprintf('O_%d.mat', jj));
    %     save(save_name2, 'label_data');
    % save_name3 = fullfile(path5, sprintf('P_%d.mat', jj)); % 存取未折叠相位
    %     save(save_name3, 'ap_truth');
    % save_name4 = fullfile(path6, sprintf('slm_%d.png', jj)); %加载到slm,折叠相位
    % imwrite(gray_img, save_name4);
    % 
   
    % figure(1); clf;
    % subplot(1,2,1);
    % imagesc(delta_phi_y); axis image; colorbar;
    % title('绝对相位 \phi(x)');
    % 
    % subplot(1,2,2);
    % imagesc(gy); axis image; colorbar; colormap gray;
    % title('剪切干涉强度 I(x)');
    % pause(1);
   
end
save('h.mat',"h_vals_clipped",'-mat')
disp('数据生成完成。');
