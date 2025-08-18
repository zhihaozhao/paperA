% Simplified 3D Architecture Visualization for Enhanced Model
% IEEE IoTJ Compatible - Fixed version for Octave

close all; clear; clc;

%% Figure 5: Enhanced Model 3D Architecture
fprintf('🧠 Generating Simplified 3D Enhanced Architecture...\n');

figure(1);
set(gcf, 'Position', [100, 100, 800, 600]);

% Enhanced Model Architecture Components (3D visualization)
% X: Processing flow, Y: Feature dimension, Z: Abstraction level

% Component positions [x, y, z] and sizes
components = struct();
components.input = [1, 2, 0.5];
components.cnn1 = [3, 2, 1];
components.cnn2 = [5, 2, 1.5];  
components.cnn3 = [7, 2, 2];
components.se = [9, 2, 2.5];
components.attention = [11, 2, 3];
components.output = [13, 2, 1];

% Colors for different components
colors = [
    0.7, 0.9, 0.7;   % Input - Light green
    0.2, 0.5, 0.8;   % CNN - Blue
    0.2, 0.5, 0.8;   % CNN - Blue
    0.2, 0.5, 0.8;   % CNN - Blue
    0.9, 0.6, 0.2;   % SE - Orange
    0.8, 0.2, 0.5;   % Attention - Purple
    0.9, 0.7, 0.7    % Output - Light red
];

% Component names
names = {'CSI Input\n(T×F×N)', 'CNN-1', 'CNN-2', 'CNN-3', 'SE Module', 'Temporal\nAttention', 'Classification'};

% Plot 3D bars/blocks for each component (using simple bar3 approach)
comp_list = {components.input, components.cnn1, components.cnn2, components.cnn3, ...
            components.se, components.attention, components.output};

hold on;

% Create simple 3D representation using plot3 and fill3
for i = 1:length(comp_list)
    pos = comp_list{i};
    
    % Define block corners
    x_size = 0.6;
    y_size = 0.6;
    z_size = 0.4;
    
    % Create simple rectangular block using fill3
    x = pos(1) + [-x_size/2, x_size/2, x_size/2, -x_size/2];
    y = pos(2) + [-y_size/2, -y_size/2, y_size/2, y_size/2];
    z_base = pos(3);
    z_top = pos(3) + z_size;
    
    % Draw top face
    fill3(x, y, [z_top, z_top, z_top, z_top], colors(i,:), 'EdgeColor', 'k');
    % Draw front face
    fill3([x(1), x(2), x(2), x(1)], [y(1), y(2), y(2), y(1)], [z_base, z_base, z_top, z_top], colors(i,:), 'EdgeColor', 'k');
    % Draw right face
    fill3([x(2), x(2), x(2), x(2)], [y(2), y(3), y(3), y(2)], [z_base, z_base, z_top, z_top], colors(i,:)*0.8, 'EdgeColor', 'k');
    
    % Add component labels (black color)
    text(pos(1), pos(2)-0.5, pos(3)+0.5, names{i}, ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
    
    % Draw connections (simple lines)
    if i < length(comp_list)
        next_pos = comp_list{i+1};
        plot3([pos(1), next_pos(1)], [pos(2), next_pos(2)], [pos(3), next_pos(3)], ...
            'r-', 'LineWidth', 2);
    end
end

% Enhanced model highlight
% Highlight the SE and Attention components (key innovations)
se_pos = components.se;
att_pos = components.attention;

% Draw highlight box around SE module
plot3([se_pos(1)-0.5, se_pos(1)+0.5, se_pos(1)+0.5, se_pos(1)-0.5, se_pos(1)-0.5], ...
      [se_pos(2)-0.5, se_pos(2)-0.5, se_pos(2)+0.5, se_pos(2)+0.5, se_pos(2)-0.5], ...
      [se_pos(3)+0.5, se_pos(3)+0.5, se_pos(3)+0.5, se_pos(3)+0.5, se_pos(3)+0.5], ...
      'color', [1, 0.84, 0], 'LineWidth', 2);

% Draw highlight box around Attention module  
plot3([att_pos(1)-0.5, att_pos(1)+0.5, att_pos(1)+0.5, att_pos(1)-0.5, att_pos(1)-0.5], ...
      [att_pos(2)-0.5, att_pos(2)-0.5, att_pos(2)+0.5, att_pos(2)+0.5, att_pos(2)-0.5], ...
      [att_pos(3)+0.5, att_pos(3)+0.5, att_pos(3)+0.5, att_pos(3)+0.5, att_pos(3)+0.5], ...
      'color', [1, 0.84, 0], 'LineWidth', 2);

% Formatting
title('Enhanced Model 3D Architecture', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Processing Flow →');
ylabel('Feature Space');
zlabel('Abstraction Level ↑');

% Set 3D view
view([-45, 25]);
grid on;
xlim([0, 15]);
ylim([1, 3]);
zlim([0, 4]);

% Add architecture advantages text (changed from white to dark colors)
text(7, 3, 3.5, '⭐ Key Innovations:', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.8, 0.4, 0]);
text(7, 2.8, 3.2, '• SE: Channel Attention', 'FontSize', 9, 'Color', [0.6, 0.3, 0]);
text(7, 2.6, 2.9, '• Temporal: Long-range Modeling', 'FontSize', 9, 'Color', [0.5, 0.1, 0.3]);
text(7, 2.4, 2.6, '• 83.0% Cross-Domain Consistency', 'FontSize', 9, 'Color', [0.1, 0.3, 0.6]);

% Save
print('figure5_enhanced_3d_architecture_fixed.pdf', '-dpdf', '-r300');
fprintf('✓ 3D Enhanced Architecture saved: figure5_enhanced_3d_architecture_fixed.pdf\n');

%% Figure 6: Physics-Guided Framework 3D Flow
fprintf('🏗️ Generating 3D Physics-Guided Framework...\n');

figure(2);
set(gcf, 'Position', [200, 200, 900, 700]);

% 3D Framework visualization
% Define 3D stages: Physics → Synthesis → Transfer → Deployment

% Stage 1: Physics Modeling (Z=3 level)
physics_x = [1, 2, 3];
physics_y = [1, 2, 3]; 
physics_z = [3, 3, 3];
physics_labels = {'Multipath', 'Human Body', 'Environment'};

% Stage 2: Synthesis (Z=2 level)
synth_x = [6];
synth_y = [2];
synth_z = [2];

% Stage 3: Transfer Learning (Z=1.5 level)
transfer_x = [9, 10, 11];
transfer_y = [1, 2, 3];
transfer_z = [1.5, 1.5, 1.5];
transfer_labels = {'Zero-shot', 'Fine-tune', 'Calibration'};

% Stage 4: Deployment (Z=1 level)
deploy_x = [14];
deploy_y = [2];
deploy_z = [1];

hold on;

% Physics modeling components
for i = 1:3
    % Create simple blocks using fill3
    x_center = physics_x(i);
    y_center = physics_y(i);
    z_center = physics_z(i);
    
    x_size = 0.5;
    y_size = 0.5;
    z_size = 0.3;
    
    x = x_center + [-x_size/2, x_size/2, x_size/2, -x_size/2];
    y = y_center + [-y_size/2, -y_size/2, y_size/2, y_size/2];
    z_base = z_center - z_size/2;
    z_top = z_center + z_size/2;
    
    % Draw physics components in light blue
    fill3(x, y, [z_top, z_top, z_top, z_top], [0.6, 0.8, 1], 'EdgeColor', 'k');
    fill3([x(1), x(2), x(2), x(1)], [y(1), y(2), y(2), y(1)], [z_base, z_base, z_top, z_top], [0.6, 0.8, 1], 'EdgeColor', 'k');
    
    text(physics_x(i), physics_y(i)-0.3, physics_z(i)+0.2, physics_labels{i}, ...
        'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold', 'Color', 'k');
end

% Synthesis stage
x = synth_x + [-0.7, 0.7, 0.7, -0.7];
y = synth_y + [-0.7, -0.7, 0.7, 0.7];
z_base = synth_z - 0.2;
z_top = synth_z + 0.2;

fill3(x, y, [z_top, z_top, z_top, z_top], [0.9, 0.4, 0.2], 'EdgeColor', 'k');
fill3([x(1), x(2), x(2), x(1)], [y(1), y(2), y(2), y(1)], [z_base, z_base, z_top, z_top], [0.9, 0.4, 0.2], 'EdgeColor', 'k');

text(synth_x, synth_y-0.3, synth_z+0.3, 'Synthetic\nCSI Generation', ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');

% Transfer learning stages
for i = 1:3
    x_center = transfer_x(i);
    y_center = transfer_y(i);
    z_center = transfer_z(i);
    
    x_size = 0.5;
    y_size = 0.5;
    z_size = 0.3;
    
    x = x_center + [-x_size/2, x_size/2, x_size/2, -x_size/2];
    y = y_center + [-y_size/2, -y_size/2, y_size/2, y_size/2];
    z_base = z_center - z_size/2;
    z_top = z_center + z_size/2;
    
    % Draw transfer components in green
    fill3(x, y, [z_top, z_top, z_top, z_top], [0.2, 0.8, 0.6], 'EdgeColor', 'k');
    fill3([x(1), x(2), x(2), x(1)], [y(1), y(2), y(2), y(1)], [z_base, z_base, z_top, z_top], [0.2, 0.8, 0.6], 'EdgeColor', 'k');
    
    text(transfer_x(i), transfer_y(i)-0.3, transfer_z(i)+0.2, transfer_labels{i}, ...
        'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold', 'Color', 'k');
end

% Deployment stage
x = deploy_x + [-0.7, 0.7, 0.7, -0.7];
y = deploy_y + [-0.7, -0.7, 0.7, 0.7];
z_base = deploy_z - 0.2;
z_top = deploy_z + 0.2;

fill3(x, y, [z_top, z_top, z_top, z_top], [0.8, 0.2, 0.5], 'EdgeColor', 'k');
fill3([x(1), x(2), x(2), x(1)], [y(1), y(2), y(2), y(1)], [z_base, z_base, z_top, z_top], [0.8, 0.2, 0.5], 'EdgeColor', 'k');

text(deploy_x, deploy_y-0.3, deploy_z+0.3, 'Enhanced\nModel', ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');

% Draw flow connections
% Physics to Synthesis
for i = 1:3
    plot3([physics_x(i), synth_x], [physics_y(i), synth_y], [physics_z(i), synth_z], ...
        'k-', 'LineWidth', 1.5);
end

% Synthesis to Transfer
for i = 1:3
    plot3([synth_x, transfer_x(i)], [synth_y, transfer_y(i)], [synth_z, transfer_z(i)], ...
        'k-', 'LineWidth', 1.5);
end

% Transfer to Deployment
plot3([transfer_x(2), deploy_x], [transfer_y(2), deploy_y], [transfer_z(2), deploy_z], ...
    'r-', 'LineWidth', 2);

% Formatting
title('Physics-Guided Sim2Real Framework (3D Flow)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Process Timeline →');
ylabel('Component Type');
zlabel('Abstraction Level ↑');

% Set optimal 3D view
view([-35, 30]);
grid on;
xlim([0, 16]);
ylim([0, 4]);
zlim([0, 4]);

% Add key results annotations (changed from white to dark colors)
text(8, 3.5, 3.5, '🎯 STEA Results:', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.8, 0.2, 0.2]);
text(8, 3.2, 3.2, '82.1% F1 @ 20% Labels', 'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.6, 0.1, 0.1]);
text(8, 2.9, 2.9, '80% Cost Reduction', 'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.1, 0.4, 0.1]);

% Save
print('figure6_physics_3d_framework_fixed.pdf', '-dpdf', '-r300');
fprintf('✓ 3D Framework saved: figure6_physics_3d_framework_fixed.pdf\n');

%% Display Summary
fprintf('\n🎨 3D Figure Generation Summary:\n');
fprintf('✓ Enhanced model 3D architecture visualization\n');
fprintf('✓ Physics-guided framework 3D flow diagram\n');
fprintf('✓ IEEE IoTJ compliance: 300 DPI PDF export\n');
fprintf('✓ Professional 3D visualization with optimal viewing angles\n');

fprintf('\n📊 Architecture Highlights Visualized:\n');
fprintf('• Enhanced model: CNN + SE + Attention 3D structure\n');
fprintf('• Physics framework: Complete Sim2Real 3D pipeline\n');
fprintf('• Key innovations: SE and Attention components highlighted\n');
fprintf('• Results integration: STEA achievements prominently displayed\n');

fprintf('\n🚀 3D figures ready for Methods section inclusion!\n');