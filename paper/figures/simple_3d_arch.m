% Simplified 3D Architecture Visualization for Enhanced Model
% IEEE IoTJ Compatible

close all; clear; clc;

%% Figure 5: Enhanced Model 3D Architecture
fprintf('🧠 Generating Simplified 3D Enhanced Architecture...\n');

figure(1);
set(gcf, 'Position', [100, 100, 800, 600]);

% Enhanced Model Architecture Components (3D visualization)
% X: Processing flow, Y: Feature dimension, Z: Abstraction level

% Component positions [x, y, z]
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

% Plot 3D bars/blocks for each component
comp_list = {components.input, components.cnn1, components.cnn2, components.cnn3, ...
            components.se, components.attention, components.output};

hold on;

for i = 1:length(comp_list)
    pos = comp_list{i};
    
    % Create 3D bar using bar3
    bar3(pos(1), pos(3), 'FaceColor', colors(i,:), 'EdgeColor', 'k', ...
        'LineWidth', 1, 'BarWidth', 0.8);
    
    % Add component labels
    text(pos(1), pos(2)-0.5, pos(3)+0.3, names{i}, ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
    
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

% Add architecture advantages text
text(7, 3, 3.5, '⭐ Key Innovations:', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [1, 0.6, 0]);
text(7, 2.8, 3.2, '• SE: Channel Attention', 'FontSize', 9, 'Color', [0.9, 0.6, 0.2]);
text(7, 2.6, 2.9, '• Temporal: Long-range Modeling', 'FontSize', 9, 'Color', [0.8, 0.2, 0.5]);
text(7, 2.4, 2.6, '• 83.0% Cross-Domain Consistency', 'FontSize', 9, 'Color', [0.2, 0.5, 0.8]);

% Save
print('figure5_enhanced_3d_architecture.pdf', '-dpdf', '-r300');
fprintf('✓ 3D Enhanced Architecture saved: figure5_enhanced_3d_architecture.pdf\n');

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
    bar3(physics_x(i), physics_z(i), 'FaceColor', [0.6, 0.8, 1], 'BarWidth', 0.6);
    text(physics_x(i), physics_y(i)-0.3, physics_z(i)+0.2, physics_labels{i}, ...
        'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
end

% Synthesis stage
bar3(synth_x, synth_z, 'FaceColor', [0.9, 0.4, 0.2], 'BarWidth', 1);
text(synth_x, synth_y-0.3, synth_z+0.3, 'Synthetic\nCSI Generation', ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'white');

% Transfer learning stages
for i = 1:3
    bar3(transfer_x(i), transfer_z(i), 'FaceColor', [0.2, 0.8, 0.6], 'BarWidth', 0.6);
    text(transfer_x(i), transfer_y(i)-0.3, transfer_z(i)+0.2, transfer_labels{i}, ...
        'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
end

% Deployment stage
bar3(deploy_x, deploy_z, 'FaceColor', [0.8, 0.2, 0.5], 'BarWidth', 1);
text(deploy_x, deploy_y-0.3, deploy_z+0.3, 'Enhanced\nModel', ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'white');

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

% Add key results annotations
text(8, 3.5, 3.5, '🎯 STEA Results:', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [1, 0.4, 0.4]);
text(8, 3.2, 3.2, '82.1% F1 @ 20% Labels', 'FontSize', 10, 'FontWeight', 'bold', 'Color', [1, 0.4, 0.4]);
text(8, 2.9, 2.9, '80% Cost Reduction', 'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.2, 0.6, 0.2]);

% Save
print('figure6_physics_3d_framework.pdf', '-dpdf', '-r300');
fprintf('✓ 3D Framework saved: figure6_physics_3d_framework.pdf\n');

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