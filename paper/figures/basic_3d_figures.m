% Basic 3D Figures using core Octave functions
% IEEE IoTJ Compatible 3D Architecture Visualization

close all; clear; clc;

%% Figure 5: Enhanced Model 3D Architecture (Using basic 3D functions)
fprintf('🧠 Generating 3D Enhanced Architecture (Basic)...\n');

figure(1);
clf;

% Define architecture flow in 3D space
x_positions = [1, 3, 5, 7, 9, 11, 13];  % X: processing flow
y_position = 2;                          % Y: fixed
z_heights = [0.5, 1, 1.5, 2, 2.5, 3, 1]; % Z: abstraction levels

component_names = {'Input', 'CNN-1', 'CNN-2', 'CNN-3', 'SE', 'Attention', 'Output'};
colors = {[0.7,0.9,0.7], [0.2,0.5,0.8], [0.2,0.5,0.8], [0.2,0.5,0.8], ...
          [0.9,0.6,0.2], [0.8,0.2,0.5], [0.9,0.7,0.7]};

hold on;

% Draw 3D architecture using cylinders and connections
for i = 1:length(x_positions)
    x = x_positions(i);
    z = z_heights(i);
    
    % Create cylinder for each component
    [X, Y, Z] = cylinder([0.4, 0.4], 20);
    Z = Z * z;  % Scale height
    X = X + x;
    Y = Y + y_position;
    
    % Draw cylinder
    surf(X, Y, Z, 'FaceColor', colors{i}, 'EdgeColor', 'k', 'LineWidth', 0.5);
    
    % Add component labels
    text(x, y_position-0.8, z+0.2, component_names{i}, ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
    
    % Draw connections
    if i < length(x_positions)
        next_x = x_positions(i+1);
        next_z = z_heights(i+1);
        
        % Connection line
        plot3([x+0.4, next_x-0.4], [y_position, y_position], [z/2, next_z/2], ...
            'r-', 'LineWidth', 2);
        
        % Arrow head (simplified)
        plot3([next_x-0.4, next_x-0.2, next_x-0.4], ...
              [y_position, y_position+0.1, y_position-0.1], ...
              [next_z/2, next_z/2, next_z/2], 'r-', 'LineWidth', 2);
    end
end

% Highlight key innovations (SE and Attention)
se_x = x_positions(5);
att_x = x_positions(6);

% SE highlight
plot3([se_x-0.6, se_x+0.6, se_x+0.6, se_x-0.6, se_x-0.6], ...
      [y_position-0.6, y_position-0.6, y_position+0.6, y_position+0.6, y_position-0.6], ...
      [z_heights(5)+0.3, z_heights(5)+0.3, z_heights(5)+0.3, z_heights(5)+0.3, z_heights(5)+0.3], ...
      'color', [1, 0.84, 0], 'LineWidth', 2, 'LineStyle', '--');

% Attention highlight  
plot3([att_x-0.6, att_x+0.6, att_x+0.6, att_x-0.6, att_x-0.6], ...
      [y_position-0.6, y_position-0.6, y_position+0.6, y_position+0.6, y_position-0.6], ...
      [z_heights(6)+0.3, z_heights(6)+0.3, z_heights(6)+0.3, z_heights(6)+0.3, z_heights(6)+0.3], ...
      'color', [1, 0.84, 0], 'LineWidth', 2, 'LineStyle', '--');

% Architecture performance annotation
text(7, y_position+1.2, 3.5, '🎯 CDAE Results: 83.0±0.1% F1', ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.2, 0.5, 0.8]);
text(7, y_position+0.9, 3.2, 'Perfect Cross-Domain Consistency', ...
    'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.2, 0.5, 0.8]);

% Formatting
title('Enhanced Model 3D Architecture', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Processing Flow →');
ylabel('Feature Dimension');
zlabel('Abstraction Level ↑');

% Optimal 3D view
view([-35, 25]);
grid on;
axis equal;
xlim([0, 15]);
ylim([1, 4]);
zlim([0, 4]);

% Save
print('figure5_enhanced_3d_arch_basic.pdf', '-dpdf', '-r300');

%% Figure 6: Physics Framework 3D Workflow
fprintf('🏗️ Generating 3D Framework Workflow...\n');

figure(3);
clf;

% Define 3D workflow stages
stages = struct();
stages.physics = [2, 2, 3];      % Physics modeling
stages.synthesis = [6, 2, 2.5];  % Synthetic generation  
stages.transfer = [10, 2, 2];    % Transfer learning
stages.deployment = [14, 2, 1.5]; % Final deployment

% Component details for each stage
details = struct();
details.physics = [2, 1; 2, 3; 2, 5];     % Multi-component physics
details.transfer = [10, 1; 10, 2; 10, 3]; % Multi-method transfer

hold on;

% Draw main workflow stages
stage_colors = {[0.6,0.8,1], [0.9,0.4,0.2], [0.2,0.8,0.6], [0.8,0.2,0.5]};
stage_names = {'Physics\nModeling', 'Synthetic\nGeneration', 'STEA\nTransfer', 'Enhanced\nModel'};
stage_positions = {stages.physics, stages.synthesis, stages.transfer, stages.deployment};

for i = 1:length(stage_positions)
    pos = stage_positions{i};
    
    % Create 3D block using patch
    x = pos(1); y = pos(2); z = pos(3);
    w = 1.5; h = 1; d = 0.8;
    
    % Define vertices for a 3D box
    vertices = [
        x-w/2, y-h/2, z-d/2;  % 1
        x+w/2, y-h/2, z-d/2;  % 2
        x+w/2, y+h/2, z-d/2;  % 3
        x-w/2, y+h/2, z-d/2;  % 4
        x-w/2, y-h/2, z+d/2;  % 5
        x+w/2, y-h/2, z+d/2;  % 6
        x+w/2, y+h/2, z+d/2;  % 7
        x-w/2, y+h/2, z+d/2   % 8
    ];
    
    % Define faces
    faces = [1 2 3 4; 5 6 7 8; 1 2 6 5; 3 4 8 7; 1 4 8 5; 2 3 7 6];
    
    % Draw the 3D block
    patch('Vertices', vertices, 'Faces', faces, 'FaceColor', stage_colors{i}, ...
        'EdgeColor', 'k', 'LineWidth', 0.5, 'FaceAlpha', 0.8);
    
    % Add stage labels
    text(x, y, z+d/2+0.3, stage_names{i}, 'HorizontalAlignment', 'center', ...
        'FontSize', 10, 'FontWeight', 'bold');
    
    % Draw connections
    if i < length(stage_positions)
        next_pos = stage_positions{i+1};
        plot3([x+w/2, next_pos(1)-w/2], [y, next_pos(2)], [z, next_pos(3)], ...
            'k-', 'LineWidth', 2);
    end
end

% Add STEA achievement annotation
text(10, 4, 3, '⭐ STEA Achievement:', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [1, 0.4, 0.4]);
text(10, 3.7, 2.7, '82.1% F1 @ 20% Labels', 'FontSize', 10, 'FontWeight', 'bold', 'Color', [1, 0.4, 0.4]);
text(10, 3.4, 2.4, '80% Cost Reduction', 'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.2, 0.6, 0.2]);

% Formatting
title('Physics-Guided Sim2Real Framework (3D)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Sim2Real Pipeline →');
ylabel('Process Components');
zlabel('Abstraction Level ↑');

% Set 3D view
view([-40, 30]);
grid on;
xlim([0, 16]);
ylim([0, 5]);
zlim([0, 4]);

% Save
print('figure6_physics_3d_framework_basic.pdf', '-dpdf', '-r300');

%% Summary
fprintf('\n🎉 3D Figure Generation Complete!\n');
fprintf('📊 Generated 3D visualizations:\n');
fprintf('  • figure5_enhanced_3d_arch_basic.pdf - Enhanced model 3D architecture\n');
fprintf('  • figure6_physics_3d_framework_basic.pdf - Physics framework 3D workflow\n');

fprintf('\n💡 3D Visualization Benefits:\n');
fprintf('• Enhanced visual impact for complex architectures\n');
fprintf('• Clear component relationship demonstration\n');
fprintf('• Professional IEEE journal presentation\n');
fprintf('• Improved reader comprehension of system design\n');