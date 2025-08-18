% 3D Enhanced Model Architecture Visualization
% IEEE IoTJ Publication Quality 3D Figures

close all; clear; clc;

%% Figure 5: Enhanced Model 3D Architecture
fprintf('🧠 Generating 3D Enhanced Model Architecture...\n');

figure(1);
set(gcf, 'Position', [100, 100, 900, 700]);

% Define architecture layers with 3D coordinates
% Format: [x, y, z, width, height, depth]
layers = struct();

% Input layer
layers.input = [1, 2, 1, 2, 1, 0.5];
% CNN layers
layers.cnn1 = [4, 2, 1, 1.5, 1.5, 0.8];
layers.cnn2 = [6, 2, 1, 1.5, 1.5, 0.8];
layers.cnn3 = [8, 2, 1, 1.5, 1.5, 0.8];
% SE Module
layers.se = [10, 2, 1.5, 1, 2, 0.6];
% Temporal Attention
layers.attention = [12, 1.5, 2, 1.5, 1, 1];
% Output layer
layers.output = [15, 2, 1, 1, 1, 0.5];

% Color scheme
colors = struct();
colors.input = [0.7, 0.9, 0.7];      % Light green
colors.cnn = [0.2, 0.5, 0.8];        % Blue
colors.se = [0.9, 0.6, 0.2];         % Orange  
colors.attention = [0.8, 0.2, 0.5];  % Purple
colors.output = [0.9, 0.7, 0.7];     % Light red

% Draw 3D blocks
hold on;

% Input block
draw_3d_block(layers.input, colors.input);
text(layers.input(1), layers.input(2), layers.input(3)+1, 'CSI Input\n(T×F×N)', ...
    'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');

% CNN blocks
cnn_layers = {layers.cnn1, layers.cnn2, layers.cnn3};
for i = 1:length(cnn_layers)
    draw_3d_block(cnn_layers{i}, colors.cnn);
    text(cnn_layers{i}(1), cnn_layers{i}(2), cnn_layers{i}(3)+1, ...
        sprintf('CNN-%d', i), 'HorizontalAlignment', 'center', ...
        'FontSize', 8, 'Color', 'white', 'FontWeight', 'bold');
end

% SE Module
draw_3d_block(layers.se, colors.se);
text(layers.se(1), layers.se(2), layers.se(3)+1.2, 'SE Module\n(Squeeze &\nExcitation)', ...
    'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');

% Temporal Attention
draw_3d_block(layers.attention, colors.attention);
text(layers.attention(1), layers.attention(2), layers.attention(3)+1.5, ...
    'Temporal\nAttention', 'HorizontalAlignment', 'center', ...
    'FontSize', 9, 'Color', 'white', 'FontWeight', 'bold');

% Output
draw_3d_block(layers.output, colors.output);
text(layers.output(1), layers.output(2), layers.output(3)+1, 'Activity\nClassification', ...
    'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');

% Draw connections
draw_3d_connection(layers.input, layers.cnn1, [0.5, 0.5, 0.5]);
draw_3d_connection(layers.cnn1, layers.cnn2, [0.5, 0.5, 0.5]);
draw_3d_connection(layers.cnn2, layers.cnn3, [0.5, 0.5, 0.5]);
draw_3d_connection(layers.cnn3, layers.se, [0.8, 0.4, 0.2]);
draw_3d_connection(layers.se, layers.attention, [0.8, 0.2, 0.5]);
draw_3d_connection(layers.attention, layers.output, [0.5, 0.5, 0.5]);

% Formatting
title('Enhanced Model 3D Architecture', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Processing Flow');
ylabel('Feature Dimension');
zlabel('Abstraction Level');

% Set optimal view
view(-30, 25);
grid on;
axis equal;
xlim([0, 17]);
ylim([0, 4]);
zlim([0, 4]);

% Add legend
legend_x = 0.5;
legend_y = 3.5;
text(legend_x, legend_y, 3, '🧠 Enhanced Architecture Components:', ...
    'FontSize', 10, 'FontWeight', 'bold');
text(legend_x, legend_y-0.3, 2.7, '• CNN: Feature Extraction', 'FontSize', 8);
text(legend_x, legend_y-0.6, 2.4, '• SE: Channel Attention', 'FontSize', 8);
text(legend_x, legend_y-0.9, 2.1, '• Attention: Temporal Modeling', 'FontSize', 8);

% Save
print('figure5_enhanced_3d_architecture.pdf', '-dpdf', '-r300');
fprintf('✓ 3D Architecture saved: figure5_enhanced_3d_architecture.pdf\n');

%% Figure 6: Physics-Guided Generation Framework 3D
fprintf('🏗️ Generating 3D Physics-Guided Framework...\n');

figure(2);
set(gcf, 'Position', [200, 200, 1000, 700]);

% Physics components in 3D space
physics_components = struct();
physics_components.multipath = [2, 1, 3, 1.5, 1, 0.8];
physics_components.human = [2, 3, 3, 1.5, 1, 0.8];
physics_components.environment = [2, 5, 3, 1.5, 1, 0.8];

% Signal processing pipeline
pipeline = struct();
pipeline.channel_model = [5, 3, 3, 1.5, 2, 0.6];
pipeline.noise_model = [5, 3, 1.5, 1.5, 2, 0.6];
pipeline.synthesis = [8, 3, 2.5, 2, 1.5, 1];

% Real data integration
real_data = struct();
real_data.benchmark = [12, 1, 3, 1.5, 1, 0.8];
real_data.finetuning = [12, 3, 2, 1.5, 1.5, 1];
real_data.validation = [12, 5, 3, 1.5, 1, 0.8];

% Output
output = struct();
output.trained_model = [15, 3, 2.5, 2, 1.5, 1];

hold on;

% Draw physics components
draw_3d_block(physics_components.multipath, [0.6, 0.8, 1]);
text(physics_components.multipath(1), physics_components.multipath(2), ...
    physics_components.multipath(3)+1, 'Multipath\nModeling', ...
    'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');

draw_3d_block(physics_components.human, [0.8, 0.6, 1]);
text(physics_components.human(1), physics_components.human(2), ...
    physics_components.human(3)+1, 'Human Body\nInteraction', ...
    'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');

draw_3d_block(physics_components.environment, [1, 0.8, 0.6]);
text(physics_components.environment(1), physics_components.environment(2), ...
    physics_components.environment(3)+1, 'Environment\nVariation', ...
    'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');

% Draw synthesis pipeline
draw_3d_block(pipeline.channel_model, [0.2, 0.7, 0.9]);
text(pipeline.channel_model(1), pipeline.channel_model(2), ...
    pipeline.channel_model(3)+1, 'Channel\nModel', ...
    'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');

draw_3d_block(pipeline.synthesis, [0.9, 0.4, 0.2]);
text(pipeline.synthesis(1), pipeline.synthesis(2), ...
    pipeline.synthesis(3)+1.2, 'Synthetic\nCSI Generation', ...
    'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold', ...
    'Color', 'white');

% Draw real data components
draw_3d_block(real_data.benchmark, [0.7, 0.9, 0.7]);
text(real_data.benchmark(1), real_data.benchmark(2), ...
    real_data.benchmark(3)+1, 'Real\nBenchmarks', ...
    'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');

draw_3d_block(real_data.finetuning, [0.2, 0.8, 0.6]);
text(real_data.finetuning(1), real_data.finetuning(2), ...
    real_data.finetuning(3)+1.2, 'STEA\nFine-tuning', ...
    'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold', ...
    'Color', 'white');

% Draw output
draw_3d_block(output.trained_model, [0.8, 0.2, 0.5]);
text(output.trained_model(1), output.trained_model(2), ...
    output.trained_model(3)+1.2, 'Enhanced\nModel', ...
    'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold', ...
    'Color', 'white');

% Draw 3D connections
draw_3d_connection(physics_components.multipath, pipeline.channel_model, [0.6, 0.6, 0.6]);
draw_3d_connection(physics_components.human, pipeline.channel_model, [0.6, 0.6, 0.6]);
draw_3d_connection(physics_components.environment, pipeline.channel_model, [0.6, 0.6, 0.6]);
draw_3d_connection(pipeline.channel_model, pipeline.synthesis, [0.8, 0.4, 0.2]);
draw_3d_connection(pipeline.synthesis, real_data.finetuning, [0.2, 0.8, 0.6]);
draw_3d_connection(real_data.benchmark, real_data.finetuning, [0.7, 0.7, 0.7]);
draw_3d_connection(real_data.finetuning, output.trained_model, [0.8, 0.2, 0.5]);

% Title and formatting
title('Physics-Guided Sim2Real Framework (3D)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Synthetic Domain');
ylabel('Process Pipeline');
zlabel('Real Domain');

% Optimal 3D view
view(-45, 35);
grid on;
axis equal;
xlim([0, 18]);
ylim([0, 7]);
zlim([0, 5]);

% Add process flow annotations
text(8, 6, 4, '📊 STEA Protocol: 82.1% F1 @ 20% Labels', ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', [1, 0.4, 0.4]);
text(8, 0.5, 4, '🎯 Physics → Synthesis → Transfer → Deployment', ...
    'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.2, 0.5, 0.8]);

% Save
print('figure6_physics_guided_3d_framework.pdf', '-dpdf', '-r300');
fprintf('✓ 3D Framework saved: figure6_physics_guided_3d_framework.pdf\n');

%% Summary
fprintf('\n🎨 3D Figure Generation Complete!\n');
fprintf('📊 Generated 3D visualizations:\n');
fprintf('  • figure5_enhanced_3d_architecture.pdf - Enhanced model 3D structure\n');
fprintf('  • figure6_physics_guided_3d_framework.pdf - Complete framework 3D view\n');

fprintf('\n💡 3D Visualization Advantages:\n');
fprintf('• Multi-dimensional data representation\n');
fprintf('• Clear architectural component relationships\n');
fprintf('• Professional IEEE journal quality\n');
fprintf('• Enhanced visual impact for complex systems\n');

%% Helper Functions

function draw_3d_block(position, color)
    % Draw a 3D rectangular block
    % position = [x, y, z, width, height, depth]
    
    x = position(1);
    y = position(2);
    z = position(3);
    w = position(4);
    h = position(5);
    d = position(6);
    
    % Define vertices
    vertices = [
        x, y, z;           % 1
        x+w, y, z;         % 2
        x+w, y+h, z;       % 3
        x, y+h, z;         % 4
        x, y, z+d;         % 5
        x+w, y, z+d;       % 6
        x+w, y+h, z+d;     % 7
        x, y+h, z+d        % 8
    ];
    
    % Define faces
    faces = [
        1, 2, 3, 4;  % Bottom
        5, 6, 7, 8;  % Top
        1, 2, 6, 5;  % Front
        3, 4, 8, 7;  % Back
        1, 4, 8, 5;  % Left
        2, 3, 7, 6   % Right
    ];
    
    % Draw the block
    patch('Vertices', vertices, 'Faces', faces, 'FaceColor', color, ...
        'EdgeColor', 'k', 'LineWidth', 0.5, 'FaceAlpha', 0.8);
end

function draw_3d_connection(block1, block2, color)
    % Draw connection between two 3D blocks
    
    % Calculate connection points (center of blocks)
    x1 = block1(1) + block1(4)/2;
    y1 = block1(2) + block1(5)/2;
    z1 = block1(3) + block1(6)/2;
    
    x2 = block2(1) + block2(4)/2;
    y2 = block2(2) + block2(5)/2;
    z2 = block2(3) + block2(6)/2;
    
    % Draw arrow
    plot3([x1, x2], [y1, y2], [z1, z2], '-', 'Color', color, 'LineWidth', 2);
    
    % Add arrowhead (simplified)
    arrow_length = 0.3;
    direction = [x2-x1, y2-y1, z2-z1];
    direction = direction / norm(direction);
    
    arrow_end = [x2, y2, z2];
    arrow_start = arrow_end - arrow_length * direction;
    
    plot3([arrow_start(1), arrow_end(1)], [arrow_start(2), arrow_end(2)], ...
        [arrow_start(3), arrow_end(3)], '-', 'Color', color, 'LineWidth', 3);
end