% Enhanced Model Architecture - Basic Compatible Version
% Inspired by reference papers, fully Octave compatible
% IEEE IoTJ Quality

close all; clear; clc;

fprintf('🏗️ Creating Enhanced Model Architecture Diagram...\n');

% Create figure
figure('Position', [100, 100, 1400, 800]);

% === Main Architecture Visualization ===
hold on;

% Component data
components = {
    struct('pos', [1, 3], 'size', [1.5, 1], 'color', [0.9, 0.9, 1.0], 'label', 'WiFi CSI\nInput\n(T×F×N)'),
    struct('pos', [3.5, 3], 'size', [1.2, 1], 'color', [1.0, 0.9, 0.7], 'label', 'Conv1D\n32 filters'),
    struct('pos', [5.2, 3], 'size', [1.2, 1], 'color', [1.0, 0.9, 0.7], 'label', 'Conv1D\n64 filters'),
    struct('pos', [6.9, 3], 'size', [1.2, 1], 'color', [1.0, 0.9, 0.7], 'label', 'Conv1D\n128 filters'),
    struct('pos', [8.8, 2.5], 'size', [1.8, 2], 'color', [1.0, 0.84, 0.0], 'label', 'SE Module\n(Channel Attention)'),
    struct('pos', [11.5, 3], 'size', [1.8, 1], 'color', [0.6, 0.98, 0.6], 'label', 'BiLSTM\n(Temporal)'),
    struct('pos', [14, 2.5], 'size', [1.8, 2], 'color', [0.87, 0.63, 0.87], 'label', 'Temporal Attention\n(Long-range)'),
    struct('pos', [16.5, 3], 'size', [1.5, 1], 'color', [0.94, 0.91, 0.55], 'label', 'Classification\nOutput')
};

% Draw components
for i = 1:length(components)
    comp = components{i};
    
    % Draw rectangle
    rectangle('Position', [comp.pos(1), comp.pos(2), comp.size(1), comp.size(2)], ...
              'FaceColor', comp.color, 'EdgeColor', 'k', 'LineWidth', 1.5);
    
    % Add label
    text(comp.pos(1) + comp.size(1)/2, comp.pos(2) + comp.size(2)/2, comp.label, ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
    
    % Draw arrows between components (simple lines with triangles)
    if i < length(components)
        next_comp = components{i+1};
        start_x = comp.pos(1) + comp.size(1);
        end_x = next_comp.pos(1);
        y_pos = comp.pos(2) + comp.size(2)/2;
        
        % Line
        plot([start_x, end_x], [y_pos, y_pos], 'r-', 'LineWidth', 2);
        
        % Simple arrow head
        arrow_size = 0.1;
        plot([end_x - arrow_size, end_x, end_x - arrow_size], ...
             [y_pos - arrow_size, y_pos, y_pos + arrow_size], 'r-', 'LineWidth', 2);
    end
end

% === Key Innovation Highlights ===
% SE Module highlight (golden dashed box)
se_comp = components{5};
rectangle('Position', [se_comp.pos(1) - 0.1, se_comp.pos(2) - 0.1, se_comp.size(1) + 0.2, se_comp.size(2) + 0.2], ...
          'FaceColor', 'none', 'EdgeColor', [1, 0.84, 0], 'LineWidth', 4, 'LineStyle', '--');

% Attention highlight
att_comp = components{7};
rectangle('Position', [att_comp.pos(1) - 0.1, att_comp.pos(2) - 0.1, att_comp.size(1) + 0.2, att_comp.size(2) + 0.2], ...
          'FaceColor', 'none', 'EdgeColor', [1, 0.84, 0], 'LineWidth', 4, 'LineStyle', '--');

% === Baseline Comparison ===
text(9, 1.5, 'Baseline Architecture Comparison', 'HorizontalAlignment', 'center', ...
     'FontWeight', 'bold', 'FontSize', 12, 'Color', [0.5, 0, 0]);

% Baseline models
baselines = {
    struct('name', 'CNN Only', 'perf', '84.2±2.5%', 'cv', 'CV=3.0%', 'color', [0.7, 0.7, 1.0]),
    struct('name', 'BiLSTM Only', 'perf', '80.3±2.2%', 'cv', 'CV=2.7%', 'color', [0.7, 1.0, 0.7]),
    struct('name', 'Conformer-lite', 'perf', '40.3±38.6%', 'cv', 'CV=95.7%', 'color', [1.0, 0.7, 0.7]),
    struct('name', 'Enhanced (Ours)', 'perf', '83.0±0.1%', 'cv', 'CV<0.2%', 'color', [1.0, 0.84, 0.0])
};

for i = 1:length(baselines)
    baseline = baselines{i};
    x_pos = 2 + (i-1) * 3.5;
    
    rectangle('Position', [x_pos, 0.5, 2.5, 0.8], 'FaceColor', baseline.color, 'EdgeColor', 'k', 'LineWidth', 1.5);
    text(x_pos + 1.25, 0.9, baseline.name, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
    text(x_pos + 1.25, 0.7, baseline.perf, 'HorizontalAlignment', 'center', 'FontSize', 8);
    text(x_pos + 1.25, 0.6, baseline.cv, 'HorizontalAlignment', 'center', 'FontSize', 8);
end

% Enhanced model special highlighting
rectangle('Position', [15.8, 0.4], [2.7, 1.0], 'FaceColor', 'none', 'EdgeColor', [1, 0.84, 0], 'LineWidth', 4);

% === Reference Citations ===
text(1, 6, 'Architecture Design Inspired by:', 'FontWeight', 'bold', 'FontSize', 10, 'Color', [0, 0, 0.8]);
text(1, 5.8, '• SenseFi: Yang et al. (2023) - Benchmark evaluation', 'FontSize', 9, 'Color', [0, 0, 0.8]);
text(1, 5.6, '• SE-Networks: Hu et al. (2018) - Channel attention', 'FontSize', 9, 'Color', [0, 0, 0.8]);
text(1, 5.4, '• CLNet: Xu et al. (2021) - Lightweight design', 'FontSize', 9, 'Color', [0, 0, 0.8]);

% === Key Achievements Box ===
rectangle('Position', [1, 6.2], [8, 1.2], 'FaceColor', [0.95, 1.0, 0.95], 'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);
text(5, 6.8, 'Enhanced Model Key Achievements', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12, 'Color', [0, 0.5, 0]);
text(1.2, 6.5, '✓ Perfect Consistency: 83.0±0.1% F1 (LOSO = LORO)', 'FontSize', 10, 'Color', [0, 0.5, 0], 'FontWeight', 'bold');
text(1.2, 6.3, '✓ Superior Stability: CV < 0.2% (vs baselines 2.7-95.7%)', 'FontSize', 10, 'Color', [0, 0.5, 0], 'FontWeight', 'bold');

% Set axis properties
xlim([0, 19]);
ylim([0, 7.5]);
axis off;

% Save the figure
print(gcf, 'enhanced_architecture_reference_inspired.pdf', '-dpdf', '-r300');
print(gcf, 'enhanced_architecture_reference_inspired.png', '-dpng', '-r300');

fprintf('✅ Architecture diagram saved: enhanced_architecture_reference_inspired.pdf\n');

fprintf('\n🎨 Reference-Inspired Architecture Features:\n');
fprintf('• Professional component-based design\n');
fprintf('• Clear data flow visualization\n');
fprintf('• Key innovations prominently highlighted\n');
fprintf('• Baseline comparison integrated\n');
fprintf('• Performance achievements emphasized\n');
fprintf('• Reference citations included\n');

fprintf('\n📚 Architecture Design Inspired by Leading Papers:\n');
fprintf('• SenseFi (Yang et al. 2023): Benchmark evaluation approach\n');
fprintf('• SE-Networks (Hu et al. 2018): Channel attention visualization\n');
fprintf('• CLNet (Xu et al. 2021): Lightweight CNN architecture\n');

fprintf('\n🚀 Professional architecture diagram ready for paper!\n');