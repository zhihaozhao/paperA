% Enhanced Model Architecture Visualization (Simple & Compatible)
% Inspired by reference papers - SenseFi, SE-Networks style
% IEEE IoTJ Compatible

close all; clear; clc;

fprintf('🏗️ Generating Enhanced Model Architecture (Reference-Inspired)...\n');

% Create main figure
figure('Position', [100, 100, 1600, 1000]);

% === Create 2D Architecture Diagram ===
hold on;

% Component positions and sizes
% Input
input_rect = rectangle('Position', [1, 5, 1.5, 1], 'FaceColor', [0.9, 0.9, 1.0], 'EdgeColor', 'k', 'LineWidth', 2);
text(1.75, 5.5, 'WiFi CSI Input\n(T×F×N)', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);

% CNN Feature Extraction
cnn_positions = [3.5, 5, 6.5];
cnn_labels = {'Conv1D\n32 filters', 'Conv1D\n64 filters', 'Conv1D\n128 filters'};
for i = 1:3
    rectangle('Position', [cnn_positions(i), 5, 1.2, 1], 'FaceColor', [1.0, 0.9, 0.7], 'EdgeColor', 'k', 'LineWidth', 1.5);
    text(cnn_positions(i) + 0.6, 5.5, cnn_labels{i}, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
    
    % Arrows between CNN layers
    if i < 3
        arrow([cnn_positions(i) + 1.2, 5.5], [cnn_positions(i+1), 5.5], 'Color', 'blue', 'LineWidth', 2);
    end
end

% SE Module (Key Innovation - Golden)
rectangle('Position', [8.5, 4.5, 2, 2], 'FaceColor', [1.0, 0.84, 0.0], 'EdgeColor', 'k', 'LineWidth', 3);
text(9.5, 5.5, 'SE Module\n(Channel Attention)', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
text(9.5, 5.1, 'Global Pool → FC → Sigmoid', 'HorizontalAlignment', 'center', 'FontSize', 8);

% BiLSTM
rectangle('Position', [11.5, 5, 2, 1], 'FaceColor', [0.6, 0.98, 0.6], 'EdgeColor', 'k', 'LineWidth', 1.5);
text(12.5, 5.5, 'BiLSTM\n(Temporal Modeling)', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);

% Temporal Attention (Key Innovation - Purple)
rectangle('Position', [14.5, 4.5, 2, 2], 'FaceColor', [0.87, 0.63, 0.87], 'EdgeColor', 'k', 'LineWidth', 3);
text(15.5, 5.5, 'Temporal Attention\n(Long-range)', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
text(15.5, 5.1, 'Q-K-V → Softmax → ∑', 'HorizontalAlignment', 'center', 'FontSize', 8);

% Output
rectangle('Position', [17.5, 5, 1.5, 1], 'FaceColor', [0.94, 0.91, 0.55], 'EdgeColor', 'k', 'LineWidth', 2);
text(18.25, 5.5, 'Classification\nOutput', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);

% === Data Flow Arrows ===
% Input to CNN
annotation('arrow', [0.16, 0.22], [0.55, 0.55], 'Color', 'blue', 'LineWidth', 3);

% CNN to SE
annotation('arrow', [0.42, 0.53], [0.55, 0.55], 'Color', 'red', 'LineWidth', 3);

% SE to BiLSTM  
annotation('arrow', [0.66, 0.72], [0.55, 0.55], 'Color', 'green', 'LineWidth', 3);

% BiLSTM to Attention
annotation('arrow', [0.85, 0.91], [0.55, 0.55], 'Color', 'purple', 'LineWidth', 3);

% Attention to Output
annotation('arrow', [1.08, 1.09], [0.55, 0.55], 'Color', 'orange', 'LineWidth', 3);

% === Key Innovation Highlights ===
% SE Module highlight box
rectangle('Position', [8.3, 4.3], [2.4, 2.4], 'FaceColor', 'none', 'EdgeColor', [1, 0.84, 0], 'LineWidth', 4, 'LineStyle', '--');

% Attention highlight box  
rectangle('Position', [14.3, 4.3], [2.4, 2.4], 'FaceColor', 'none', 'EdgeColor', [1, 0.84, 0], 'LineWidth', 4, 'LineStyle', '--');

% === Baseline Comparison ===
text(10, 3.5, 'vs. Baseline Architectures:', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12, 'Color', 'darkred');

baseline_data = {
    'names': {'CNN Only', 'BiLSTM Only', 'Conformer-lite', 'Enhanced (Ours)'},
    'performance': [84.2, 80.3, 40.3, 83.0],
    'consistency': [3.0, 2.7, 95.7, 0.2],
    'colors': {[0.7, 0.7, 1.0], [0.7, 1.0, 0.7], [1.0, 0.7, 0.7], [1.0, 0.84, 0.0]}
};

for i = 1:4
    x_pos = 3 + (i-1) * 3.5;
    rectangle('Position', [x_pos, 2, 2.5, 0.8], 'FaceColor', baseline_data.colors{i}, 'EdgeColor', 'k', 'LineWidth', 1.5);
    text(x_pos + 1.25, 2.4, baseline_data.names{i}, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
    text(x_pos + 1.25, 2.15, sprintf('F1: %.1f%%\nCV: %.1f%%', baseline_data.performance(i), baseline_data.consistency(i)), ...
         'HorizontalAlignment', 'center', 'FontSize', 8);
end

% Enhanced model special highlighting
rectangle('Position', [13.8, 1.8], [2.9, 1.2], 'FaceColor', 'none', 'EdgeColor', [1, 0.84, 0], 'LineWidth', 4);

% === Performance Results ===
text(10, 1, 'Key Performance Results:', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12, 'Color', 'darkgreen');
text(5, 0.5, '• Perfect Cross-Domain Consistency: 83.0±0.1% F1 (LOSO = LORO)', 'FontSize', 10, 'Color', 'darkgreen', 'FontWeight', 'bold');
text(5, 0.3, '• Exceptional Stability: CV < 0.2% (vs. baselines 2.7-95.7%)', 'FontSize', 10, 'Color', 'darkgreen', 'FontWeight', 'bold');
text(5, 0.1, '• Label Efficiency Breakthrough: 82.1% F1 using only 20% real data', 'FontSize', 10, 'Color', 'darkgreen', 'FontWeight', 'bold');

% === Title and Layout ===
title('Enhanced Model Architecture: Reference-Inspired Professional Design\n(CNN + SE + BiLSTM + Temporal Attention)', ...
      'FontSize', 16, 'FontWeight', 'bold');

xlim([0, 20]);
ylim([0, 7]);
axis off;

% Add grid for organization
for i = 1:20
    line([i, i], [0, 7], 'Color', [0.9, 0.9, 0.9], 'LineWidth', 0.5);
end
for i = 1:7
    line([0, 20], [i, i], 'Color', [0.9, 0.9, 0.9], 'LineWidth', 0.5);
end

% === Add Reference Citation ===
text(1, 6.7, 'Inspired by:', 'FontWeight', 'bold', 'FontSize', 10, 'Color', 'darkblue');
text(1, 6.5, '• SenseFi: Yang et al. (2023)', 'FontSize', 9, 'Color', 'darkblue');
text(1, 6.3, '• SE-Networks: Hu et al. (2018)', 'FontSize', 9, 'Color', 'darkblue');
text(1, 6.1, '• CLNet: Xu et al. (2021)', 'FontSize', 9, 'Color', 'darkblue');

% Save the figure
print(gcf, 'enhanced_model_architecture_final.pdf', '-dpdf', '-r300');
print(gcf, 'enhanced_model_architecture_final.png', '-dpng', '-r300');

fprintf('✅ Enhanced Model Architecture saved: enhanced_model_architecture_final.pdf\n');

fprintf('\n🎨 Architecture Visualization Complete!\n');
fprintf('🏗️ Features:\n');
fprintf('• Professional 2D architecture diagram\n');
fprintf('• Reference paper-inspired design\n');
fprintf('• Key innovations highlighted (SE + Attention)\n');
fprintf('• Baseline model comparison included\n');
fprintf('• Performance results integration\n');
fprintf('• IEEE IoTJ publication quality\n');

fprintf('\n📚 Reference Integration:\n');
fprintf('• SenseFi benchmark visualization style\n');
fprintf('• SE-Networks architecture representation\n');
fprintf('• Modern deep learning diagram conventions\n');
fprintf('• Clear data flow and component relationships\n');

fprintf('\n🚀 Ready for paper integration!\n');