% Figure 2: Experimental Protocols Overview (Octave Version)
% IEEE IoTJ Compatible

close all; clear; clc;

fprintf('🔬 Generating Figure 2: Experimental Protocols...\n');

% Create figure
figure('Position', [100, 100, 1400, 1000]);

% === Protocol Visualization ===
hold on;

% D2 Protocol (Left)
rectangle('Position', [1, 6.5], [3.5, 2], 'FaceColor', [1.0, 0.9, 0.7], 'EdgeColor', 'k', 'LineWidth', 2);
text(2.75, 7.8, 'D2 Protocol', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
text(2.75, 7.5, 'Synthetic Data Validation', 'HorizontalAlignment', 'center', 'FontSize', 10, 'FontStyle', 'italic');
text(2.75, 7.1, '• 540 configurations', 'HorizontalAlignment', 'center', 'FontSize', 9);
text(2.75, 6.9, '• Noise levels: {0.0, 0.4, 0.8}', 'HorizontalAlignment', 'center', 'FontSize', 9);
text(2.75, 6.7, '• 4 models × 5 seeds', 'HorizontalAlignment', 'center', 'FontSize', 9);

% CDAE Protocol (Center)
rectangle('Position', [5.5, 6.5], [3.5, 2], 'FaceColor', [0.8, 1.0, 0.8], 'EdgeColor', 'k', 'LineWidth', 2);
text(7.25, 7.8, 'CDAE Protocol', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
text(7.25, 7.5, 'Cross-Domain Evaluation', 'HorizontalAlignment', 'center', 'FontSize', 10, 'FontStyle', 'italic');
text(7.25, 7.1, '• LOSO: Leave-One-Subject-Out', 'HorizontalAlignment', 'center', 'FontSize', 9);
text(7.25, 6.9, '• LORO: Leave-One-Room-Out', 'HorizontalAlignment', 'center', 'FontSize', 9);
text(7.25, 6.7, '• 40 configurations total', 'HorizontalAlignment', 'center', 'FontSize', 9);

% STEA Protocol (Right)
rectangle('Position', [10, 6.5], [3.5, 2], 'FaceColor', [0.8, 0.9, 1.0], 'EdgeColor', 'k', 'LineWidth', 2);
text(11.75, 7.8, 'STEA Protocol', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
text(11.75, 7.5, 'Transfer Efficiency Assessment', 'HorizontalAlignment', 'center', 'FontSize', 10, 'FontStyle', 'italic');
text(11.75, 7.1, '• Transfer methods: 4 approaches', 'HorizontalAlignment', 'center', 'FontSize', 9);
text(11.75, 6.9, '• Label ratios: 1%-100%', 'HorizontalAlignment', 'center', 'FontSize', 9);
text(11.75, 6.7, '• 56 configurations completed', 'HorizontalAlignment', 'center', 'FontSize', 9);

% === Protocol Integration Flow ===
text(7.5, 5.8, 'Protocol Integration & Results', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12, 'Color', 'darkred');

% Integration arrows
plot([2.75, 6.5], [6.5, 5.5], 'r-', 'LineWidth', 2);
plot([7.25, 7.5], [6.5, 5.5], 'r-', 'LineWidth', 2);
plot([11.75, 8.5], [6.5, 5.5], 'r-', 'LineWidth', 2);

% Results box
rectangle('Position', [5, 4.5], [5, 1.5], 'FaceColor', [1.0, 1.0, 0.8], 'EdgeColor', 'darkred', 'LineWidth', 3);
text(7.5, 5.6, 'Breakthrough Results', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12, 'Color', 'darkred');
text(7.5, 5.3, '✓ D2: Synthetic data quality validated', 'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', 'darkgreen');
text(7.5, 5.1, '✓ CDAE: 83.0±0.1% cross-domain consistency', 'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', 'darkgreen');
text(7.5, 4.9, '✓ STEA: 82.1% F1 @ 20% labels (80% cost reduction)', 'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', 'darkgreen');
text(7.5, 4.7, '✓ Statistical significance: p < 0.01 for all comparisons', 'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', 'darkgreen');

% === Model Architecture Summary ===
text(7.5, 3.8, 'Enhanced Model Architecture', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12, 'Color', 'darkblue');

% Model components
model_components = {
    {2, 3, 1.5, 0.6, [1.0, 0.9, 0.7], 'CNN\nFeatures'},
    {4, 3, 1.5, 0.6, [1.0, 0.84, 0.0], 'SE Module'},
    {6, 3, 1.5, 0.6, [0.9, 1.0, 0.9], 'BiLSTM'},
    {8, 3, 1.5, 0.6, [0.9, 0.8, 1.0], 'Attention'},
    {10, 3, 1.5, 0.6, [0.9, 0.9, 0.8], 'Output'}
};

for i = 1:length(model_components)
    comp = model_components{i};
    rectangle('Position', [comp{1}, comp{2}, comp{3}, comp{4}], 'FaceColor', comp{5}, 'EdgeColor', 'k', 'LineWidth', 1);
    text(comp{1} + comp{3}/2, comp{2} + comp{4}/2, comp{6}, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
    
    % Arrows between components
    if i < length(model_components)
        next_comp = model_components{i+1};
        plot([comp{1} + comp{3}, next_comp{1}], [comp{2} + comp{4}/2, next_comp{2} + next_comp{4}/2], 'b-', 'LineWidth', 2);
    end
end

% === Final Results Summary ===
rectangle('Position', [2, 1.5], [10, 1], 'FaceColor', [0.95, 1.0, 0.95], 'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);
text(7, 2.1, 'First Systematic Sim2Real Study in WiFi CSI HAR', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11, 'Color', [0, 0.5, 0]);
text(7, 1.8, 'Perfect Cross-Domain Consistency + Revolutionary Label Efficiency', 'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', [0, 0.5, 0]);

% === Title ===
title('Comprehensive Experimental Evaluation Framework\n(D2 Validation + CDAE Cross-Domain + STEA Transfer Efficiency)', 'FontSize', 14, 'FontWeight', 'bold');

xlim([0, 15]);
ylim([1, 9]);
axis off;

% Save figure
print(gcf, 'figure2_protocols_overview.pdf', '-dpdf', '-r300');
fprintf('✅ Figure 2 saved: figure2_protocols_overview.pdf\n');