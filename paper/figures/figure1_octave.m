% Figure 1: System Architecture Overview (Octave Version)
% IEEE IoTJ Compatible

close all; clear; clc;

fprintf('🏗️ Generating Figure 1: System Architecture Overview...\n');

% Create figure
figure('Position', [100, 100, 1400, 1000]);

% === System Architecture Components ===
hold on;

% Define component specifications
components = {
    % [x, y, width, height, color, label]
    {1, 7, 2.5, 1, [0.9, 0.9, 1.0], 'Real WiFi CSI\nBenchmark Data\n(SenseFi)'},
    {1, 5.5, 1.8, 0.8, [0.8, 1.0, 0.8], 'Multipath\nPropagation'},
    {3, 5.5, 1.8, 0.8, [0.8, 1.0, 0.8], 'Human Body\nInteraction'},
    {5, 5.5, 1.8, 0.8, [0.8, 1.0, 0.8], 'Environmental\nVariations'},
    {2.5, 4, 3, 1, [1.0, 0.95, 0.8], 'Physics-Guided\nSynthetic CSI Generator'},
    {1, 2.5, 1.5, 0.8, [1.0, 0.9, 0.7], 'D2 Protocol\n540 Configs'},
    {3, 2.5, 1.5, 0.8, [1.0, 0.9, 0.7], 'CDAE Protocol\n40 Configs'},
    {4.8, 2.5, 1.5, 0.8, [1.0, 0.9, 0.7], 'STEA Protocol\n56 Configs'},
    {8, 6.5, 2, 0.8, [1.0, 0.8, 0.8], 'CNN Feature\nExtraction'},
    {10.5, 6.5, 2, 0.8, [1.0, 0.84, 0.0], 'SE Module\n(Channel Attention)'},
    {8, 5.5, 2, 0.8, [0.9, 1.0, 0.9], 'BiLSTM\n(Temporal)'},
    {10.5, 5.5, 2, 0.8, [0.9, 0.8, 1.0], 'Temporal Attention\n(Long-range)'},
    {9.2, 4.2, 2.5, 0.8, [0.8, 0.9, 1.0], 'Sim2Real Transfer\nLearning'},
    {8, 2.8, 1.8, 0.6, [0.9, 0.9, 0.9], 'Calibration\nAnalysis'},
    {10.2, 2.8, 1.8, 0.6, [0.9, 0.9, 0.9], 'Cross-Domain\nRobustness'},
    {4, 1, 6, 0.8, [0.8, 1.0, 1.0], 'Results: 83.0±0.1% F1 + 82.1% @ 20% Labels'}
};

% Draw all components
for i = 1:length(components)
    comp = components{i};
    x = comp{1}; y = comp{2}; w = comp{3}; h = comp{4}; 
    color = comp{5}; label = comp{6};
    
    % Draw rectangle
    rectangle('Position', [x, y, w, h], 'FaceColor', color, 'EdgeColor', 'k', 'LineWidth', 1.5);
    
    % Add label
    text(x + w/2, y + h/2, label, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
end

% === Data Flow Arrows ===
% Real data to physics modeling
plot([2.5, 3], [7, 6.3], 'b-', 'LineWidth', 2);
plot([3 - 0.1, 3, 3 - 0.1], [6.3 - 0.1, 6.3, 6.3 + 0.1], 'b-', 'LineWidth', 2);

% Physics to generator
plot([2.5, 4], [5.5, 5], 'g-', 'LineWidth', 2);
plot([4 - 0.1, 4, 4 - 0.1], [5 - 0.1, 5, 5 + 0.1], 'g-', 'LineWidth', 2);

% Generator to protocols
plot([3.5, 2.5], [4, 3.3], 'Color', [1, 0.5, 0], 'LineWidth', 2);
plot([3.7, 4.7], [4, 3.3], 'Color', [1, 0.5, 0], 'LineWidth', 2);
plot([4.5, 5.5], [4, 3.3], 'Color', [1, 0.5, 0], 'LineWidth', 2);

% Real + synthetic to model
plot([3, 8.5], [7.5, 7], 'Color', [0.5, 0, 0.5], 'LineWidth', 2);
plot([4, 8.5], [3, 6.8], 'Color', [1, 0.5, 0], 'LineWidth', 2);

% Model flow
plot([9, 10.5], [6.9, 6.9], 'r-', 'LineWidth', 2);
plot([9, 10.5], [5.9, 5.9], 'r-', 'LineWidth', 2);
plot([9.8, 9.8], [6.5, 5], 'r-', 'LineWidth', 2);

% To evaluation
plot([10, 9], [4.2, 3.4], 'Color', [0.5, 0, 0], 'LineWidth', 2);

% To results
plot([9, 7], [2.8, 1.8], 'Color', [0, 0.5, 0], 'LineWidth', 2);

% === Add section labels ===
text(2.5, 8.5, 'Input Data Layer', 'FontWeight', 'bold', 'FontSize', 12, 'Color', [0, 0, 0.5]);
text(3, 6.8, 'Physics Modeling', 'FontWeight', 'bold', 'FontSize', 12, 'Color', [0, 0.5, 0]);
text(4, 3.8, 'Data Generation', 'FontWeight', 'bold', 'FontSize', 12, 'Color', [1, 0.5, 0]);
text(10, 7.5, 'Enhanced Model', 'FontWeight', 'bold', 'FontSize', 12, 'Color', [0.5, 0, 0]);
text(9.5, 3.5, 'Evaluation', 'FontWeight', 'bold', 'FontSize', 12, 'Color', [0.5, 0.5, 0.5]);

% === Key Innovation Highlights ===
% SE module highlight
rectangle('Position', [10.3, 6.3], [2.4, 1.0], 'FaceColor', 'none', 'EdgeColor', [1, 0.84, 0], 'LineWidth', 3, 'LineStyle', '--');

% === Title and formatting ===
title('Physics-Guided Synthetic WiFi CSI Framework: System Architecture Overview', 'FontSize', 14, 'FontWeight', 'bold');

xlim([0, 13]);
ylim([0.5, 9]);
axis off;

% Save figure
print(gcf, 'figure1_system_overview.pdf', '-dpdf', '-r300');
fprintf('✅ Figure 1 saved: figure1_system_overview.pdf\n');