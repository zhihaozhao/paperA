% Figure 1: System Overview (Simplified Compatible)
% IEEE IoTJ Compatible

close all; clear; clc;

fprintf('🏗️ Generating Figure 1: System Overview...\n');

figure('Position', [100, 100, 1200, 800]);
hold on;

% === Simple Component Boxes ===
% Input Data
fill([1, 3, 3, 1], [6, 6, 7, 7], [0.9, 0.9, 1.0], 'EdgeColor', 'k', 'LineWidth', 2);
text(2, 6.5, 'Real WiFi CSI\nBenchmark Data', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% Physics Modeling
fill([1, 2.5, 2.5, 1], [4.5, 4.5, 5.5, 5.5], [0.8, 1.0, 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
text(1.75, 5, 'Physics\nModeling', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

fill([3, 4.5, 4.5, 3], [4.5, 4.5, 5.5, 5.5], [0.8, 1.0, 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
text(3.75, 5, 'Synthetic\nGeneration', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% Enhanced Model
fill([6, 9, 9, 6], [5, 5, 6.5, 6.5], [1.0, 0.84, 0.0], 'EdgeColor', 'k', 'LineWidth', 2);
text(7.5, 5.75, 'Enhanced Model\n(CNN+SE+Attention)', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% Evaluation
fill([6, 9, 9, 6], [3, 3, 4, 4], [0.9, 0.9, 0.9], 'EdgeColor', 'k', 'LineWidth', 1.5);
text(7.5, 3.5, 'Trustworthy Evaluation\n(D2+CDAE+STEA)', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% Results
fill([3, 11, 11, 3], [1, 1, 2, 2], [0.95, 1.0, 0.95], 'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);
text(7, 1.5, 'Results: 83.0±0.1% F1 Cross-Domain + 82.1% @ 20% Labels', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% === Flow Arrows ===
% Simple arrows using plot
plot([2.5, 3.5], [6.2, 5.8], 'b-', 'LineWidth', 3);
plot([4.5, 6], [5.2, 5.8], 'r-', 'LineWidth', 3);
plot([7.5, 7.5], [5, 4], 'Color', [0.5, 0, 0.5], 'LineWidth', 3);
plot([7.5, 7], [3, 2], 'Color', [0, 0.5, 0], 'LineWidth', 3);

% === Title ===
title('Physics-Guided Synthetic WiFi CSI Framework: System Overview', 'FontSize', 14, 'FontWeight', 'bold');

xlim([0, 12]);
ylim([0.5, 7.5]);
axis off;

% Save
print(gcf, 'figure1_system_overview.pdf', '-dpdf', '-r300');
fprintf('✅ Figure 1 generated: figure1_system_overview.pdf\n');