% Method 4: MATLAB Professional Scientific Plotting
% 适用于IEEE期刊投稿的高质量图表制作
% 需要MATLAB R2019b或更高版本

%% Setup for IEEE IoTJ compliance
close all; clear; clc;

% IEEE IoTJ color scheme (colorblind-friendly)
ieee_colors = [46, 134, 171; 232, 72, 85; 60, 179, 113; 220, 20, 60] / 255;

%% Figure 3: D3 Cross-Domain Performance
figure('Position', [100, 100, 1200, 600]); % Size for IEEE IoTJ double column

% D3 Data
models = {'Enhanced', 'CNN', 'BiLSTM', 'Conformer-lite'};
loso_f1 = [0.830, 0.842, 0.803, 0.403];
loso_err = [0.001, 0.025, 0.022, 0.386];
loro_f1 = [0.830, 0.796, 0.789, 0.841];
loro_err = [0.001, 0.097, 0.044, 0.040];

% Create grouped bar chart
x = 1:length(models);
width = 0.35;

% Plot bars with error bars
b1 = bar(x - width/2, loso_f1, width, 'FaceColor', ieee_colors(1,:), ...
    'EdgeColor', 'k', 'LineWidth', 0.5, 'FaceAlpha', 0.8);
hold on;
b2 = bar(x + width/2, loro_f1, width, 'FaceColor', ieee_colors(2,:), ...
    'EdgeColor', 'k', 'LineWidth', 0.5, 'FaceAlpha', 0.8);

% Add error bars
errorbar(x - width/2, loso_f1, loso_err, 'k', 'LineStyle', 'none', ...
    'LineWidth', 1, 'CapSize', 3);
errorbar(x + width/2, loro_f1, loro_err, 'k', 'LineStyle', 'none', ...
    'LineWidth', 1, 'CapSize', 3);

% Formatting for IEEE IoTJ
set(gca, 'XTick', x, 'XTickLabel', models);
xlabel('Model Architecture', 'FontName', 'Times', 'FontSize', 10);
ylabel('Macro F1 Score', 'FontName', 'Times', 'FontSize', 10);
title('Cross-Domain Generalization Performance', 'FontName', 'Times', 'FontSize', 12);
legend({'LOSO', 'LORO'}, 'Location', 'northeast', 'FontName', 'Times', 'FontSize', 9);
grid on; alpha(0.3);
ylim([0, 1.0]);
set(gca, 'YTick', 0:0.1:1.0);

% Add value labels on bars
for i = 1:length(models)
    text(i - width/2, loso_f1(i) + loso_err(i) + 0.02, ...
        sprintf('%.3f±%.3f', loso_f1(i), loso_err(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8, 'FontName', 'Times');
    text(i + width/2, loro_f1(i) + loro_err(i) + 0.02, ...
        sprintf('%.3f±%.3f', loro_f1(i), loro_err(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8, 'FontName', 'Times');
end

% IEEE IoTJ export settings
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperSize', [17.1, 10]);
set(gcf, 'PaperPosition', [0, 0, 17.1, 10]);
print(gcf, 'figure3_d3_cross_domain_matlab.pdf', '-dpdf', '-r300');

fprintf('✓ Figure 3 saved: figure3_d3_cross_domain_matlab.pdf\n');

%% Figure 4: D4 Label Efficiency Curve
figure('Position', [200, 200, 1200, 800]); % IEEE IoTJ double column size

% D4 Data
label_ratios = [1.0, 5.0, 10.0, 20.0, 100.0];
f1_scores = [0.455, 0.780, 0.730, 0.821, 0.833];
error_bars = [0.050, 0.016, 0.104, 0.003, 0.000];

% Efficient range background
patch([0, 20, 20, 0], [0, 0, 1, 1], [144, 238, 144]/255, ...
    'FaceAlpha', 0.2, 'EdgeColor', 'none');
hold on;

% Error ribbon
x_smooth = linspace(1, 100, 100);
y_smooth = interp1(label_ratios, f1_scores, x_smooth, 'pchip');
upper = interp1(label_ratios, f1_scores + error_bars, x_smooth, 'pchip');
lower = interp1(label_ratios, f1_scores - error_bars, x_smooth, 'pchip');

fill([x_smooth, fliplr(x_smooth)], [upper, fliplr(lower)], ...
    ieee_colors(1,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');

% Main efficiency curve
plot(label_ratios, f1_scores, 'o-', 'Color', ieee_colors(1,:), ...
    'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerFaceColor', 'white', ...
    'MarkerEdgeColor', ieee_colors(1,:), 'MarkerEdgeWidth', 2);

% Target lines
yline(0.80, '--r', 'LineWidth', 1.5, 'Label', 'Target: 80% F1', ...
    'FontName', 'Times', 'FontSize', 9);
yline(0.90, ':orange', 'LineWidth', 1, 'Label', 'Ideal: 90% F1', ...
    'FontName', 'Times', 'FontSize', 9);

% Key achievement annotation
annotation('textarrow', [0.35, 0.20], [0.87, 0.821], ...
    'String', sprintf('Key Achievement:\n82.1%% F1 @ 20%% Labels'), ...
    'FontName', 'Times', 'FontSize', 10, 'FontWeight', 'bold', ...
    'Color', [1, 0.4, 0.4], 'BackgroundColor', [1, 1, 0.8]);

% Formatting
xlabel('Label Ratio (%)', 'FontName', 'Times', 'FontSize', 10);
ylabel('Macro F1 Score', 'FontName', 'Times', 'FontSize', 10);
title('Sim2Real Label Efficiency Breakthrough', 'FontName', 'Times', 'FontSize', 12);
xlim([0, 105]);
ylim([0, 1.0]);
grid on; alpha(0.3);

% Add data point labels
for i = 1:length(label_ratios)
    text(label_ratios(i), f1_scores(i) + error_bars(i) + 0.03, ...
        sprintf('%.3f', f1_scores(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8, ...
        'FontName', 'Times', 'Color', ieee_colors(1,:));
end

% IEEE IoTJ export settings
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperSize', [17.1, 12]);
set(gcf, 'PaperPosition', [0, 0, 17.1, 12]);
print(gcf, 'figure4_d4_label_efficiency_matlab.pdf', '-dpdf', '-r300');

fprintf('✓ Figure 4 saved: figure4_d4_label_efficiency_matlab.pdf\n');

%% Summary Statistics Display
fprintf('\n📊 MATLAB Method Summary:\n');
fprintf('✓ IEEE IoTJ compliant sizing and resolution\n');
fprintf('✓ Professional scientific graphics quality\n');
fprintf('✓ Precise numerical annotations\n');
fprintf('✓ Publication-ready PDF export\n');

fprintf('\n🎯 Key Results Visualized:\n');
fprintf('• Enhanced model consistency: 83.0±0.1%% F1 across LOSO/LORO\n');
fprintf('• Label efficiency breakthrough: 82.1%% F1 @ 20%% labels\n');
fprintf('• Cost-benefit achievement: 80%% labeling cost reduction\n');

fprintf('\n💡 MATLAB Advantages:\n');
fprintf('• IEEE journal standard compliance\n');
fprintf('• High-precision numerical plotting\n');
fprintf('• Professional annotation capabilities\n');
fprintf('• Direct PDF export at publication quality\n');