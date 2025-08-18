% Simplified Octave Figure Generation for IEEE IoTJ
% Compatible with Octave 9.x - simplified version

close all; clear; clc;

% IEEE IoTJ color scheme
blue = [46, 134, 171] / 255;
red = [232, 72, 85] / 255;
green = [60, 179, 113] / 255;
crimson = [220, 20, 60] / 255;

%% Figure 3: D3 Cross-Domain Performance
fprintf('📊 Generating Figure 3: CDAE Cross-Domain Performance...\n');

figure(1);

% D3 Data
models = {'Enhanced', 'CNN', 'BiLSTM', 'Conformer'};
loso_f1 = [0.830, 0.842, 0.803, 0.403];
loso_err = [0.001, 0.025, 0.022, 0.386];
loro_f1 = [0.830, 0.796, 0.789, 0.841];
loro_err = [0.001, 0.097, 0.044, 0.040];

% Create subplot layout
x = 1:length(models);
width = 0.35;

% LOSO bars
b1 = bar(x - width/2, loso_f1, width);
set(b1, 'FaceColor', blue, 'EdgeColor', 'k');
hold on;

% LORO bars
b2 = bar(x + width/2, loro_f1, width);
set(b2, 'FaceColor', red, 'EdgeColor', 'k');

% Error bars
errorbar(x - width/2, loso_f1, loso_err, 'k.', 'LineWidth', 1, 'MarkerSize', 1);
errorbar(x + width/2, loro_f1, loro_err, 'k.', 'LineWidth', 1, 'MarkerSize', 1);

% Formatting
set(gca, 'XTick', x);
set(gca, 'XTickLabel', models);
xlabel('Model Architecture');
ylabel('Macro F1 Score');
title('CDAE: Cross-Domain Generalization Performance');
legend('LOSO', 'LORO', 'Location', 'northeast');
grid on;
ylim([0, 1.0]);

% Value labels
for i = 1:length(models)
    text(i - width/2, loso_f1(i) + 0.05, sprintf('%.3f', loso_f1(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8);
    text(i + width/2, loro_f1(i) + 0.05, sprintf('%.3f', loro_f1(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8);
end

% Enhanced highlight text
text(1, 0.95, 'Enhanced: 83.0% Consistency', 'HorizontalAlignment', 'center', ...
    'FontSize', 10, 'FontWeight', 'bold', 'Color', [1, 0.84, 0]);

% Save Figure 3
print('figure3_cdae_octave.pdf', '-dpdf', '-r300');

%% Figure 4: D4 Label Efficiency
fprintf('📊 Generating Figure 4: STEA Label Efficiency...\n');

figure(2);

% D4 Data
x_data = [1, 5, 10, 20, 100];
y_data = [0.455, 0.780, 0.730, 0.821, 0.833];
y_err = [0.050, 0.016, 0.104, 0.003, 0.000];

% Efficient range background (simple rectangle)
rectangle('Position', [0, 0, 20, 1], 'FaceColor', [0.8, 1, 0.8], 'EdgeColor', 'none');
hold on;

% Main curve
plot(x_data, y_data, 'o-', 'Color', blue, 'LineWidth', 2.5, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'white', 'MarkerEdgeColor', blue);

% Error bars
errorbar(x_data, y_data, y_err, 'Color', blue, 'LineStyle', 'none', 'LineWidth', 1);

% Target line
line([0, 105], [0.80, 0.80], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5);

% Key point highlight (20% labels)
plot(20, 0.821, 'o', 'MarkerSize', 12, 'MarkerFaceColor', [1, 1, 0], ...
    'MarkerEdgeColor', red, 'LineWidth', 3);

% Formatting
xlabel('Label Ratio (%)');
ylabel('Macro F1 Score');
title('STEA: Sim2Real Label Efficiency Breakthrough');
xlim([0, 105]);
ylim([0, 1.0]);
grid on;

% Add data labels
for i = 1:length(x_data)
    text(x_data(i), y_data(i) + y_err(i) + 0.04, sprintf('%.3f', y_data(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', blue);
end

% Key achievement text
text(50, 0.92, 'Key Achievement: 82.1% F1 @ 20% Labels', ...
    'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', ...
    'Color', red);

text(50, 0.85, '80% Cost Reduction', ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', ...
    'Color', [0.2, 0.6, 0.2]);

% Target line label
text(80, 0.82, 'Target: 80% F1', 'FontSize', 9, 'Color', 'r');

% Efficient range label
text(10, 0.95, 'Efficient Range', 'FontSize', 9, 'Color', [0.2, 0.6, 0.2]);

% Save Figure 4
print('figure4_stea_octave.pdf', '-dpdf', '-r300');

%% Display Results
fprintf('\n✅ Octave Figure Generation Complete!\n');
fprintf('📊 Generated files:\n');
fprintf('  • figure3_cdae_octave.pdf - CDAE cross-domain performance\n');
fprintf('  • figure4_stea_octave.pdf - STEA label efficiency breakthrough\n');

fprintf('\n🎯 Key Results Visualized:\n');
fprintf('• CDAE: Enhanced 83.0±0.1%% consistency across LOSO/LORO\n');
fprintf('• STEA: 82.1%% F1 @ 20%% labels (80%% cost reduction)\n');

fprintf('\n📋 IEEE IoTJ Compliance:\n');
fprintf('• Resolution: 300 DPI PDF export\n');
fprintf('• Format: Professional scientific graphics\n');
fprintf('• Data: Based on verified D3/D4 experimental results\n');

fprintf('\n🚀 Ready for paper inclusion and IEEE IoTJ submission!\n');