% Simple Figure 3: Cross-Domain Performance Analysis
% Clean bar plot with Enhanced model highlighting
% IEEE IoTJ Compatible

close all; clear; clc;

fprintf('📊 Generating Simple Enhanced Cross-Domain Analysis...\n');

% Performance data from paper
models = {'Enhanced', 'CNN', 'BiLSTM', 'Conformer-lite'};
loso_f1 = [0.830, 0.842, 0.803, 0.403];
loso_err = [0.001, 0.025, 0.022, 0.386];
loro_f1 = [0.830, 0.796, 0.789, 0.841];
loro_err = [0.001, 0.097, 0.044, 0.040];

% Create main figure
figure('Position', [100, 100, 1200, 800]);

% === Main Performance Comparison ===
x = 1:length(models);
width = 0.35;

% Create grouped bar chart
bars1 = bar(x - width/2, loso_f1, width, 'FaceColor', [0.18 0.80 0.44], 'EdgeColor', 'k', 'LineWidth', 1);
hold on;
bars2 = bar(x + width/2, loro_f1, width, 'FaceColor', [0.20 0.60 0.87], 'EdgeColor', 'k', 'LineWidth', 1);

% Enhanced model golden highlighting - manual approach
% Create golden border for Enhanced model (first bars)
bar(1 - width/2, loso_f1(1), width, 'FaceColor', 'none', 'EdgeColor', [1 0.84 0], 'LineWidth', 4);
bar(1 + width/2, loro_f1(1), width, 'FaceColor', 'none', 'EdgeColor', [1 0.84 0], 'LineWidth', 4);

% Add simple error bars manually
for i = 1:length(models)
    % LOSO error bars
    plot([i - width/2, i - width/2], [loso_f1(i) - loso_err(i), loso_f1(i) + loso_err(i)], ...
         'k-', 'LineWidth', 2);
    plot([i - width/2 - 0.02, i - width/2 + 0.02], [loso_f1(i) - loso_err(i), loso_f1(i) - loso_err(i)], ...
         'k-', 'LineWidth', 2);
    plot([i - width/2 - 0.02, i - width/2 + 0.02], [loso_f1(i) + loso_err(i), loso_f1(i) + loso_err(i)], ...
         'k-', 'LineWidth', 2);
    
    % LORO error bars
    plot([i + width/2, i + width/2], [loro_f1(i) - loro_err(i), loro_f1(i) + loro_err(i)], ...
         'k-', 'LineWidth', 2);
    plot([i + width/2 - 0.02, i + width/2 + 0.02], [loro_f1(i) - loro_err(i), loro_f1(i) - loro_err(i)], ...
         'k-', 'LineWidth', 2);
    plot([i + width/2 - 0.02, i + width/2 + 0.02], [loro_f1(i) + loro_err(i), loro_f1(i) + loro_err(i)], ...
         'k-', 'LineWidth', 2);
    
    % Add performance value labels
    text(i - width/2, loso_f1(i) + loso_err(i) + 0.03, sprintf('%.3f', loso_f1(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
    text(i + width/2, loro_f1(i) + loro_err(i) + 0.03, sprintf('%.3f', loro_f1(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
end

set(gca, 'XTick', x, 'XTickLabel', models);
xlabel('Model Architecture', 'FontWeight', 'bold');
ylabel('Macro F1 Score', 'FontWeight', 'bold');
title('Cross-Domain Performance Comparison\n(Enhanced Model in Gold)', 'FontWeight', 'bold');
legend({'LOSO', 'LORO'}, 'Location', 'northeast', 'FontWeight', 'bold');
grid on;
ylim([0, 1.0]);
set(gca, 'YTick', 0:0.1:1.0);

% Add perfect consistency annotation
text(1.5, 0.9, 'Perfect Consistency\n83.0±0.1% (LOSO = LORO)', ...
     'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10, ...
     'BackgroundColor', 'yellow', 'EdgeColor', 'red', 'LineWidth', 2);

% Add arrow pointing to Enhanced model
annotation('arrow', [0.35, 0.25], [0.82, 0.83], 'Color', 'red', 'LineWidth', 2);

fprintf('✅ Enhanced cross-domain performance plot completed\n');

% Save the figure
print(gcf, 'figure3_enhanced_compatible.pdf', '-dpdf', '-r300');
print(gcf, 'figure3_enhanced_compatible.png', '-dpng', '-r300');

fprintf('📁 Files saved:\n');
fprintf('• figure3_enhanced_compatible.pdf (Publication quality)\n');
fprintf('• figure3_enhanced_compatible.png (Preview)\n');

fprintf('\n🎉 Simple Enhanced Figure 3 Generation Complete!\n');
fprintf('🏆 Key Features:\n');
fprintf('• Golden Enhanced model highlighting\n');
fprintf('• Clear error bars and performance labels\n');
fprintf('• Perfect for small datasets\n');
fprintf('• High visual contrast and differentiation\n');
fprintf('• IEEE IoTJ publication quality\n');
fprintf('• Enhanced model advantages clearly shown\n');