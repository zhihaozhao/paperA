%% Motion Control Statistical Analysis for Fruit-Picking Robot Literature
% MATLAB script for high-order motion control performance analysis
% 
% Author: Research Team
% Date: 2024
% Purpose: Generate publication-quality motion control analysis figures

function motion_control_analysis()
    % Main function for motion control meta-analysis
    
    fprintf('=== Motion Control Statistical Analysis ===\n');
    
    % Load data
    try
        data = readtable('fruit_picking_literature_data.csv');
        fprintf('Loaded %d studies for analysis\n', height(data));
    catch ME
        fprintf('Error loading data: %s\n', ME.message);
        return;
    end
    
    % Filter motion control relevant data
    motion_data = data(~isnan(data.success_rate) | ~isnan(data.cycle_time), :);
    fprintf('Found %d studies with motion control metrics\n', height(motion_data));
    
    % Create comprehensive figure
    create_motion_control_figure(motion_data);
    
    % Statistical analysis
    perform_motion_statistical_tests(motion_data);
    
    fprintf('Motion control analysis completed!\n');
end

function create_motion_control_figure(data)
    % Create comprehensive motion control analysis figure
    
    figure('Position', [100, 100, 1400, 1000]);
    
    % Panel A: Success Rate Distribution with Statistical Fit
    subplot(2, 3, 1);
    success_rates = data.success_rate(~isnan(data.success_rate)) * 100;
    
    if ~isempty(success_rates)
        % Histogram
        [counts, centers] = hist(success_rates, 12);
        bar(centers, counts, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'black', 'FaceAlpha', 0.7);
        hold on;
        
        % Fit normal distribution
        pd = fitdist(success_rates, 'Normal');
        x_fit = linspace(min(success_rates), max(success_rates), 100);
        y_fit = pdf(pd, x_fit);
        y_scaled = y_fit * length(success_rates) * (max(success_rates) - min(success_rates)) / 12;
        plot(x_fit, y_scaled, 'r-', 'LineWidth', 2);
        
        xlabel('Success Rate (%)');
        ylabel('Frequency');
        title('(A) Success Rate Distribution');
        legend(['Data', sprintf('Normal Fit (μ=%.1f%%, σ=%.1f%%)', pd.mu, pd.sigma)], 'Location', 'best');
        grid on; grid minor;
    end
    
    % Panel B: Cycle Time Analysis
    subplot(2, 3, 2);
    cycle_data = data(~isnan(data.cycle_time) & ~isnan(data.success_rate), :);
    
    if height(cycle_data) > 3
        x = cycle_data.cycle_time;
        y = cycle_data.success_rate * 100;
        
        scatter(x, y, 60, [0.8, 0.2, 0.2], 'filled', 'MarkerEdgeColor', 'black');
        
        % Add regression line
        coeffs = polyfit(x, y, 1);
        x_fit = linspace(min(x), max(x), 100);
        y_fit = polyval(coeffs, x_fit);
        hold on;
        plot(x_fit, y_fit, '--', 'Color', [0.2, 0.2, 0.2], 'LineWidth', 2);
        
        % Calculate correlation
        [R, P] = corrcoef(x, y);
        text(0.05, 0.95, sprintf('r = %.3f\np = %.3f', R(1,2), P(1,2)), ...
             'Units', 'normalized', 'BackgroundColor', 'white', 'EdgeColor', 'black');
        
        xlabel('Cycle Time (seconds)');
        ylabel('Success Rate (%)');
        title('(B) Cycle Time vs Success Rate');
        grid on; grid minor;
    end
    
    % Panel C: DOF vs Performance Analysis
    subplot(2, 3, 3);
    
    % Extract DOF data (would need to be extracted from text)
    % For now, use model complexity as proxy
    complexity_data = data(~isnan(data.model_size) & ~isnan(data.accuracy), :);
    
    if height(complexity_data) > 5
        algorithms = unique(complexity_data.algorithm_family);
        colors = lines(length(algorithms));
        
        for i = 1:length(algorithms)
            if ~strcmp(algorithms{i}, 'NaN') && ~isempty(algorithms{i})
                algo_subset = complexity_data(strcmp(complexity_data.algorithm_family, algorithms{i}), :);
                
                if height(algo_subset) > 0
                    x = algo_subset.model_size;
                    y = algo_subset.accuracy * 100;
                    sizes = (algo_subset.year - 2014) * 10; % Size represents recency
                    
                    scatter(x, y, sizes, colors(i,:), 'filled', 'MarkerEdgeColor', 'black', ...
                           'DisplayName', algorithms{i});
                    hold on;
                end
            end
        end
        
        xlabel('Model Complexity (MB)');
        ylabel('Detection Accuracy (%)');
        title('(C) Complexity vs Performance');
        legend('Location', 'best');
        grid on; grid minor;
    end
    
    % Panel D: Algorithm Performance Summary
    subplot(2, 3, 4);
    
    % Performance summary by algorithm family
    algorithms = {'R-CNN', 'YOLO', 'SSD', 'Hybrid'};
    metrics = {'accuracy', 'success_rate', 'speed_performance'};
    
    performance_matrix = zeros(length(algorithms), length(metrics));
    
    for i = 1:length(algorithms)
        algo_data = data(strcmp(data.algorithm_family, algorithms{i}), :);
        
        if height(algo_data) > 0
            performance_matrix(i, 1) = nanmean(algo_data.accuracy) * 100;
            performance_matrix(i, 2) = nanmean(algo_data.success_rate) * 100;
            % Invert speed (lower is better)
            speeds = algo_data.speed_ms(~isnan(algo_data.speed_ms));
            if ~isempty(speeds)
                performance_matrix(i, 3) = 100 - (nanmean(speeds) / max(speeds) * 100);
            end
        end
    end
    
    % Create grouped bar chart
    bar_handle = bar(performance_matrix);
    set(gca, 'XTickLabel', algorithms);
    xlabel('Algorithm Family');
    ylabel('Performance Score (%)');
    title('(D) Overall Performance Summary');
    legend({'Accuracy', 'Success Rate', 'Speed Score'}, 'Location', 'best');
    grid on;
    
    % Panel E: Temporal Evolution Analysis
    subplot(2, 3, 5);
    
    years = unique(data.year);
    years = years(~isnan(years));
    yearly_performance = zeros(size(years));
    yearly_std = zeros(size(years));
    
    for i = 1:length(years)
        year_data = data(data.year == years(i), :);
        accuracies = year_data.accuracy(~isnan(year_data.accuracy));
        
        if ~isempty(accuracies)
            yearly_performance(i) = mean(accuracies) * 100;
            yearly_std(i) = std(accuracies) * 100;
        end
    end
    
    % Plot with error bars
    errorbar(years, yearly_performance, yearly_std, 'o-', 'LineWidth', 2, ...
            'MarkerSize', 8, 'Color', [0.2, 0.3, 0.5]);
    
    % Add trend line
    if length(years) > 3
        coeffs = polyfit(years, yearly_performance, 1);
        x_trend = linspace(min(years), max(years), 100);
        y_trend = polyval(coeffs, x_trend);
        hold on;
        plot(x_trend, y_trend, '--', 'Color', [0.8, 0.2, 0.2], 'LineWidth', 2);
        
        % Future projection
        future_years = [2025, 2026, 2027];
        future_performance = polyval(coeffs, future_years);
        plot(future_years, future_performance, 's-', 'Color', [0.8, 0.2, 0.2], ...
            'LineWidth', 2, 'MarkerSize', 8);
    end
    
    xlabel('Publication Year');
    ylabel('Average Accuracy (%)');
    title('(E) Performance Evolution Over Time');
    grid on;
    
    % Panel F: Environmental Impact Analysis
    subplot(2, 3, 6);
    
    environments = unique(data.environment);
    environments = environments(~cellfun(@isempty, environments));
    
    env_performance = zeros(size(environments));
    env_counts = zeros(size(environments));
    
    for i = 1:length(environments)
        if ~strcmp(environments{i}, 'NaN')
            env_data = data(strcmp(data.environment, environments{i}), :);
            accuracies = env_data.accuracy(~isnan(env_data.accuracy));
            
            if ~isempty(accuracies)
                env_performance(i) = mean(accuracies) * 100;
                env_counts(i) = length(accuracies);
            end
        end
    end
    
    % Remove empty entries
    valid_idx = env_performance > 0;
    env_performance = env_performance(valid_idx);
    env_counts = env_counts(valid_idx);
    environments = environments(valid_idx);
    
    if ~isempty(env_performance)
        % Horizontal bar chart
        barh(1:length(environments), env_performance, 'FaceColor', [0.3, 0.7, 0.3], ...
            'EdgeColor', 'black', 'FaceAlpha', 0.8);
        
        % Add count labels
        for i = 1:length(environments)
            text(env_performance(i) + 1, i, sprintf('n=%d', env_counts(i)), ...
                'VerticalAlignment', 'middle', 'FontWeight', 'bold');
        end
        
        set(gca, 'YTick', 1:length(environments), 'YTickLabel', environments);
        xlabel('Average Accuracy (%)');
        title('(F) Environmental Performance Impact');
        grid on;
    end
    
    % Save figure
    sgtitle('Motion Control Performance Statistical Analysis', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Save in multiple formats
    saveas(gcf, 'fig_motion_control_matlab.png');
    saveas(gcf, 'fig_motion_control_matlab.pdf');
    saveas(gcf, 'fig_motion_control_matlab.fig');
    
    fprintf('Motion control figure saved: fig_motion_control_matlab.png/.pdf/.fig\n');
end

function perform_motion_statistical_tests(data)
    % Perform comprehensive statistical tests
    
    fprintf('\n=== Statistical Test Results ===\n');
    
    % Test 1: Algorithm family performance differences
    algorithms = {'R-CNN', 'YOLO', 'SSD', 'Hybrid'};
    groups = cell(length(algorithms), 1);
    
    for i = 1:length(algorithms)
        algo_data = data(strcmp(data.algorithm_family, algorithms{i}), :);
        groups{i} = algo_data.accuracy(~isnan(algo_data.accuracy));
    end
    
    % Remove empty groups
    non_empty = ~cellfun(@isempty, groups);
    groups = groups(non_empty);
    algorithms = algorithms(non_empty);
    
    if length(groups) > 1
        % One-way ANOVA
        group_data = [];
        group_labels = [];
        
        for i = 1:length(groups)
            group_data = [group_data; groups{i}];
            group_labels = [group_labels; repmat(i, length(groups{i}), 1)];
        end
        
        [p_anova, tbl, stats] = anova1(group_data, group_labels, 'off');
        fprintf('Algorithm ANOVA: F = %.3f, p = %.3f\n', tbl{2,5}, p_anova);
        
        % Post-hoc pairwise comparisons
        if p_anova < 0.05
            fprintf('Significant differences found. Pairwise comparisons:\n');
            for i = 1:length(groups)
                for j = i+1:length(groups)
                    [~, p_ttest] = ttest2(groups{i}, groups{j});
                    effect_size = (mean(groups{i}) - mean(groups{j})) / ...
                                 sqrt(((length(groups{i})-1)*var(groups{i}) + ...
                                      (length(groups{j})-1)*var(groups{j})) / ...
                                      (length(groups{i}) + length(groups{j}) - 2));
                    fprintf('  %s vs %s: p = %.3f, Cohen''s d = %.3f\n', ...
                           algorithms{i}, algorithms{j}, p_ttest, effect_size);
                end
            end
        end
    end
    
    % Test 2: Temporal trend analysis
    years = unique(data.year);
    years = years(~isnan(years));
    
    if length(years) > 3
        yearly_accuracy = zeros(size(years));
        for i = 1:length(years)
            year_data = data(data.year == years(i), :);
            accuracies = year_data.accuracy(~isnan(year_data.accuracy));
            if ~isempty(accuracies)
                yearly_accuracy(i) = mean(accuracies);
            end
        end
        
        % Linear regression for trend
        [coeffs, S] = polyfit(years, yearly_accuracy, 1);
        [y_fit, delta] = polyval(coeffs, years, S);
        
        % Calculate R-squared
        y_mean = mean(yearly_accuracy);
        SS_tot = sum((yearly_accuracy - y_mean).^2);
        SS_res = sum((yearly_accuracy - y_fit).^2);
        R_squared = 1 - SS_res/SS_tot;
        
        fprintf('Temporal trend: %.4f accuracy/year (R² = %.3f)\n', coeffs(1), R_squared);
        
        % Future projections
        future_years = [2025, 2026, 2027];
        future_accuracy = polyval(coeffs, future_years);
        fprintf('Projected accuracy in 2025: %.2f%%\n', future_accuracy(1)*100);
    end
    
    % Test 3: Environmental impact assessment
    environments = unique(data.environment);
    environments = environments(~cellfun(@isempty, environments));
    
    env_groups = cell(length(environments), 1);
    for i = 1:length(environments)
        if ~strcmp(environments{i}, 'NaN')
            env_data = data(strcmp(data.environment, environments{i}), :);
            env_groups{i} = env_data.accuracy(~isnan(env_data.accuracy));
        end
    end
    
    % Remove empty groups
    non_empty = ~cellfun(@isempty, env_groups);
    env_groups = env_groups(non_empty);
    environments = environments(non_empty);
    
    if length(env_groups) > 1
        % Environmental ANOVA
        env_data_combined = [];
        env_labels_combined = [];
        
        for i = 1:length(env_groups)
            env_data_combined = [env_data_combined; env_groups{i}];
            env_labels_combined = [env_labels_combined; repmat(i, length(env_groups{i}), 1)];
        end
        
        [p_env, tbl_env] = anova1(env_data_combined, env_labels_combined, 'off');
        fprintf('Environmental ANOVA: F = %.3f, p = %.3f\n', tbl_env{2,5}, p_env);
        
        % Environmental means
        fprintf('Environmental performance ranking:\n');
        env_means = zeros(size(environments));
        for i = 1:length(environments)
            env_means(i) = mean(env_groups{i}) * 100;
        end
        
        [sorted_means, sort_idx] = sort(env_means, 'descend');
        for i = 1:length(environments)
            fprintf('  %s: %.2f%% (n=%d)\n', environments{sort_idx(i)}, ...
                   sorted_means(i), length(env_groups{sort_idx(i)}));
        end
    end
end

function save_analysis_results(results)
    % Save analysis results to file
    
    % Convert to JSON-like structure for export
    results_file = 'motion_control_matlab_results.txt';
    fid = fopen(results_file, 'w');
    
    fprintf(fid, 'Motion Control Statistical Analysis Results\n');
    fprintf(fid, '==========================================\n\n');
    
    fprintf(fid, 'Analysis completed: %s\n', datestr(now));
    fprintf(fid, 'MATLAB version: %s\n', version);
    
    fclose(fid);
    fprintf('Results saved to: %s\n', results_file);
end

%% Helper functions

function export_figure_for_latex(fig_handle, filename)
    % Export figure in multiple formats suitable for LaTeX
    
    % High-resolution PNG
    print(fig_handle, [filename '.png'], '-dpng', '-r300');
    
    % PDF for LaTeX
    print(fig_handle, [filename '.pdf'], '-dpdf', '-r300');
    
    % EPS for compatibility
    print(fig_handle, [filename '.eps'], '-depsc', '-r300');
    
    fprintf('Figure exported: %s.png/.pdf/.eps\n', filename);
end

function data_summary = get_motion_data_summary(data)
    % Generate comprehensive data summary
    
    data_summary = struct();
    
    % Basic statistics
    data_summary.total_studies = height(data);
    data_summary.year_range = [min(data.year), max(data.year)];
    
    % Performance statistics
    data_summary.accuracy_stats = struct();
    accuracies = data.accuracy(~isnan(data.accuracy));
    if ~isempty(accuracies)
        data_summary.accuracy_stats.mean = mean(accuracies);
        data_summary.accuracy_stats.std = std(accuracies);
        data_summary.accuracy_stats.median = median(accuracies);
        data_summary.accuracy_stats.range = [min(accuracies), max(accuracies)];
    end
    
    % Algorithm distribution
    algorithms = unique(data.algorithm_family);
    algorithms = algorithms(~cellfun(@isempty, algorithms));
    
    data_summary.algorithm_distribution = struct();
    for i = 1:length(algorithms)
        if ~strcmp(algorithms{i}, 'NaN')
            count = sum(strcmp(data.algorithm_family, algorithms{i}));
            field_name = matlab.lang.makeValidName(algorithms{i});
            data_summary.algorithm_distribution.(field_name) = count;
        end
    end
    
    fprintf('Data summary generated with %d studies\n', data_summary.total_studies);
    
    % Run the analysis if called as main function
    if nargin == 0
        % Call main analysis function here if needed
        fprintf('Motion control analysis completed.\n');
    end
end