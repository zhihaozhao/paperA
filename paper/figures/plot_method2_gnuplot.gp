# Method 2: Gnuplot Professional Scientific Plotting
# 适用于高质量IEEE期刊图表，专业科学绘图工具

# Set terminal for IEEE IoTJ quality output
set terminal pdfcairo enhanced color font "Times-Roman,10" size 17.1cm,10cm
set output "figure3_d3_cross_domain_gnuplot.pdf"

# Figure 3: D3 Cross-Domain Performance
set title "Cross-Domain Generalization Performance" font "Times-Roman,12"
set xlabel "Model Architecture" font "Times-Roman,10"
set ylabel "Macro F1 Score" font "Times-Roman,10"

# Set style
set style data histograms
set style histogram clustered gap 1
set style fill solid 0.8 border lt -1
set boxwidth 0.8
set grid ytics alpha 0.3

# Set colors (IEEE IoTJ colorblind-friendly)
set palette defined (1 '#2E86AB', 2 '#E84855', 3 '#3CB371', 4 '#DC143C')

# Set y-axis range
set yrange [0:1.0]
set ytics 0.1

# Set x-axis labels
set xtics ("Enhanced" 1, "CNN" 2, "BiLSTM" 3, "Conformer" 4)

# Plot data with error bars
plot 'figure3_data_gnuplot.dat' using 2:3:xtic(1) title "LOSO" linecolor rgb "#2E86AB" with histograms, \
     '' using 4:5 title "LORO" linecolor rgb "#E84855" with histograms

# Generate second figure
set output "figure4_d4_label_efficiency_gnuplot.pdf"
set size 17.1cm,12cm

set title "Sim2Real Label Efficiency Breakthrough" font "Times-Roman,12"
set xlabel "Label Ratio (%)" font "Times-Roman,10"
set ylabel "Macro F1 Score" font "Times-Roman,10"

# Reset style for line plot
unset style
set style line 1 linecolor rgb "#2E86AB" linewidth 3 pointtype 7 pointsize 1.5
set style line 2 linecolor rgb "#FF6B6B" linewidth 2 linetype 2
set style line 3 linecolor rgb "#FFA500" linewidth 1.5 linetype 3

# Set ranges
set xrange [0:105]
set yrange [0:1.0]
set grid

# Add target lines
set arrow from 0,0.80 to 105,0.80 nohead linestyle 2
set label "Target: 80% F1" at 50,0.82 textcolor rgb "#FF6B6B"

# Add key achievement annotation
set arrow from 30,0.85 to 20,0.821 head filled linestyle 2
set label "Key Achievement:\n82.1% F1 @ 20% Labels" at 35,0.87 textcolor rgb "#FF6B6B"

# Plot efficiency curve with error bars
plot 'figure4_data_gnuplot.dat' using 1:2:3 with yerrorbars linestyle 1 title "Enhanced Fine-tune", \
     '' using 1:2 with lines linestyle 1 notitle