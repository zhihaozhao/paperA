# Method 3: R ggplot2 Professional Publication Graphics
# 适用于IEEE期刊的publication-quality图表
# 安装依赖: install.packages(c("ggplot2", "dplyr", "gridExtra", "RColorBrewer"))

library(ggplot2)
library(dplyr)
library(gridExtra)
library(RColorBrewer)

# IEEE IoTJ color palette (colorblind-friendly)
ieee_colors <- c("#2E86AB", "#E84855", "#3CB371", "#DC143C")

# ============ Figure 3: D3 Cross-Domain Performance ============

# D3 data preparation
d3_data <- data.frame(
  Model = rep(c("Enhanced", "CNN", "BiLSTM", "Conformer-lite"), 2),
  Protocol = rep(c("LOSO", "LORO"), each = 4),
  F1_Score = c(0.830, 0.842, 0.803, 0.403,  # LOSO
               0.830, 0.796, 0.789, 0.841), # LORO
  Std_Error = c(0.001, 0.025, 0.022, 0.386,  # LOSO
                0.001, 0.097, 0.044, 0.040)  # LORO
)

# Create Figure 3
figure3 <- ggplot(d3_data, aes(x = Model, y = F1_Score, fill = Protocol)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, alpha = 0.8) +
  geom_errorbar(aes(ymin = F1_Score - Std_Error, ymax = F1_Score + Std_Error),
                position = position_dodge(width = 0.8), width = 0.25, size = 0.5) +
  
  # IEEE IoTJ styling
  scale_fill_manual(values = c("LOSO" = "#2E86AB", "LORO" = "#E84855")) +
  scale_y_continuous(limits = c(0, 1.0), breaks = seq(0, 1, 0.1)) +
  
  # Labels and titles
  labs(title = "Cross-Domain Generalization Performance",
       x = "Model Architecture", 
       y = "Macro F1 Score",
       fill = "Protocol") +
  
  # IEEE IoTJ theme
  theme_minimal() +
  theme(text = element_text(family = "Times", size = 10),
        plot.title = element_text(size = 12, hjust = 0.5),
        axis.title = element_text(size = 10),
        axis.text = element_text(size = 8),
        legend.text = element_text(size = 9),
        legend.position = "top",
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank()) +
  
  # Add value labels on bars
  geom_text(aes(label = sprintf("%.3f±%.3f", F1_Score, Std_Error)),
            position = position_dodge(width = 0.8), 
            vjust = -0.5, size = 2.5)

# Save Figure 3
ggsave("figure3_d3_cross_domain_ggplot2.pdf", figure3, 
       width = 17.1, height = 10, units = "cm", dpi = 300)

# ============ Figure 4: D4 Label Efficiency Curve ============

# D4 data preparation
d4_data <- data.frame(
  Label_Percent = c(1.0, 5.0, 10.0, 20.0, 100.0),
  F1_Score = c(0.455, 0.780, 0.730, 0.821, 0.833),
  Std_Error = c(0.050, 0.016, 0.104, 0.003, 0.000),
  Seeds = c(12, 6, 5, 5, 5)
)

# Create Figure 4
figure4 <- ggplot(d4_data, aes(x = Label_Percent, y = F1_Score)) +
  
  # Efficiency range background
  annotate("rect", xmin = 0, xmax = 20, ymin = 0, ymax = 1, 
           alpha = 0.2, fill = "#90EE90") +
  annotate("text", x = 10, y = 0.95, label = "Efficient Range (≤20%)",
           size = 3, color = "#2E8B57") +
  
  # Error ribbon
  geom_ribbon(aes(ymin = F1_Score - Std_Error, ymax = F1_Score + Std_Error),
              alpha = 0.3, fill = "#2E86AB") +
  
  # Main efficiency curve
  geom_line(size = 2, color = "#2E86AB") +
  geom_point(size = 3, color = "#2E86AB", shape = 21, fill = "white", stroke = 2) +
  
  # Target lines
  geom_hline(yintercept = 0.80, linetype = "dashed", color = "#FF6B6B", size = 1.5) +
  geom_hline(yintercept = 0.90, linetype = "dotted", color = "#FFA500", size = 1) +
  
  # Key achievement annotation
  annotate("text", x = 35, y = 0.87, 
           label = "Key Achievement:\n82.1% F1 @ 20% Labels",
           size = 3.5, color = "#FF6B6B", 
           hjust = 0.5, vjust = 0.5,
           fontface = "bold") +
  annotate("segment", x = 30, y = 0.85, xend = 20, yend = 0.821,
           arrow = arrow(length = unit(0.3, "cm")), 
           color = "#FF6B6B", size = 1.2) +
  
  # Styling
  scale_x_continuous(limits = c(0, 105), breaks = c(1, 5, 10, 20, 50, 100)) +
  scale_y_continuous(limits = c(0, 1.0), breaks = seq(0, 1, 0.1)) +
  
  labs(title = "Sim2Real Label Efficiency Breakthrough",
       x = "Label Ratio (%)",
       y = "Macro F1 Score") +
  
  # IEEE IoTJ theme
  theme_minimal() +
  theme(text = element_text(family = "Times", size = 10),
        plot.title = element_text(size = 12, hjust = 0.5),
        axis.title = element_text(size = 10),
        axis.text = element_text(size = 8),
        panel.grid.minor = element_blank()) +
  
  # Add data point labels
  geom_text(aes(label = sprintf("%.3f", F1_Score)), 
            vjust = -1.5, size = 2.5, color = "#2E86AB")

# Save Figure 4
ggsave("figure4_d4_label_efficiency_ggplot2.pdf", figure4,
       width = 17.1, height = 12, units = "cm", dpi = 300)

# ============ Summary Statistics ============
cat("\n📊 R ggplot2 Method Summary:\n")
cat("✓ Figure 3: Cross-domain performance with error bars\n")
cat("✓ Figure 4: Label efficiency with achievement annotation\n") 
cat("✓ IEEE IoTJ compliant: 300 DPI, Times font, proper sizing\n")
cat("✓ Professional quality: Color-blind friendly, clean design\n")
cat("\n🎯 Key Results Visualized:\n")
cat("• Enhanced consistency: 83.0±0.1% F1 across protocols\n")
cat("• Label efficiency breakthrough: 82.1% F1 @ 20% labels\n")
cat("• Cost reduction: 80% labeling savings demonstrated\n")

cat("\n💡 R ggplot2 Advantages:\n")
cat("• Publication-quality output\n")
cat("• IEEE journal compliance\n") 
cat("• Professional statistical graphics\n")
cat("• Easy customization and refinement\n")