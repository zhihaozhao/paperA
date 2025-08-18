#!/usr/bin/env python3
"""
Method 1: Pure Python ASCII Art Plotting (No external dependencies)
适用于快速预览和debug，无需安装任何包
"""

import csv
import math

def ascii_bar_chart(data, title, max_width=60):
    """Create ASCII bar chart."""
    print(f"\n{title}")
    print("=" * len(title))
    
    # Find max value for scaling
    max_val = max(data.values())
    
    for label, value in data.items():
        # Calculate bar width
        bar_width = int((value / max_val) * max_width)
        bar = "█" * bar_width
        
        # Add value label
        print(f"{label:15} │{bar:<{max_width}} {value:.3f}")
    
    print(f"{'':15} └{'─' * max_width}┘")

def ascii_line_chart(x_data, y_data, title, width=60, height=20):
    """Create ASCII line chart."""
    print(f"\n{title}")
    print("=" * len(title))
    
    # Normalize data
    x_min, x_max = min(x_data), max(x_data)
    y_min, y_max = min(y_data), max(y_data)
    
    # Create canvas
    canvas = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Plot points
    for i in range(len(x_data)):
        x_pos = int((x_data[i] - x_min) / (x_max - x_min) * (width - 1))
        y_pos = height - 1 - int((y_data[i] - y_min) / (y_max - y_min) * (height - 1))
        
        if 0 <= x_pos < width and 0 <= y_pos < height:
            canvas[y_pos][x_pos] = '●'
            
            # Connect with lines
            if i > 0:
                prev_x = int((x_data[i-1] - x_min) / (x_max - x_min) * (width - 1))
                prev_y = height - 1 - int((y_data[i-1] - y_min) / (y_max - y_min) * (height - 1))
                
                # Simple line drawing
                steps = max(abs(x_pos - prev_x), abs(y_pos - prev_y))
                if steps > 0:
                    for step in range(steps):
                        interp_x = prev_x + (x_pos - prev_x) * step // steps
                        interp_y = prev_y + (y_pos - prev_y) * step // steps
                        if 0 <= interp_x < width and 0 <= interp_y < height:
                            canvas[interp_y][interp_x] = '─' if canvas[interp_y][interp_x] == ' ' else '●'
    
    # Print canvas
    print(f"  {y_max:.3f} ┤")
    for row in canvas:
        print(f"         │{''.join(row)}")
    print(f"  {y_min:.3f} └{'─' * width}┐")
    print(f"         {x_min:.1f}{'':>{width-10}}{x_max:.1f}")

def main():
    print("📊 Figure 3: D3 Cross-Domain Performance (ASCII Method)")
    
    # Figure 3 data
    loso_data = {
        'Enhanced': 0.830,
        'CNN': 0.842, 
        'BiLSTM': 0.803,
        'Conformer': 0.403
    }
    
    loro_data = {
        'Enhanced': 0.830,
        'CNN': 0.796,
        'BiLSTM': 0.789,
        'Conformer': 0.841
    }
    
    ascii_bar_chart(loso_data, "LOSO Protocol Performance")
    ascii_bar_chart(loro_data, "LORO Protocol Performance")
    
    print("\n📊 Figure 4: D4 Label Efficiency Curve (ASCII Method)")
    
    # Figure 4 data
    x_labels = [1.0, 5.0, 10.0, 20.0, 100.0]
    y_values = [0.455, 0.780, 0.730, 0.821, 0.833]
    
    ascii_line_chart(x_labels, y_values, "Enhanced Fine-tune Label Efficiency")
    
    print(f"\n🎯 Key Findings:")
    print(f"• Enhanced model: 83.0% F1 consistent across LOSO/LORO")
    print(f"• Label efficiency: 82.1% F1 @ 20% labels") 
    print(f"• Cost reduction: 80% labeling cost savings")
    
    print(f"\n💡 ASCII Method Advantages:")
    print(f"• No dependencies required")
    print(f"• Quick data visualization")
    print(f"• Platform independent")
    print(f"• Good for initial review")

if __name__ == "__main__":
    main()