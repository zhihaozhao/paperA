#!/usr/bin/env python3
"""
Task 1: 合并视觉文献到表4 (tab:algorithm_comparison)
分析并去重来自表5、表6、表11的视觉相关文献
"""

import re
from collections import defaultdict

def extract_table_data():
    """提取各表格中的视觉文献数据"""
    
    # 表4当前数据 (tab:algorithm_comparison)
    table4_current = [
        ("sa2016deepfruits", "R-CNN Series", "DeepFruits (Faster R-CNN)", "84.8", "393", "n=450", "Outdoor/Greenhouse", "Early RGB+NIR fusion"),
        ("wan2020faster", "R-CNN Series", "Faster R-CNN (Improved)", "90.7", "58", "n=1200", "Outdoor", "Optimized conv/pooling"),
        ("fu2020faster", "R-CNN Series", "Faster R-CNN (VGG16)", "89.3", "181", "n=800", "Outdoor", "RGB+depth filtering"),
        ("gene2019multi", "R-CNN Series", "Mask R-CNN (ResNet-101)", "94.8", "136", "n=1000", "Outdoor", "Instance segmentation"),
        ("tu2020passion", "R-CNN Series", "MS-FRCNN", "96.2", "120", "n=850", "Orchard", "Multi-scale detection"),
        ("fu2018kiwifruit", "R-CNN Series", "Faster R-CNN (ZFNet)", "92.3", "274", "n=600", "Outdoor", "Kiwifruit-specific"),
        ("kuznetsova2020using", "YOLO Series", "YOLOv3-based", "89.1", "84", "n=750", "Outdoor", "Multi-scale detection"),
        ("liu2020yolo", "YOLO Series", "YOLO-Tomato", "96.4", "54", "n=950", "Greenhouse", "Robust tomato detection"),
        ("lawal2021tomato", "YOLO Series", "YOLOv3 (Modified)", "99.5", "52", "n=800", "Greenhouse", "SPP + Mish activation"),
        ("gai2023detection", "YOLO Series", "YOLOv4-dense", "94.7", "467", "n=1100", "Orchard", "DenseNet backbone"),
        ("li2021real", "YOLO Series", "YOLO-Grape", "91.1", "12.3", "n=680", "Vineyard", "Depthwise separable conv"),
        ("tang2023fruit", "YOLO Series", "YOLOv4-tiny (Enhanced)", "92.1", "31", "n=720", "Orchard", "K-means++ clustering"),
        ("yu2019fruit", "Mask R-CNN Variants", "Strawberry Mask R-CNN", "95.8", "820", "n=1000", "Field", "Instance segmentation"),
        ("jia2020detection", "Mask R-CNN Variants", "Optimized Mask R-CNN", "97.3", "250", "n=1020", "Orchard", "ResNet+DenseNet"),
        ("chu2021deep", "Mask R-CNN Variants", "Suppression Mask R-CNN", "90.5", "250", "n=1500", "Outdoor", "Robust lighting handling"),
        ("ge2019fruit", "Mask R-CNN Variants", "Safety Mask R-CNN", "90.0", "820", "n=450", "Table-top", "3D localization + safety"),
        ("magalhaes2021evaluating", "SSD-based Methods", "SSD MobileNet v2", "66.2", "16.4", "n=400", "Greenhouse", "TPU-compatible"),
        ("onishi2019automated", "SSD-based Methods", "SSD + Stereo vision", "92.3", "160", "n=350", "V-shaped", "Deep learning + 3D"),
        ("peng2018general", "SSD-based Methods", "Multi-class SSD", "89.5", "125", "n=520", "General", "Multi-fruit detection"),
        ("hameed2018comprehensive", "CNN Classifiers", "Custom CNN", "87.5", "125", "n=520", "Laboratory", "Basic classification"),
        ("mavridou2019machine", "CNN Classifiers", "Multi-class CNN", "89.3", "92", "n=680", "Greenhouse", "Precision agriculture"),
        ("saleem2021automation", "CNN Classifiers", "CNN+ML hybrid", "91.0", "156", "n=420", "Natural", "Deep learning integration"),
        ("goel2015fuzzy", "Traditional Methods", "Fuzzy classification", "94.3", "85", "n=300", "Pre-harvest", "Rule-based learning"),
        ("zhao2016detecting", "Traditional Methods", "AdaBoost + Color analysis", "88.7", "65", "n=480", "Greenhouse", "Ensemble learning"),
        ("wei2014automatic", "Traditional Methods", "Support Vector Machine", "89.2", "78", "n=350", "Field", "Feature-based classification"),
    ]
    
    # 表5视觉文献 (tab:literature_support_summary) - 前9个
    table5_vision = [
        ("sa2016deepfruits", "R-CNN", "DeepFruits", "84.8", "393", "n=450", "Figure 4(a,c)", "R-CNN precision advantage"),
        ("wan2020faster", "R-CNN", "Faster", "90.7", "58", "n=1200", "Figure 4(a,c)", "R-CNN processing improvement"),
        ("gene2019multi", "YOLO", "YOLOv4", "91.2", "84", "n=1100", "Figure 4(a,b,d)", "YOLO optimal balance"),
        ("tang2020recognition", "YOLO", "YOLOv5", "89.8", "92", "n=750", "Figure 4(a,b,d)", "YOLO commercial viability"),
        ("li2020detection", "YOLO", "Custom", "88.7", "95", "n=600", "Figure 4(b,c)", "YOLO real-time performance"),
        ("gai2023detection", "YOLO", "YOLOv8", "92.1", "71", "n=1300", "Figure 4(a,c,d)", "YOLO latest advancement"),
        ("zhang2020state", "YOLO", "YOLOv9", "91.5", "83", "n=1150", "Figure 4(b,c)", "YOLO continued evolution"),
        ("chu2021deep", "R-CNN", "Mask", "87.8", "94", "n=950", "Figure 4(a,b)", "R-CNN segmentation capability"),
        ("williams2019robotic", "Hybrid", "YOLO+RL", "85.9", "128", "n=820", "Figure 4(a,b,c,d)", "Hybrid approach potential"),
    ]
    
    # 表11文献 (tab:figure4_support) - 12个
    table11_vision = [
        ("sa2016deepfruits", "R-CNN", "84.8", "393", "n=450", "Fig 4(a,c)"),
        ("wan2020faster", "R-CNN", "90.7", "58", "n=1200", "Fig 4(a,c)"),
        ("fu2020faster", "R-CNN", "88.5", "125", "n=800", "Fig 4(a,c)"),
        ("xiong2020autonomous", "R-CNN", "87.2", "89", "n=650", "Fig 4(a,c)"),
        ("yu2019fruit", "YOLO", "91.2", "84", "n=1100", "Fig 4(a,b,d)"),
        ("tang2020recognition", "YOLO", "89.8", "92", "n=750", "Fig 4(a,b,d)"),
        ("kang2020fast", "YOLO", "90.9", "78", "n=950", "Fig 4(a,b,d)"),
        ("li2020detection", "YOLO", "88.7", "95", "n=600", "Fig 4(b,c)"),
        ("gai2023detection", "YOLO", "92.1", "71", "n=1300", "Fig 4(a,c,d)"),
        ("zhang2020state", "YOLO", "91.5", "83", "n=1150", "Fig 4(b,c)"),
        ("chu2021deep", "R-CNN", "87.8", "94", "n=950", "Fig 4(a,b)"),
        ("williams2019robotic", "Hybrid", "85.9", "128", "n=820", "Fig 4(a,b,c,d)"),
    ]
    
    return table4_current, table5_vision, table11_vision

def analyze_duplicates():
    """分析重复文献"""
    table4_current, table5_vision, table11_vision = extract_table_data()
    
    print("🔍 重复文献分析")
    print("=" * 60)
    
    # 提取表4现有的引用
    table4_refs = set([item[0] for item in table4_current])
    print(f"表4当前文献数: {len(table4_refs)}")
    
    # 分析表5重复
    table5_refs = set([item[0] for item in table5_vision])
    table5_duplicates = table5_refs.intersection(table4_refs)
    table5_new = table5_refs - table4_refs
    
    print(f"表5视觉文献数: {len(table5_refs)}")
    print(f"表5与表4重复: {len(table5_duplicates)} - {table5_duplicates}")
    print(f"表5新增文献: {len(table5_new)} - {table5_new}")
    
    # 分析表11重复
    table11_refs = set([item[0] for item in table11_vision])
    table11_duplicates = table11_refs.intersection(table4_refs)
    table11_new = table11_refs - table4_refs
    
    print(f"表11视觉文献数: {len(table11_refs)}")
    print(f"表11与表4重复: {len(table11_duplicates)} - {table11_duplicates}")
    print(f"表11新增文献: {len(table11_new)} - {table11_new}")
    
    # 总新增文献
    all_new = table5_new.union(table11_new)
    print(f"\n📊 总计新增文献: {len(all_new)} - {all_new}")
    
    return table4_current, table5_vision, table11_vision, all_new

def identify_missing_literature():
    """识别需要从PDF中提取的文献"""
    
    # 基于Table4RCNN和Table5YOLO文件夹识别更多文献
    rcnn_papers = [
        "Multi-modal deep learning for Fuji apple detection using RGB-D cameras",
        "Multi-class fruit-on-plant detection for apple in SNAP system using Faster R-CNN", 
        "Kiwifruit detection in field images using Faster R-CNN with ZFNet",
        "Intact detection of highly occluded immature tomatoes using deep learning",
        "Detection and segmentation of overlapped fruits based on optimized mask R-CNN"
    ]
    
    yolo_papers = [
        "Using YOLOv3 Algorithm with Pre- and Post-Processing for Apple Detection",
        "Towards an Efficient Tomato Harvesting Robot 3D Perception",
        "Single-Shot Convolution Neural Networks for Real-Time Fruit Detection",
        "Real-Time Visual Localization of the Picking Points for a Ridge-Planting Strawberry",
        "Multi-scale feature adaptive fusion model for real-time detection"
    ]
    
    print("\n📚 需要从PDF提取数据的文献:")
    print("R-CNN相关:")
    for i, paper in enumerate(rcnn_papers, 1):
        print(f"  {i}. {paper}")
    
    print("YOLO相关:")
    for i, paper in enumerate(yolo_papers, 1):
        print(f"  {i}. {paper}")
    
    return rcnn_papers + yolo_papers

if __name__ == "__main__":
    print("🚀 Task 1: 视觉文献合并分析")
    print("=" * 60)
    
    # 分析重复文献
    table4_current, table5_vision, table11_vision, new_refs = analyze_duplicates()
    
    # 识别需要补充的文献
    missing_papers = identify_missing_literature()
    
    print(f"\n✅ 分析完成")
    print(f"当前表4文献: {len(table4_current)}")
    print(f"需要新增文献: {len(new_refs)}")
    print(f"需要从PDF提取: {len(missing_papers)}")
    print(f"预计合并后总数: {len(table4_current) + len(new_refs) + len(missing_papers)}")