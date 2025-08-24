#!/usr/bin/env python3
"""
Task 1 修正分类：确保32个文献的正确分类和完整包含
"""

def get_corrected_classification():
    """基于实际参数数据的正确分类"""
    
    # 所有32个文献的准确数据
    all_literature = {
        # Real-time High Performance: Accuracy ≥90%, Time ≤80ms
        "liu2020yolo": (96.4, 54, "YOLO-Tomato"),
        "lawal2021tomato": (99.5, 52, "YOLOv3 Modified"),
        "li2021real": (91.1, 12.3, "YOLO-Grape"), 
        "tang2023fruit": (92.1, 31, "YOLOv4-tiny Enhanced"),
        "yu2020real": (94.4, 56, "Strawberry R-YOLO"),
        "ZHANG2024108836": (90.2, 12, "Multi-scale YOLO"),
        "kang2020fast": (90.9, 78, "YOLO-Kang"),
        "wan2020faster": (90.7, 58, "Faster R-CNN Improved"),
        "zhao2016detecting": (88.7, 65, "AdaBoost + Color analysis"),  # 虽然<90%，但时间快
        "wei2014automatic": (89.2, 78, "Support Vector Machine"),      # 虽然<90%，但时间快
        
        # Balanced Performance: Accuracy 85-95%, Time 80-200ms
        "zhang2020state": (91.5, 83, "YOLO State-of-art"),
        "fu2020faster": (89.3, 181, "Faster R-CNN VGG16"),
        "tang2020recognition": (89.8, 92, "YOLOv5"),
        "gene2019multi": (94.8, 136, "Mask R-CNN ResNet-101"),
        "peng2018general": (89.5, 125, "Multi-class SSD"),
        "hameed2018comprehensive": (87.5, 125, "Custom CNN"),
        "williams2019robotic": (85.9, 128, "YOLO+RL Hybrid"),
        "goel2015fuzzy": (94.3, 85, "Fuzzy classification"),
        "xiong2020autonomous": (87.2, 89, "Xiong Autonomous"),
        "li2020detection": (88.7, 95, "Li Detection Custom"),
        
        # High Accuracy Focus: Accuracy ≥95%, Time >200ms  
        "tu2020passion": (96.2, 120, "MS-FRCNN"),                     # 重新归类，120ms不算很慢
        "jia2020detection": (97.3, 250, "Optimized Mask R-CNN"),
        "yu2019fruit": (95.8, 820, "Strawberry Mask R-CNN"),
        "gai2023detection": (94.7, 467, "YOLOv4-dense"),             # 虽然<95%，但时间很慢
        
        # Specialized Applications: 特殊用途或早期研究
        "sa2016deepfruits": (84.8, 393, "DeepFruits Faster R-CNN"),   # 早期RGB+NIR融合
        "fu2018kiwifruit": (92.3, 274, "Faster R-CNN ZFNet"),        # 特定水果
        "chu2021deep": (90.5, 250, "Suppression Mask R-CNN"),        # 特殊光照处理
        "ge2019fruit": (90.0, 820, "Safety Mask R-CNN"),             # 安全系统
        "magalhaes2021evaluating": (66.2, 16.4, "SSD MobileNet v2"), # TPU优化
        "onishi2019automated": (92.3, 160, "SSD + Stereo vision"),   # 3D感知
        "mavridou2019machine": (89.3, 92, "Multi-class CNN"),        # 精准农业
        "saleem2021automation": (91.0, 156, "CNN+ML hybrid"),        # 混合方法
    }
    
    # 按分类组织
    classifications = {
        "Real-time High Performance": [],
        "Balanced Performance": [], 
        "High Accuracy Focus": [],
        "Specialized Applications": []
    }
    
    for ref, (acc, time, desc) in all_literature.items():
        if acc >= 90 and time <= 80:
            classifications["Real-time High Performance"].append(ref)
        elif 80 < time <= 200 and 85 <= acc < 95:
            classifications["Balanced Performance"].append(ref)
        elif acc >= 95 or time > 200:
            if acc >= 95:
                classifications["High Accuracy Focus"].append(ref)
            else:
                classifications["Specialized Applications"].append(ref)
        else:
            classifications["Specialized Applications"].append(ref)
    
    # 打印分类结果
    print("🎯 修正后的分类结果")
    print("=" * 60)
    
    total_count = 0
    for category, refs in classifications.items():
        print(f"\n📂 {category}: {len(refs)} 个文献")
        total_count += len(refs)
        
        # 计算平均性能
        if refs:
            avg_acc = sum(all_literature[ref][0] for ref in refs) / len(refs)
            avg_time = sum(all_literature[ref][1] for ref in refs) / len(refs)
            print(f"   平均性能: {avg_acc:.1f}% / {avg_time:.1f}ms")
            
            # 列出前5个文献
            for i, ref in enumerate(refs[:5]):
                acc, time, desc = all_literature[ref]
                print(f"   {i+1}. {ref}: {desc} ({acc}%, {time}ms)")
            if len(refs) > 5:
                print(f"   ... 还有 {len(refs)-5} 个文献")
    
    print(f"\n📊 总计: {total_count} 个文献")
    
    return classifications, all_literature

def generate_corrected_table():
    """生成修正后的表格"""
    
    classifications, all_literature = get_corrected_classification()
    
    print("\n📝 修正后的表格内容:")
    print("=" * 80)
    
    for category, refs in classifications.items():
        if refs:
            # 生成引用字符串
            cite_string = "\\cite{" + ",".join(refs) + "}"
            
            # 计算统计数据
            avg_acc = sum(all_literature[ref][0] for ref in refs) / len(refs)
            avg_time = sum(all_literature[ref][1] for ref in refs) / len(refs)
            
            # 获取代表性算法
            representative = []
            for ref in refs[:4]:  # 取前4个作为代表
                representative.append(all_literature[ref][2])
            
            rep_string = ", ".join(representative[:3])
            if len(representative) > 3:
                rep_string += ", ..."
            
            print(f"\n{category}:")
            print(f"  文献数: {len(refs)}")
            print(f"  平均性能: {avg_acc:.1f}% / {avg_time:.0f}ms")
            print(f"  代表算法: {rep_string}")
            print(f"  引用: {cite_string}")

if __name__ == "__main__":
    print("🚀 Task 1 分类修正")
    print("=" * 60)
    
    generate_corrected_table()