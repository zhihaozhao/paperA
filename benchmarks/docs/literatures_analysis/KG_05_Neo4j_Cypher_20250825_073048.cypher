// Agricultural Robotics Literature Knowledge Graph
// Generated: 20250825_073048
// Total Papers: 159

// Create constraints
CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT algorithm_id IF NOT EXISTS FOR (a:Algorithm) REQUIRE a.id IS UNIQUE;

// Create Paper nodes (sample)
CREATE (p0:Paper {id: 'PAPER_0001', title: 'DeepFruits: A Fruit Detection System Using Deep Ne', year: 2016, citations: 662});
CREATE (p1:Paper {id: 'PAPER_0002', title: 'Harvesting Robots for High-value Crops: State-of-t', year: 2014, citations: 388});
CREATE (p2:Paper {id: 'PAPER_0003', title: 'Sensors and systems for fruit detection and locali', year: 2015, citations: 364});
CREATE (p3:Paper {id: 'PAPER_0004', title: 'Fruit detection for strawberry harvesting robot in', year: 2019, citations: 373});
CREATE (p4:Paper {id: 'PAPER_0005', title: 'Deep Count: Fruit Counting Based on Deep Simulated', year: 2017, citations: 332});
CREATE (p5:Paper {id: 'PAPER_0006', title: 'Recognition and Localization Methods for Vision-Ba', year: 2020, citations: 298});
CREATE (p6:Paper {id: 'PAPER_0007', title: 'Faster R-CNN for multi-class fruit detection using', year: 2020, citations: 258});
CREATE (p7:Paper {id: 'PAPER_0008', title: 'A review of key techniques of vision-based control', year: 2016, citations: 241});
CREATE (p8:Paper {id: 'PAPER_0009', title: 'Research and development in agricultural robotics:', year: 2018, citations: 219});
CREATE (p9:Paper {id: 'PAPER_0010', title: 'YOLO-Tomato: A Robust Algorithm for Tomato Detecti', year: 2020, citations: 223});

// Create Algorithm nodes
CREATE (a0:Algorithm {id: 'Faster_RCNN', name: 'Faster_RCNN', usage_count: 7});
CREATE (a1:Algorithm {id: 'RCNN', name: 'RCNN', usage_count: 7});
CREATE (a2:Algorithm {id: 'Traditional', name: 'Traditional', usage_count: 141});
CREATE (a3:Algorithm {id: 'ResNet', name: 'ResNet', usage_count: 2});
CREATE (a4:Algorithm {id: 'Inception', name: 'Inception', usage_count: 1});
CREATE (a5:Algorithm {id: 'YOLO', name: 'YOLO', usage_count: 8});
CREATE (a6:Algorithm {id: 'YOLOv3', name: 'YOLOv3', usage_count: 4});
CREATE (a7:Algorithm {id: 'YOLOv4', name: 'YOLOv4', usage_count: 4});
CREATE (a8:Algorithm {id: 'MobileNet', name: 'MobileNet', usage_count: 1});
CREATE (a9:Algorithm {id: 'YOLOv5', name: 'YOLOv5', usage_count: 1});
