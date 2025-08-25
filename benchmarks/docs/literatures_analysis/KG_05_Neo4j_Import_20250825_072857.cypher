// Agricultural Robotics Literature Knowledge Graph
// Generated: 20250825_072857

// Create Paper Nodes
CREATE (p:Paper {id: 'PAPER_0001_DeepFruits_A_Fruit_Detection_System_Using_Deep_Ne', title: 'DeepFruits: A Fruit Detection System Using Deep Ne...', year: 2016, citations: 662});
CREATE (p:Paper {id: 'PAPER_0002_Harvesting_Robots_for_High_value_Crops_State_of_t', title: 'Harvesting Robots for High-value Crops: State-of-t...', year: 2014, citations: 388});
CREATE (p:Paper {id: 'PAPER_0003_Sensors_and_systems_for_fruit_detection_and_locali', title: 'Sensors and systems for fruit detection and locali...', year: 2015, citations: 364});
CREATE (p:Paper {id: 'PAPER_0004_Fruit_detection_for_strawberry_harvesting_robot_in', title: 'Fruit detection for strawberry harvesting robot in...', year: 2019, citations: 373});
CREATE (p:Paper {id: 'PAPER_0005_Deep_Count_Fruit_Counting_Based_on_Deep_Simulated', title: 'Deep Count: Fruit Counting Based on Deep Simulated...', year: 2017, citations: 332});

// Create Algorithm Nodes
CREATE (a:Algorithm {id: 'ALGORITHM_0001_Faster_RCNN', name: 'Faster_RCNN', category: 'Deep_Learning', usage_count: 1});
CREATE (a:Algorithm {id: 'ALGORITHM_0002_Traditional_CV', name: 'Traditional_CV', category: 'Classical_Computer_Vision', usage_count: 1});
CREATE (a:Algorithm {id: 'ALGORITHM_0003_Traditional_CV', name: 'Traditional_CV', category: 'Classical_Computer_Vision', usage_count: 1});
CREATE (a:Algorithm {id: 'ALGORITHM_0004_Mask_Rcnn', name: 'Mask_Rcnn', category: 'Other', usage_count: 1});
CREATE (a:Algorithm {id: 'ALGORITHM_0005_ResNet', name: 'ResNet', category: 'Deep_Learning', usage_count: 1});
CREATE (a:Algorithm {id: 'ALGORITHM_0006_ResNet', name: 'ResNet', category: 'Deep_Learning', usage_count: 1});
CREATE (a:Algorithm {id: 'ALGORITHM_0007_Traditional_CV', name: 'Traditional_CV', category: 'Classical_Computer_Vision', usage_count: 1});

// Create Relationships (Sample)
MATCH (s {id: 'PAPER_0001_DeepFruits_A_Fruit_Detection_System_Using_Deep_Ne'}), (t {id: 'AUTHOR_0001_Sa'}) CREATE (s)-[:AUTHORED_BY]->(t);
MATCH (s {id: 'PAPER_0001_DeepFruits_A_Fruit_Detection_System_Using_Deep_Ne'}), (t {id: 'AUTHOR_0002_Ge'}) CREATE (s)-[:AUTHORED_BY]->(t);
MATCH (s {id: 'PAPER_0001_DeepFruits_A_Fruit_Detection_System_Using_Deep_Ne'}), (t {id: 'AUTHOR_0003_ZY'}) CREATE (s)-[:AUTHORED_BY]->(t);
MATCH (s {id: 'PAPER_0001_DeepFruits_A_Fruit_Detection_System_Using_Deep_Ne'}), (t {id: 'AUTHOR_0004_Dayoub'}) CREATE (s)-[:AUTHORED_BY]->(t);
MATCH (s {id: 'PAPER_0001_DeepFruits_A_Fruit_Detection_System_Using_Deep_Ne'}), (t {id: 'AUTHOR_0005_Upcroft'}) CREATE (s)-[:AUTHORED_BY]->(t);
MATCH (s {id: 'PAPER_0001_DeepFruits_A_Fruit_Detection_System_Using_Deep_Ne'}), (t {id: 'AUTHOR_0006_Perez'}) CREATE (s)-[:AUTHORED_BY]->(t);
MATCH (s {id: 'PAPER_0001_DeepFruits_A_Fruit_Detection_System_Using_Deep_Ne'}), (t {id: 'AUTHOR_0007_McCool'}) CREATE (s)-[:AUTHORED_BY]->(t);
MATCH (s {id: 'PAPER_0001_DeepFruits_A_Fruit_Detection_System_Using_Deep_Ne'}), (t {id: 'ALGORITHM_0001_Faster_RCNN'}) CREATE (s)-[:USES_ALGORITHM]->(t);
MATCH (s {id: 'PAPER_0001_DeepFruits_A_Fruit_Detection_System_Using_Deep_Ne'}), (t {id: 'ENVIRONMENT_0001_Rgb_Nir'}) CREATE (s)-[:TESTED_IN_ENVIRONMENT]->(t);
MATCH (s {id: 'PAPER_0001_DeepFruits_A_Fruit_Detection_System_Using_Deep_Ne'}), (t {id: 'FRUIT_0001_sweet_pepper'}) CREATE (s)-[:TARGETS_FRUIT]->(t);
MATCH (s {id: 'PAPER_0001_DeepFruits_A_Fruit_Detection_System_Using_Deep_Ne'}), (t {id: 'CHALLENGE_0001_Weather_Conditions'}) CREATE (s)-[:ADDRESSES_CHALLENGE]->(t);
MATCH (s {id: 'PAPER_0001_DeepFruits_A_Fruit_Detection_System_Using_Deep_Ne'}), (t {id: 'CHALLENGE_0002_Generalization'}) CREATE (s)-[:ADDRESSES_CHALLENGE]->(t);
MATCH (s {id: 'PAPER_0002_Harvesting_Robots_for_High_value_Crops_State_of_t'}), (t {id: 'AUTHOR_0008_Bac'}) CREATE (s)-[:AUTHORED_BY]->(t);
MATCH (s {id: 'PAPER_0002_Harvesting_Robots_for_High_value_Crops_State_of_t'}), (t {id: 'AUTHOR_0009_CW'}) CREATE (s)-[:AUTHORED_BY]->(t);
MATCH (s {id: 'PAPER_0002_Harvesting_Robots_for_High_value_Crops_State_of_t'}), (t {id: 'AUTHOR_0010_van_Henten'}) CREATE (s)-[:AUTHORED_BY]->(t);
MATCH (s {id: 'PAPER_0002_Harvesting_Robots_for_High_value_Crops_State_of_t'}), (t {id: 'AUTHOR_0011_EJ'}) CREATE (s)-[:AUTHORED_BY]->(t);
MATCH (s {id: 'PAPER_0002_Harvesting_Robots_for_High_value_Crops_State_of_t'}), (t {id: 'AUTHOR_0012_Hemming'}) CREATE (s)-[:AUTHORED_BY]->(t);
MATCH (s {id: 'PAPER_0002_Harvesting_Robots_for_High_value_Crops_State_of_t'}), (t {id: 'AUTHOR_0013_Edan'}) CREATE (s)-[:AUTHORED_BY]->(t);
MATCH (s {id: 'PAPER_0002_Harvesting_Robots_for_High_value_Crops_State_of_t'}), (t {id: 'ALGORITHM_0002_Traditional_CV'}) CREATE (s)-[:USES_ALGORITHM]->(t);
MATCH (s {id: 'PAPER_0003_Sensors_and_systems_for_fruit_detection_and_locali'}), (t {id: 'AUTHOR_0014_Gongal'}) CREATE (s)-[:AUTHORED_BY]->(t);
