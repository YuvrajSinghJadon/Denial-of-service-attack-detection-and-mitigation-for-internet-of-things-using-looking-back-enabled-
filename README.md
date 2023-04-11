# Denial-of-service-attack-detection-and-mitigation-for-internet-of-things-using-looking-back-enabled-

YUVRAJ SINGH JADON
* yuvrajsingjjadon.201it268@nitk.edu.in 
Compiled March 16, 2023

Abstract: Internet of Things (IoT) systems are vulnera- ble to various types of cyberattacks, including Denial of Service (DoS) and Distributed Denial of Service (DDoS). In this paper, we propose a new architecture to detect and defend against DoS/DDoS attacks on IoT systems using machine learning. The proposed architecture con- sists of two components, namely DoS/DDoS detection and DoS/DDoS defence. The detection component uses a multi-class classifier that applies the looking-back con- cept and is evaluated on the bot IoT dataset. The eval- uation results are promising: a looking-back capable random forest classifier achieves 99.81 

1. INTRODUCTION
IoT systems have become an integral part of our daily lives, and their integration into different areas of life has made them vulnerable to various types of cyberattacks. One of the most com- mon types of attacks on IoT systems are Denial of Service (DoS) and Distributed Denial of Service (DDoS) attacks. DoS/DDoS attacks can cause significant damage to IoT systems, resulting in the loss of critical data and system downtime. Therefore, there is a need for an efficient and effective mechanism to detect and mitigate DoS/DDoS attacks in IoT systems.

2. METHODOLOGY
In this paper, we propose a new architecture to detect and de- fend against DoS/DDoS attacks on IoT systems using machine learning. The proposed architecture consists of two compo- nents, namely DoS/DDoS detection and DoS/DDoS defence. The detection component enables fine-grained detection by iden- tifying the specific type of attack and the type of packet used for the attack. The proposed DoS/DDoS detection component is a multi-class classifier that applies the "looking back" concept and is evaluated against the Bot-IoT dataset.

• Dataset: The experiment used a preprocessed dataset cre- ated at the Research Cyber Range Laboratory at the Univer- sity of New South Wales Canberra. The dataset contained
features and laboratoryels for classifying different types of
attacks in a network.

• Data preprocessing: The unwanted features were removed
from the training and testing datasets and previous attack types were extracted for each sample in the training and testing datasets.
• Algorithm selection: Four algorithms were selected for this experiment: Decision Tree Classifier, Random Forest Classifier, K-Nearest Neighbor Classifier, and Multi-Layer Perceptron Classifier.
• Model training and testing: The experiment was per- formed by setting a looking-back time step p and train- ing each of the four classifiers with the training data. The models were then tested with the test data to evaluate their performance. For each value of p, the accuracy and kappa values of each classifier were calculated and stored.
• Performance evaluation: Three metrics were used to eval- uate the performance of each model: Accuracy value, f1 value, and Kappa value. The accuracy value measures the proportion of correctly classified samples, the f1 value con- siders both precision and recognition of each class, and the kappa value measures the agreement between the pre- dicted and actual labels, taking into account the possibility of random agreement.
• Results analysis: Finally, the results of the experiment were analyzed to determine the optimal looking-back time step p that yields the best performance for each algorithm. This was done by examining the accuracy and kappa values for different values of p and selecting the value that gave the best

3. RESULTS
The proposed looking-back enabled random forest classifier achieves 99.81% accuracy in detecting DoS/DDoS attacks. The proposed architecture is able to effectively detect and mitigate DoS/DDoS attacks in IoT systems by providing a fine-grained detection mechanism that identifies the specific type of attack and the packet type used in the attack.
• 1.Based on the given accuracy and kappa values, the follow- ing findings can be derived from the experiment:
• 2.The decision tree (DT) and random forest (RF) models perform equally well in terms of accuracy and kappa across
  

   Fig. 1. Plots of Accuracy over multiple Looking-Back steps
all look-back steps. This suggests that these models are robust to changes in the amount of historical data used for training.
• 3.The K-Nearest Neighbors (KNN) and Multi-Layer Percep- tron (MLP) models have lower accuracy and kappa values compared to the DT and RF models. This suggests that these models may not be well suited for cyberattack predic- tion or may need further optimization.
• 4.The performance of the MLP model decreases as the look- back step increases. This suggests that the model may not be able to learn effectively from a large amount of historical data.
• 5.The performance of the KNN model remains constant across all lookback steps. This suggests that beyond a cer- tain point, the model can no longer benefit from additional historical data.
• 6.The RF model outperforms the other models in terms of kappa values, suggesting that it’s better at capturing the correspondence between predicted and actual labels.
These results suggest that the DT and RF models are well suited for cyberattack prediction and that the amount of historical data used for training can be varied without significantly affecting the model’s performance. However, for the KNN and MLP models, further optimization may be required to improve their performance.

4. CONCLUSION
In conclusion, this paper proposes a new architecture for DoS/DDoS attack detection and mitigation for IoT systems us-
Fig. 3. Kappa results over multiple Looking-Back steps
Fig. 2. Accuracy and kappa results (%) for all classifiers using multiple Looking-Back steps
 

ing machine learning techniques. The proposed architecture provides a fine-granularity detection mechanism that identifies the specific type of attack and the packet type used in the attack. The proposed Looking-Back-enabled Random Forest classifier achieves an accuracy of 99.81% in detecting DoS/DDoS attacks. The proposed architecture can effectively detect and mitigate DoS/DDoS attacks in IoT systems, providing a robust defense mechanism against cyber-attacks.
5. REFERENCES
Alaeddine Mihoub a, Ouissem Ben Fredj b, Omar Cheikhrouhou c f, Abdelouahid Derhab d, Moez Krichen. Denial of service at- tack detection and mitigation for internet of things using looking- back-enabled machine learning techniques
