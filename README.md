# Intrusion-Detection-System

Basic Idea:
Two staged IDS specific to IoT networks where Signature based IDS and Anomaly based IDS which is trained and classified using machine learning in this case CNN-LSTM is used together in component based architecture.

Download UNSW NB15 Dataset from : https://www.unsw.adfa.edu.au

Project Title:
IoT Intrusion Detection System

Abstract:
To build a cloud-based application which will be positioned at the gateway
of IOT devices in the flowchart. This application is a combination of both Signature-based IDS and
Anomaly-based IDS. 
It will take any packet as input(basically the packet headers) and then classify it into one of the
categories ie. teardrop, buffer-overflow, rootkit etc. One of the categories will be unknown. If the
IDS is not able to classify the packet into any of the attack types then the packet will be labeled
as unknown.
On the other hand, the Anomaly-based IDS will know the nature of a normal packet and on
seeing a new input packet it will compare the nature of the packet with the nature of a normal
packet. If the difference is found to be large(beyond some threshold) then the packet will be
classified as an anomaly or else, it will be labeled as a normal packet.
This way, we aim to set up an efficient IDS for the IoT networks.
