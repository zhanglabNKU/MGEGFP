# MGEGFP: A multi-view graph embedding method for gene function prediction based on adaptive estimation with GCN


## Install:
git clone https://github.com/nkuweili/MGEGFP.git  
cd MGEGFP/

## Environment:
* python 3.7.7

* pytorch 1.3.1

* scikit-learn 0.23.2


## Overall workflow:

![The overall framework of MGEGFP][Workflow.jpg]



Fig. 1. (a) Intra-view graph representation learning. For each individual view, we exploit RWR to obtain the high-quality initial feature representations. Then the original adjacency matrix and the initial features undergo a $L$-layer GCN equipped with a jump connection to generate the learned representation matrix, which is then fed into the decoder to calculate the reconstructed adjacency matrix. The jump connection effectively combines the low-layer information and high-layer information. (b) Inter-view graph representation learning on the basis of intra-view learning. For simplicity, we use two views as an example. In multi-view representation learning, a dual-channel GCN encoder is constructed to disentangle the view-specific information and the common pattern across all views. Then the obtained embeddings in each view pass through the multi-gate module and the outputs are used to decode the topology structure of each view. Finally, the learned gene representations are concatenated and used to train the plug-in classifier to annotate gene functions.




## Contact
Email:nkuweili@mail.nankai.edu.cn
