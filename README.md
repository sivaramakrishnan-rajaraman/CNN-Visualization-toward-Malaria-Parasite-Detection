# CNN-Visualization-toward-Malaria-Parasite-Detection

Feel free to use the attached notebooks for your own model and data. The repository also includes Matlab codes to extract and visualize learned weights, saliencies, and class activation maps in a custom trained model. Kindly cite the publication as these codes and data are part of this published work:

### Rajaraman S, Silamut K, Hossain MA, Ersoy I, Maude RJ, Jaeger S, Thoma GR, Antani SK. Understanding the learned behavior of customized convolutional neural networks toward malaria parasite detection in thin blood smear images. J Med Imaging (Bellingham). 2018 Jul;5(3):034501. doi: 10.1117/1.JMI.5.3.034501. Epub 2018 Jul 18.

# Prerequisites:

Keras>=2.2.0

Tensorflow-GPU>=1.9.0

OpenCV>=3.3

Matlab R2018b

# Goal

Convolutional neural networks (CNNs) have become the architecture of choice for visual recognition tasks. However, these models are perceived as black boxes since there is a lack of understanding of the learned behavior from the underlying task of interest. This lack of transparency is a serious drawback, particularly in applications involving medical screening and diagnosis since poorly understood model behavior could adversely impact subsequent clinical decision-making. Recently, researchers have begun working on this issue and several methods have been proposed to visualize and understand the behavior of these models. We highlight the advantages offered through visualizing and understanding the weights, saliencies, class activation maps, and region of interest localizations in customized CNNs applied to the challenge of classifying parasitized and uninfected cells to aid in malaria screening. We provide an explanation for the models’ classification decisions. We characterize, evaluate, and statistically validate the performance of different customized CNNs keeping every training subject’s data separate from the validation set.

# Data Availability

The data used in study is taken from https://ceb.nlm.nih.gov/repositories/malaria-datasets/. This page hosts a repository of segmented cells from the thin blood smear slide images from the Malaria Screener research activity. To reduce the burden for microscopists in resource-constrained regions and improve diagnostic accuracy, researchers at the Lister Hill National Center for Biomedical Communications (LHNCBC), part of National Library of Medicine (NLM), have developed a mobile application that runs on a standard Android smartphone attached to a conventional light microscope. Giemsa-stained thin blood smear slides from 150 P. falciparum-infected and 50 healthy patients were collected and photographed at Chittagong Medical College Hospital, Bangladesh. The smartphone’s built-in camera acquired images of slides for each microscopic field of view. The images were manually annotated by an expert slide reader at the Mahidol-Oxford Tropical Medicine Research Unit in Bangkok, Thailand. The de-identified images and annotations are archived at NLM (IRB#12972). A level-set based algorithm is applied to detect and segment the red blood cells. The dataset contains a total of 27,558 cell images with equal instances of parasitized and uninfected cells. 

# Model Configurations

We evaluated the performance of six customized CNNs including (a) sequential CNN, (b) VGG-16, (c) ResNet-50, (d) Xception, (e) Inception-V3, and (f) DenseNet-121, customized for the underlying task. We used the trainable and non-trainable layers of these models, everything up to the fully connected layers. We added a global average pooling (GAP) layer, followed by a dense fully connected, dropout, and logistic layer. The untrained models were in the process, customized for the classification task of our interest. The models were optimized for hyperparameters by the randomized grid search method. 

# Performance Evaluation

The performance of the customized CNNs was evaluated in terms of accuracy, the area under receiver operating characteristic curve (AUC), sensitivity, specificity, F1-score, and Matthews correlation coefficient (MCC).

# Visualization studies:

The interpretation and understanding of CNNs is a hotly debated topic in ML, particularly in the context of clinical decision-making. CNNs are perceived as black boxes and it is imperative to explain their working to build trust in their predictions. The methods of visualizing CNNs are broadly categorized into (i) preliminary methods that help to visualize the overall structure of the model; and, (ii) gradient-based methods that manipulate the gradients from the forward and backward pass during training. We performed class activation maps (CAM) visualizations to understand the learned model behavior. The details are available in the published paper made available in this repository. 
