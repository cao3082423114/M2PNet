# Multimodal Motion Perception Network
This code accompanies the paper: Motion Perception Driven Multimodal Self-Supervised Video Object Segmentation



#### Requirements :
    conda env create -f environment.yml
    conda activate env
    

#### Datasets :
* DAVIS 2016 can be used as-is.The dataset can be downloaded at the following address: https://davischallenge.org/davis2016/code.html
* The rest has to be converted to DAVIS format. Some helper functions are available in tools.
* MoCA needs to be processed. Helper functions are available in tools. The (already filtered) dataset is also available on google drive: https://drive.google.com/drive/u/2/folders/1x-owzr9Voz65NQghrN_H1LEYDaaQP5n1, which can be used as-is after download.
* Precomputed flows can be generated from raft/run_inference.py or FlowFormerPlusPlus/visualize_flow.py

#### Training :
    python train.py --dataset DAVIS --flow_to_rgb

#### Inference :
    python eval.py --dataset DAVIS  --flow_to_rgb --inference --resume_path {}

#### Benchmark :
* For DAVIS: use the official evaluation code: https://github.com/fperazzi/davis
* For MOCA: use tools/MoCA_eval.py

