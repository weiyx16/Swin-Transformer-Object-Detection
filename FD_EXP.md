
git clone -b dev https://github.com/weiyx16/Swin-Transformer-Object-Detection Swin-Transformer-Object-Detection-dev
cd Swin-Transformer-Object-Detection-dev
mkdir data; ln -s /mntdata/amldata/coco2017 ./data/coco; ls ./data/coco

/mntdata/yx_results/azcopy copy "https://bingdatawu2premium.blob.core.windows.net/fwd-data/amldata/coco2017/val2017?st=2022-03-02T04%3A23%3A21Z&se=2024-03-03T04%3A23%3A00Z&sp=racwl&sv=2018-03-28&sr=c&sig=FHM8hf7ZAIItCEPdxc3o06TeYjPakjPpZvExBLK1IKA%3D" ./ --recursive; /mntdata/yx_results/azcopy copy "https://bingdatawu2premium.blob.core.windows.net/fwd-data/amldata/coco2017/train2017?st=2022-03-02T04%3A23%3A21Z&se=2024-03-03T04%3A23%3A00Z&sp=racwl&sv=2018-03-28&sr=c&sig=FHM8hf7ZAIItCEPdxc3o06TeYjPakjPpZvExBLK1IKA%3D" ./ --recursive; /mntdata/yx_results/azcopy copy "https://bingdatawu2premium.blob.core.windows.net/fwd-data/amldata/coco2017/annotations?st=2022-03-02T04%3A23%3A21Z&se=2024-03-03T04%3A23%3A00Z&sp=racwl&sv=2018-03-28&sr=c&sig=FHM8hf7ZAIItCEPdxc3o06TeYjPakjPpZvExBLK1IKA%3D" ./ --recursive

# esvit
tools/dist_train.sh configs/swin/mask_rcnn_swin_base_patch4_window14_mstrain_480-800_adamw_1x_coco_fromesvit.py 8 --cfg-options model.pretrained=/mntdata/yx_results/pretrain_models/esvit_base_v1_w14_300ep.pth model.backbone.use_checkpoint=False --work-dir /mntdata/yx_results/output/FD/OD/SRC_ESVIT_1X  --deterministic 

# distilled esvit
tools/dist_train.sh configs/swin/mask_rcnn_swinv2_base_patch4_window14_mstrain_480-800_adamw_1x_coco_fromfd.py 8 --cfg-options model.pretrained=/mntdata/yx_results/pretrain_models/fd_esvit_300ep.pth model.backbone.use_checkpoint=False optimizer.lr=2e-4 --work-dir /mntdata/yx_results/output/FD/OD/FD_ESVIT_1X_LR2e4  --deterministic 