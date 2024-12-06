#############
# Copy the provided files and folders in mmsegmentaion folder. It should merge with existing folders automatically.


############# One time tasks

## Setup dataset and model path

# Go to configs/_base_/datasets/ade20k.py,
# Change the data_root = '<root_path>/mmsegmentation/data/ade/<datafolder>'.

# Go to tools/mmseg/datasets/ade.py,
# Change the classes and palette as provided in test_change_label_color.py.

## Setup model config
# Go to configs/swin/cave-seg-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py,
# Change the num_classes = 13 and other parameters if necessary.



############# Start from here everytime you open a new terminal

conda activate <root_path>/.conda/envs/mmseg
cd <root_path>/mmsegmentation



############# Test (to see scores) with single GPU

python tools/test.py configs/swin/cave-seg-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py work_dirs/cave-seg-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512/iter_160000.pth



############# Inference with single GPU (single image)

python demo/inference_single.py <image_path> configs/swin/cave-seg-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py /work_dirs/cave-seg-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512/iter_16000.pth --opacity 1.0 --out-file demo/output/sample_output.jpg



############# Inference with single GPU (list of images)
# Change path_testset, file_to_write, output_dir in test_txt_prepare.py
python test_txt_prepare.py

python demo/inference_sequence.py demo/test_caveseg.txt configs/swin/cave-seg-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py work_dirs/cave-seg-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512/iter_160000.pth

# Change path in test_change_label_color.py
python test_change_label_color.py



############# General utils

# Check the output of the following command
which python
# It should be as following
<root_path>/.conda/envs/mmseg/bin/python

# Print config
python tools/misc/print_config.py configs/swin/cave-seg-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py

# Get flops and params
python tools/analysis_tools/get_flops.py configs/swin/cave-seg-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py --shape 3 960 540

