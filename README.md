## Reward models with human feedback for compositionality

### Datasets setup

#### Downloading images
You can download the images for TIFA and AIGCIQA2023 datasets [here](https://drive.google.com/file/d/1etw5Fb-wuiIb6KbUu7-gAm_fRRy_dp2r/view?usp=sharing).
Once downloaded, you can unzip them and inside the `data-transfer` folder you should see two files: `tifa_annotated_images.zip` and `allimg.zip`. Unzip these two files, and you should get two folders: `annotated_images` and `allimg`.

#### TIFA
The preference ranking data for TIFA can be accessed here: `csvs/tifa.csv`. To setup the images, ensure there is a folder called `annotated_images` in the same directory as the csv file.

#### AIGCIQA2023
The preference ranking data for AIGCIQA2023 can be accessed here: `csvs/aigciqa2023.csv`. To setup the images, ensure there is a folder called `allimg` in the same directory as the csv file.

#### ECCV_Caption
The preference ranking for ECCV-Caption can be accessed here: `eccv_caption.csv`. To setup the images, download the coco2014 val split from [here](http://images.cocodataset.org/zips/val2014.zip). Unzip and then store the folder `val2014` in the same directory as the csv file. 

#### D3PO
Download the images using `wget https://huggingface.co/datasets/yangkaiSIGS/d3po_datasets/resolve/main/text2img_evaluation.7z`. Extract the images and ensure that the folder `img` is inside the same directory as the csv file.
