# HBLALS_pytorch
This is the official code for "Hierarchical Bilevel Learning with Architecture and Loss Search for Hadamard-based Image Restoration" (https://www.ijcai.org/proceedings/2022/0245.pdf)
## pretrained model
The pretrained model of image classification network and pre-trained network are available at [Google Drive](https://drive.google.com/drive/folders/1sMbhGsJ-u99sUzKWQ9bmTO6Q3sI1Leud?usp=share_link)


## test
```Shell
python test.py
```
## search loss
```Shell
python train_search_loss.py
```
## search architecture
```Shell
python train_search_architecture.py
```
## train model
```Shell
python train.py
```
## Reference
If you find this code useful, please cite:
```Shell
@article{zhuhierarchical,
  title={Hierarchical Bilevel Learning with Architecture and Loss Search for Hadamard-based Image Restoration},
  author={Zhu, Guijing and Ma, Long and Fan, Xin and Liu, Risheng},
  booktitle = {IJCAI 2022},
  year = {2022},
}
```

