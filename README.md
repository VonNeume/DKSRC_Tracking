# Discriminative Non-Linear Sparse Coding for Visual Object Tracking

![image](https://github.com/wahahamyt/DKSRC_Tracking/blob/main/5ehecb.gif)

**1. Requirements:**

You need firstly install Anaconda.

```shell
pip install opencv-python
pip install visdom
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install got10k
pip install scikit-learn
pip install skfeature-chappers
```
The pretrained weights of CCP-DIM can be found at:


**Baidu Net Disk:**

> https://pan.baidu.com/s/1WKpDtjhqdgkgmzjqCxQ2rw 
Code: drap 


**GoogleNet Disk:**

> https://drive.google.com/drive/folders/1VLsG_lcv95M83A554KmoduV43eNLCvk7?usp=sharing 

Both the files should be located on the `./checkpoints/`.

If you want to train the CCP-DIM in your own data, you can visit the following link for more details. 
> https://github.com/wahahamyt/CCP-DIM



**2.Start:**

Run the tracker:
```shell
python demo.py
```

You can debug the tracker by setting the ```p.debug = True``` in ```./tracker/params.py```.
Then run following command:
```
python -m visdom.server
```






