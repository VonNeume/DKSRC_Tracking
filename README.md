# Discriminative Non-Linear Sparse Coding for Visual Object Tracking

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


> https://pan.baidu.com/s/19-vw4cdQDquk1ucLtF7Mdg, Code: 7ygx

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






