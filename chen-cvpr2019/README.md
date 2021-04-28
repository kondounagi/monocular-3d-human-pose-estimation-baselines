# chen-cvpr2019
**[To see all the contents, please check this page on HackMD !](https://hackmd.io/@BFfHyumSTF6-Uy3zSgDrPw/Sy1R91wLu)**
## Abstruct
This project is aiming to train monocular 3D pose estimation model only from 2D pose dataset ([MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/)).
Please check [this Google Slide](https://docs.google.com/presentation/d/1i4rG6PuUL60iPAe9Jdj127xp0wsCg7SieyDyRqhmVq0/edit#slide=id.gc12fc09c67_0_40) for more detailed concept of this project !

## Requirements
```
$ docker --version
Docker version 20.10.4, build d3cb89e
$ docker-compose --version
docker-compose version 1.28.5, build unknown
```
## Run
### Docker
#### Build Image
```
$ ./build.sh
```
#### Run Container
```
$ ./run.sh
```
### Train & Test
In order to reproduce the training of the result above, you should type the following command.
```
$ python main.py --gpus 1 --batch_size 256
```
You can see detailed options in help.
```
$ python main.py --help
```
#### Options

| Option | Explanation | Default |
| -------- | -------- | -------- |
| `--gen_lr`     | Learning rate of generator's optimizer (Adam)     | 0.001     |
| `--dis_lr`     | Learning rate of discriminator's optimizer (Adam)     | 0.001     |
| `--gen_eps`     | Epsilon of generaotr's optimizer (Adam)     | 1e-8     |
| `--dis_eps`     | Epsilon of discriminator's optimizer (Adam)     | 1e-8     |
| `--gan_accuracy_cap`     | Accuracy cap for discriminator     | 0.9     |


### Demo
Given an input image, 2D pose is predicted by pretrained model, and 3D pose is estimated.
The result is in `./demo_out`.
At first time, please download 2d pose estimator model by just running `bash ./download_2d_pose_estimator.sh`.
You can download pretrained pytorch model from [this google drive link](https://drive.google.com/file/d/1qFL4WXVS7Atll1qaqa-gG8LtA1j7BIyJ/view?usp=sharing).
Specify the path to pretrained_model.ckpt file by `--ckpt` option, or place it to the default directory (`./test/checkpoints/pretrained_model.ckpt`).
```
$ python demo.py
```

#### Options

| Option | Explanation | Default |
| -------- | -------- | -------- |
| `--input`     | Input image path     | ./test/input_sample.png     |
| `--width`     | Resize input to specific width    | 368     |
| `--height`     | Resize input to specific height | 368     |
| `--thr`    | Threshold value for pose parts heat map     | 0.1     |
| `--ckpt`     | Checkpoint file path from trained pytorch-lightning     | ./test/checkpoints/pretrained_model.ckpt|


## Result
### [Human3.6M](http://vision.imar.ro/human3.6m/description.php) Evaluation


| Method   | MPJPE (mm) |
| -------- | --------   |
| [Kudo+, arxiv18](https://arxiv.org/pdf/1803.08244.pdf)    | 131      |
| [Drover+, ECCV18](https://arxiv.org/pdf/1803.08244.pdf)    | 127      |
|[Chen+, CVPR19](https://arxiv.org/pdf/1904.04812.pdf) with [Kinetics dataset](https://deepmind.com/research/open-source/kinetics)| 61|
|**Ours** <br>Reproduction implementation of [Chen+, CVPR19] <br>without Kinetics dataset | 91|

### Training Logs
You can reproduce exactly the same training process with below by `$ python main.py --batch_size 256`
#### MPJPE
![](https://i.imgur.com/NcXlb1l.png)
#### P_MPJPE (MPJPE after rigid transformation)
![](https://i.imgur.com/4bXehaW.png)

### Demo
You can reproduce exactly same demo with below by `$ python demo.py`


| input image | 2D Pose Estimation | 
| -------- | -------- |
| ![](https://i.imgur.com/5lz1TdU.jpg)     | ![](https://i.imgur.com/wvgVvHZ.jpg)
 

| $0^\circ$|$15^\circ$| $165^\circ$  |
| -------- | -------- | -------- |
| ![](https://i.imgur.com/QR6SyF9.png) | ![](https://i.imgur.com/FcRemFJ.png)  |![](https://i.imgur.com/BfY8c3w.png) |


### Logging
You can check training log with tensorbarod or mlflow.
```
$ tensorboard --logdir /workspace/lighting_logs --port port
```
Best model in terms of `val_mpjpe_epoch` is saved in `/workspace/lighting_logs/versions?`.


## Related Paper
### Unsupervised 3D Pose Estimation with Geometric Self-Supervision [Chen+, CVPR20]
* [paper](https://arxiv.org/pdf/1904.04812.pdf)
* No official implementation
![](https://i.imgur.com/SLD7X9a.png)
### Unsupervised Adversarial Learning of 3D Human Pose from 2D Joint Locations [Kudo+, arxiv18]
$L_{adv}$ in [Chen+, CVPR19] is based on this paper. Heuristic loss, which was proposed in this paper, was not adopted in [Chen+, CVPR19].
* [paper](https://github.com/DwangoMediaVillage/3dpose_gan)
* [code (chainer)](https://github.com/DwangoMediaVillage/3dpose_gan)
![](https://i.imgur.com/i56UOFw.png)

### A simple yet effective baseline for 3d human pose estimation [Martinez+, ICCV17]
The network architecture of lifting model in [Chen+, CVPR19] is based on this paper.
[Chen+, CVPR19] built its lifting network (generator) repeating the network block (below figure) for 4 times, and discriminator for 3 times.
* [paper](https://arxiv.org/abs/1705.03098)
* [code (tensorflow)](https://github.com/una-dinosauria/3d-pose-baseline)
![](https://i.imgur.com/Zbs8nJM.png)



## Credit
```
@inproceedings{martinez_2017_3dbaseline,
  title={A simple yet effective baseline for 3d human pose estimation},
  author={Martinez, Julieta and Hossain, Rayat and Romero, Javier and Little, James J.},
  booktitle={ICCV},
  year={2017}
}
@inproceedings{chen2019unsupervised,
  title={Unsupervised 3D Pose Estimation with Geometric Self-Supervision}, 
  author={Ching-Hang Chen and Ambrish Tyagi and Amit Agrawal and Dylan Drover and Rohith MV and Stefan Stojanov and James M. Rehg},
  year={2019},
  booktitle={CVPR}
}
@inproceedings{kudo2018unsupervised,
  title={Unsupervised Adversarial Learning of 3D Human Pose from 2D Joint Locations}, 
  author={Yasunori Kudo and Keisuke Ogaki and Yusuke Matsui and Yuri Odagiri},
  booktitle={arxiv},
  year={2018}
}
```
