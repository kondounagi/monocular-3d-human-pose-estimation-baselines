# chen-cvpr2019
**[To see all the contents, please check this page on HackMD !](https://hackmd.io/@BFfHyumSTF6-Uy3zSgDrPw/Sy1R91wLu)**

## Related Paper
### Unsupervised 3D Pose Estimation with Geometric Self-Supervision [Chen+, CVPR20]
* [paper](https://arxiv.org/pdf/1904.04812.pdf)
* No code
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
In order to train the model in $n$ gpus, do the command below.
```
$ python main.py --gpus n --batch_size batch_size
```
You can see detailed options in help.
```
$ python main.py --help
```
### Logging
You can check training log with tensorbarod or mlflow.
```
$ tensorboard --logdir /workspace/lighting_logs --port port
```
Best 3 model in `val_mpjpe_epoch` is saved in `/workspace/lighting_logs/versions?`.
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