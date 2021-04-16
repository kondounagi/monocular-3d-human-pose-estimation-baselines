# chen-cvpr2019

## Paper
* Unsupervised 3D Pose Estimation with Geometric Self-Supervision
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
$ python main.py --gpus n
```
You can see detailed options in help.
```
$ python main.py --help
```

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