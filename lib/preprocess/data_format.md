
We use MPI15 skeletion:
(note: the imgs in the dataset are extraced from internet videos)
```
0 : neck,
1 : head,
2 : pelvis,
3 : LShoulder,
4 : LElbow,
5 : LWrist,
6 : LHip,
7 : LKnee,
8 : LAnkle,
9 : RShoulder,
10: RElbow,
11: RWrist,
12: RHip,
13: RKnee,
14: RAnkle 

Pairs:
[[0, 1], [0, 2], [0, 9], [9, 10], [10, 11], [0, 3], [3, 4], [4, 5], [2, 12], [12, 13], [13, 14], [2, 6], [6, 7], [7, 8]]
```

CMU data format:
(but the dataset has not note the visiual ablity, so ..
if use CMU, we can use coco's nose keypoint because CMU do not has headtop as well)
````
'0': Neck
'1': Nose
'2': BodyCenter (center of hips)
'3': lShoulder
'4': lElbow
'5': lWrist,
'6': lHip
'7': lKnee
'8': lAnkle
'9': rShoulder
'10': rElbow
'11': rWrist
'12': rHip
'13': rKnee
'14': rAnkle
'15': lEye
'16': lEar
'17': rEye
'18': rEar
````


Our json is organized as follows:

```
xxx.json
{ "root":
  # image 0  
  [     
    "img_height"   : int,  
    "img_width"    : int,  
    "img_paths"    : "images/path/xxxxxx.jpg",  
    "dataset"      : str,  # dataset name 
    "isValidation" : 0 for train, 1 for test  
    "bodys"        : nested list, N x J x 11
  ],  
  [...],  # image 1 
  [...],  # image 2
}  


For "bodys": 
  N : the number of people in the image,
  J : the number of joints. mpi15 is used,
  11: [x, y, Z, v, X, Y, Z, fx, fy, cx, cy]
    x, y    : 2D keypoints,
    X, Y, Z : 3D keypoints (cm),
    v       : 0: not labeled, 1: occluded, 2: visible,
    fx, fy  : focal length,
    cx, cy  : principal point.

Note that for internet images (unknown intrinsics), fx, fy equal the image width, and cx, cy equal the image center.
```