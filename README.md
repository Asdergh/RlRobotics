# Project structer
```
.
├── MUJOCO_LOG.TXT
├── README.md
├── src
│   └── algorithm
│       ├── REINFORCE.py
│       ├── __init__.py
│       ├── base
│       │   ├── __pycache__
│       │   │   └── base_alg.cpython-311.pyc
│       │   └── base_alg.py
│       ├── models
│       │   ├── __pycache__
│       │   │   └── layers.cpython-311.pyc
│       │   ├── base_module.py
│       │   └── layers.py
│       └── mujoco_env
│           ├── __pycache__
│           │   └── env.cpython-311.pyc
│           ├── cassie
│           │   ├── assets
│           │   │   ├── achilles-rod.obj
│           │   │   ├── cassie-texture.png
│           │   │   ├── foot-crank.obj
│           │   │   ├── foot.obj
│           │   │   ├── heel-spring.obj
│           │   │   ├── hip-pitch.obj
│           │   │   ├── hip-roll.obj
│           │   │   ├── hip-yaw.obj
│           │   │   ├── knee-spring.obj
│           │   │   ├── knee.obj
│           │   │   ├── pelvis.obj
│           │   │   ├── plantar-rod.obj
│           │   │   ├── shin.obj
│           │   │   └── tarsus.obj
│           │   ├── cassie.xml
│           │   └── scene.xml
│           ├── env.py
│           └── g1_robot
│               ├── assets
│               │   ├── head_link.STL
│               │   ├── left_ankle_pitch_link.STL
│               │   ├── left_ankle_roll_link.STL
│               │   ├── left_elbow_link.STL
│               │   ├── left_hand_index_0_link.STL
│               │   ├── left_hand_index_1_link.STL
│               │   ├── left_hand_middle_0_link.STL
│               │   ├── left_hand_middle_1_link.STL
│               │   ├── left_hand_palm_link.STL
│               │   ├── left_hand_thumb_0_link.STL
│               │   ├── left_hand_thumb_1_link.STL
│               │   ├── left_hand_thumb_2_link.STL
│               │   ├── left_hip_pitch_link.STL
│               │   ├── left_hip_roll_link.STL
│               │   ├── left_hip_yaw_link.STL
│               │   ├── left_knee_link.STL
│               │   ├── left_rubber_hand.STL
│               │   ├── left_shoulder_pitch_link.STL
│               │   ├── left_shoulder_roll_link.STL
│               │   ├── left_shoulder_yaw_link.STL
│               │   ├── left_wrist_pitch_link.STL
│               │   ├── left_wrist_roll_link.STL
│               │   ├── left_wrist_yaw_link.STL
│               │   ├── logo_link.STL
│               │   ├── pelvis.STL
│               │   ├── pelvis_contour_link.STL
│               │   ├── right_ankle_pitch_link.STL
│               │   ├── right_ankle_roll_link.STL
│               │   ├── right_elbow_link.STL
│               │   ├── right_hand_index_0_link.STL
│               │   ├── right_hand_index_1_link.STL
│               │   ├── right_hand_middle_0_link.STL
│               │   ├── right_hand_middle_1_link.STL
│               │   ├── right_hand_palm_link.STL
│               │   ├── right_hand_thumb_0_link.STL
│               │   ├── right_hand_thumb_1_link.STL
│               │   ├── right_hand_thumb_2_link.STL
│               │   ├── right_hip_pitch_link.STL
│               │   ├── right_hip_roll_link.STL
│               │   ├── right_hip_yaw_link.STL
│               │   ├── right_knee_link.STL
│               │   ├── right_rubber_hand.STL
│               │   ├── right_shoulder_pitch_link.STL
│               │   ├── right_shoulder_roll_link.STL
│               │   ├── right_shoulder_yaw_link.STL
│               │   ├── right_wrist_pitch_link.STL
│               │   ├── right_wrist_roll_link.STL
│               │   ├── right_wrist_yaw_link.STL
│               │   ├── torso_link_rev_1_0.STL
│               │   ├── waist_roll_link_rev_1_0.STL
│               │   └── waist_yaw_link_rev_1_0.STL
│               ├── g1.xml
│               └── scene.xml
├── test
│   ├── casie_simulation.py
│   ├── frist_agent.py
│   ├── mujoco_tests
│   ├── sarsa_res.py
│   └── test.py
├── test.mp4
└── weights
    └── vanila_reinforce.pt
```
