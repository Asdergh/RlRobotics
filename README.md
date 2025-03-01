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
│           │   └── model
│           │       ├── assets
│           │       │   ├── achilles-rod.obj
│           │       │   ├── cassie-texture.png
│           │       │   ├── foot-crank.obj
│           │       │   ├── foot.obj
│           │       │   ├── heel-spring.obj
│           │       │   ├── hip-pitch.obj
│           │       │   ├── hip-roll.obj
│           │       │   ├── hip-yaw.obj
│           │       │   ├── knee-spring.obj
│           │       │   ├── knee.obj
│           │       │   ├── pelvis.obj
│           │       │   ├── plantar-rod.obj
│           │       │   ├── shin.obj
│           │       │   └── tarsus.obj
│           │       ├── cassie.xml
│           │       └── scene.xml
│           └── env.py
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
