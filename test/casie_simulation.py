import mujoco 
import torch as th
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import write_video
import cv2



# xml_string = """
# <mujoco model="tippe top">
#   <option integrator="RK4"/>

#   <asset>
#     <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
#      rgb2=".2 .3 .4" width="300" height="300"/>
#     <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
#   </asset>

#   <worldbody>
#     <geom size=".2 .2 .01" type="plane" material="grid"/>
#     <light pos="0 0 .6"/>
#     <camera name="closeup" pos="0 -.3 .17" xyaxes="1 0 0 0 1 2" mode="targetbody" target="top"/>
#     <body name="top" pos="0 0 .02">
#       <freejoint/>
#       <geom name="ball" type="sphere" size=".02" />
#       <geom name="stem" type="cylinder" pos="0 0 .02" size="0.004 .008"/>
#       <geom name="ballast" type="box" size=".023 .023 0.005"  pos="0 0 -.015"
#        contype="0" conaffinity="0" group="3"/>
#     </body>
#   </worldbody>

#   <keyframe>
#     <key name="spinning" qpos="0 0 0.02 1 0 0 0" qvel="0 0 0 0 1 1100" />
#   </keyframe>
# </mujoco>
# """

# model = mujoco.MjModel.from_xml_string(xml_string)
# data = mujoco.MjData(model)
frames = []
# scene_option = mujoco.MjvOption()
# scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

# grav_vector = np.array([0., 0., 1.])
model = mujoco.MjModel.from_xml_path("C:\\Users\\1\\Desktop\\PythonProjects\\RL_robotics\\src\\algorithm\\mujoco_env\\cassie\\model\\scene.xml")
data = mujoco.MjData(model)
# mujoco.mj_resetDataKeyframe(model, data, 0) 
# data.qpos[:6] = np.cos(np.random.normal(0.12, 0.12, 6))
with mujoco.Renderer(model, 480, 640) as renderer:

    for _ in range(1000):
        
        # model.opt.gravity = grav_vector
        
        # data.ctrl[2] -= 0.23
        data.qpos[4] += 0.01
        # data.ctrl[0] += 0.23
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        frame = renderer.render()
        frames.append(frame)
        # grav_vector += np.random.randint(-1, 1) * np.random.normal(0.12, 0.012, 3)

if len(frames) == 1:
    _, axis = plt.subplots()
    axis.imshow((frames[0].numpy() / 256.0))
    plt.show()

else:

    for frame in frames:
        cv2.imshow("robot_test_scene", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    
    
    
# import mujoco
# import numpy as np
# from scipy.optimize import minimize

# # Загрузка модели Cassie
# model = mujoco.MjModel.from_xml_path("C:\\\\\\\\Users\\\\\\\\1\\\\\\\\Desktop\\\\\\\\PythonProjects\\\\\\\\RL_robotics\\\\\\\\test\\\\\\\\mujoco_tests\\\\\\\\scene.xml")
# data = mujoco.MjData(model)

# # Параметры симуляции
# sim_time = 10.0  # Время симуляции (секунды)
# dt = 0.01  # Шаг симуляции
# n_steps = int(sim_time / dt)  # Количество шагов
# n_ctrl = model.nu  # Количество управляющих сигналов (моторов)

# # Целевая функция: максимизация высоты прыжка
# def objective(ctrl_seq):
#     """
#     Целевая функция для оптимизации.
#     ctrl_seq: Последовательность управляющих сигналов (n_steps x n_ctrl).
#     """
#     # Сброс симуляции
#     mujoco.mj_resetData(model, data)
    
#     # Применение управляющих сигналов
#     max_height = 0
#     for i in range(n_steps):
#         # Применяем управляющие сигналы
#         data.ctrl[:] = ctrl_seq[i * n_ctrl : (i + 1) * n_ctrl]
        
#         # Шаг симуляции
#         mujoco.mj_step(model, data)
        
#         # Обновляем максимальную высоту
#         max_height = max(max_height, data.qpos[2])  # qpos[2] — вертикальная координата
        
#     # Минимизируем отрицательную высоту (чтобы максимизировать высоту)
#     return -max_height

# # Ограничения на управляющие сигналы
# def control_constraints(ctrl_seq):
#     """
#     Ограничения на управляющие сигналы.
#     ctrl_seq: Последовательность управляющих сигналов.
#     """
#     # Пример: ограничение на моменты моторов (например, -10 <= tau <= 10)
#     return np.clip(ctrl_seq, -10, 10)

# # Начальное предположение для управляющих сигналов
# initial_ctrl = np.zeros(n_steps * n_ctrl)

# # Оптимизация
# result = minimize(
#     objective,  # Целевая функция
#     initial_ctrl,  # Начальное предположение
#     method='L-BFGS-B',  # Метод оптимизации
#     bounds=[(-10, 10)] * (n_steps * n_ctrl),  # Ограничения на управляющие сигналы
#     options={'maxiter': 100, 'disp': True}  # Параметры оптимизации
# )

# # Оптимальные управляющие сигналы
# optimal_ctrl = result.x.reshape(n_steps, n_ctrl)

# # Визуализация результата
# def simulate_and_visualize(ctrl_seq):
#     """
#     Симуляция и визуализация результата.
#     ctrl_seq: Оптимальная последовательность управляющих сигналов.
#     """
#     # Сброс симуляции
#     mujoco.mj_resetData(model, data)
    
#     # Визуализация
#     frames = []
#     with mujoco.Renderer(model) as renderer:
#         for i in range(n_steps):
#             # Применяем управляющие сигналы
#             data.ctrl[:] = ctrl_seq[i]
#             mujoco.mj_step(model, data)
#             renderer.update_scene(data)
#             frame = th.Tensor(renderer.render()).unsqueeze(dim=0)
            
#             frames.append(frame)
        
#     write_video(
#         "test.mp4",
#         video_array=th.cat(frames, dim=0),
#         fps=100
#     )

# # Запуск симуляции с оптимальными управляющими сигналами
# simulate_and_visualize(optimal_ctrl)
    
    


