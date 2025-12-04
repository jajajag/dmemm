import os
import numpy as np
import imageio

import diffuser.sampling as sampling
import diffuser.utils as utils


# -----------------------------------------------------------------------------#
#                                   Parser                                      #
# -----------------------------------------------------------------------------#

class Parser(utils.Parser):
    # 按需要改 dataset，默认给你 walker2d
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')


# -----------------------------------------------------------------------------#
#                          helpers: unwrap & restore env                        #
# -----------------------------------------------------------------------------#

def unwrap_env(env):
    """把 NormalizedEnv / TimeLimit 等 wrapper 一层层拆掉，拿到底层 MujocoEnv。"""
    e = env
    while hasattr(e, 'env'):
        e = e.env
    return e


def restore_env_state(env, state):
    """
    把环境恢复到给定 MUJOCO state（qpos+qvel），并且把 wrapper 的 _elapsed_steps 归零。
    state: shape (nq + nv,)
    """
    base_env = unwrap_env(env)
    model = base_env.model
    nq = model.nq
    nv = model.nv

    qpos = state[:nq]
    qvel = state[nq:]

    base_env.set_state(qpos, qvel)

    e = env
    while True:
        if hasattr(e, "_elapsed_steps"):
            e._elapsed_steps = 0
        if hasattr(e, "env"):
            e = e.env
        else:
            break


# -----------------------------------------------------------------------------#
#                              Load experiments                                 #
# -----------------------------------------------------------------------------#

diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)

utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

policy = policy_config()

env = dataset.env
base_env = unwrap_env(env)

# -----------------------------------------------------------------------------#
#                         Reset once, get initial state                         #
# -----------------------------------------------------------------------------#

obs0 = env.reset()
init_state = base_env.state_vector().copy()

print("[Info] Initial obs shape:", obs0.shape)
print("[Info] diffusion horizon:", diffusion.horizon)

H = diffusion.horizon          # 例如 4
TARGET_STEPS = 32              # 想要最多跑多少步（两个 rollout 都用这个上限）


# -----------------------------------------------------------------------------#
# 1) Rollout A: 每 1 步 replan（标准 MPC：step-wise replan）                    #
# -----------------------------------------------------------------------------#

restore_env_state(env, init_state)
obs = obs0.copy()

states_plan_every_1 = [base_env.state_vector().copy()]

for t in range(TARGET_STEPS):
    if t % 10 == 0:
        print(f"[Plan-every-1] t={t}", flush=True)

    conditions = {0: obs}
    # action_1 是第一步 action，samples_1.actions[0] 是整条 plan
    action_1, samples_1 = policy(conditions, batch_size=1, verbose=False)

    obs, r, done, info = env.step(action_1)
    states_plan_every_1.append(base_env.state_vector().copy())

    if done:
        print(f"[Plan-every-1] episode done at t={t}")
        break

states_plan_every_1 = np.stack(states_plan_every_1, axis=0)
print("[Info] Plan-every-1 rollout length (env steps):",
      len(states_plan_every_1) - 1)


# -----------------------------------------------------------------------------#
# 2) Rollout B: 每 H 步 replan（segment-wise：执行完整 horizon 的 plan）        #
# -----------------------------------------------------------------------------#

restore_env_state(env, init_state)
obs = obs0.copy()

states_plan_every_H = [base_env.state_vector().copy()]

current_plan_actions = None   # 当前 segment 的 action 序列 [H, act_dim]
step_in_segment = 0           # 当前 segment 内第几步

for t in range(TARGET_STEPS):
    if current_plan_actions is None or step_in_segment >= H:
        # 需要重新 plan 一次：从当前 obs 出发，采样一条长度 H 的 plan
        conditions = {0: obs}
        _, samples_seg = policy(conditions, batch_size=1, verbose=False)

        if not hasattr(samples_seg, "actions"):
            raise RuntimeError(
                "samples_seg 没有 actions 字段，policy 没有返回整条 action 计划？"
            )

        current_plan_actions = samples_seg.actions[0]   # [H, act_dim]
        step_in_segment = 0

        print(f"[Plan-every-H] new plan at t={t}, will execute up to {H} steps")

    # 执行当前 plan 内的第 step_in_segment 个 action
    a = current_plan_actions[step_in_segment]
    obs, r, done, info = env.step(a)
    states_plan_every_H.append(base_env.state_vector().copy())
    step_in_segment += 1

    if done:
        print(f"[Plan-every-H] episode done at t={t}")
        break

states_plan_every_H = np.stack(states_plan_every_H, axis=0)
print("[Info] Plan-every-H rollout length (env steps):",
      len(states_plan_every_H) - 1)


# -----------------------------------------------------------------------------#
#                             Rendering to images                               #
# -----------------------------------------------------------------------------#

# 保存路径自动推一推
if args.savepath is None:
    base_save = os.path.join(args.loadbase, args.dataset)
    if args.diffusion_loadpath is not None:
        base_save = os.path.join(base_save, args.diffusion_loadpath)
    save_root = os.path.join(base_save,
                             f"two_rollout_images_plan1_vs_planH_T{TARGET_STEPS}")
else:
    save_root = os.path.join(args.savepath,
                             f"two_rollout_images_plan1_vs_planH_T{TARGET_STEPS}")

dir_every_1 = os.path.join(save_root, "plan_every_1")
dir_every_H = os.path.join(save_root, "plan_every_H")
os.makedirs(dir_every_1, exist_ok=True)
os.makedirs(dir_every_H, exist_ok=True)

print(f"[Saving] Plan-every-1 images -> {dir_every_1}")
print(f"[Saving] Plan-every-H images -> {dir_every_H}")


def render_state_sequence(states, out_dir, prefix):
    """
    给定一串 MUJOCO state（qpos+qvel），逐帧用 renderer.render() 存成 png.
    states: [N, nq+nv]，包含 t=0 的初始 state。
    """
    for i, st in enumerate(states):
        img = renderer.render(st)
        fname = os.path.join(out_dir, f"{prefix}_{i:04d}.png")
        imageio.imwrite(fname, img)


render_state_sequence(states_plan_every_1, dir_every_1, prefix="plan1")
render_state_sequence(states_plan_every_H, dir_every_H, prefix="planH")

print("[Done] All frames saved.")

