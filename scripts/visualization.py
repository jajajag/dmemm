import os
import numpy as np
import imageio

import diffuser.sampling as sampling
import diffuser.utils as utils


# -----------------------------------------------------------------------------#
#                                   Parser                                      #
# -----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'   # 按需要改
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')


# -----------------------------------------------------------------------------#
#                              Helper: unwrap env                               #
# -----------------------------------------------------------------------------#

def unwrap_env(env):
    """把 NormalizedEnv / TimeLimit 这一类 wrapper 一层层拆掉，拿到最底层 MujocoEnv."""
    e = env
    while hasattr(e, 'env'):
        e = e.env
    return e


def obs_to_state(env, obs, ref_state=None):
    """
    将 gym Mujoco 的 observation 转回 state = [qpos, qvel].
    对于 halfcheetah / walker2d 这一类，obs = [qpos[1:], qvel].

    参数:
        env: dataset.env
        obs: (obs_dim,) 例如 17 维
        ref_state: (nq+nv,) 用来提供 root x（qpos[0]）；如果为 None，就用当前 env.state_vector().
    """
    base_env = unwrap_env(env)
    model = base_env.model
    nq = model.nq
    nv = model.nv

    obs = np.asarray(obs)
    assert obs.shape[-1] == (nq - 1 + nv), \
        f"Unexpected obs dim {obs.shape[-1]}, expected {nq-1+nv} for nq={nq}, nv={nv}"

    if ref_state is None:
        ref_state = base_env.state_vector()
    ref_state = np.asarray(ref_state)
    assert ref_state.shape[-1] == nq + nv

    ref_qpos = ref_state[:nq]
    # root x 用 ref_qpos[0]，使得画出来的位置大致合理
    root_x = ref_qpos[0]

    qpos = np.zeros(nq, dtype=np.float32)
    qvel = np.zeros(nv, dtype=np.float32)

    # obs = [qpos[1:], qvel]
    qpos[0] = root_x
    qpos[1:] = obs[:nq-1]
    qvel[:] = obs[nq-1:]

    state = np.concatenate([qpos, qvel], axis=0)
    return state


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
#                        1) Open-loop: 单次 plan 的 trajectory                  #
# -----------------------------------------------------------------------------#

obs = env.reset()
init_state = base_env.state_vector().copy()

horizon = diffusion.horizon    # diffusion 模型的预测长度
print(f"[Info] diffusion horizon = {horizon}")

conditions = {0: obs}
action0, samples0 = policy(conditions, batch_size=1, verbose=args.verbose)
# samples0.observations: 形状 [1, horizon, obs_dim]
plan_obs_traj = samples0.observations[0]      # [H, obs_dim]

print(f"[Info] Got open-loop plan observations with shape {plan_obs_traj.shape}")

# 把 plan 的 obs 转成 state，基于初始 state 的 root x
plan_states = []
for t in range(horizon):
    st = obs_to_state(env, plan_obs_traj[t], ref_state=init_state)
    plan_states.append(st)
plan_states = np.stack(plan_states, axis=0)   # [H, nq+nv]


# -----------------------------------------------------------------------------#
#                       2) Closed-loop: 每步重新 plan 的 rollout               #
# -----------------------------------------------------------------------------#

closed_states = [init_state.copy()]
closed_obs = [obs.copy()]

max_steps = horizon    # 你也可以设成其他，比如 200

for t in range(max_steps):
    if t % 10 == 0:
        print(f"[Closed-loop] t = {t}", flush=True)

    conditions = {0: obs}
    action, samples_t = policy(conditions, batch_size=1, verbose=False)

    next_obs, reward, done, _ = env.step(action)

    closed_obs.append(next_obs.copy())
    closed_states.append(base_env.state_vector().copy())

    obs = next_obs
    if done:
        print(f"[Closed-loop] Episode done at t = {t}")
        break

closed_states = np.stack(closed_states, axis=0)   # [T_cl+1, nq+nv]
print(f"[Info] Closed-loop rollout length = {len(closed_states)}")


# -----------------------------------------------------------------------------#
#                               Rendering to images                             #
# -----------------------------------------------------------------------------#

save_root = os.path.join(args.savepath, "two_traj_images")
open_dir = os.path.join(save_root, "open_loop_plan")
closed_dir = os.path.join(save_root, "closed_loop")
os.makedirs(open_dir, exist_ok=True)
os.makedirs(closed_dir, exist_ok=True)

print(f"[Saving] Open-loop images -> {open_dir}")
print(f"[Saving] Closed-loop images -> {closed_dir}")


def render_state_sequence(states, out_dir, prefix="traj"):
    """
    给定一串 MUJOCO state（qpos+qvel），逐帧用 renderer.render() 存成 png.
    """
    for i, st in enumerate(states):
        img = renderer.render(st)     # 这里传 state（长度 nq+nv），就不会再报 18 vs 17 的错了
        fname = os.path.join(out_dir, f"{prefix}_{i:04d}.png")
        imageio.imwrite(fname, img)


# 1) 渲染 open-loop 计划出来的 trajectory
render_state_sequence(plan_states, open_dir, prefix="open")

# 2) 渲染 closed-loop rollout 的真实轨迹
render_state_sequence(closed_states, closed_dir, prefix="closed")

print("[Done] All frames saved.")

