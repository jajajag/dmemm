import os
import numpy as np
import imageio

import diffuser.sampling as sampling
import diffuser.utils as utils


# -----------------------------------------------------------------------------#
#                                   Parser                                      #
# -----------------------------------------------------------------------------#

class Parser(utils.Parser):
    # 默认 walker2d，按需改成 halfcheetah 等
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

    # 把外层 wrapper 的 step 计数也重置
    e = env
    while True:
        if hasattr(e, "_elapsed_steps"):
            e._elapsed_steps = 0
        if hasattr(e, "env"):
            e = e.env
        else:
            break


def obs_to_state(env, obs, ref_state=None):
    """
    将 mujoco-style obs 转为完整 state = [qpos, qvel]
    假设 obs = [qpos[1:], qvel]（walker2d/halfcheetah 这种结构）。
    """
    base_env = unwrap_env(env)
    model = base_env.model
    nq = model.nq
    nv = model.nv

    obs = np.asarray(obs)
    assert obs.shape[-1] == (nq - 1 + nv), \
        f"obs dim {obs.shape[-1]} != {nq-1+nv} (nq-1+nv)"

    if ref_state is None:
        ref_state = base_env.state_vector()
    ref_state = np.asarray(ref_state)
    assert ref_state.shape[-1] == nq + nv

    qpos0 = ref_state[:nq]
    root_x = qpos0[0]

    qpos = np.zeros(nq, dtype=np.float32)
    qvel = np.zeros(nv, dtype=np.float32)

    qpos[0] = root_x
    qpos[1:] = obs[:nq-1]
    qvel[:] = obs[nq-1:]

    return np.concatenate([qpos, qvel], axis=0)


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

H = diffusion.horizon      # e.g., 4
T_MAX = 32                 # 想 env / model rollout 到多少步（上限）


# -----------------------------------------------------------------------------#
# 1) plan1: 每 1 步 replan（MPC-style，真实 env）                                #
# -----------------------------------------------------------------------------#

restore_env_state(env, init_state)
obs = obs0.copy()

states_plan1 = [base_env.state_vector().copy()]

for t in range(T_MAX):
    if t % 10 == 0:
        print(f"[plan1] t={t}", flush=True)

    conditions = {0: obs}
    act_1, samples_1 = policy(conditions, batch_size=1, verbose=False)

    obs, r, done, info = env.step(act_1)
    states_plan1.append(base_env.state_vector().copy())

    if done:
        print(f"[plan1] episode done at t={t}")
        break

states_plan1 = np.stack(states_plan1, axis=0)
print("[Info] plan1 rollout length (env steps):", len(states_plan1) - 1)


# -----------------------------------------------------------------------------#
# 2) planH: 每 H 步 replan（执行完整 horizon，真实 env）                         #
# -----------------------------------------------------------------------------#

restore_env_state(env, init_state)
obs = obs0.copy()

states_planH = [base_env.state_vector().copy()]

current_plan_actions = None   # [H, act_dim]
step_in_segment = 0

for t in range(T_MAX):
    if current_plan_actions is None or step_in_segment >= H:
        conditions = {0: obs}
        _, samples_seg = policy(conditions, batch_size=1, verbose=False)

        if not hasattr(samples_seg, "actions"):
            raise RuntimeError("samples_seg 没有 actions 字段？")

        current_plan_actions = samples_seg.actions[0]   # [H, act_dim]
        step_in_segment = 0
        print(f"[planH] new plan at t={t}")

    a = current_plan_actions[step_in_segment]
    obs, r, done, info = env.step(a)
    states_planH.append(base_env.state_vector().copy())
    step_in_segment += 1

    if done:
        print(f"[planH] episode done at t={t}")
        break

states_planH = np.stack(states_planH, axis=0)
print("[Info] planH rollout length (env steps):", len(states_planH) - 1)


# -----------------------------------------------------------------------------#
# 3) dir1: 每 1 步 replan，但只在 model 里滚（不 step env）                      #
#    每次只用预测 obs 的第 0 步，接上去继续 plan                                 #
# -----------------------------------------------------------------------------#

print("\n[dir1] model-only rollout with step-wise replan...")
obs_d1 = obs0.copy()
ref_state_d1 = init_state.copy()

direct1_states = [init_state.copy()]

for t in range(T_MAX):
    conditions = {0: obs_d1}
    _, samples_d1 = policy(conditions, batch_size=1, verbose=False)

    if not hasattr(samples_d1, "observations"):
        raise RuntimeError("samples_d1 没有 observations 字段？")

    traj = samples_d1.observations[0]   # [H, obs_dim]
    next_obs = traj[0]                  # 只拿第一步的 obs

    st = obs_to_state(env, next_obs, ref_state=ref_state_d1)
    direct1_states.append(st)

    # 下一个 step 的条件
    ref_state_d1 = st
    obs_d1 = next_obs

direct1_states = np.stack(direct1_states, axis=0)
print("[Info] dir1 (model-only) length:", len(direct1_states) - 1)


# -----------------------------------------------------------------------------#
# 4) dirH: 每 H 步 replan，model-only，多段 horizon obs 接起来直到 T_MAX        #
# -----------------------------------------------------------------------------#

print("\n[dirH] model-only rollout with H-step segments...")
obs_dH = obs0.copy()
ref_state_dH = init_state.copy()

directH_states = [init_state.copy()]
steps = 0

while steps < T_MAX:
    conditions = {0: obs_dH}
    _, samples_dH = policy(conditions, batch_size=1, verbose=False)

    if not hasattr(samples_dH, "observations"):
        raise RuntimeError("samples_dH 没有 observations 字段？")

    traj = samples_dH.observations[0]   # [H, obs_dim]

    for k in range(H):
        if steps >= T_MAX:
            break
        next_obs = traj[k]
        st = obs_to_state(env, next_obs, ref_state=ref_state_dH)
        directH_states.append(st)

        ref_state_dH = st
        obs_dH = next_obs
        steps += 1

directH_states = np.stack(directH_states, axis=0)
print("[Info] dirH (model-only) length:", len(directH_states) - 1)


# -----------------------------------------------------------------------------#
#                             Rendering to images                               #
# -----------------------------------------------------------------------------#

# 保存目录：短名字 traj_vis
if args.savepath is None:
    base_save = os.path.join(args.loadbase, args.dataset)
    if args.diffusion_loadpath is not None:
        base_save = os.path.join(base_save, args.diffusion_loadpath)
    save_root = os.path.join(base_save, "traj_vis")
else:
    save_root = os.path.join(args.savepath, "traj_vis")

dir_plan1 = os.path.join(save_root, "plan1")
dir_planH = os.path.join(save_root, "planH")
dir_dir1 = os.path.join(save_root, "dir1")
dir_dirH = os.path.join(save_root, "dirH")

os.makedirs(dir_plan1, exist_ok=True)
os.makedirs(dir_planH, exist_ok=True)
os.makedirs(dir_dir1, exist_ok=True)
os.makedirs(dir_dirH, exist_ok=True)

print(f"[Saving] plan1 -> {dir_plan1}")
print(f"[Saving] planH -> {dir_planH}")
print(f"[Saving] dir1  -> {dir_dir1}")
print(f"[Saving] dirH  -> {dir_dirH}")


def render_state_sequence(states, out_dir, prefix):
    """
    states: [N, nq+nv]，逐帧渲成 png.
    """
    for i, st in enumerate(states):
        img = renderer.render(st)
        fname = os.path.join(out_dir, f"{prefix}_{i:04d}.png")
        imageio.imwrite(fname, img)


render_state_sequence(states_plan1, dir_plan1, prefix="p1")
render_state_sequence(states_planH, dir_planH, prefix="pH")
render_state_sequence(direct1_states, dir_dir1, prefix="d1")
render_state_sequence(directH_states, dir_dirH, prefix="dH")

print("[Done] All frames saved.")

