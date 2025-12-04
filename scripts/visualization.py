import os
import numpy as np
import imageio

import diffuser.sampling as sampling
import diffuser.utils as utils


# -----------------------------------------------------------------------------#
#                                   Parser                                      #
# -----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'   # 按需改
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

    # 把外面 TimeLimit 等 wrapper 的步骤计数也重置一下
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

horizon = diffusion.horizon


# -----------------------------------------------------------------------------#
#                  1) Fixed-plan rollout: 执行第一次 plan 的整条 action         #
# -----------------------------------------------------------------------------#

# 从初始 obs0 sample 一条 plan
conditions0 = {0: obs0}
action0, samples0 = policy(conditions0, batch_size=1, verbose=args.verbose)

if not hasattr(samples0, "actions"):
    raise RuntimeError("samples0 没有 actions 字段，policy 没有返回整条 action 计划？")

plan_actions = samples0.actions[0]   # shape: [H, act_dim]
print("[Info] Got fixed plan actions with shape:", plan_actions.shape)

# 确保从 init_state 开始
restore_env_state(env, init_state)
obs = obs0.copy()

fixed_states = [base_env.state_vector().copy()]

for t in range(horizon):
    a = plan_actions[t]
    obs, r, done, info = env.step(a)
    fixed_states.append(base_env.state_vector().copy())

    if done:
        print(f"[Fixed-plan] episode done at t={t}")
        break

fixed_states = np.stack(fixed_states, axis=0)
print("[Info] Fixed-plan rollout length:", len(fixed_states))


# -----------------------------------------------------------------------------#
#              2) Replan rollout: 每一步都重新 plan，只执行第一步 action       #
# -----------------------------------------------------------------------------#

restore_env_state(env, init_state)
obs = obs0.copy()

replan_states = [base_env.state_vector().copy()]

max_steps = horizon  # 为了公平起见也跑 horizon 步，你也可以单独设一个更大的 T

for t in range(max_steps):
    if t % 10 == 0:
        print(f"[Replan] t={t}", flush=True)

    conditions = {0: obs}
    action_t, samples_t = policy(conditions, batch_size=1, verbose=False)

    obs, r, done, info = env.step(action_t)
    replan_states.append(base_env.state_vector().copy())

    if done:
        print(f"[Replan] episode done at t={t}")
        break

replan_states = np.stack(replan_states, axis=0)
print("[Info] Replan rollout length:", len(replan_states))


# -----------------------------------------------------------------------------#
#                             Rendering to images                               #
# -----------------------------------------------------------------------------#

if args.savepath is None:
    save_root = os.path.join(args.loadbase, args.dataset, "two_rollout_images")
else:
    save_root = os.path.join(args.savepath, "two_rollout_images")

fixed_dir = os.path.join(save_root, "fixed_plan")
replan_dir = os.path.join(save_root, "replan")
os.makedirs(fixed_dir, exist_ok=True)
os.makedirs(replan_dir, exist_ok=True)

print(f"[Saving] Fixed-plan images -> {fixed_dir}")
print(f"[Saving] Replan images     -> {replan_dir}")


def render_state_sequence(states, out_dir, prefix):
    """
    给定一串 MUJOCO state（qpos+qvel），逐帧用 renderer.render() 存成 png.
    """
    for i, st in enumerate(states):
        img = renderer.render(st)   # 这里传的是完整 state，不是 obs
        fname = os.path.join(out_dir, f"{prefix}_{i:04d}.png")
        imageio.imwrite(fname, img)


render_state_sequence(fixed_states, fixed_dir, prefix="fixed")
render_state_sequence(replan_states, replan_dir, prefix="replan")

print("[Done] All frames saved.")

