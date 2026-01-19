"""
PushWorld PLR Training with Benchmark Puzzles.

This script trains an RL agent on PushWorld puzzle environments using PLR
(Prioritized Level Replay) with a FIXED BENCHMARK of puzzles instead of 
randomly generating levels.

Key difference from pushworld_plr.py:
- Uses a pre-loaded benchmark of puzzles as the training set
- PLR prioritizes which puzzles from the benchmark to train on
- No random level generation - all levels come from the benchmark

Usage:
    python pushworld_plr_benchmark.py                              # Use default config
    python pushworld_plr_benchmark.py benchmark_path=/path/to.pkl  # Use custom benchmark
    python pushworld_plr_benchmark.py seed=42 num_updates=1000     # Override parameters
"""

import json
import os
import time
from enum import IntEnum
from typing import Sequence, Tuple

import chex
import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import core, struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState as BaseTrainState
from omegaconf import DictConfig, OmegaConf

import wandb
from jaxued.environments.pushworld import (
    Actions,
    EnvParams,
    EnvState,
    Level,
    Observation,
    PushWorld,
    PushWorldRenderer,
    Benchmark,
)
from jaxued.environments.underspecified_env import UnderspecifiedEnv
from jaxued.level_sampler import LevelSampler
from jaxued.linen import ResetRNN
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss
from jaxued.wrappers import AutoReplayWrapper


class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1


class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    # === Below is used for logging ===
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)


# region PPO helper functions
def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Compute Generalized Advantage Estimation."""

    def compute_gae_at_timestep(carry, x):
        gae, next_value = carry
        value, reward, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values


def sample_trajectories_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
) -> Tuple[
    Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array],
    Tuple[
        Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict
    ],
]:
    """Sample trajectories using the RNN policy."""

    def sample_step(carry, _):
        rng, train_state, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, last_done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        carry = (rng, train_state, hstate, next_obs, env_state, done)
        return carry, (obs, action, reward, done, log_prob, value, info)

    (rng, train_state, hstate, last_obs, last_env_state, last_done), traj = (
        jax.lax.scan(
            sample_step,
            (
                rng,
                train_state,
                init_hstate,
                init_obs,
                init_env_state,
                jnp.zeros(num_envs, dtype=bool),
            ),
            None,
            length=max_episode_length,
        )
    )

    x = jax.tree_util.tree_map(lambda x: x[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)

    return (
        rng,
        train_state,
        hstate,
        last_obs,
        last_env_state,
        last_value.squeeze(0),
    ), traj


def evaluate_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    max_episode_length: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Evaluate the RNN policy on given levels."""
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]

    def step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, _ = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(rng_step, num_levels), state, action, env_params
        )

        next_mask = mask & ~done
        episode_length += mask

        return (rng, hstate, obs, next_state, done, next_mask, episode_length), (
            state,
            reward,
        )

    (_, _, _, _, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (
            rng,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_levels, dtype=bool),
            jnp.ones(num_levels, dtype=bool),
            jnp.zeros(num_levels, dtype=jnp.int32),
        ),
        None,
        length=max_episode_length,
    )

    return states, rewards, episode_lengths


def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool = True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """Update the actor-critic network using PPO."""
    obs, actions, dones, log_probs, values, targets, advantages = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages

    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            (
                init_hstate,
                obs,
                actions,
                last_dones,
                log_probs,
                values,
                targets,
                advantages,
            ) = minibatch

            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn(
                    params, (obs, last_dones), init_hstate
                )
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                ratio = jnp.exp(log_probs_pred - log_probs)
                A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                l_clip = (
                    -jnp.minimum(
                        ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A
                    )
                ).mean()

                values_pred_clipped = values + (values_pred - values).clip(
                    -clip_eps, clip_eps
                )
                l_vf = (
                    0.5
                    * jnp.maximum(
                        (values_pred - targets) ** 2,
                        (values_pred_clipped - targets) ** 2,
                    ).mean()
                )

                loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy

                return loss, (l_vf, l_clip, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0).reshape(
                    n_minibatch, -1, *x.shape[1:]
                ),
                init_hstate,
            ),
            *jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1)
                .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                .swapaxes(0, 1),
                batch,
            ),
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)


class PushWorldActorCritic(nn.Module):
    """Actor-Critic network for PushWorld environment."""

    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs
        img = obs.image

        # CNN embedding
        img_embed = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(img)
        img_embed = nn.relu(img_embed)
        img_embed = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(
            img_embed
        )
        img_embed = nn.relu(img_embed)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)

        # Dense layer
        embedding = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="embed",
        )(img_embed)
        embedding = nn.relu(embedding)

        # LSTM
        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))(
            (embedding, dones), initial_carry=hidden
        )

        # Actor head
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0"
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor1",
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic head
        critic = nn.Dense(
            64, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0"
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1"
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(
            jax.random.PRNGKey(0), (*batch_dims, 256)
        )


# endregion


# region checkpointing
def setup_checkpointing(
    config: dict, train_state: TrainState, env: UnderspecifiedEnv, env_params: EnvParams
) -> ocp.CheckpointManager:
    """Set up checkpoint manager for saving training progress."""
    overall_save_dir = os.path.join(
        os.getcwd(), "checkpoints", f"{config['run_name']}", str(config["seed"])
    )
    os.makedirs(overall_save_dir, exist_ok=True)

    with open(os.path.join(overall_save_dir, "config.json"), "w+") as f:
        f.write(json.dumps(config, indent=True))

    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, "models"),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config["checkpoint_save_interval"],
            max_to_keep=config["max_number_of_checkpoints"],
        ),
    )
    return checkpoint_manager


# endregion


def train_state_to_log_dict(
    train_state: TrainState, level_sampler: LevelSampler
) -> dict:
    """Extract logging information from train state."""
    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    s = jnp.maximum(idx.sum(), 1)
    return {
        "log": {
            "level_sampler/size": sampler["size"],
            "level_sampler/episode_count": sampler["episode_count"],
            "level_sampler/max_score": sampler["scores"].max(),
            "level_sampler/weighted_score": (
                sampler["scores"] * level_sampler.level_weights(sampler)
            ).sum(),
            "level_sampler/mean_score": (sampler["scores"] * idx).sum() / s,
        },
        "info": {
            "num_dr_updates": train_state.num_dr_updates,
            "num_replay_updates": train_state.num_replay_updates,
            "num_mutation_updates": train_state.num_mutation_updates,
        },
    }


def compute_score(config, dones, values, max_returns, advantages):
    """Compute PLR score for levels."""
    if config["score_function"] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config["score_function"] == "pvl":
        return positive_value_loss(dones, advantages)
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")


@hydra.main(version_base=None, config_path="../configs", config_name="pushworld_plr_benchmark")
def main(cfg: DictConfig):
    # Convert OmegaConf to dict for compatibility
    config = OmegaConf.to_container(cfg, resolve=True)

    # Determine tags for wandb
    tags = []
    if not config["exploratory_grad_updates"]:
        tags.append("robust")
    tags.append("PLR-benchmark")
    tags.append("pushworld")

    # Initialize wandb
    if config["use_wandb"]:
        wandb.init(
            config=config,
            project=config["wandb_project"],
            group=config["run_name"],
            tags=tags,
            mode=config.get("wandb_mode", "online"),
        )
        config = dict(wandb.config)

    if config["use_wandb"]:
        wandb.define_metric("num_updates")
        wandb.define_metric("num_env_steps")
        wandb.define_metric("solve_rate/*", step_metric="num_updates")
        wandb.define_metric("level_sampler/*", step_metric="num_updates")
        wandb.define_metric("agent/*", step_metric="num_updates")
        wandb.define_metric("return/*", step_metric="num_updates")
        wandb.define_metric("eval_ep_lengths/*", step_metric="num_updates")

    def log_eval(stats, train_state_info):
        print(f"Logging update: {stats['update_count']}")

        env_steps = (
            stats["update_count"] * config["num_train_envs"] * config["num_steps"]
        )
        log_dict = {
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps / stats["time_delta"],
        }

        solve_rates = stats["eval_solve_rates"]
        returns = stats["eval_returns"]
        log_dict.update(
            {
                f"solve_rate/{name}": solve_rate
                for name, solve_rate in zip(config["eval_levels"], solve_rates)
            }
        )
        log_dict.update({"solve_rate/mean": solve_rates.mean()})
        log_dict.update(
            {f"return/{name}": ret for name, ret in zip(config["eval_levels"], returns)}
        )
        log_dict.update({"return/mean": returns.mean()})
        log_dict.update({"eval_ep_lengths/mean": stats["eval_ep_lengths"].mean()})

        log_dict.update(train_state_info["log"])

        log_dict.update(
            {
                "images/highest_scoring_level": wandb.Image(
                    np.array(stats["highest_scoring_level"]),
                    caption="Highest scoring level",
                )
            }
        )
        log_dict.update(
            {
                "images/highest_weighted_level": wandb.Image(
                    np.array(stats["highest_weighted_level"]),
                    caption="Highest weighted level",
                )
            }
        )

        for s in ["dr", "replay", "mutation"]:
            if train_state_info["info"][f"num_{s}_updates"] > 0:
                log_dict.update(
                    {
                        f"images/{s}_levels": [
                            wandb.Image(np.array(image))
                            for image in stats[f"{s}_levels"]
                        ]
                    }
                )

        for i, level_name in enumerate(config["eval_levels"]):
            frames, episode_length = (
                stats["eval_animation"][0][:, i],
                stats["eval_animation"][1][i],
            )
            frames = np.array(frames[:episode_length])
            log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4)})

        if config["use_wandb"]:
            wandb.log(log_dict)

    # =========================================================================
    # LOAD BENCHMARK - This is the key difference from pushworld_plr.py!
    # =========================================================================
    benchmark_name = config.get("benchmark_name")
    benchmark_path = config.get("benchmark_path")
    
    if benchmark_name:
        # Load by name (auto-downloads from HuggingFace if needed)
        print(f"Loading benchmark by name: {benchmark_name}")
        benchmark = Benchmark.load(benchmark_name)
    elif benchmark_path:
        # Load from explicit path
        print(f"Loading benchmark from path: {benchmark_path}")
        benchmark = Benchmark.load_from_path(benchmark_path)
    else:
        raise ValueError("Must specify either 'benchmark_name' or 'benchmark_path' in config")
    num_train_puzzles = benchmark.num_train_puzzles()
    num_test_puzzles = benchmark.num_test_puzzles()
    print(f"Loaded benchmark with {num_train_puzzles} train puzzles and {num_test_puzzles} test puzzles")

    # Create a function that samples from the benchmark
    def sample_level_from_benchmark(key: chex.PRNGKey) -> Level:
        """Sample a random level from the benchmark."""
        return benchmark.sample_puzzle(key, puzzle_type="train")

    # Setup the PushWorld environment
    env = PushWorld(
        penalize_time=config["penalize_time"],
        reward_shaping=config["reward_shaping"],
    )
    eval_env = env

    # Create renderer for visualization
    env_renderer = PushWorldRenderer(env, tile_size=8, render_grid_lines=False)

    # Wrap environment for auto-replay
    env = AutoReplayWrapper(env)
    env_params = EnvParams(max_steps_in_episode=config["max_steps_in_episode"])

    # Level sampler - we set capacity to num_train_puzzles since we have a fixed set
    # We'll fill the buffer with ALL training puzzles upfront
    level_sampler = LevelSampler(
        capacity=num_train_puzzles,  # Buffer size = number of training puzzles
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=0.0,  # Start replaying immediately since we pre-fill
        prioritization=config["prioritization"],
        prioritization_params={
            "temperature": config["temperature"],
            "k": config["topk_k"],
        },
        duplicate_check=False,  # No duplicates since we load fixed puzzles
    )

    @jax.jit
    def create_train_state(rng) -> TrainState:
        """Creates the train state with pre-filled level buffer."""
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["num_minibatches"] * config["epoch_ppo"]))
                / config["num_updates"]
            )
            return config["lr"] * frac

        # Sample a puzzle for network initialization
        rng, rng_level = jax.random.split(rng)
        sample_level = sample_level_from_benchmark(rng_level)
        obs, _ = env.reset_to_level(rng, sample_level, env_params)
        obs = jax.tree_util.tree_map(
            lambda x: jnp.repeat(
                jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...],
                256,
                axis=0,
            ),
            obs,
        )
        init_x = (obs, jnp.zeros((256, config["num_train_envs"])))

        # Initialize network
        network = PushWorldActorCritic(len(Actions))
        network_params = network.init(
            rng,
            init_x,
            PushWorldActorCritic.initialize_carry((config["num_train_envs"],)),
        )
        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )

        # Initialize sampler with a placeholder
        pholder_level = sample_level_from_benchmark(jax.random.PRNGKey(0))
        sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})

        # Pre-fill the buffer with ALL training puzzles
        all_train_levels = benchmark.get_all_train_puzzles()
        initial_scores = jnp.zeros(num_train_puzzles)  # Start with equal scores
        initial_max_returns = jnp.full(num_train_puzzles, -jnp.inf)
        sampler, _ = level_sampler.insert_batch(
            sampler, all_train_levels, initial_scores, {"max_return": initial_max_returns}
        )

        pholder_level_batch = jax.tree_util.tree_map(
            lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0),
            pholder_level,
        )
        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            update_state=0,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
            mutation_last_level_batch=pholder_level_batch,
        )

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        """Main training step - always replays from the benchmark buffer."""

        def on_replay_levels(rng: chex.PRNGKey, train_state: TrainState):
            """Sample levels from buffer and update policy.
            
            Since we use a fixed benchmark, this is the ONLY update type.
            We always sample from the pre-filled buffer.
            """
            sampler = train_state.sampler

            # Sample levels from buffer (weighted by PLR scores)
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(
                sampler, rng_levels, config["num_train_envs"]
            )

            # Reset environments to sampled levels
            init_obs, init_env_state = jax.vmap(
                env.reset_to_level, in_axes=(0, 0, None)
            )(jax.random.split(rng_reset, config["num_train_envs"]), levels, env_params)

            # Collect trajectories
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                PushWorldActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs,
                init_env_state,
                config["num_train_envs"],
                config["num_steps"],
            )

            # Compute advantages and update scores
            advantages, targets = compute_gae(
                config["gamma"],
                config["gae_lambda"],
                last_value,
                values,
                rewards,
                dones,
            )
            max_returns = jnp.maximum(
                level_sampler.get_levels_extra(sampler, level_inds)["max_return"],
                compute_max_returns(dones, rewards),
            )
            scores = compute_score(config, dones, values, max_returns, advantages)

            # Update scores in buffer
            sampler = level_sampler.update_batch(
                sampler, level_inds, scores, {"max_return": max_returns}
            )

            # Update the policy
            (rng, train_state), losses = update_actor_critic_rnn(
                rng,
                train_state,
                PushWorldActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=True,
            )

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_walls": levels.wall_map.sum() / config["num_train_envs"],
            }

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.REPLAY,
                num_replay_updates=train_state.num_replay_updates + 1,
                replay_last_level_batch=levels,
            )
            return (rng, train_state), metrics

        rng, train_state = carry
        # Always do replay since we have a fixed benchmark
        return on_replay_levels(rng, train_state)

    def eval(rng: chex.PRNGKey, train_state: TrainState):
        """Evaluate the current policy on evaluation levels."""
        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config["eval_levels"])
        num_levels = len(config["eval_levels"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_env_to_level, (0, 0, None))(
            jax.random.split(rng_reset, num_levels), levels, env_params
        )
        states, rewards, episode_lengths = evaluate_rnn(
            rng,
            eval_env,
            env_params,
            train_state,
            PushWorldActorCritic.initialize_carry((num_levels,)),
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return (
            states,
            cum_rewards,
            episode_lengths,
        )

    @jax.jit
    def train_and_eval_step(runner_state, _):
        """Run training for eval_freq steps, then evaluate."""
        # Train
        (rng, train_state), metrics = jax.lax.scan(
            train_step, runner_state, None, config["eval_freq"]
        )

        # Eval
        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(
            jax.random.split(rng_eval, config["eval_num_attempts"]), train_state
        )

        # Collect Metrics
        eval_solve_rates = jnp.where(cum_rewards > 0, 1.0, 0.0).mean(axis=0)
        eval_returns = cum_rewards.mean(axis=0)

        states, episode_lengths = jax.tree_util.tree_map(
            lambda x: x[0], (states, episode_lengths)
        )

        # Render states for animation
        images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(
            states, env_params
        )
        frames = images.transpose(0, 1, 4, 2, 3)

        metrics["update_count"] = (
            train_state.num_dr_updates
            + train_state.num_replay_updates
            + train_state.num_mutation_updates
        )
        metrics["eval_returns"] = eval_returns
        metrics["eval_solve_rates"] = eval_solve_rates
        metrics["eval_ep_lengths"] = episode_lengths
        metrics["eval_animation"] = (frames, episode_lengths)
        metrics["dr_levels"] = jax.vmap(env_renderer.render_level, (0, None))(
            train_state.dr_last_level_batch, env_params
        )
        metrics["replay_levels"] = jax.vmap(env_renderer.render_level, (0, None))(
            train_state.replay_last_level_batch, env_params
        )
        metrics["mutation_levels"] = jax.vmap(env_renderer.render_level, (0, None))(
            train_state.mutation_last_level_batch, env_params
        )

        highest_scoring_level = level_sampler.get_levels(
            train_state.sampler, train_state.sampler["scores"].argmax()
        )
        highest_weighted_level = level_sampler.get_levels(
            train_state.sampler,
            level_sampler.level_weights(train_state.sampler).argmax(),
        )

        metrics["highest_scoring_level"] = env_renderer.render_level(
            highest_scoring_level, env_params
        )
        metrics["highest_weighted_level"] = env_renderer.render_level(
            highest_weighted_level, env_params
        )

        return (rng, train_state), metrics

    # Set up the train states
    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)

    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    print(f"Buffer pre-filled with {train_state.sampler['size']} puzzles")

    # And run the train_eval_sep function for the specified number of updates
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)
    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics["time_delta"] = curr_time - start_time
        log_eval(metrics, train_state_to_log_dict(runner_state[1], level_sampler))
        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(
                eval_step, args=ocp.args.StandardSave(runner_state[1])
            )
            checkpoint_manager.wait_until_finished()
    return runner_state[1]


if __name__ == "__main__":
    main()
