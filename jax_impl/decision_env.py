"""High-level decision environment for pedestrian crossing + SDC driving.

Pedestrian locomotion is scripted (graph pathfinding).
RL controls high-level decisions only:
  - Pedestrian: go or wait (jaywalking is personality-driven)
  - SDC: acceleration and steering

Runs entirely in JAX on GPU.
"""

import os
import jax
import jax.numpy as jnp
from functools import partial
import heapq

# Map constants (same layout)
MAP_W = 120.0
MAP_H = 120.0
NUM_PEDS = int(os.environ.get("MARL_NUM_PEDS", "12"))
JW_MULT = float(os.environ.get("MARL_JW_MULT", "0.25"))
DT = 0.1
MAX_STEPS = 500
SDC_WHEELBASE = 2.5
SDC_MAX_SPEED = 8.33
SDC_MAX_ACCEL = 3.0
SDC_MAX_STEER = 0.52
COLLISION_DIST = 1.5

ROAD_RECTS = jnp.array([
    [0, 52, 120, 60], [52, 0, 60, 120], [60, 82, 120, 90],
], dtype=jnp.float32)

SIDEWALK_RECTS = jnp.array([
    [0, 50, 50, 52], [62, 50, 120, 52],
    [0, 60, 50, 62], [62, 60, 120, 62],
    [50, 0, 52, 50], [50, 62, 52, 80], [50, 92, 52, 120],
    [60, 0, 62, 50], [60, 62, 62, 80], [60, 92, 62, 120],
    [62, 80, 120, 82], [62, 90, 120, 92],
    [50, 50, 52, 52], [60, 50, 62, 52],
    [50, 60, 52, 62], [60, 60, 62, 62],
    [50, 80, 52, 82], [60, 80, 62, 82],
    [50, 90, 52, 92], [60, 90, 62, 92],
], dtype=jnp.float32)

CROSSWALK_RECTS = jnp.array([
    [48, 50, 52, 62], [60, 50, 64, 62],
    [50, 48, 62, 52], [50, 60, 62, 64],
    [48, 80, 52, 92], [60, 80, 64, 92],
], dtype=jnp.float32)

WAYPOINTS = jnp.array([
    [5,51],[25,51],[47,51],[65,51],[90,51],[115,51],
    [5,61],[25,61],[47,61],[65,61],[90,61],[115,61],
    [51,5],[51,25],[51,47],[51,65],[51,75],[51,79],[51,93],[51,110],
    [61,5],[61,25],[61,47],[61,65],[61,75],[61,79],[61,93],[61,110],
    [75,81],[100,81],[115,81],[75,91],[100,91],[115,91],
], dtype=jnp.float32)
NUM_WAYPOINTS = WAYPOINTS.shape[0]

# Navigation graph nodes
GRAPH_NODES = jnp.array([
    [5,51],[25,51],[47,51],[65,51],[90,51],[115,51],         # 0-5
    [5,61],[25,61],[47,61],[65,61],[90,61],[115,61],         # 6-11
    [51,5],[51,25],[51,47],[51,65],[51,75],[51,79],          # 12-17
    [51,93],[51,110],                                         # 18-19
    [61,5],[61,25],[61,47],[61,65],[61,75],[61,79],          # 20-25
    [61,93],[61,110],                                         # 26-27
    [75,81],[100,81],[115,81],                                # 28-30
    [75,91],[100,91],[115,91],                                # 31-33
    [50,56],[62,56],[56,50],[56,62],[50,86],[62,86],         # 34-39: crosswalk mids
], dtype=jnp.float32)
NUM_GRAPH = GRAPH_NODES.shape[0]

# Pre-computed shortest paths (all pairs) as next-hop table
# _next_hop[i, j] = the next node to visit from i to reach j
# Computed once at module load time
EDGES = [
    (0,1),(1,2),(3,4),(4,5),(6,7),(7,8),(9,10),(10,11),
    (12,13),(13,14),(15,16),(16,17),(18,19),
    (20,21),(21,22),(23,24),(24,25),(26,27),
    (28,29),(29,30),(31,32),(32,33),
    (2,14),(3,22),(8,15),(9,23),(17,28),(25,28),(18,31),(26,31),
    (2,34),(34,8),(3,35),(35,9),(14,36),(36,22),(15,37),(37,23),
    (17,38),(38,18),(25,39),(39,26),
]

def _build_next_hop():
    """Build all-pairs next-hop table using Floyd-Warshall style BFS."""
    n = NUM_GRAPH
    adj = [[] for _ in range(n)]
    costs = {}
    for a, b in EDGES:
        d = float(jnp.sqrt((GRAPH_NODES[a,0]-GRAPH_NODES[b,0])**2 + (GRAPH_NODES[a,1]-GRAPH_NODES[b,1])**2))
        adj[a].append(b); adj[b].append(a)
        costs[(a,b)] = d; costs[(b,a)] = d

    next_hop = [[i for _ in range(n)] for i in range(n)]  # next_hop[src][dst]

    for src in range(n):
        dist = [float('inf')] * n
        dist[src] = 0.0
        prev = [-1] * n
        pq = [(0.0, src)]
        visited = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited: continue
            visited.add(u)
            for v in adj[u]:
                nd = d + costs[(u,v)]
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        # Build next-hop from prev
        for dst in range(n):
            if dst == src:
                next_hop[src][dst] = src
                continue
            node = dst
            while prev[node] != src and prev[node] != -1:
                node = prev[node]
            next_hop[src][dst] = node if prev[node] == src else src

    return jnp.array(next_hop, dtype=jnp.int32)

NEXT_HOP = _build_next_hop()

# Crosswalk midpoint node indices
CW_MID_NODES = jnp.array([34, 35, 36, 37, 38, 39], dtype=jnp.int32)

# Pre-compute which edges are crosswalk crossings
# A crossing edge leads to or from a crosswalk midpoint
CW_EDGE_NODES = set()
for a, b in EDGES:
    if a >= 34 or b >= 34:
        CW_EDGE_NODES.add(a)
        CW_EDGE_NODES.add(b)

# Geometry helpers
def point_in_rects(x, y, rects):
    return jnp.any((x >= rects[:,0]) & (x <= rects[:,2]) & (y >= rects[:,1]) & (y <= rects[:,3]))

def dist_to_rects(x, y, rects):
    cx = jnp.clip(x, rects[:,0], rects[:,2])
    cy = jnp.clip(y, rects[:,1], rects[:,3])
    return jnp.min(jnp.sqrt((x-cx)**2 + (y-cy)**2))

def normalize_angle(a):
    return a - 2*jnp.pi * jnp.floor((a + jnp.pi) / (2*jnp.pi))

def nearest_node(x, y):
    return jnp.argmin((x - GRAPH_NODES[:,0])**2 + (y - GRAPH_NODES[:,1])**2)

# Pedestrian high-level RL action space: Discrete(2)
#   0 = go (cross; crosswalk vs jaywalk decided by personality)
#   1 = wait (stop moving)

PED_OBS_SIZE = 20  # compact observation for high-level decisions
SDC_OBS_SIZE = 34  # SDC observation (30 base + 4 lane info)

# State and step functions

PED_SPAWN_IDXS = jnp.array([0, 7, 2, 9, 4, 11, 12, 14, 21, 24, 18, 27], dtype=jnp.int32)
SDC_SPAWNS = jnp.array([
    [10,56],[30,56],[90,56],[110,56],[56,10],[56,30],[56,100],[56,110],[80,86],[100,86],
], dtype=jnp.float32)


def reset(key):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # Spawn peds at graph nodes
    spawn_nodes = jax.random.choice(k1, NUM_GRAPH - 6, shape=(NUM_PEDS,))  # avoid crosswalk mids
    ped_x = GRAPH_NODES[spawn_nodes, 0] + jax.random.uniform(k2, (NUM_PEDS,), minval=-0.5, maxval=0.5)
    ped_y = GRAPH_NODES[spawn_nodes, 1] + jax.random.uniform(k3, (NUM_PEDS,), minval=-0.5, maxval=0.5)
    traits = jax.random.uniform(k4, (NUM_PEDS, 4))
    wp_idx = jax.random.randint(k5, (NUM_PEDS,), 0, NUM_WAYPOINTS)

    # Ped navigation state
    ped_curr_node = jax.vmap(nearest_node)(ped_x, ped_y)
    ped_goal_node = jax.vmap(nearest_node)(WAYPOINTS[wp_idx, 0], WAYPOINTS[wp_idx, 1])
    ped_path_progress = jnp.zeros(NUM_PEDS, dtype=jnp.int32)  # steps along path

    # SDC
    k5a, k5b = jax.random.split(k5)
    si = jax.random.randint(k5a, (), 0, 10)
    gi = (si + jax.random.randint(k5b, (), 1, 10)) % 10
    sdc_x, sdc_y = SDC_SPAWNS[si, 0], SDC_SPAWNS[si, 1]
    sdc_gx, sdc_gy = SDC_SPAWNS[gi, 0], SDC_SPAWNS[gi, 1]
    sdc_heading = jnp.arctan2(sdc_gy - sdc_y, sdc_gx - sdc_x)

    return {
        "ped_x": ped_x, "ped_y": ped_y,
        "ped_heading": jnp.zeros(NUM_PEDS),
        "ped_speed": jnp.zeros(NUM_PEDS),
        "ped_traits": traits,
        "ped_wp_idx": wp_idx,
        "ped_wp_reached": jnp.zeros(NUM_PEDS, dtype=jnp.int32),
        "ped_curr_node": ped_curr_node,
        "ped_goal_node": ped_goal_node,
        "ped_on_road_steps": jnp.zeros(NUM_PEDS, dtype=jnp.int32),
        "ped_alive": jnp.ones(NUM_PEDS, dtype=jnp.bool_),
        "sdc_x": sdc_x, "sdc_y": sdc_y,
        "sdc_heading": sdc_heading,
        "sdc_speed": jnp.float32(0),
        "sdc_steering": jnp.float32(0),
        "sdc_goal_x": sdc_gx, "sdc_goal_y": sdc_gy,
        "sdc_prev_speed": jnp.float32(0),
        "sdc_dist_to_goal_prev": jnp.sqrt((sdc_gx-sdc_x)**2 + (sdc_gy-sdc_y)**2),
        "step_count": jnp.int32(0),
        "done": jnp.bool_(False),
    }


def _move_ped_along_path(i, state, ped_action, key):
    """Move one pedestrian based on high-level RL action."""
    px, py = state["ped_x"][i], state["ped_y"][i]
    wp = WAYPOINTS[state["ped_wp_idx"][i]]
    curr = state["ped_curr_node"][i]
    goal = state["ped_goal_node"][i]
    traits = state["ped_traits"][i]
    speed_pref = traits[3]

    # Base speed from personality
    base_speed = 1.0 + speed_pref * 1.0  # 1.0 to 2.0 m/s

    # RL decides: GO (action=0) or WAIT (action=1)
    # If GO: personality determines crosswalk vs jaywalk
    should_wait = (ped_action == 1)

    # If going, jaywalking depends on personality
    jt = traits[0]
    key, subkey = jax.random.split(key)
    jaywalk_roll = jax.random.uniform(subkey)
    should_jaywalk = (jaywalk_roll < jt * JW_MULT) & ~should_wait

    # Get next node along graph path
    next_node = NEXT_HOP[curr, goal]
    next_pos = GRAPH_NODES[next_node]

    # Target
    target_x = jnp.where(should_wait, px,
               jnp.where(should_jaywalk, wp[0], next_pos[0]))
    target_y = jnp.where(should_wait, py,
               jnp.where(should_jaywalk, wp[1], next_pos[1]))
    speed = jnp.where(should_wait, 0.0, base_speed)

    # Move toward target
    dx = target_x - px
    dy = target_y - py
    dist = jnp.sqrt(dx**2 + dy**2) + 1e-6
    move = jnp.minimum(speed * DT, dist)
    nx, ny = dx / dist, dy / dist

    new_x = px + nx * move * state["ped_alive"][i]
    new_y = py + ny * move * state["ped_alive"][i]
    new_x = jnp.clip(new_x, 1.0, MAP_W - 1.0)
    new_y = jnp.clip(new_y, 1.0, MAP_H - 1.0)

    # Update current node if we reached the next one
    d_to_next = jnp.sqrt((new_x - next_pos[0])**2 + (new_y - next_pos[1])**2)
    new_curr = jnp.where(d_to_next < 1.5, next_node, curr)

    # Check waypoint reached
    d_to_wp = jnp.sqrt((new_x - wp[0])**2 + (new_y - wp[1])**2)
    reached = d_to_wp < 2.5
    key, subkey = jax.random.split(key)
    new_wp_idx = jnp.where(reached,
                            jax.random.randint(subkey, (), 0, NUM_WAYPOINTS),
                            state["ped_wp_idx"][i])
    new_goal = jnp.where(reached,
                          nearest_node(WAYPOINTS[new_wp_idx, 0], WAYPOINTS[new_wp_idx, 1]),
                          goal)

    # Track road steps
    on_road = point_in_rects(new_x, new_y, ROAD_RECTS) & ~point_in_rects(new_x, new_y, CROSSWALK_RECTS)
    new_road_steps = jnp.where(on_road, state["ped_on_road_steps"][i] + 1,
                                jnp.maximum(state["ped_on_road_steps"][i] - 1, 0))

    heading = jnp.arctan2(ny, nx)

    return (new_x, new_y, heading, speed, new_curr, new_goal, new_wp_idx,
            state["ped_wp_reached"][i] + reached.astype(jnp.int32), new_road_steps, key)


def step(state, ped_actions, sdc_action, key):
    """Step environment. MARL: peds decide go/wait, SDC drives.

    ped_actions: (NUM_PEDS,) int32: 0=go, 1=wait
    sdc_action: (2,) float: [accel_norm, steer_norm] in [-1,1]
    """
    new_px = jnp.zeros(NUM_PEDS)
    new_py = jnp.zeros(NUM_PEDS)
    new_heading = jnp.zeros(NUM_PEDS)
    new_speed = jnp.zeros(NUM_PEDS)
    new_curr = jnp.zeros(NUM_PEDS, dtype=jnp.int32)
    new_goal = jnp.zeros(NUM_PEDS, dtype=jnp.int32)
    new_wp_idx = jnp.zeros(NUM_PEDS, dtype=jnp.int32)
    new_wp_reached = jnp.zeros(NUM_PEDS, dtype=jnp.int32)
    new_road_steps = jnp.zeros(NUM_PEDS, dtype=jnp.int32)

    def ped_loop(i, carry):
        (px, py, hd, sp, cn, gn, wi, wr, rs, key) = carry
        result = _move_ped_along_path(i, state, ped_actions[i], key)
        npx, npy, nhd, nsp, ncn, ngn, nwi, nwr, nrs, key = result
        px = px.at[i].set(npx)
        py = py.at[i].set(npy)
        hd = hd.at[i].set(nhd)
        sp = sp.at[i].set(nsp)
        cn = cn.at[i].set(ncn)
        gn = gn.at[i].set(ngn)
        wi = wi.at[i].set(nwi)
        wr = wr.at[i].set(nwr)
        rs = rs.at[i].set(nrs)
        return (px, py, hd, sp, cn, gn, wi, wr, rs, key)

    carry = (new_px, new_py, new_heading, new_speed, new_curr, new_goal,
             new_wp_idx, new_wp_reached, new_road_steps, key)
    carry = jax.lax.fori_loop(0, NUM_PEDS, ped_loop, carry)
    new_px, new_py, new_heading, new_speed, new_curr, new_goal, new_wp_idx, new_wp_reached, new_road_steps, key = carry

    # Step SDC
    accel = sdc_action[0] * SDC_MAX_ACCEL
    steering = sdc_action[1] * SDC_MAX_STEER
    sdc_new_x = state["sdc_x"] + state["sdc_speed"] * jnp.cos(state["sdc_heading"]) * DT
    sdc_new_y = state["sdc_y"] + state["sdc_speed"] * jnp.sin(state["sdc_heading"]) * DT
    heading_delta = jnp.where(jnp.abs(state["sdc_speed"]) > 0.01,
                              (state["sdc_speed"] / SDC_WHEELBASE) * jnp.tan(steering) * DT, 0.0)
    sdc_new_heading = normalize_angle(state["sdc_heading"] + heading_delta)
    sdc_new_speed = jnp.clip(state["sdc_speed"] + accel * DT, 0.0, SDC_MAX_SPEED)

    # SDC road constraint (0.5m margin: keeps on road without making driving too hard)
    on_road = point_in_rects(sdc_new_x, sdc_new_y, ROAD_RECTS)
    snap_rx = jnp.clip(sdc_new_x, ROAD_RECTS[:,0] + 0.5, ROAD_RECTS[:,2] - 0.5)
    snap_ry = jnp.clip(sdc_new_y, ROAD_RECTS[:,1] + 0.5, ROAD_RECTS[:,3] - 0.5)
    snap_d = jnp.sqrt((sdc_new_x-snap_rx)**2 + (sdc_new_y-snap_ry)**2)
    nr = jnp.argmin(snap_d)
    sdc_new_x = jnp.where(on_road, sdc_new_x, snap_rx[nr])
    sdc_new_y = jnp.where(on_road, sdc_new_y, snap_ry[nr])
    sdc_new_speed = jnp.where(on_road, sdc_new_speed, sdc_new_speed * 0.3)
    sdc_new_x = jnp.clip(sdc_new_x, 1.0, MAP_W - 1.0)
    sdc_new_y = jnp.clip(sdc_new_y, 1.0, MAP_H - 1.0)

    # Collisions
    ped_dists = jnp.sqrt((new_px - sdc_new_x)**2 + (new_py - sdc_new_y)**2)
    collisions = (ped_dists < COLLISION_DIST) & state["ped_alive"]
    any_collision = jnp.any(collisions)

    # Goal check
    goal_dist = jnp.sqrt((sdc_new_x - state["sdc_goal_x"])**2 + (sdc_new_y - state["sdc_goal_y"])**2)
    reached_goal = goal_dist < 3.0

    new_step = state["step_count"] + 1
    done = (new_step >= MAX_STEPS) | any_collision | reached_goal

    # Build new state
    new_state = {
        "ped_x": new_px, "ped_y": new_py,
        "ped_heading": new_heading, "ped_speed": new_speed,
        "ped_traits": state["ped_traits"],
        "ped_wp_idx": new_wp_idx,
        "ped_wp_reached": new_wp_reached,
        "ped_curr_node": new_curr,
        "ped_goal_node": new_goal,
        "ped_on_road_steps": new_road_steps,
        "ped_alive": state["ped_alive"] & ~collisions,
        "sdc_x": sdc_new_x, "sdc_y": sdc_new_y,
        "sdc_heading": sdc_new_heading,
        "sdc_speed": sdc_new_speed,
        "sdc_steering": steering,
        "sdc_goal_x": state["sdc_goal_x"], "sdc_goal_y": state["sdc_goal_y"],
        "sdc_prev_speed": state["sdc_speed"],
        "sdc_dist_to_goal_prev": goal_dist,
        "step_count": new_step,
        "done": done,
    }

    # Rewards
    ped_rewards = _ped_rewards(new_state, state, collisions)
    sdc_reward = _sdc_reward(new_state, state, any_collision, reached_goal)

    return new_state, ped_rewards, sdc_reward, done, {"collisions": jnp.sum(collisions), "sdc_reached_goal": reached_goal}


def _ped_rewards(new_state, old_state, collisions):
    """Ped rewards for MARL. Peds learn WHEN to go vs wait."""
    def single(i):
        r = jnp.float32(0.0)
        alive = new_state["ped_alive"][i]
        px, py = new_state["ped_x"][i], new_state["ped_y"][i]
        speed = new_state["ped_speed"][i]

        # Waypoint progress: reward for moving toward goal
        wp = WAYPOINTS[new_state["ped_wp_idx"][i]]
        d_new = jnp.sqrt((px-wp[0])**2 + (py-wp[1])**2)
        d_old = jnp.sqrt((old_state["ped_x"][i]-wp[0])**2 + (old_state["ped_y"][i]-wp[1])**2)
        r = r + (d_old - d_new) * 2.0

        # Waypoint reached
        r = r + jnp.where(d_new < 2.5, 5.0, 0.0)

        # Collision: devastating (this is what peds learn to avoid by waiting)
        r = r - jnp.where(collisions[i], 25.0, 0.0)

        # Waiting penalty (small): don't wait forever, only when needed
        r = r - jnp.where(speed < 0.1, 0.05, 0.0)

        # Smart waiting: near SDC + waiting = good decision
        sdc_dist = jnp.sqrt((px-new_state["sdc_x"])**2 + (py-new_state["sdc_y"])**2)
        sdc_fast = new_state["sdc_speed"] > 2.0
        r = r + jnp.where((sdc_dist < 8.0) & sdc_fast & (speed < 0.1), 0.3, 0.0)

        # Dumb waiting: no SDC nearby + waiting = bad
        r = r - jnp.where((sdc_dist > 15.0) & (speed < 0.1), 0.1, 0.0)

        return r * alive

    return jax.vmap(single)(jnp.arange(NUM_PEDS))


def _sdc_reward(new_state, old_state, any_collision, reached_goal):
    """Shaped reward for the SDC."""
    r = jnp.float32(0.0)
    sx, sy = new_state["sdc_x"], new_state["sdc_y"]
    speed = new_state["sdc_speed"]
    ped_dists = jnp.sqrt((new_state["ped_x"]-sx)**2 + (new_state["ped_y"]-sy)**2)
    alive = new_state["ped_alive"]

    # Goal progress
    d_new = jnp.sqrt((sx-new_state["sdc_goal_x"])**2 + (sy-new_state["sdc_goal_y"])**2)
    r = r + (old_state["sdc_dist_to_goal_prev"] - d_new) * 2.0

    # Goal reached (primary objective)
    r = r + jnp.where(reached_goal, 50.0, 0.0)

    # Collision: catastrophic
    r = r - jnp.where(any_collision, 50.0, 0.0)

    # Only consider peds on the road or crosswalk as threats
    peds_on_road = jax.vmap(lambda i: (
        point_in_rects(new_state["ped_x"][i], new_state["ped_y"][i], ROAD_RECTS) & alive[i]
    ))(jnp.arange(NUM_PEDS))
    threat_dists = jnp.where(peds_on_road, ped_dists, 999.0)
    nearest_threat = jnp.min(threat_dists)

    # Idle penalty: penalize stopping unless a ped is on the road nearby
    cw_dist = dist_to_rects(sx, sy, CROSSWALK_RECTS)
    ped_blocking = nearest_threat < 8.0  # ped on road within 8m
    r = r + jnp.where((speed < 0.5) & ~ped_blocking, -0.2, 0.0)  # stop = bad when road clear
    # Speed reward when road is clear
    r = r + jnp.where(~ped_blocking & (speed > 2.0), 0.05, 0.0)

    # Smoothness
    r = r - 0.005 * jnp.abs(speed - old_state["sdc_prev_speed"]) / DT

    # Off-road penalty
    on_road = point_in_rects(sx, sy, ROAD_RECTS)
    r = r - jnp.where(~on_road, 0.5, 0.0)

    # Crosswalk yielding
    peds_on_cw = jnp.any(
        jax.vmap(lambda i: point_in_rects(new_state["ped_x"][i], new_state["ped_y"][i], CROSSWALK_RECTS) & alive[i])(jnp.arange(NUM_PEDS))
    )
    # Penalty for speeding through an occupied crosswalk
    r = r - jnp.where((cw_dist < 5.0) & peds_on_cw & (speed > 4.0), 2.0, 0.0)
    r = r - jnp.where((cw_dist < 3.0) & peds_on_cw & (speed > 2.0), 1.0, 0.0)

    # Jaywalker response
    jaywalkers = jax.vmap(lambda i: (
        point_in_rects(new_state["ped_x"][i], new_state["ped_y"][i], ROAD_RECTS) &
        ~point_in_rects(new_state["ped_x"][i], new_state["ped_y"][i], CROSSWALK_RECTS) & alive[i]
    ))(jnp.arange(NUM_PEDS))
    jw_dists = jnp.where(jaywalkers, ped_dists, 999.0)
    nearest_jw = jnp.min(jw_dists)
    # Only penalize speeding toward jaywalkers
    r = r - jnp.where((nearest_jw < 5.0) & (speed > 3.0), 1.0, 0.0)
    r = r - jnp.where((nearest_jw < 3.0) & (speed > 1.0), 0.5, 0.0)

    # Near-miss penalty (any ped, road or not)
    near_miss = jnp.any((ped_dists < 2.5) & alive & (speed > 3.0))
    r = r - jnp.where(near_miss, 3.0, 0.0)

    # Discourage creeping when the road is clear
    r = r - jnp.where(~ped_blocking & (speed > 0.1) & (speed < 1.5), 0.05, 0.0)

    # Lane centering
    # Horizontal road: y=52-60, center=56. Vertical road: x=52-60, center=56.
    on_h_road = (sy >= 52.0) & (sy <= 60.0) & on_road
    on_v_road = (sx >= 52.0) & (sx <= 60.0) & on_road
    # Distance from lane center
    h_lane_off = jnp.abs(sy - 56.0)  # 0=centered, 4=edge
    v_lane_off = jnp.abs(sx - 56.0)
    lane_off = jnp.where(on_h_road, h_lane_off, jnp.where(on_v_road, v_lane_off, 0.0))
    # Reward for staying centered, penalty for riding edges
    r = r + jnp.where(on_road & (lane_off < 1.5), 0.03, 0.0)
    r = r - jnp.where(on_road & (lane_off > 3.0), 0.1, 0.0)

    # Heading alignment with road direction
    # On horizontal road: heading should be ~0 (east) or ~pi (west)
    # On vertical road: heading should be ~pi/2 (south) or ~-pi/2 (north)
    heading = new_state["sdc_heading"]
    h_align = jnp.minimum(jnp.abs(heading), jnp.abs(jnp.abs(heading) - jnp.pi))  # 0=aligned with E/W
    v_align = jnp.minimum(jnp.abs(heading - jnp.pi/2), jnp.abs(heading + jnp.pi/2))  # 0=aligned with N/S
    misalign = jnp.where(on_h_road, h_align, jnp.where(on_v_road, v_align, 0.0))
    r = r - jnp.where(on_road & (misalign > 0.5), 0.1 * misalign, 0.0)

    return r


# Observations

def get_ped_obs(state, i):
    """Compact ped observation for high-level decisions (20 floats)."""
    obs = jnp.zeros(PED_OBS_SIZE)
    px, py = state["ped_x"][i], state["ped_y"][i]
    wp = WAYPOINTS[state["ped_wp_idx"][i]]
    traits = state["ped_traits"][i]

    # Own state (4)
    obs = obs.at[0].set(px / MAP_W)
    obs = obs.at[1].set(py / MAP_H)
    obs = obs.at[2].set(state["ped_speed"][i] / 2.5)
    obs = obs.at[3].set(state["ped_heading"][i] / jnp.pi)
    # Traits (4)
    obs = obs.at[4:8].set(traits)
    # Waypoint direction (3)
    dx, dy = wp[0]-px, wp[1]-py
    d = jnp.sqrt(dx**2+dy**2)+1e-6
    obs = obs.at[8].set(d / MAP_W)
    obs = obs.at[9].set(dx / d)
    obs = obs.at[10].set(dy / d)
    # Surface (3)
    obs = obs.at[11].set(point_in_rects(px,py,SIDEWALK_RECTS).astype(jnp.float32))
    obs = obs.at[12].set(point_in_rects(px,py,CROSSWALK_RECTS).astype(jnp.float32))
    obs = obs.at[13].set((point_in_rects(px,py,ROAD_RECTS) & ~point_in_rects(px,py,CROSSWALK_RECTS)).astype(jnp.float32))
    # Nearest crosswalk dist (1)
    obs = obs.at[14].set(jnp.minimum(dist_to_rects(px,py,CROSSWALK_RECTS)/MAP_W, 1.0))
    # SDC relative (5)
    obs = obs.at[15].set((state["sdc_x"]-px)/MAP_W)
    obs = obs.at[16].set((state["sdc_y"]-py)/MAP_H)
    obs = obs.at[17].set(state["sdc_speed"]/SDC_MAX_SPEED)
    obs = obs.at[18].set(state["sdc_heading"]/jnp.pi)
    sdc_d = jnp.sqrt((state["sdc_x"]-px)**2+(state["sdc_y"]-py)**2)
    obs = obs.at[19].set(jnp.minimum(sdc_d/50.0, 1.0))
    return obs


def get_sdc_obs(state):
    """SDC observation (30 floats)."""
    obs = jnp.zeros(SDC_OBS_SIZE)
    x, y = state["sdc_x"], state["sdc_y"]
    # Own (6)
    obs = obs.at[0].set(x/MAP_W)
    obs = obs.at[1].set(y/MAP_H)
    obs = obs.at[2].set(state["sdc_speed"]*jnp.cos(state["sdc_heading"])/SDC_MAX_SPEED)
    obs = obs.at[3].set(state["sdc_speed"]*jnp.sin(state["sdc_heading"])/SDC_MAX_SPEED)
    obs = obs.at[4].set(state["sdc_heading"]/jnp.pi)
    obs = obs.at[5].set(state["sdc_speed"]/SDC_MAX_SPEED)
    # Goal (3)
    dx = state["sdc_goal_x"]-x; dy = state["sdc_goal_y"]-y
    gd = jnp.sqrt(dx**2+dy**2)+1e-6
    obs = obs.at[6].set(jnp.minimum(gd/MAP_W,1.0))
    obs = obs.at[7].set(dx/gd)
    obs = obs.at[8].set(dy/gd)
    # Road (2)
    obs = obs.at[9].set(point_in_rects(x,y,ROAD_RECTS).astype(jnp.float32))
    obs = obs.at[10].set(jnp.minimum(dist_to_rects(x,y,ROAD_RECTS)/10.0, 1.0))
    # Crosswalk (1)
    obs = obs.at[11].set(jnp.minimum(dist_to_rects(x,y,CROSSWALK_RECTS)/MAP_W, 1.0))
    # 6 nearest peds (18 = 6*3)
    pd = jnp.sqrt((state["ped_x"]-x)**2+(state["ped_y"]-y)**2)
    pd = jnp.where(state["ped_alive"], pd, 1e6)
    si = jnp.argsort(pd)
    def write_ped(j, obs):
        idx = si[j]
        v = pd[idx] < 1e5
        obs = obs.at[12+j*3].set(jnp.where(v,(state["ped_x"][idx]-x)/MAP_W,0.0))
        obs = obs.at[12+j*3+1].set(jnp.where(v,(state["ped_y"][idx]-y)/MAP_H,0.0))
        obs = obs.at[12+j*3+2].set(jnp.where(v,state["ped_speed"][idx]/2.5,0.0))
        return obs
    obs = jax.lax.fori_loop(0, 6, write_ped, obs)

    # Lane info (4 floats): tells SDC which way the road goes
    heading = state["sdc_heading"]
    on_h = (y >= 52.0) & (y <= 60.0)  # on horizontal road
    on_v = (x >= 52.0) & (x <= 60.0)  # on vertical road
    # Lane offset from center (0=centered, 1=edge)
    lane_off = jnp.where(on_h, jnp.abs(y - 56.0) / 4.0,
               jnp.where(on_v, jnp.abs(x - 56.0) / 4.0, 0.5))
    obs = obs.at[30].set(lane_off)
    # Road direction: 1=horizontal, -1=vertical, 0=intersection/other
    obs = obs.at[31].set(jnp.where(on_h & ~on_v, 1.0, jnp.where(on_v & ~on_h, -1.0, 0.0)))
    # Heading alignment with road (0=aligned, 1=perpendicular)
    h_align = jnp.minimum(jnp.abs(heading), jnp.abs(jnp.abs(heading) - jnp.pi))
    v_align = jnp.minimum(jnp.abs(heading - jnp.pi/2), jnp.abs(heading + jnp.pi/2))
    alignment = jnp.where(on_h, h_align / (jnp.pi/2), jnp.where(on_v, v_align / (jnp.pi/2), 0.5))
    obs = obs.at[32].set(alignment)
    # At intersection (both h and v road)
    obs = obs.at[33].set((on_h & on_v).astype(jnp.float32))

    return obs


def get_all_ped_obs(state):
    return jax.vmap(lambda i: get_ped_obs(state, i))(jnp.arange(NUM_PEDS))
