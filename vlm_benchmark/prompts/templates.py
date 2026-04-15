"""
Prompt templates for VLM spatial reasoning benchmark.

All prompts are in English and include coordinate system definition.
"""

# Coordinate system definition (from robosuite documentation)
COORDINATE_SYSTEM = """[Coordinate System]
- Origin: Robot base center
- X-axis: Points forward (away from robot base, toward the table) (+X = forward)
- Y-axis: Points to the robot's left (+Y = left)
- Z-axis: Points upward (+Z = up)
- Unit: meters (m)
- Table height: Z ≈ 0.8m
- Typical workspace: X ∈ [0.3, 0.7], Y ∈ [-0.3, 0.3], Z ∈ [0.8, 1.2]

[Camera Views]
- First image: "agentview" - Third-person view looking at the robot from the front
  - Image left = robot's right = -Y direction
  - Image right = robot's left = +Y direction
  - Image up = +Z direction (upward)
  - Image depth (away from camera) = -X direction (toward robot)
- Second image: "eye_in_hand" - Gripper camera view (first-person from gripper)
"""

# Q1: Object coordinate localization
OBJECT_LOCALIZATION_PROMPT = """You are a robotic vision system analyzing workspace images.

[Known Information]
- Current gripper position: [{eef_x:.3f}, {eef_y:.3f}, {eef_z:.3f}]

{coordinate_system}

[Task]
Looking at the images, estimate the 3D coordinates [x, y, z] of the "{target_object}" in the world frame.

Important notes:
- The coordinates should be in meters
- Consider the object's center position
- Use the gripper position as a reference point

Output ONLY in JSON format: {{"x": <float>, "y": <float>, "z": <float>}}
"""

# Q2: Gripper close timing judgment
GRIPPER_CLOSE_PROMPT = """You are a robotic vision system analyzing workspace images.

[Known Information]
- Current gripper position: [{eef_x:.3f}, {eef_y:.3f}, {eef_z:.3f}]

{coordinate_system}

[Task]
Looking at the images, determine whether the gripper is currently positioned to successfully close and grasp the "{target_object}".

Consider:
- Is the gripper directly above or around the object?
- Is the distance close enough for a successful grasp (typically < 5cm)?
- Is the gripper orientation suitable for grasping?

Output ONLY in JSON format: {{"can_close": <true/false>, "reason": "<brief explanation>"}}
"""

# Q3: Next move direction prediction
MOVE_DIRECTION_PROMPT = """You are a robotic vision system analyzing workspace images.

[Known Information]
- Current gripper position: [{eef_x:.3f}, {eef_y:.3f}, {eef_z:.3f}]

{coordinate_system}

[Task]
Looking at the images, if the goal is to grasp the "{target_object}", which direction should the gripper move next?

Answer with a normalized 3D direction vector. The vector should:
- Point from the current gripper position toward the target object
- Have a magnitude of approximately 1 (unit vector)
- Use the coordinate system defined above

Output ONLY in JSON format: {{"dx": <float>, "dy": <float>, "dz": <float>}}
"""


def get_object_localization_prompt(eef_pos, target_object):
    """Generate object localization prompt."""
    return OBJECT_LOCALIZATION_PROMPT.format(
        eef_x=eef_pos[0],
        eef_y=eef_pos[1],
        eef_z=eef_pos[2],
        target_object=target_object,
        coordinate_system=COORDINATE_SYSTEM
    )


def get_gripper_close_prompt(eef_pos, target_object):
    """Generate gripper close timing prompt."""
    return GRIPPER_CLOSE_PROMPT.format(
        eef_x=eef_pos[0],
        eef_y=eef_pos[1],
        eef_z=eef_pos[2],
        target_object=target_object,
        coordinate_system=COORDINATE_SYSTEM
    )


def get_move_direction_prompt(eef_pos, target_object):
    """Generate move direction prediction prompt."""
    return MOVE_DIRECTION_PROMPT.format(
        eef_x=eef_pos[0],
        eef_y=eef_pos[1],
        eef_z=eef_pos[2],
        target_object=target_object,
        coordinate_system=COORDINATE_SYSTEM
    )


# All question types
QUESTION_TYPES = {
    "object_localization": {
        "prompt_fn": get_object_localization_prompt,
        "description": "Estimate 3D coordinates of target object",
        "output_keys": ["x", "y", "z"],
    },
    "gripper_close": {
        "prompt_fn": get_gripper_close_prompt,
        "description": "Judge if gripper can close to grasp",
        "output_keys": ["can_close", "reason"],
    },
    "move_direction": {
        "prompt_fn": get_move_direction_prompt,
        "description": "Predict next move direction",
        "output_keys": ["dx", "dy", "dz"],
    },
}
