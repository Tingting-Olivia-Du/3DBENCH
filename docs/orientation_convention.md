# LIBERO EE Orientation Convention

## World Coordinate Frame (Robot Base = Origin)

- **X-axis**: forward, away from robot body, into the workspace
- **Y-axis**: right, robot's right when facing forward
- **Z-axis**: up, vertical

### Top-Down View (looking from ceiling)

```
          +X (forward, into workspace)
           ↑
           |
           |
  -Y ------●------→ +Y (robot's right)
  (robot   |  (origin =
   left)   |   robot base)
           ↓
          -X (toward robot)
```

### Agent View (camera is in front-above, looking back toward robot)

The agent-view camera is positioned at the front-top of the workspace,
looking back toward the robot. This **mirrors left/right**: robot's +Y
(right) appears on the **left side** of the image.

```
  Agent View Image:
  ┌────────────────────────────┐
  │          +Z (up)           │
  │           ↑                │
  │           |                │
  │   +Y ←---●---→ -Y         │
  │  (robot   |  (robot       │
  │   right)  |   left)       │
  │  IMAGE    ↓   IMAGE       │
  │  LEFT    -Z   RIGHT       │
  │                            │
  │  +X comes TOWARD camera    │
  │  (objects get bigger)      │
  │  -X goes AWAY from camera  │
  │  (objects get smaller)     │
  └────────────────────────────┘

  Verified with actual images:
  - +Y nudge → gripper moves LEFT in agent view ✓
  - +Z nudge → gripper moves UP in agent view ✓
  - +X nudge → gripper gets BIGGER (toward camera) ✓
```

### Wrist View (camera on gripper, looking down when gripper faces down)

When the gripper points straight down (default), the wrist camera
looks at the table from above.

```
  Wrist View Image (gripper pointing down):
  ┌────────────────────────────┐
  │      +X (into workspace)   │
  │           ↑                │
  │           |                │
  │   -Y ←---●---→ +Y         │
  │  (robot   |  (robot       │
  │   left)   |   right)      │
  │  IMAGE    ↓   IMAGE       │
  │  LEFT    -X   RIGHT       │
  │     (toward robot)        │
  │                            │
  │  -Z goes INTO the image    │
  │  (toward table surface)    │
  │  +Z comes OUT of image     │
  │  (away from table)         │
  └────────────────────────────┘

  Verified with actual images:
  - +X nudge → new far objects appear at IMAGE TOP → +X at IMAGE TOP ✓
  - -X nudge → robot arm enters from IMAGE BOTTOM → -X at IMAGE BOTTOM ✓
  - +Y nudge → scene shifts LEFT → +Y at IMAGE RIGHT ✓
  - -Y nudge → scene shifts RIGHT → -Y at IMAGE LEFT ✓
```

## Default Gripper Pose (Pointing Straight Down)

When the Panda gripper points straight down at the table:
- `roll ≈ -180 (or +180)`, `pitch ≈ 0`, `yaw ≈ 0`
- EE local X-axis → +X_world (forward)
- EE local Y-axis → -Y_world (flipped, pointing left)
- EE local Z-axis → -Z_world (pointing down)

## Euler Convention: XYZ Intrinsic, Right-Hand Rule

All rotations use **XYZ intrinsic** Euler angles (scipy convention).
Positive direction follows the **right-hand rule**: point right thumb along
the +axis, fingers curl in the + rotation direction.

### Right-Hand Rule vs Agent View: Why Rotation Looks Mirrored

When you use the right-hand rule (imagine the YZ plane in your head) vs
when you look at the agent view image, the **Y axis is flipped**:

|  | Right-hand rule (math) | Agent view (image) |
|---|---|---|
| +Y | right side | **left** side (mirrored) |
| -Y | left side | **right** side (mirrored) |
| +Z | up | up (same) |
| -Z | down | down (same) |

**Why?** The agent-view camera faces the robot from the front. The robot's
right hand (+Y) appears on the LEFT side of the image — like looking in a
mirror.

**Consequence**: A rotation that is counter-clockwise in the right-hand rule
(in the YZ world plane) looks **clockwise** in the agent view image, because
the Y axis is mirrored.

| +Roll | Right-hand rule (YZ plane) | Agent view image |
|---|---|---|
| Direction | Counter-clockwise: +Y→+Z→-Y→-Z | Clockwise: left→up→right→down |
| Y-axis moves toward | +Z (up) ✓ | up ✓ (same) |
| Z-axis moves toward | -Y (right in math) | image RIGHT ✓ (= -Y world) |

Both are correct — they describe the same physical rotation from different viewpoints.

---

## Roll (Rotation about X-axis / Forward)

**Right-hand rule**: Thumb → +X (forward). Fingers curl from +Y toward +Z.
From +X looking toward origin: **counter-clockwise** in world YZ plane.
In agent view image: appears **clockwise** (due to Y-axis mirror).

```
+Roll: Y-axis → +Z, Z-axis → -Y   (verified with scipy)
-Roll: Y-axis → -Z, Z-axis → +Y
```

### Four Quadrants (EE Z-axis position in YZ plane)

Starting from roll=0 (EE_Z pointing +Z/up):

```
              +Z (up)
               |
     roll=-45  |  roll=+45
          \    |    /
           \   |   /
    +Y --------0-------- -Y
           /   |   \
          /    |    \
    roll=-135  |  roll=+135
               |
              -Z (down)
```

| Roll Range | EE_Z moves from → to | Description |
|---|---|---|
| 0 → +90 | +Z → -Y | Z-axis tilts toward -Y (robot's left) |
| +90 → +180 | -Y → -Z | Continues flipping to face down |
| 0 → -90 | +Z → +Y | Z-axis tilts toward +Y (robot's right) |
| -90 → -180 | +Y → -Z | Continues flipping to face down |

**In LIBERO** (gripper default = roll ≈ -180, Z points down):
- Roll increases (e.g. -180 → -135): gripper tip tilts toward +Y (right)
- Roll decreases (e.g. -180 → +135, wrapping): gripper tip tilts toward -Y (left)

### Agent View

```
  +Roll (world: Z→-Y):
    → Z tilts toward -Y = IMAGE RIGHT (world -Y = image right)
    → looks CLOCKWISE in the image
    (even though it's counter-clockwise in world coordinates,
     because agent view mirrors the Y axis)

  -Roll (world: Z→+Y):
    → Z tilts toward +Y = IMAGE LEFT
    → looks COUNTER-CLOCKWISE in the image
```

### Wrist View

```
  +Roll → scene rotates clockwise in image
  -Roll → scene rotates counter-clockwise in image
```

---

## Pitch (Rotation about Y-axis / Right)

**Right-hand rule**: Thumb → +Y (right). Fingers curl from +Z toward +X.

In LIBERO default pose (roll ≈ -180), **pitch=0 means EE_Z points -Z (straight down)**.

### Four Quadrants (EE_Z tip direction in XZ plane, from LIBERO default)

```
              +Z (up)
               |
  pitch=±180   |   pitch=-90
  (tip up)     |   (tip toward +X/
          \    |    forward/camera)
           \   |   /
  -X --------[0]-------- +X (forward)
  (away from / |   \
   camera)  /  |    \
  pitch=+90    |   pitch=0
  (tip toward  |   (tip down = DEFAULT)
   -X/robot)   |
              -Z (down)
```

| Pitch | EE_Z points toward | Agent View |
|---|---|---|
| **0°** | **-Z (straight down)** | **DEFAULT** |
| 0 → +90 | -Z → -X (tip tilts backward toward robot) | tip toward camera (bigger) |
| **+90°** | **-X (horizontal, toward robot)** | tip fully toward camera |
| +90 → +180 | -X → +Z (tip tilts upward) | tip goes up |
| 0 → -90 | -Z → +X (tip tilts forward into workspace) | tip away from camera (smaller) |
| **-90°** | **+X (horizontal, into workspace)** | tip fully away from camera |
| -90 → -180 | +X → +Z (tip tilts upward) | tip goes up |

### Agent View

```
  +Pitch → tip tilts toward -X (robot body) = TOWARD camera (gets bigger)
  -Pitch → tip tilts toward +X (workspace) = AWAY from camera (gets smaller)
```

### Wrist View

```
  +Pitch → view shifts to show area closer to robot
  -Pitch → view shifts to show area further into workspace
```

---

## Yaw (Rotation about Z-axis / Up)

**Right-hand rule**: Thumb → +Z (up). Fingers curl from +X toward +Y.

```
+Yaw: X-axis → +Y, Y-axis → -X (counter-clockwise from above)
-Yaw: X-axis → -Y, Y-axis → +X (clockwise from above)
```

### Rotation in the Horizontal Plane

Yaw only rotates the gripper in the horizontal plane. The Z-axis direction
does not change (still points down in default pose).

```
Top-down view (looking from +Z down):

              +X (forward)
               |
    -Yaw       |      +Yaw
    (CW)       |      (CCW)
          \    |    /
           \   |   /
    -Y --------0-------- +Y (right)

  +Yaw = counter-clockwise from above (X rotates toward +Y)
  -Yaw = clockwise from above (X rotates toward -Y)
```

| Yaw Range | EE_X moves from → to | Description |
|---|---|---|
| 0 → +90 | +X → +Y | X-axis rotates to point right |
| +90 → +180 | +Y → -X | Continues to point backward |
| 0 → -90 | +X → -Y | X-axis rotates to point left |
| -90 → -180 | -Y → -X | Continues to point backward |

### Agent View

```
  +Yaw (CCW from above) → gripper rotates CCW in image
  -Yaw (CW from above) → gripper rotates CW in image
  (agent view is roughly top-down-ish, so rotation direction matches)
```

### Wrist View

```
  +Yaw (CCW from above) → scene rotates CW in image
  -Yaw (CW from above) → scene rotates CCW in image
  (camera rotates, scene appears to rotate opposite direction)
```

---

## Summary Table

| Angle | Axis | + Direction (from roll=0) | + Direction (LIBERO default, roll≈-180) |
|---|---|---|---|
| **Roll** | X (forward) | Z→-Y (tip toward robot left) | tip tilts toward +Y (robot right) |
| **Pitch** | Y (right) | Z→+X (tip forward) | tip tilts toward -X (toward robot) |
| **Yaw** | Z (up) | X→+Y (CCW from above) | swivels CCW from above |

## Quick Reference for LIBERO Default Pose

Default: `roll ≈ ±180, pitch ≈ 0, yaw ≈ 0` (gripper straight down)

| Change | World Effect | Agent View | Wrist View |
|---|---|---|---|
| Roll ↑ (toward -150) | tip → +Y (right) | tip → image right | scene rotates CW |
| Roll ↓ (toward +150) | tip → -Y (left) | tip → image left | scene rotates CCW |
| Pitch ↑ (+10) | tip → -X (toward robot) | tip toward camera | view shifts up |
| Pitch ↓ (-10) | tip → +X (into workspace) | tip away from camera | view shifts down |
| Yaw ↑ (+30) | swivels CCW from above | rotates CCW in image | scene rotates CW |
| Yaw ↓ (-30) | swivels CW from above | rotates CW in image | scene rotates CCW |
