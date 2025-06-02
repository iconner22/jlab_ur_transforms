from scipy.spatial.transform import Rotation as R
import numpy as np
# Constant Gocator in TCP Transform
GOCATOR_IN_TCP_MM = np.array([[-1,-0.014,0.007,152.868], [-0.014,1.00,0.017,1.462], [-0.008,0.017,-1.00,375.739], [0.0,0.0,0.0,1.0]])
GOCATOR_TRANSLATION_M = GOCATOR_IN_TCP_MM[:3, 3] * 0.001
GOCATOR_IN_TCP_M = GOCATOR_IN_TCP_MM.copy()
GOCATOR_IN_TCP_M[:3, 3] = GOCATOR_TRANSLATION_M
print("Gocator in TCP in meters: \n", GOCATOR_IN_TCP_M)

## Waveguide One
### TCP in Base Transform
WG1_TCP_IN_BASE_TRANSLATION = np.array([-1.02957,0.47028,0.464929])
WG1_TCP_ROTATION_MATRIX = R.from_rotvec(np.array([-0.0581889,-1.5676,-0.0625987])).as_matrix()
WG1_TCP_IN_BASE = np.eye(4)
WG1_TCP_IN_BASE[:3, :3] = WG1_TCP_ROTATION_MATRIX
WG1_TCP_IN_BASE[:3, 3] = WG1_TCP_IN_BASE_TRANSLATION
print("WG1 TCP in Base: \n", WG1_TCP_IN_BASE)

### Plane Center Point in Gocator Frame (y-coordinate flipped)
WG1_GOCATOR_CENTER_MM = np.array([4.006, 15.5, 39.182])
WG1_GOCATOR_CENTER_M = WG1_GOCATOR_CENTER_MM * 0.001
WG1_GOCATOR_CENTER_HOMOGENEOUS = np.array([WG1_GOCATOR_CENTER_M[0], WG1_GOCATOR_CENTER_M[1], WG1_GOCATOR_CENTER_M[2], 1.0]).T
print("WG1 Center Point Homogenous: ", WG1_GOCATOR_CENTER_HOMOGENEOUS)

# Transform Gocator center to Base frame
print("WG1 Center Point in Base Frame\n", WG1_TCP_IN_BASE @ GOCATOR_IN_TCP_M @ WG1_GOCATOR_CENTER_HOMOGENEOUS)
WG1_CP = WG1_TCP_IN_BASE @ GOCATOR_IN_TCP_M @ WG1_GOCATOR_CENTER_HOMOGENEOUS

## Waveguide Two
### TCP in Base 
WG2_TCP_IN_BASE_TRANSLATION = np.array([-1.00273,0.81859,0.465914])
WG2_TCP_ROTATION_MATRIX = R.from_rotvec(np.array([-0.0581629,-1.56748,-0.0625936])).as_matrix()
WG2_TCP_IN_BASE = np.eye(4)
WG2_TCP_IN_BASE[:3, :3] = WG2_TCP_ROTATION_MATRIX
WG2_TCP_IN_BASE[:3, 3] = WG2_TCP_IN_BASE_TRANSLATION
print("WG2 TCP in Base: \n", WG2_TCP_IN_BASE)

### Plane Center Point in Gocator Frame (y-coordinate flipped)
WG2_GOCATOR_CENTER_MM = np.array([3.486, -5.033, 59.69])
WG2_GOCATOR_CENTER_M = WG2_GOCATOR_CENTER_MM * 0.001
WG2_GOCATOR_CENTER_HOMOGENEOUS = np.array([WG2_GOCATOR_CENTER_M[0], WG2_GOCATOR_CENTER_M[1], WG2_GOCATOR_CENTER_M[2], 1.0]).T

print("WG2 Center Point Homogenous: ", WG2_GOCATOR_CENTER_HOMOGENEOUS)

# Transform Gocator center to Base frame
print("WG2 Center Point in Base Frame\n", WG2_TCP_IN_BASE @ GOCATOR_IN_TCP_M @ WG2_GOCATOR_CENTER_HOMOGENEOUS)
WG2_CP = WG2_TCP_IN_BASE @ GOCATOR_IN_TCP_M @ WG2_GOCATOR_CENTER_HOMOGENEOUS
# Normal Vector Transformations
print("************* Plane Calculations **************")

WG1_NV = np.array([0.001,-0.064,0.998])
WG1_NV_HOMOGENEOUS = np.array([WG1_NV[0], WG1_NV[1], WG1_NV[2], 0.0]).T
WG1_NV_IN_BASE  = WG1_TCP_IN_BASE@ GOCATOR_IN_TCP_M @ WG1_NV_HOMOGENEOUS
print("WG1 Normal Vector in Base Frame: ", WG1_NV_IN_BASE[:3])

WG2_NV = np.array([-0.02,-0.021,1])
WG2_NV_HOMOGENEOUS = np.array([WG2_NV[0], WG2_NV[1], WG2_NV[2], 0.0]).T
WG2_NV_IN_BASE  = WG2_TCP_IN_BASE@ GOCATOR_IN_TCP_M @ WG2_NV_HOMOGENEOUS
print("WG2 Normal Vector in Base Frame: ", WG2_NV_IN_BASE[:3])

# Calculate the angle between the two normal vectors
cos_angle = np.dot(WG1_NV_IN_BASE[:3], WG2_NV_IN_BASE[:3]) / (np.linalg.norm(WG1_NV_IN_BASE[:3]) * np.linalg.norm(WG2_NV_IN_BASE[:3]))
print("Cosine of angle between WG1 and WG2 normal vectors: ", cos_angle)
angle_rad = np.arccos(cos_angle)
angle_deg = np.degrees(angle_rad)
print("Angle between WG1 and WG2 normal vectors (degrees): ", angle_deg)


#Calculate the cross product
cross_product = np.cross(WG1_NV_IN_BASE[:3], WG2_NV_IN_BASE[:3])
print("Cross product of WG1 and WG2 normal vectors: ", cross_product)
parallel = np.allclose(cross_product, [0, 0, 0])
print("Parallel:", parallel)

# Distance between the points
distance = abs(np.dot(WG1_NV_IN_BASE[:3], WG2_CP[:3] - WG1_CP[:3])) / np.linalg.norm(WG1_NV_IN_BASE[:3])
print("Distance between WG1 and WG2 planes: ", distance)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define plane 1 (normal and point)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Create rectangular plane geometry given a normal and a point
def create_plane(normal, point, size=5):
    normal = normal / np.linalg.norm(normal)
    if np.allclose(normal, [0, 0, 1]):
        v = np.array([1, 0, 0])
    else:
        v = np.cross(normal, [0, 0, 1])
    v /= np.linalg.norm(v)
    u = np.cross(normal, v)

    corners = []
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            corner = point + size * (dx * u + dy * v)
            corners.append(corner)
    return [corners]



# Slice to get 3D vector and point
n1 = WG1_NV_IN_BASE[:3]
p1 = WG1_CP[:3]
n2 = WG2_NV_IN_BASE[:3]
p2 = WG2_CP[:3]

# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create and add planes
plane1 = create_plane(n1, p1)
plane2 = create_plane(n2, p2)

ax.add_collection3d(Poly3DCollection(plane1, color='skyblue', alpha=0.5))
ax.add_collection3d(Poly3DCollection(plane2, color='salmon', alpha=0.5))

# Add normal vectors as arrows
ax.quiver(*p1, *n1, length=2, color='blue', normalize=True)
ax.quiver(*p2, *n2, length=2, color='red', normalize=True)

# Mark the points
ax.scatter(*p1, color='blue', s=50)
ax.scatter(*p2, color='red', s=50)

# Set plot limits and labels
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.tight_layout()
plt.show()
