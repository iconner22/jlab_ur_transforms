from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# Gocator in TCP frame Transform TCP_T_Gocator
GOCATOR_IN_TCP_MM = np.array([[-1,-0.014,0.007,152.868], [-0.014,1.00,0.017,1.462], [-0.008,0.017,-1.00,375.739], [0.0,0.0,0.0,1.0]])
GOCATOR_TRANSLATION_M = GOCATOR_IN_TCP_MM[:3, 3] * 0.001
GOCATOR_IN_TCP_M = GOCATOR_IN_TCP_MM.copy()
GOCATOR_IN_TCP_M[:3, 3] = GOCATOR_TRANSLATION_M
print("Gocator in TCP in meters: \n", GOCATOR_IN_TCP_M)

def build_homogenous_transform_matrix(tv, rv):
    '''
    Creates a 4x4 homogeneous transform matrix
    Parameters: tv - numpy array of the translation vector [x,y,z]. rv - numpy array of the rotation vector in the angle-axis format from the UR robot
    '''
    # Creates the homogeneous 4x4 transform matrix.
    #
    rotation_matrix = R.from_rotvec(rv).as_matrix() #derive rotation matrix from rotation vector
    transform_matrix = np.eye(4) # 4x4 Identity Matrix
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = tv
    return transform_matrix

def transform_to_base(b_t_tcp, tcp_t_go, point):
    # perform the dot product to get the point from the gocator into the base frame
    return b_t_tcp @ tcp_t_go @ point.T

def prepare_point(point):
    # Prepare the Gocator x,y,z point for transforms
    point_meters = point * 0.001 # turn to meters
    point_m_flipped = point_meters * np.array([1,-1,1]) #flip across x-axis (negate the y-value)
    point_m_flipped_homogeneous = np.array([point_m_flipped[0], point_m_flipped[1], point_m_flipped[2], 1]) #turn into homogeneous form for the matrix
    return point_m_flipped_homogeneous
def prepare_plane(plane):
    # Prepare the Gocator plane normal vector for transforms
    plane_flipped = plane * np.array([1,-1,1]) #flip across x-axis (negate the y-value)
    plane_flipped_homogeneous = np.array([plane_flipped[0], plane_flipped[1], plane_flipped[2], 0]) #turn into homogeneous form for the matrix
    return plane_flipped_homogeneous
def angle_between(plane1_nv, plane2_nv):
    # Calculate the angle between the two planes based off their normal vectors
    cos_angle = np.dot(plane1_nv[:3], plane2_nv[:3])/(np.linalg.norm(plane1_nv[:3]) * np.linalg.norm(plane2_nv[:3]))
    angle_rads = np.arccos(cos_angle)
    angle_degs = np.degrees(angle_rads)
    return angle_rads, angle_degs
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
def plot_plane(nv1, p1, nv2, p2):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create and add planes
    plane1 = create_plane(nv1, p1)
    plane2 = create_plane(nv2, p2)

    ax.add_collection3d(Poly3DCollection(plane1, color='skyblue', alpha=0.5))
    ax.add_collection3d(Poly3DCollection(plane2, color='salmon', alpha=0.5))

    # Add normal vectors as arrows
    ax.quiver(*p1, *nv1, length=2, color='blue', normalize=True)
    ax.quiver(*p2, *nv2, length=2, color='red', normalize=True)

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


print("Calculations for WG1")
base_t_wg1_tcp = build_homogenous_transform_matrix(np.array([-1.02957,0.47028,0.464929]), np.array(([-0.0581889,-1.5676,-0.0625987])))
wg1_center_point = prepare_point(np.array([4.00, -15.5, 39.182]))
wg1_point_in_base = transform_to_base(base_t_wg1_tcp, GOCATOR_IN_TCP_M, wg1_center_point)
wg1_point_in_base_rb = np.array([-1.36365,0.513362,0.614754])
print(f"Calculated center point in base frame: {wg1_point_in_base[:3]}")
print(f"Robot center point in base frame: {wg1_point_in_base_rb}")
print(f"Distance between points: {np.linalg.norm(wg1_point_in_base[:3] - wg1_point_in_base_rb)} meters")
print(f"Distance between points: {np.linalg.norm(wg1_point_in_base[:3] - wg1_point_in_base_rb) * 1000} mm")
wg1_nv = np.array([0.001,0.064,0.998])
wg1_nv_h = prepare_plane(wg1_nv)
wg1_nv_base = transform_to_base(base_t_wg1_tcp, GOCATOR_IN_TCP_M, wg1_nv_h)
print("Wave Guide 1 Normal Vector in Base", wg1_nv_base)
print("Calculations for WG2")
base_t_wg2_tcp = build_homogenous_transform_matrix(np.array([-1.00273,0.81859,0.465914]), np.array(([-0.0581629,-1.56748,-0.0625936])))
wg2_center_point = prepare_point(np.array([3.486, 5.033, 59.69]))
wg2_point_in_base = transform_to_base(base_t_wg2_tcp, GOCATOR_IN_TCP_M, wg2_center_point)
wg2_point_in_base_rb = np.array([-1.3176, 0.83974, 0.616758])
print(f"Calculated center point in base frame: {wg2_point_in_base[:3]}")
print(f"Robot center point in base frame: {wg2_point_in_base_rb}")
print(f"Distance between points: {np.linalg.norm(wg2_point_in_base[:3] - wg2_point_in_base_rb)} meters")
print(f"Distance between points: {np.linalg.norm(wg2_point_in_base[:3] - wg2_point_in_base_rb) * 1000} mm")
wg2_nv = np.array([-0.02, 0.021,1])
wg2_nv_h = prepare_plane(wg2_nv)
wg2_nv_base = transform_to_base(base_t_wg2_tcp, GOCATOR_IN_TCP_M, wg2_nv_h)
print("Wave Guide 2 Normal Vector in Base", wg2_nv_base)


offset_rads, offset_degs = angle_between(wg1_nv_base, wg2_nv_base)
print(f"The wave guides are off by: {offset_rads} rads, {offset_degs} degs")

plot_plane(wg1_nv_base[:3], wg1_point_in_base[:3], wg2_nv_base[:3], wg2_point_in_base[:3])