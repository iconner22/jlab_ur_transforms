import numpy as np
from scipy.spatial.transform import Rotation as R

#Previously calculated plane values
wg1_normal = [ 0.99253846, -0.12370374, 0.0042997]
wg1_center = [-1.36367748,  0.51324915, 0.61472839]

wg2_normal = [0.99701622, -0.08062956, 0.02470565]
wg2_center = [-1.3176, 0.83974, 0.616758]


#make sure that the normal vectors are normalized
wg1_normal = wg1_normal/np.linalg.norm(wg1_normal)
wg2_normal = wg2_normal/np.linalg.norm(wg2_normal)

print(wg1_normal)
print(wg2_normal)

#take the cross product of the two vectors. This gives the axis perpendicular to the two vectors
cross_product = np.clip(np.cross(wg1_normal, wg2_normal), -1, 1)
print("Cross Product: ",cross_product)

#take the dot product of the two vectors, used to find the angle between the two vectors. If dot is zero, then the vectors are orthoganal
dot_product = np.dot(wg1_normal, wg2_normal)
print("Dot Product: ", dot_product)

if np.linalg.norm(cross_product) < 1e-8: # if the cross product is very close to zero then they are parallel or antiparallel
    if dot_product > 0: #this means that the two vectors are not perpendicular
        rotation_vector = np.array([0,0,0])
    else:
        if abs(wg2_normal[0]) < 0.9:
            axis  = np.array([1,0,0])
        else:
            axis = np.array([0,1,0])
        rotation_vector = np.pi * axis # rotate 180 degrees in the appropriate direction

else:
    axis = cross_product/np.linalg.norm(cross_product) #normalize the cross product
    angle = np.arccos(dot_product) #find the angle between the two planes
    rotation_vector = angle * axis
print("Rotation Vector: ",rotation_vector)

r = R.from_rotvec(rotation_vector)
euler_angles_degree = r.as_euler('xyz', degrees = True)
rot_matrix = r.as_matrix()
print("Rotation Matrix: \n",rot_matrix)

print("Roll Pitch Yaw for Wave Guide 2 (degs):", euler_angles_degree)

wg2_normal_rot = rot_matrix @ wg2_normal.T
print("WG1:", wg1_normal)
print("WG2 Rotated:", wg2_normal_rot)

print(np.cross(wg1_normal, wg2_normal_rot))