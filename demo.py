from mpl_toolkits.mplot3d import Axes3D 
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt 
import open3d as o3d 
from scipy.spatial import Delaunay
import numpy as np
# 加载图像
image_path = "imgs/fresh.jpg"
image = cv2.imread(image_path)

# 初始化MediaPipe的FaceMesh模型
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def show_print():
        
    # 初始化FaceMesh模型
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

        # 将图像从BGR格式转换为RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 处理RGB图像，查找面部关键点
        results = face_mesh.process(image_rgb)

        # 绘制面部轮廓
        if results.multi_face_landmarks:
            
            for face_landmarks in results.multi_face_landmarks:
                # 可视化绘制关键点
                mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))
                # 打印关键点信息
                # for idx, landmark in enumerate(face_landmarks.landmark):
                #    print(f"关键点 {idx}: (x={landmark.x}, y={landmark.y}, z={landmark.z})")

        # 显示结果图像
        cv2.imshow('Face Mesh', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def plot_3d():
    '''关键点展示成3d模型'''
    # 初始化FaceMesh模型
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

        # 将图像从BGR格式转换为RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 处理RGB图像，查找面部关键点
        results = face_mesh.process(image_rgb)

        # 绘制面部轮廓
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmark_points = []
            for landmark in face_landmarks.landmark:
                landmark_points.append((landmark.x, landmark.y, landmark.z))
            
            # 可视化关键点
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            xs = [p[0] for p in landmark_points]
            ys = [p[1] for p in landmark_points]
            zs = [p[2] for p in landmark_points]
            ax.scatter(xs, ys, zs, c='r', marker='o')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()
            
def facial_segmentation():

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) 

    # 将图像从BGR格式转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 处理RGB图像，查找面部关键点
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            x_coords = [landmark.x for landmark in face_landmarks.landmark]
            y_coords = [landmark.y for landmark in face_landmarks.landmark]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            # 创建黑色图像
            black_image = np.zeros_like(image)

            # 在黑色图像上绘制人脸区域
            cv2.rectangle(black_image, (int(min_x * image.shape[1]), int(min_y * image.shape[0])),
                        (int(max_x * image.shape[1]), int(max_y * image.shape[0])), (255, 255, 255), -1)

            # 将黑色图像与原始图像进行按位与运算，保留人脸部分
            segmented_image = cv2.bitwise_and(image, black_image)

            # 显示结果图像
            cv2.imshow('Segmented Image', segmented_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
def generate_point_cloud():
    '''生成带脸的点云信息'''
        # 初始化FaceMesh模型
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

        # 将图像从BGR格式转换为RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 处理RGB图像，查找面部关键点
        results = face_mesh.process(image_rgb)

        # 绘制面部轮廓
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmark_points = []
            landmark_point_xy = []

            for landmark in face_landmarks.landmark:
                landmark_points.append((landmark.x, landmark.y, landmark.z))
                landmark_point_xy.append([landmark.x,landmark.y])
            

            filled_image = facial_segmentation(image_rgb,landmark_point_xy)
            # 显示结果图像
            cv2.imshow('Face Mesh', filled_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 创建点云对象
            point_cloud =  o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(landmark_points)

            # 保存点云 不仅仅是点云还需要的是面部信息
            # save_path = "mesh.ply"
            # o3d.io.write_point_cloud(save_path, point_cloud)

def rebuild():
    '''使用保存的点云生成人脸模型.需要进行几何重建'''
    point_cloud_path = 'mesh.ply'
    # 读取保存的点云
    point_cloud = o3d.io.read_point_cloud(point_cloud_path)
    # 计算点云的法线
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 进行三角划分生成网格模型
    mesh,_ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud)
    # 保存生成的模型
    mesh_path = "mesh.obj"
    o3d.io.write_triangle_mesh(mesh_path, mesh)

if __name__ =='__main__':
    facial_segmentation()