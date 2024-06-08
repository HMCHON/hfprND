import pandas as pd
import numpy as np
from scipy.stats import norm

class HOFS_E:
    def __init__(self, csvpath, n_frames, sigma=None, augment=False):
        self.csvpath = csvpath
        self.n_frames = n_frames
        self.data = pd.read_csv(self.csvpath)
        self.augment = augment

        if augment != False:
            self.sigma = sigma

            max_class_num_counts = self.data['class_no'].value_counts().max()  # 가장 많은 class의 갯수
            self.max_class_no = self.data['class_no'].value_counts().idxmax()  # 가장 많은 class의 no

            # 데이터 증강을 하기 위한 사전 정보 탐
            class_num_counts = self.data['class_no'].value_counts()
            self.pre_info = []
            for class_no, class_count in class_num_counts.items():
                if class_no == self.max_class_no:
                    self.pre_info.append([class_no, int(class_count/n_frames), 0, 0])  # class_no, class_count, nce, per_nce
                else:
                    class_group_count = int(class_count/n_frames)
                    nce_group_count = int(max_class_num_counts/n_frames) - class_group_count
                    # print('class_no :', class_no, "class_count :", class_count, "nce :", nce)
                    # print("first :", nce/int(class_count/n_frames), "second :", int(nce/class_count))
                    # calculate_per_nce = lambda nce, class_count: calculate_per_nce(nce, class_count, n_frames)
                    per_nce_group = self.calculate_per_nce(nce_group_count, class_group_count)
                    self.pre_info.append([class_no, class_group_count, nce_group_count, per_nce_group])
            self.pre_info = sorted(self.pre_info, key=lambda x: x[0])

    def calculate_per_nce(self, nce_group_count, class_group_count):
            return int(nce_group_count/class_group_count)



    def read(self):
            self.label = self.data[['class_no']]
            label = []
            self.label = self.label.to_numpy()
            for i in range(0, len(self.label), self.n_frames):
                if i + self.n_frames <= len(self.label):
                    grouped_section = self.label[i:i + self.n_frames]
                    most_frequent_label = np.bincount(grouped_section.flatten()).argmax()
                    label.append(most_frequent_label)


            # 각 좌표와 점수에 대한 변수 할당
            self.whole_body = self.data[["nose_x", "nose_y",
                                         "left_eye_x", "left_eye_y",
                                         "right_eye_x", "right_eye_y",
                                         "left_ear_x", "left_ear_y",
                                         "right_ear_x", "right_ear_y",
                                         "left_shoulder_x", "left_shoulder_y",
                                         "right_shoulder_x", "right_shoulder_y",
                                         "left_elbow_x", "left_elbow_y",
                                         "right_elbow_x", "right_elbow_y",
                                         "left_wrist_x", "left_wrist_y",
                                         "right_wrist_x", "right_wrist_y",
                                         "left_hip_x", "left_hip_y",
                                         "right_hip_x", "right_hip_y",
                                         "left_knee_x", "left_knee_y",
                                         "right_knee_x", "right_knee_y",
                                         "left_ankle_x", "left_ankle_y",
                                         "right_ankle_x", "right_ankle_y"]]
            whole_body = []
            self.whole_body = self.whole_body.to_numpy()
            for i in range(0, len(self.whole_body), self.n_frames):
                if i + self.n_frames <= len(self.whole_body):
                    grouped_section = self.whole_body[i:i + self.n_frames]
                    whole_body.append(grouped_section)



            self.left_arm = self.data[['left_shoulder_x', 'left_shoulder_y',
                                       'left_elbow_x', 'left_elbow_y',
                                       'left_wrist_x', 'left_wrist_y']]
            left_arm = []
            self.left_arm = self.left_arm.to_numpy()
            for i in range(0, len(self.left_arm), self.n_frames):
                if i + self.n_frames <= len(self.left_arm):
                    grouped_section = self.left_arm[i:i + self.n_frames]
                    left_arm.append(grouped_section)



            self.right_arm = self.data[['right_shoulder_x', 'right_shoulder_y',
                                        'right_elbow_x', 'right_elbow_y',
                                        'right_wrist_x', 'right_wrist_y']]
            right_arm = []
            self.right_arm = self.right_arm.to_numpy()
            for i in range(0, len(self.right_arm), self.n_frames):
                if i + self.n_frames <= len(self.right_arm):
                    grouped_section = self.right_arm[i:i + self.n_frames]
                    right_arm.append(grouped_section)


            self.upper_body = self.data[['right_shoulder_x', 'right_shoulder_y',
                                         'left_shoulder_x', 'left_shoulder_y',
                                         'right_hip_x', 'right_hip_y',
                                         'left_hip_x', 'left_hip_y']]
            upper_body = []
            self.upper_body = self.upper_body.to_numpy()
            for i in range(0, len(self.upper_body), self.n_frames):
                if i + self.n_frames <= len(self.upper_body):
                    grouped_section = self.upper_body[i:i + self.n_frames]
                    upper_body.append(grouped_section)


            self.lower_body = self.data[['right_hip_x', 'right_hip_y',
                                         'right_knee_x', 'right_knee_y',
                                         'right_ankle_x', 'right_ankle_y',
                                         'left_hip_x', 'left_hip_y',
                                         'left_knee_x', 'left_knee_y',
                                         'left_ankle_x', 'left_ankle_y']]
            lower_body = []
            self.lower_body = self.lower_body.to_numpy()
            for i in range(0, len(self.lower_body), self.n_frames):
                if i + self.n_frames <= len(self.lower_body):
                    grouped_section = self.lower_body[i:i + self.n_frames]
                    lower_body.append(grouped_section)


            self.head = self.data[['nose_x', 'nose_y',
                                   'right_eye_x', 'right_eye_y',
                                   'left_eye_x', 'left_eye_y',
                                   'right_ear_x', 'right_ear_y',
                                   'left_ear_x', 'left_ear_y']]
            head = []
            self.head = self.head.to_numpy()
            for i in range(0, len(self.head), self.n_frames):
                if i + self.n_frames <= len(self.head):
                    grouped_section = self.head[i:i + self.n_frames]
                    head.append(grouped_section)





            # # LSTM layer
            # self.label = self.data[['class_no']]
            # label = []
            # self.label = self.label.to_numpy()
            # for i in range(0, len(self.label), self.n_frames):
            #     if i + self.n_frames <= len(self.label):
            #         grouped_section = self.label[i:i + self.n_frames]
            #         # 1이 하나라도 있는지 확인
            #         if 1 in grouped_section:
            #             label.append(1)
            #         else:
            #             most_frequent_label = np.bincount(grouped_section.flatten()).argmax()
            #             label.append(most_frequent_label)

            if self.augment == False:
                return [whole_body, right_arm, left_arm, upper_body, lower_body, head, label]
            else:
                print("pre_info :", self.pre_info)
                new_whole_body = []
                new_right_arm = []
                new_left_arm = []
                new_upper_body = []
                new_lower_body = []
                new_head = []
                new_label = []

                for index, category in enumerate(label):
                    group = whole_body[index]

                    if category == self.max_class_no:
                        new_whole_body.append(whole_body[index])
                        new_right_arm.append(right_arm[index])
                        new_left_arm.append(left_arm[index])
                        new_upper_body.append(upper_body[index])
                        new_lower_body.append(lower_body[index])
                        new_head.append(head[index])
                        new_label.append(label[index])

                    else:
                        pre_info = self.pre_info[category]
                        per_nce = pre_info[3]
                        i = 0

                        # 원본 한번 append
                        new_whole_body.append(whole_body[index])
                        new_right_arm.append(right_arm[index])
                        new_left_arm.append(left_arm[index])
                        new_upper_body.append(upper_body[index])
                        new_lower_body.append(lower_body[index])
                        new_head.append(head[index])
                        new_label.append(label[index])

                        # per_nce 만큼 변형 후 append
                        while i < per_nce:
                            new_whole_body_element_group = []
                            new_right_arm_element_group = []
                            new_left_arm_element_group = []
                            new_upper_body_element_group = []
                            new_lower_body_element_group = []
                            new_head_element_group = []
                            for frame in range(self.n_frames):
                                element = group[frame]
                                new_nose = [element[0], element[1]]
                                new_left_eye = [element[2], element[3]]
                                new_right_eye = [element[4], element[5]]
                                new_left_ear = [element[6], element[7]]
                                new_right_ear = [element[8], element[9]]
                                new_left_shoulder = self.augment_keypoint([element[10], element[11]], self.sigma)
                                new_right_shoulder = self.augment_keypoint([element[12], element[13]], self.sigma)
                                new_left_elbow = self.augment_keypoint([element[14], element[15]], self.sigma)
                                new_right_elbow = self.augment_keypoint([element[16], element[17]], self.sigma)
                                new_left_wrist = self.augment_keypoint([element[18], element[19]], self.sigma)
                                new_right_wrist = self.augment_keypoint([element[20], element[21]], self.sigma)
                                new_left_hip = self.augment_keypoint([element[22], element[23]], self.sigma)
                                new_right_hip = self.augment_keypoint([element[24], element[25]], self.sigma)
                                new_left_knee = self.augment_keypoint([element[26], element[27]], self.sigma)
                                new_right_knee = self.augment_keypoint([element[28], element[29]], self.sigma)
                                new_left_ankle = self.augment_keypoint([element[30], element[31]], self.sigma)
                                new_right_ankle = self.augment_keypoint([element[32], element[33]], self.sigma)

                                new_whole_body_element = [new_nose[0], new_nose[1],
                                                  new_left_eye[0], new_left_eye[1],
                                                  new_right_eye[0], new_right_eye[1],
                                                  new_left_ear[0], new_left_ear[1],
                                                  new_right_ear[0], new_right_ear[1],
                                                  new_left_shoulder[0], new_left_shoulder[1],
                                                  new_right_shoulder[0], new_right_shoulder[1],
                                                  new_left_elbow[0], new_left_elbow[1],
                                                  new_right_elbow[0], new_right_elbow[1],
                                                  new_left_wrist[0], new_left_wrist[1],
                                                  new_right_wrist[0], new_right_wrist[1],
                                                  new_left_hip[0], new_left_hip[1],
                                                  new_right_hip[0], new_right_hip[1],
                                                  new_left_knee[0], new_left_knee[1],
                                                  new_right_knee[0], new_right_knee[1],
                                                  new_left_ankle[0], new_left_ankle[1],
                                                  new_right_ankle[0], new_right_ankle[1]]

                                new_left_arm_element = [new_left_shoulder[0], new_left_shoulder[1],
                                                        new_left_elbow[0], new_left_elbow[1],
                                                        new_left_wrist[0], new_left_wrist[1]]

                                new_right_arm_element = [new_right_shoulder[0], new_right_shoulder[1],
                                                         new_right_elbow[0], new_right_elbow[1],
                                                         new_right_wrist[0], new_right_wrist[1]]

                                new_upper_body_element = [new_right_shoulder[0], new_right_shoulder[1],
                                                          new_left_shoulder[0], new_left_shoulder[1],
                                                          new_right_hip[0], new_right_hip[1],
                                                          new_left_hip[0], new_left_hip[1]]

                                new_lower_body_element = [new_right_hip[0], new_right_hip[1],
                                                          new_right_knee[0], new_right_knee[1],
                                                          new_right_ankle[0], new_right_ankle[1],
                                                          new_left_hip[0], new_left_hip[1],
                                                          new_left_knee[0], new_left_knee[1],
                                                          new_left_ankle[0], new_left_ankle[1]]

                                new_head_element = [new_nose[0], new_nose[1],
                                                    new_right_eye[0], new_right_eye[1],
                                                    new_left_eye[0], new_left_eye[1],
                                                    new_right_ear[0], new_right_ear[1],
                                                    new_left_ear[0], new_left_ear[1]]

                                # argument 한번 append
                                new_whole_body_element_group.append(new_whole_body_element)
                                new_right_arm_element_group.append(new_right_arm_element)
                                new_left_arm_element_group.append(new_left_arm_element)
                                new_upper_body_element_group.append(new_upper_body_element)
                                new_lower_body_element_group.append(new_lower_body_element)
                                new_head_element_group.append(new_head_element)

                            new_whole_body.append(np.array(new_whole_body_element_group))
                            new_right_arm.append(np.array(new_right_arm_element_group))
                            new_left_arm.append(np.array(new_left_arm_element_group))
                            new_upper_body.append(np.array(new_upper_body_element_group))
                            new_lower_body.append(np.array(new_lower_body_element_group))
                            new_head.append(np.array(new_head_element_group))
                            new_label.append(category)

                            i += 1

                return [new_whole_body, new_right_arm, new_left_arm, new_upper_body, new_lower_body, new_head, new_label]



    def augment_keypoint(self, keypoint, sigma):
        z_score = norm.ppf(0.975)  # 95% 신뢰구간
        confidence_interval = z_score * sigma

        while True:
            new_x = keypoint[0] + np.random.normal(0, sigma)
            new_y = keypoint[1] + np.random.normal(0, sigma)
            if np.abs(new_x - keypoint[0]) <= confidence_interval and np.abs(new_y - keypoint[1]) <= confidence_interval:
                return [new_x, new_y]

    def augment_data(self, data):
        augmented_data = []
        for group in data:
            augmented_group = []
            for frame in group:
                augmented_frame = []
                for j in range(0, len(frame), 2):
                    augmented_point = self.augment_keypoint([frame[j], frame[j+1]], self.sigma)
                    augmented_frame.extend(augmented_point)
                augmented_group.append(augmented_frame)
            augmented_data.append(np.array(augmented_group))
        return augmented_data

    def get_dataset(self):
        # self.read()
        self.predata = self.read()
        return self.predata

if __name__ == "__main__":
    train_csv_path = "C:\\Users\\LAMS\\Desktop\\hfprNSD\\data\\hofs_e\\csves1\\HOFS_E_3_val.csv"
    dataset_instance = HOFS_E(train_csv_path, 3, 1, True)
    train_dataset = dataset_instance.get_dataset()
    train_whole_body = np.array(train_dataset[0])
