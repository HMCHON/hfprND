import os
import pandas as pd

# 작업 디렉토리 설정
work_dir = '../work_dir'  # 실제 작업 디렉토리로 변경하세요

# 결과를 저장할 데이터프레임 초기화
result_df = pd.DataFrame(columns=['model_name', 'augmented', 'frame', 'loss', 'accuracy'])

# 작업 디렉토리 내 모든 폴더 순회
for folder_name in os.listdir(work_dir):
    folder_path = os.path.join(work_dir, folder_name)
    if os.path.isdir(folder_path):
        # 폴더 이름에서 model_name과 augmented 추출
        parts = folder_name.split('_')
        model_name = parts[0] + '_' + parts[1]
        augmented = parts[2]

        # if parts[1] in ['model23', 'model24', 'model25', 'model26', 'model27', 'model28']:
        # if parts[1] in ['model17', 'model18', 'model19', 'model20', 'model21', 'model22']:
        # if parts[1] in ['model11', 'model12', 'model13', 'model14', 'model15', 'model16']:
        # if parts[1] in ['model5', 'model6', 'model7', 'model8', 'model9', 'model10']:

        # 폴더 내 CSV 파일 읽기
        for file_name in os.listdir(folder_path):
            if file_name == 'pre_result.csv':
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)

                # 필요한 열만 추출하여 새로운 데이터프레임 생성
                df['model_name'] = model_name
                df['augmented'] = augmented
                df = df[['model_name', 'augmented', 'frame', 'loss', 'accuracy']]

                # 결과 데이터프레임에 추가
                result_df = pd.concat([result_df, df], ignore_index=True)

# 결과를 CSV 파일로 저장
result_df.to_csv('../check_combined_results.csv', index=False)
print("CSV 파일이 성공적으로 생성되었습니다.")
