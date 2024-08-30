import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import csv
import streamlit as st


# 데이터 입력받기
def indata():

    fields = ['id','model','year','transmission','mileage','fuelType','tax','mpg','engineSize']   # 리스트
    data = {}   # 딕셔너리

    # 데이터 입력
    for i in fields:
        data[i] = st.text_input(f'{i} : ')
    filename = 'aidata/used_car_x_test_indata.csv'

    with open(filename, 'w', newline='') as f:    # f는 파일 개체 변수
        w = csv.DictWriter(f, fieldnames=fields)    # 딕셔너리 데이터임을 선언.
        w.writeheader()   # 열제목 작성
        w.writerow(data)    # 데이터 작성

    return filename

# 인공지능 모델
def model(filename):

    # 1. 파일 열기 ===================================================
    df_X_train = pd.read_csv('aidata/used_car_x_train.csv')
    df_y_train = pd.read_csv('aidata/used_car_y_train.csv')
    df_X_test = pd.read_csv(filename)

    # 2. 전처리 ======================================================

    # 1) 범주형, 숫자형 데이터로 분리
    # 범주형 데이터 - 스페이스값 있는 데이터 주의!
    df_X_train['model'] = df_X_train['model'].str.replace(" ","")   # 공백 없애기(str.replace와 replace 다름. 또는 stream함수 사용)
    df_X_test['model'] = df_X_test['model'].str.replace(" ","")

    X_train_word = df_X_train[['model', 'transmission', 'fuelType']]
    X_test_word = df_X_test[['model', 'transmission', 'fuelType']]

    # 숫자형 데이터 (범주형을 제외한 것)
    X_train_num = df_X_train.drop(['id', 'model', 'transmission', 'fuelType'], axis=1)
    X_test_num = df_X_test.drop(['id', 'model', 'transmission', 'fuelType'], axis=1)

    # 2) 데이터 스케일링 : min-max 정규화 - 수치형 데이터
    # 객체 생성
    scaler = MinMaxScaler()

    # 학습
    X_train_num_scale = scaler.fit_transform(X_train_num)   # transform은 적용시키는 것.
    X_test_num_scale = scaler.transform(X_test_num)     # 테스트 데이터는 fit(학습) 하지 않음.

    # 데이터 프레임 설정
    df_train_num = pd.DataFrame(X_train_num_scale, columns=X_train_num.columns)
    df_test_num = pd.DataFrame(X_test_num_scale, columns=X_test_num.columns)

    # 3) 원핫 인코딩 - 범주형 데이터
    df_train_word = pd.get_dummies(X_train_word)    # 데이터프레임 형태이므로 그대로 사용.
    df_test_word = pd.get_dummies(X_test_word)

    # 원핫 인코딩 후에 훈련데이터와 테스트데이터 컬럼 체크.
    # 집합{}으로 만드는 파이썬 명령어: set

    # 훈련데이터 목록
    train_cols = set(df_train_word.columns)  # 집합: 키 값이 없는 딕셔너리 형태. 연산 가능
    test_cols = set(df_test_word)

    missing_test = train_cols - test_cols   # {'model_ A2', 'model_ RS7', 'model_ S5', 'model_ S8'}
    missing_train = test_cols - train_cols

    # df_test_word['model_ A2'] = 0 이런 식으로 위의 missing 컬럼들을 0값으로 추가해야 함.
    if len(missing_test) > 0:
        for i in missing_test:
            df_test_word[i] = 0

    if len(missing_train) > 0:
        for i in missing_train:
            df_train_word[i] = 0

    # 4) 위에서 구분한 문자 데이터와 숫자 데이터를 합쳐야 함.
    # 데이터 합칠 때 : merge는 키값이 있어야 하므로 지금 같은 경우는 합칠 수 없다.
    #                 concat은 키 없이 옆으로도 합칠 수 있다.

    df_train = pd.concat([df_train_num, df_train_word], axis=1)    # axis=0이면 아래로, 1이면 옆으로.
    df_test = pd.concat([df_test_num, df_test_word], axis=1)


    # 3. 모델링 =========================================================
    # 머신러닝 - 지도학습

    # 1) 독립변수(X), 종속변수(y)
    X = df_train
    y = df_y_train['price']

    # 2) 데이터 7:3으로 나누기
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0) # , stratify=

    # 회귀예측모델 생성 및 학습
    # 3) 모델 객체 생성
    rforest_model = RandomForestRegressor(random_state=0) # n_estimate 기본값은 100

    # 4) 학습
    rforest_model.fit(X_train, y_train)

    # 4. 평가
    # 평가1 : score 이용
    st.write('LandomForest R2 score : ', rforest_model.score(X_val, y_val))

    # 평가2 : RMSE 이용
    X_predict_f = rforest_model.predict(X_val)
    st.write('LandomForest RMSE : ', root_mean_squared_error(y_val, X_predict_f))  # 예측값

    # 5. 활용
    # 컬럼명, 위치 같아야 함.
    df_test2 = df_test[['year', 'mileage', 'tax', 'mpg', 'engineSize', 'model_A1', 'model_A2',
        'model_A3', 'model_A4', 'model_A5', 'model_A6', 'model_A7',
        'model_A8', 'model_Q2', 'model_Q3', 'model_Q5', 'model_Q7',
        'model_Q8', 'model_R8', 'model_RS3', 'model_RS4', 'model_RS5',
        'model_RS6', 'model_RS7', 'model_S3', 'model_S4', 'model_S5',
        'model_S8', 'model_SQ5', 'model_SQ7', 'model_TT',
        'transmission_Automatic', 'transmission_Manual',
        'transmission_Semi-Auto', 'fuelType_Diesel', 'fuelType_Hybrid', 'fuelType_Petrol']]

    y_predict = rforest_model.predict(df_test2)

    st.write(f'입력하신 중고차 예상 가격은 {y_predict[0]}원입니다.')


# 외부 호출용 함수
def aiml_main():
    filename = indata()
    if st.button('예상 가격 조회'):
        model(filename)



if __name__ == '__main__':
    aiml_main()
