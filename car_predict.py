import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import csv
import streamlit as st


# 데이터 입력받기
def indata():

    # fields = ['id','model','year','transmission','mileage','fuelType','tax','mpg','engineSize']   # 리스트
    # data = {}   # 딕셔너리

    # 데이터 입력
    # for i in fields:
    #     data[i] = st.text_input(f'{i} : ')

    # df = pd.DataFrame(data)
    # print(data)

    input_data = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        input_data['id'] = st.number_input('ID : ', min_value=0, value=8001, step=1, format="%d")
        input_data['model'] = st.text_input('Model : ', value=None, placeholder='ex) A1, S8, RS6, ...')
        input_data['year'] = st.number_input('Year : ', min_value=1900, value=2020, step=1, format="%d")

    with col2:
        input_data['transmission'] = st.selectbox('Transmission : ', ('Automatic', 'Semi-Auto', 'Manual'), index=0)
        input_data['mileage'] = st.number_input('Mileage : ', min_value=0, step=1000, format="%d")
        input_data['fuelType'] = st.selectbox('Fuel Type : ', ('Petrol', 'Diesel', 'Hybrid'), index=0)
        

    with col3:
        input_data['tax'] = st.number_input('Tax : ', min_value=0, step=1, format="%d")
        input_data['mpg'] = st.number_input('Mpg : ', min_value=0.0, step=0.1, format="%.1f")
        input_data['engineSize'] = st.number_input('Engine Size : ', min_value=0.0, step=0.1, format="%.1f")
    
    list_data = []
    list_data.append(input_data)

    df_data = pd.DataFrame(list_data)

    return df_data

# 인공지능 모델
def model(data):

    # 1. 파일 열기 ===================================================
    df_X_train = pd.read_csv('data/used_car_x_train.csv')
    df_y_train = pd.read_csv('data/used_car_y_train.csv')
    df_X_test = data

    # 2. 전처리 ======================================================

    # 1) 범주형, 숫자형 데이터로 분리
    
    # 숫자형 데이터
    X_train_num = df_X_train.select_dtypes(include='number')
    X_test_num = df_X_test.select_dtypes(include='number')

    # 범주형 데이터
    X_train_word = df_X_train.loc[:, ~df_X_train.columns.isin(X_train_num.columns.to_list())]
    X_test_word = df_X_test.loc[:, ~df_X_test.columns.isin(X_test_num.columns.to_list())]

    # 문자열 공백 제거
    X_train_word = X_train_word.apply(lambda x: x.str.strip(), axis = 1)
    X_test_word = X_test_word.apply(lambda x: x.str.strip(), axis = 1)

    # 스케일링 전 id 삭제
    X_train_num = X_train_num.drop(columns=['id'])
    X_test_num = X_test_num.drop(columns=['id'])


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

    # 컬럼명으로 Sort
    train_word_list = df_train_word.columns.to_list()
    train_word_list.sort()

    test_word_list = df_test_word.columns.to_list()
    test_word_list.sort()

    df_train = pd.concat([df_train_num, df_train_word[train_word_list]], axis=1)    # axis=0이면 아래로, 1이면 옆으로.
    df_test = pd.concat([df_test_num, df_test_word[test_word_list]], axis=1)


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
    X_predict_rf_score = rforest_model.score(X_val, y_val)

    # 평가2 : RMSE 이용
    X_predict_rf_rmse = root_mean_squared_error(y_val, rforest_model.predict(X_val))    # 예측값

    # 5. 활용
    y_predict = rforest_model.predict(df_test)

    st.write(f'입력하신 중고차 예상 가격은 {format(round(y_predict[0]), ',')}원입니다.')
    st.divider()

    st.write('이 예측은 Random Forest를 이용하여 모델링한 것으로, 성능평가지표는 다음과 같습니다.')
    st.write('R2 score : ', round(X_predict_rf_score, 2))
    st.write('RMSE : ', round(X_predict_rf_rmse, 2))


# 외부 호출용 함수
def aiml_main():
    data = indata()

    if st.button('예상 가격 조회'):
        if data['model'].values == [None]:
            st.write("'Model'을 입력하세요.")
        else:
            model(data)


if __name__ == '__main__':
    aiml_main()
