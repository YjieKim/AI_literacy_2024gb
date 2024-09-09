import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import streamlit as st

# 경고 메시지 출력 안 함.
warnings.filterwarnings(action='ignore')

# 한글 글꼴 설정
# plt.rc('font', family='malgun gothic')
plt.rc('font', family='Gulim')

# 기본 실행 함수
def basic():

    # 데이터 가져오기
    df = pd.read_csv('data/electric_car_data.csv', encoding='EUC-KR')

    # melt를 위해 전체 컬럼명이 필요함.
    columns = df.columns.to_list()
    df_melt = pd.melt(df, id_vars=['기준일'], value_vars=columns, var_name='지역', value_name='자동차수')

    # 파생변수 생성
    df_melt['연'] = df_melt['기준일'].str[:4]
    df_melt['월'] = df_melt['기준일'].str[5:7]

    return df_melt


### 지역별 연도별 분석 ###
def region_year_mean(df_melt):

    # 지역별, 연도별 자동차수 평균
    df_region_year = round(pd.pivot_table(df_melt, index='지역', columns='연', values='자동차수', aggfunc='mean'), ndigits=2)
    
    # 시각화: 간단하게 이용할 경우 pandas 사용.
    # df.plot(kind="차트종류", x="필드", y="필드")
    # 인덱스가 x축인 그래프가 기본임.
    # rot=0 : 글자 가로로.
    # df_region_year.plot(kind='bar', rot=0)
    # plt.show()

    # 지역에서 '합계'를 제외.
    # 지역은 index임.
    # 행의 데이터 추출, df[조건]. 조건이 여러개면 df[(조건) (연산자: |, & 등) (조건)]
    # df_region_year_n = df_region_year[df_region_year.index != '합계']
    df_region_year_n = df_region_year[df_region_year.index != '합계']

    st.subheader("지역별/연도별 분석")

    # Streamlit에서 표 형식으로 출력하려면 dataframe 사용.
    # 데이터프레임의 행렬을 전환 : T
    st.dataframe(df_region_year_n.T)

    # 보통 차트 변수는 ax(=axes) 사용 많이 함.
    ax = df_region_year_n.plot(kind='bar', rot=0)
    
    # Streamlit에서 차트 형식으로 출력.
    fig = ax.get_figure()
    st.pyplot(fig)


### 2023년 지역별 월별 분석 ###
def region_2023_month(df_melt):

    st.subheader("2023년 지역별 분석")

    df_2023 = df_melt[(df_melt['연'] == '2023') & (df_melt['지역'] != '합계')]

    df_2023_g = round(pd.pivot_table(df_2023, index='지역', columns='월', values='자동차수', aggfunc='mean'), 2)

    # table은 dataframe과는 달리 아무런 인터렉션이 없음.
    # st.table(df_2023_g)
    st.dataframe(df_2023_g.T)

    ax = df_2023_g.plot(kind='bar', rot=0)
    fig = ax.get_figure()
    st.pyplot(fig)


### 2022년 지역별 분기별 분석 ###
def region_2022_quarter(df_melt):

    st.subheader("2022년 분기별 분석")

    df_2022 = df_melt[(df_melt['연'] == '2022') & (df_melt['지역'] != '합계')]

    # 분기 컬럼 추가
    ## 1. 데이터 타입을 정수로 변경
    df_2022_n = df_2022
    df_2022_n['월'] = df_2022['월'].astype(int)

    ## 2. where: 조건 비교 함수
    df_2022_n['분기'] = np.where((df_2022_n['월'] >= 1) & (df_2022_n['월'] <= 3), '1분기',
                            np.where((df_2022_n['월'] >= 4) & (df_2022_n['월'] <= 6), '2분기',
                            np.where((df_2022_n['월'] >= 7) & (df_2022_n['월'] <= 9), '3분기', '4분기')))

    # 피벗테이블 생성
    df_2022_g = round(pd.pivot_table(df_2022_n, index='지역', columns='분기', values='자동차수', aggfunc='mean'), 2)
    st.dataframe(df_2022_g.T)

    # 차트 생성
    ax = df_2022_g.plot(kind='bar', rot=0)
    fig = ax.get_figure()
    st.pyplot(fig)
    
    # 참고) Group by 이용 : 기준 컬럼, 계산할 컬럼, 계산방법
    # df_2022_g2 = round(df_2022_n.groupby(['지역','분기'])[['자동차수']].mean().reset_index(), 2)
    # print(df_2022_g2)

# main 실행 함수
def elec_exe():
    
    # menu = st.selectbox("분석내용", ['지역별/연도별 분석', '2023년 지역별 분석', '2022년 분기별 분석'], index=None, placeholder="선택하세요.")
    menu = st.selectbox("분석내용", ['선택하세요.', '지역별/연도별 분석', '2023년 지역별 분석', '2022년 분기별 분석'])

    df_basic = basic()

    if menu == '지역별/연도별 분석':
        region_year_mean(df_basic)
    elif menu == '2023년 지역별 분석':
        region_2023_month(df_basic)
    elif menu == '2022년 분기별 분석':
        region_2022_quarter(df_basic)
    else:
        st.image("image/Moses.png", width=600)

# 자신이 호출할 때만 실행되는 코드. (다른 파일에서 호출할 때는 실행 안 됨.)
if __name__=='__main__':
    elec_exe()
