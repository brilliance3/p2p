import streamlit as st
import numpy as np
import pandas as pd
import joblib

# 모델, 피처 목록 불러오기 (상환률 예측용)
model = joblib.load('rf_repayment_model.pkl')
feature_names = joblib.load('feature_names.pkl')

st.title('📈 대출 상환률 예측기')

st.markdown("""
사용자의 신용 점수, 부채비율, 재직기간 등 정보를 기반으로
**예상 상환률**을 예측합니다.
""")

# 사용자 입력 받기
risk_score = st.slider('신용 위험 점수 (FICO Range Low)', 300, 900, 650)
dti = st.number_input('부채 대비 소득 비율 (DTI, %)', value=15.0)
emp_length = st.slider('재직 기간 (년)', 0.0, 10.0, 2.0)
state = st.selectbox('거주 지역 (State)', ['CA', 'NY', 'TX', 'FL', 'IL'])

# 입력값 구성
user_input = {
    'fico_range_low': risk_score,
    'dti': dti,
    'emp_length': emp_length,
    'addr_state_' + state: 1
}

# 빈 입력 데이터프레임 생성
input_df = pd.DataFrame(columns=feature_names)
input_df.loc[0] = 0  # 기본값 0으로 채우기

# 사용자 입력 적용
for k, v in user_input.items():
    if k in input_df.columns:
        input_df.at[0, k] = v

# 예측 버튼
if st.button('상환률 예측하기'):
    pred = model.predict(input_df)[0]
    st.success(f'📊 예측된 대출 상환률은 **{pred * 100:.2f}%** 입니다.')
