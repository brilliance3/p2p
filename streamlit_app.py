import streamlit as st
import numpy as np
import pandas as pd
import joblib

# 모델과 피처 목록 불러오기
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
policy_code = st.selectbox('정책 코드 (Policy Code)', [1])  # 대부분 1로 설정되어 있음

# 빈 입력 데이터프레임 생성
input_df = pd.DataFrame(columns=feature_names)
input_df.loc[0] = 0  # 모든 피처 0으로 초기화

# 사용자 입력 반영
input_df.at[0, 'fico_range_low'] = risk_score
input_df.at[0, 'dti'] = dti
input_df.at[0, 'emp_length'] = emp_length
input_df.at[0, 'policy_code'] = policy_code

# 지역 선택 반영 (one-hot encoding)
state_col = f'addr_state_{state}'
if state_col in input_df.columns:
    input_df.at[0, state_col] = 1
else:
    st.warning(f"⚠️ 선택한 주(state): {state}는 학습 데이터에 포함되지 않았습니다.")

# 예측 버튼
if st.button('상환률 예측하기'):
    pred = model.predict(input_df)[0]
    st.success(f'📊 예측된 대출 상환률은 **{pred * 100:.2f}%** 입니다.')
