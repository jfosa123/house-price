import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="房价预测模型", page_icon="🏠", layout="wide")

st.title('🏠 房价预测模型 - 专业版')
st.warning("⚠️ 注意：此版本使用特征映射，准确性可能受影响")

# 加载模型
@st.cache_resource
def load_model():
    try:
        with open('house_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

model = load_model()

# 简化版特征映射 - 只收集最重要的特征
st.header('📊 请输入房屋信息')

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider('整体质量评分', 1, 10, 6)
    gr_liv_area = st.number_input('地上居住面积 (平方英尺)', value=1500, min_value=500, max_value=10000)
    total_bsmt_sf = st.number_input('地下室总面积', value=1000, min_value=0, max_value=5000)
    garage_cars = st.slider('车库容量', 0, 4, 2)

with col2:
    year_built = st.slider('建造年份', 1900, 2020, 2000)
    full_bath = st.slider('全卫数量', 0, 4, 2)
    bedrooms = st.slider('卧室数量', 1, 8, 3)
    tot_rms_abv_grd = st.slider('总房间数', 2, 15, 6)

# 预测函数
def create_smart_features(input_dict):
    """智能创建262维特征向量"""
    features = np.zeros(262)
    
    # 基于Kaggle竞赛的常见特征位置映射
    mapping = {
        'OverallQual': 0, 'GrLivArea': 1, 'GarageCars': 2, 'TotalBsmtSF': 3,
        'YearBuilt': 4, 'FullBath': 5, 'TotRmsAbvGrd': 6, 'BedroomAbvGr': 7,
        'YearRemodAdd': 8, 'Fireplaces': 9, 'GarageArea': 10, '1stFlrSF': 11,
        'WoodDeckSF': 12, 'OpenPorchSF': 13, 'LotArea': 14, 'OverallCond': 15
    }
    
    # 设置已知特征
    for feature, value in input_dict.items():
        if feature in mapping:
            features[mapping[feature]] = value
    
    # 设置合理的默认值给其他特征
    features[16:50] = 1    # 各种存在性标志
    features[50:100] = 2000  # 年份相关特征
    
    return features.reshape(1, -1)

if st.button('🚀 预测房价', type='primary') and model is not None:
    user_input = {
        'OverallQual': overall_qual,
        'GrLivArea': gr_liv_area,
        'GarageCars': garage_cars,
        'TotalBsmtSF': total_bsmt_sf,
        'YearBuilt': year_built,
        'FullBath': full_bath,
        'TotRmsAbvGrd': tot_rms_abv_grd,
        'BedroomAbvGr': bedrooms
    }
    
    try:
        features = create_smart_features(user_input)
        prediction = model.predict(features)[0]
        
        st.success(f'## 预测房价: ${prediction:,.2f}')
        st.info(f'特征维度: {features.shape}')
        
    except Exception as e:
        st.error(f'预测错误: {e}')

st.markdown("---")
st.markdown("需要更精确的预测？请提供模型训练时的特征列表")
