import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="房价预测模型", page_icon="🏠", layout="wide")

# 应用标题
st.title('🏠 Kaggle房价预测模型')
st.markdown('基于XGBoost的房价预测系统')

# 加载模型
@st.cache_resource
def load_model():
    try:
        with open('house_price_model.pkl', 'rb') as f:
            model_bundle = pickle.load(f)
        
        # 从model_bundle中提取所有组件
        model = model_bundle['model']
        feature_names = model_bundle['feature_names']
        scaler = model_bundle['scaler']  # 获取缩放器
        
        st.success(f"✅ 模型加载成功！特征数量: {len(feature_names)}")
        return model, feature_names, scaler
    except Exception as e:
        st.error(f"❌ 加载模型失败: {e}")
        return None, None, None

model, feature_names, scaler = load_model()

# 创建特征向量的函数
def create_features(user_input):
    """
    基于用户输入创建特征向量，并确保与训练时特征顺序一致
    """
    # 创建一个全零的特征向量，长度与训练时相同
    features = np.zeros(len(feature_names))
    
    # 创建特征名称到索引的映射
    feature_to_index = {name: idx for idx, name in enumerate(feature_names)}
    
    # 设置用户提供的特征值
    feature_mapping = {
        'OverallQual': user_input['OverallQual'],
        'GrLivArea': user_input['GrLivArea'],
        'GarageCars': user_input['GarageCars'],
        'TotalBsmtSF': user_input['TotalBsmtSF'],
        'YearBuilt': user_input['YearBuilt'],
        '1stFlrSF': user_input['1stFlrSF'],
        'FullBath': user_input['FullBath'],
        'TotRmsAbvGrd': user_input['TotRmsAbvGrd'],
        'YearRemodAdd': user_input['YearRemodAdd'],
        'GarageArea': user_input['GarageArea'],
        'BedroomAbvGr': user_input['BedroomAbvGr'],
        'Fireplaces': user_input['Fireplaces'],
        'LotArea': user_input['LotArea'],
        'WoodDeckSF': user_input['WoodDeckSF'],
        'OpenPorchSF': user_input['OpenPorchSF'],
        'MoSold': user_input['MoSold'],
        'YrSold': user_input['YrSold']
    }
    
    # 将特征值设置到正确的位置
    for feature_name, value in feature_mapping.items():
        if feature_name in feature_to_index:
            idx = feature_to_index[feature_name]
            features[idx] = value
    
    return features.reshape(1, -1)

# 用户输入界面（保持不变）
st.header('📊 请输入房屋信息')

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 核心特征")
    overall_qual = st.slider('整体质量评分 (OverallQual)', 1, 10, 6)
    gr_liv_area = st.number_input('地上居住面积 (GrLivArea)', value=1500, min_value=500, max_value=10000)
    total_bsmt_sf = st.number_input('地下室总面积 (TotalBsmtSF)', value=1000, min_value=0, max_value=5000)
    garage_cars = st.slider('车库容量 (GarageCars)', 0, 4, 2)
    year_built = st.slider('建造年份 (YearBuilt)', 1900, 2023, 2000)

with col2:
    st.subheader("🛏️ 房间配置")
    first_fl_sf = st.number_input('一层面积 (1stFlrSF)', value=1000, min_value=500, max_value=5000)
    full_bath = st.slider('全卫数量 (FullBath)', 1, 4, 2)
    tot_rms_abv_grd = st.slider('总房间数 (TotRmsAbvGrd)', 2, 15, 6)
    bedrooms = st.slider('卧室数量 (BedroomAbvGr)', 1, 6, 3)
    fireplaces = st.slider('壁炉数量 (Fireplaces)', 0, 3, 1)

# 其他特征
col3, col4 = st.columns(2)

with col3:
    st.subheader("🚗 其他特征")
    year_remod_add = st.slider('改建年份 (YearRemodAdd)', 1950, 2023, 2000)
    garage_area = st.number_input('车库面积 (GarageArea)', value=500, min_value=0, max_value=1500)

with col4:
    st.subheader("🌳 外部特征")
    lot_area = st.number_input('地块面积 (LotArea)', value=10000, min_value=1000, max_value=50000)
    wood_deck_sf = st.number_input('木平台面积 (WoodDeckSF)', value=0, min_value=0, max_value=500)
    open_porch_sf = st.number_input('开放式门廊面积 (OpenPorchSF)', value=50, min_value=0, max_value=500)

# 固定特征
mo_sold = st.slider('销售月份 (MoSold)', 1, 12, 6)
yr_sold = st.slider('销售年份 (YrSold)', 2000, 2023, 2020)

# 预测按钮
st.markdown("---")
predict_button = st.button('🚀 预测房价', type='primary', use_container_width=True)

if predict_button and model is not None and scaler is not None:
    try:
        # 收集用户输入
        user_input = {
            'OverallQual': overall_qual,
            'GrLivArea': gr_liv_area,
            'GarageCars': garage_cars,
            'TotalBsmtSF': total_bsmt_sf,
            'YearBuilt': year_built,
            '1stFlrSF': first_fl_sf,
            'FullBath': full_bath,
            'TotRmsAbvGrd': tot_rms_abv_grd,
            'YearRemodAdd': year_remod_add,
            'GarageArea': garage_area,
            'BedroomAbvGr': bedrooms,
            'Fireplaces': fireplaces,
            'LotArea': lot_area,
            'WoodDeckSF': wood_deck_sf,
            'OpenPorchSF': open_porch_sf,
            'MoSold': mo_sold,
            'YrSold': yr_sold
        }
        
        # 创建特征向量
        features = create_features(user_input)
        
        # 关键：使用训练时的缩放器进行特征缩放
        features_scaled = scaler.transform(features)
        
        # 进行预测
        log_prediction = model.predict(features_scaled)[0]
        
        # 关键：用指数转换还原为实际价格
        prediction = np.expm1(log_prediction)
        
        # 显示结果
        st.success('## 📈 预测结果')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="预测房价", value=f"${prediction:,.0f}")
        
        with col2:
            st.metric(label="人民币", value=f"¥{prediction * 7.2:,.0f}")
        
        with col3:
            if prediction > 500000:
                level = "豪华"
            elif prediction > 300000:
                level = "中等偏上"
            elif prediction > 150000:
                level = "中等"
            else:
                level = "经济"
            st.metric(label="房屋等级", value=level)
            
    except Exception as e:
        st.error(f"❌ 预测错误: {e}")

elif predict_button:
    st.error("❌ 模型或缩放器未加载，无法预测")

# 页脚
st.markdown("---")
st.markdown("基于XGBoost模型 | 特征缩放 | 对数价格转换 | Streamlit部署")
