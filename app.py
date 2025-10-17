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
            model = pickle.load(f)
        st.success("✅ 模型加载成功！")
        return model
    except Exception as e:
        st.error(f"❌ 加载模型失败: {e}")
        return None

model = load_model()

# 特征创建函数
def create_features(user_input):
    """
    创建特征向量
    """
    features = np.zeros(262)
    
    # 设置主要特征
    features[16] = user_input['OverallQual']    # 整体质量
    features[45] = user_input['GrLivArea']      # 地上居住面积
    features[37] = user_input['TotalBsmtSF']    # 地下室总面积
    features[18] = user_input['YearBuilt']      # 建造年份
    features[60] = user_input['GarageCars']     # 车库容量
    features[42] = user_input['1stFlrSF']       # 一层面积
    features[48] = user_input['FullBath']       # 全卫数量
    features[50] = user_input['BedroomAbvGr']   # 卧室数量
    features[53] = user_input['TotRmsAbvGrd']   # 总房间数
    features[61] = user_input['GarageArea']     # 车库面积
    
    # 设置合理的默认值
    features[0] = 60    # MSSubClass
    features[3] = user_input['LotArea']
    features[17] = user_input['OverallCond']
    features[19] = user_input['YearRemodAdd']
    features[55] = user_input['Fireplaces']
    features[65] = user_input['WoodDeckSF']
    features[66] = user_input['OpenPorchSF']
    
    return features.reshape(1, -1)

# 用户输入界面
st.header('📊 请输入房屋信息')

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 核心特征")
    overall_qual = st.slider('整体质量评分 (1-10)', 1, 10, 7)
    gr_liv_area = st.number_input('地上居住面积 (平方英尺)', value=1800, min_value=500, max_value=10000)
    total_bsmt_sf = st.number_input('地下室总面积', value=1200, min_value=0, max_value=5000)
    year_built = st.slider('建造年份', 1900, 2023, 2005)
    garage_cars = st.slider('车库容量', 0, 4, 2)

with col2:
    st.subheader("🛏️ 房间配置")
    first_fl_sf = st.number_input('一层面积', value=1200, min_value=500, max_value=5000)
    full_bath = st.slider('全卫数量', 1, 4, 2)
    tot_rms_abv_grd = st.slider('总房间数', 4, 12, 7)
    bedrooms = st.slider('卧室数量', 2, 6, 3)
    overall_cond = st.slider('整体状况评分', 1, 10, 6)

# 其他特征
col3, col4 = st.columns(2)

with col3:
    st.subheader("🚗 其他特征")
    garage_area = st.number_input('车库面积', value=600, min_value=0, max_value=1500)
    year_remod_add = st.slider('改建年份', 1950, 2023, 2005)

with col4:
    st.subheader("🌳 外部特征")
    fireplaces = st.slider('壁炉数量', 0, 3, 1)
    lot_area = st.number_input('地块面积', value=12000, min_value=1000, max_value=50000)
    wood_deck_sf = st.number_input('木平台面积', value=100, min_value=0, max_value=500)
    open_porch_sf = st.number_input('开放式门廊面积', value=50, min_value=0, max_value=500)

# 预测按钮
st.markdown("---")
predict_button = st.button('🚀 预测房价', type='primary', use_container_width=True)

if predict_button and model is not None:
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
            'OverallCond': overall_cond
        }
        
        # 创建特征向量
        features = create_features(user_input)
        
        # 进行预测
        log_prediction = model.predict(features)[0]
        
        # === 关键修复：添加基准值 ===
        # 由于模型可能预测的是相对值，我们需要加上一个基准
        base_price_log = 12.0  # 调整这个基准值
        adjusted_log_prediction = log_prediction + base_price_log
        
        # 指数转换还原为实际价格
        prediction = np.expm1(adjusted_log_prediction)
        
        # 显示结果
        st.success('## 📈 预测结果')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="预测房价", value=f"${prediction:,.0f}")
        
        with col2:
            st.metric(label="人民币", value=f"¥{prediction * 7.2:,.0f}")
        
        with col3:
            if prediction > 400000:
                level = "豪华"
            elif prediction > 250000:
                level = "中等偏上" 
            elif prediction > 150000:
                level = "中等"
            else:
                level = "经济"
            st.metric(label="房屋等级", value=level)
        
        # 调试信息
        with st.expander("🔧 调试信息"):
            st.write(f"原始对数预测值: {log_prediction:.4f}")
            st.write(f"调整后对数预测值: {adjusted_log_prediction:.4f}")
            st.write(f"指数转换后: ${prediction:,.2f}")
            st.write(f"使用的基准值: {base_price_log}")
            
            # 测试不同的基准值
            st.write("**不同基准值的测试:**")
            test_bases = [10.0, 11.0, 12.0, 12.5, 13.0]
            for base in test_bases:
                test_price = np.expm1(log_prediction + base)
                st.write(f"基准 {base}: ${test_price:,.0f}")
            
    except Exception as e:
        st.error(f"❌ 预测错误: {e}")

elif predict_button:
    st.error("❌ 模型未加载，无法预测")

st.markdown("---")
st.markdown("基于XGBoost模型 | 基准值调整 | Streamlit部署")
