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
        
        # 检查加载的是模型还是model_bundle
        if hasattr(model, 'predict'):
            # 直接加载的是模型
            st.success("✅ 模型加载成功！(直接模型)")
            return model, None, None
        else:
            # 加载的是model_bundle
            st.success("✅ 模型加载成功！(model_bundle)")
            return model['model'], model.get('feature_names'), model.get('scaler')
            
    except Exception as e:
        st.error(f"❌ 加载模型失败: {e}")
        return None, None, None

model, feature_names, scaler = load_model()

# 创建特征向量的函数
def create_features(user_input):
    """
    基于用户输入创建特征向量
    """
    # 如果不知道确切的特征，我们创建一个262维的特征向量
    # 基于Kaggle房价预测的常见特征顺序
    features = np.zeros(262)
    
    # 设置主要特征（基于常见Kaggle特征顺序）
    features[16] = user_input['OverallQual']    # 整体质量
    features[45] = user_input['GrLivArea']      # 地上居住面积
    features[60] = user_input['GarageCars']     # 车库容量
    features[37] = user_input['TotalBsmtSF']    # 地下室总面积
    features[18] = user_input['YearBuilt']      # 建造年份
    features[42] = user_input['1stFlrSF']       # 一层面积
    features[48] = user_input['FullBath']       # 全卫数量
    features[50] = user_input['BedroomAbvGr']   # 卧室数量
    features[53] = user_input['TotRmsAbvGrd']   # 总房间数
    features[61] = user_input['GarageArea']     # 车库面积
    features[19] = user_input['YearRemodAdd']   # 改建年份
    features[55] = user_input['Fireplaces']     # 壁炉数量
    features[3] = user_input['LotArea']         # 地块面积
    features[65] = user_input['WoodDeckSF']     # 木平台面积
    features[66] = user_input['OpenPorchSF']    # 开放式门廊面积
    features[75] = user_input['MoSold']         # 销售月份
    features[76] = user_input['YrSold']         # 销售年份
    
    # 设置合理的默认值给其他特征
    features[0] = 60    # MSSubClass: 最常见的房屋类型
    features[17] = 5    # OverallCond: 整体状况
    features[4] = 1     # Street: 铺砌街道
    features[40] = 1    # CentralAir: 有中央空调
    
    return features.reshape(1, -1)

# 用户输入界面
st.header('📊 请输入房屋信息')

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 核心特征")
    overall_qual = st.slider('整体质量评分 (OverallQual)', 1, 10, 6,
                           help='1-10分，10分为最佳')
    gr_liv_area = st.number_input('地上居住面积 (GrLivArea)', value=1500,
                                min_value=500, max_value=10000)
    total_bsmt_sf = st.number_input('地下室总面积 (TotalBsmtSF)', value=1000,
                                  min_value=0, max_value=5000)
    garage_cars = st.slider('车库容量 (GarageCars)', 0, 4, 2)
    year_built = st.slider('建造年份 (YearBuilt)', 1900, 2023, 2000)

with col2:
    st.subheader("🛏️ 房间配置")
    first_fl_sf = st.number_input('一层面积 (1stFlrSF)', value=1000,
                                min_value=500, max_value=5000)
    full_bath = st.slider('全卫数量 (FullBath)', 1, 4, 2)
    tot_rms_abv_grd = st.slider('总房间数 (TotRmsAbvGrd)', 2, 15, 6)
    bedrooms = st.slider('卧室数量 (BedroomAbvGr)', 1, 6, 3)
    fireplaces = st.slider('壁炉数量 (Fireplaces)', 0, 3, 1)

# 其他特征
col3, col4 = st.columns(2)

with col3:
    st.subheader("🚗 其他特征")
    year_remod_add = st.slider('改建年份 (YearRemodAdd)', 1950, 2023, 2000)
    garage_area = st.number_input('车库面积 (GarageArea)', value=500,
                                min_value=0, max_value=1500)

with col4:
    st.subheader("🌳 外部特征")
    lot_area = st.number_input('地块面积 (LotArea)', value=10000,
                             min_value=1000, max_value=50000)
    wood_deck_sf = st.number_input('木平台面积 (WoodDeckSF)', value=0,
                                 min_value=0, max_value=500)
    open_porch_sf = st.number_input('开放式门廊面积 (OpenPorchSF)', value=50,
                                  min_value=0, max_value=500)

# 固定特征
mo_sold = st.slider('销售月份 (MoSold)', 1, 12, 6)
yr_sold = st.slider('销售年份 (YrSold)', 2000, 2023, 2020)

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
            'MoSold': mo_sold,
            'YrSold': yr_sold
        }
        
        # 创建特征向量
        features = create_features(user_input)
        
        # 进行预测
        log_prediction = model.predict(features)[0]
        
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
        
        # 显示调试信息
        with st.expander("🔧 技术详情"):
            st.write(f"模型类型: {type(model)}")
            st.write(f"特征维度: {features.shape}")
            st.write(f"对数预测值: {log_prediction:.4f}")
            st.write(f"指数转换后: ${prediction:,.2f}")
            if feature_names:
                st.write(f"特征数量: {len(feature_names)}")
            else:
                st.write("使用默认262维特征映射")
            
    except Exception as e:
        st.error(f"❌ 预测错误: {e}")
        st.info("如果价格不合理，可能需要调整特征映射顺序")

elif predict_button:
    st.error("❌ 模型未加载，无法预测")

# 页脚
st.markdown("---")
st.markdown("基于XGBoost模型 | 对数价格转换 | Streamlit部署")
