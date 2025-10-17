import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="房价预测模型", page_icon="🏠", layout="wide")

# 应用标题
st.title('🏠 Kaggle房价预测模型')
st.markdown('基于XGBoost的房价预测系统 (262个特征)')

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

# 创建262维特征向量的函数
def create_262_features(user_input):
    """
    基于Kaggle房价预测竞赛的标准特征创建262维特征向量
    """
    features = np.zeros(262)
    
    # 设置主要数值特征（基于Kaggle竞赛的常见特征顺序）
    # 数值特征
    features[0] = user_input['MSSubClass']          # 房屋类型
    features[1] = user_input['MSZoning']            # 分区分类
    features[2] = user_input['LotFrontage']         # 临街距离
    features[3] = user_input['LotArea']             # 地块面积
    features[4] = user_input['Street']              # 街道类型
    features[5] = user_input['Alley']               # 小巷通道
    features[6] = user_input['LotShape']            # 地块形状
    features[7] = user_input['LandContour']         # 平整度
    features[8] = user_input['Utilities']           # 公用设施
    features[9] = user_input['LotConfig']           # 地块配置
    
    features[10] = user_input['LandSlope']          # 坡度
    features[11] = user_input['Neighborhood']       # 社区
    features[12] = user_input['Condition1']         # 邻近条件1
    features[13] = user_input['Condition2']         # 邻近条件2
    features[14] = user_input['BldgType']           # 建筑类型
    features[15] = user_input['HouseStyle']         # 房屋风格
    features[16] = user_input['OverallQual']        # 整体质量 ★重要
    features[17] = user_input['OverallCond']        # 整体状况 ★重要
    features[18] = user_input['YearBuilt']          # 建造年份 ★重要
    features[19] = user_input['YearRemodAdd']       # 改建年份
    
    features[20] = user_input['RoofStyle']          # 屋顶风格
    features[21] = user_input['RoofMatl']           # 屋顶材料
    features[22] = user_input['Exterior1st']        # 外部覆盖1
    features[23] = user_input['Exterior2nd']        # 外部覆盖2
    features[24] = user_input['MasVnrType']         # 砖石类型
    features[25] = user_input['MasVnrArea']         # 砖石面积
    features[26] = user_input['ExterQual']          # 外部质量
    features[27] = user_input['ExterCond']          # 外部状况
    features[28] = user_input['Foundation']         # 地基类型
    features[29] = user_input['BsmtQual']           # 地下室高度
    
    features[30] = user_input['BsmtCond']           # 地下室状况
    features[31] = user_input['BsmtExposure']       # 地下室暴露
    features[32] = user_input['BsmtFinType1']       # 地下室完成类型1
    features[33] = user_input['BsmtFinSF1']         # 地下室完成面积1
    features[34] = user_input['BsmtFinType2']       # 地下室完成类型2
    features[35] = user_input['BsmtFinSF2']         # 地下室完成面积2
    features[36] = user_input['BsmtUnfSF']          # 地下室未完成面积
    features[37] = user_input['TotalBsmtSF']        # 地下室总面积 ★重要
    features[38] = user_input['Heating']            # 供暖类型
    features[39] = user_input['HeatingQC']          # 供暖质量
    
    features[40] = user_input['CentralAir']         # 中央空调
    features[41] = user_input['Electrical']         # 电气系统
    features[42] = user_input['1stFlrSF']           # 一层面积 ★重要
    features[43] = user_input['2ndFlrSF']           # 二层面积
    features[44] = user_input['LowQualFinSF']       # 低质量完成面积
    features[45] = user_input['GrLivArea']          # 地上居住面积 ★重要
    features[46] = user_input['BsmtFullBath']       # 地下室全卫
    features[47] = user_input['BsmtHalfBath']       # 地下室半卫
    features[48] = user_input['FullBath']           # 全卫数量 ★重要
    features[49] = user_input['HalfBath']           # 半卫数量
    
    features[50] = user_input['BedroomAbvGr']       # 地上卧室数 ★重要
    features[51] = user_input['KitchenAbvGr']       # 地上厨房数
    features[52] = user_input['KitchenQual']        # 厨房质量
    features[53] = user_input['TotRmsAbvGrd']       # 地上总房间数 ★重要
    features[54] = user_input['Functional']         # 功能评级
    features[55] = user_input['Fireplaces']         # 壁炉数量
    features[56] = user_input['FireplaceQu']        # 壁炉质量
    features[57] = user_input['GarageType']         # 车库类型
    features[58] = user_input['GarageYrBlt']        # 车库建造年份
    features[59] = user_input['GarageFinish']       # 车库完成情况
    
    features[60] = user_input['GarageCars']         # 车库容量 ★重要
    features[61] = user_input['GarageArea']         # 车库面积 ★重要
    features[62] = user_input['GarageQual']         # 车库质量
    features[63] = user_input['GarageCond']         # 车库状况
    features[64] = user_input['PavedDrive']         # 车道铺设
    features[65] = user_input['WoodDeckSF']         # 木平台面积
    features[66] = user_input['OpenPorchSF']        # 开放式门廊面积
    features[67] = user_input['EnclosedPorch']      # 封闭式门廊面积
    features[68] = user_input['3SsnPorch']          # 三季门廊面积
    features[69] = user_input['ScreenPorch']        # 纱窗门廊面积
    
    features[70] = user_input['PoolArea']           # 泳池面积
    features[71] = user_input['PoolQC']             # 泳池质量
    features[72] = user_input['Fence']              # 围栏质量
    features[73] = user_input['MiscFeature']        # 其他特征
    features[74] = user_input['MiscVal']            # 其他特征价值
    features[75] = user_input['MoSold']             # 销售月份
    features[76] = user_input['YrSold']             # 销售年份
    features[77] = user_input['SaleType']           # 销售类型
    features[78] = user_input['SaleCondition']      # 销售条件
    
    # 设置合理的默认值给剩余的特征 (79-261)
    features[79:150] = 0    # 各种分类变量的one-hot编码默认值
    features[150:200] = 1   # 各种存在性标志默认值
    features[200:262] = 0.5 # 各种质量评分的默认值
    
    return features.reshape(1, -1)

# 创建用户输入界面
st.header('📊 请输入房屋信息')

tab1, tab2, tab3 = st.tabs(["🏠 基本信息", "🛏️ 房间配置", "🚗 外部设施"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("房屋质量与年份")
        overall_qual = st.slider('整体质量评分 (OverallQual)', 1, 10, 6, 
                               help='整体材料和装修质量，10为最佳')
        overall_cond = st.slider('整体状况评分 (OverallCond)', 1, 10, 5)
        year_built = st.slider('建造年份 (YearBuilt)', 1900, 2020, 2000)
        year_remod_add = st.slider('改建年份 (YearRemodAdd)', 1950, 2020, 2000)
    
    with col2:
        st.subheader("面积信息")
        gr_liv_area = st.number_input('地上居住面积 (GrLivArea)', value=1500, 
                                    min_value=500, max_value=10000,
                                    help='地上总居住面积')
        total_bsmt_sf = st.number_input('地下室总面积 (TotalBsmtSF)', value=1000, 
                                      min_value=0, max_value=5000)
        first_fl_sf = st.number_input('一层面积 (1stFlrSF)', value=1000, 
                                    min_value=500, max_value=5000)
        lot_area = st.number_input('地块面积 (LotArea)', value=10000, 
                                 min_value=1000, max_value=50000)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("房间信息")
        tot_rms_abv_grd = st.slider('地上总房间数 (TotRmsAbvGrd)', 2, 15, 6)
        bedrooms = st.slider('卧室数量 (BedroomAbvGr)', 1, 8, 3)
        full_bath = st.slider('全卫数量 (FullBath)', 0, 4, 2)
        half_bath = st.slider('半卫数量 (HalfBath)', 0, 3, 1)
    
    with col2:
        st.subheader("厨房与壁炉")
        kitchen_qual = st.slider('厨房质量 (KitchenQual)', 1, 5, 3)
        fireplaces = st.slider('壁炉数量 (Fireplaces)', 0, 3, 1)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("车库信息")
        garage_cars = st.slider('车库容量 (GarageCars)', 0, 4, 2,
                              help='可容纳车辆数量')
        garage_area = st.number_input('车库面积 (GarageArea)', value=500, 
                                    min_value=0, max_value=1500)
    
    with col2:
        st.subheader("外部设施")
        wood_deck_sf = st.number_input('木制平台面积 (WoodDeckSF)', value=0, 
                                     min_value=0, max_value=500)
        open_porch_sf = st.number_input('开放式门廊面积 (OpenPorchSF)', value=50, 
                                      min_value=0, max_value=500)

# 固定特征值（基于常见默认值）
default_features = {
    'MSSubClass': 60,      # 最常见的房屋类型
    'MSZoning': 3,         # 住宅低密度
    'LotFrontage': 70,     # 平均临街距离
    'Street': 1,           # 铺砌街道
    'Alley': 0,            # 无小巷通道
    'LotShape': 3,         # 规则形状
    'LandContour': 3,      # 平坦
    'Utilities': 2,        # 所有公用设施
    'LotConfig': 3,        # 内部地块
    'LandSlope': 1,        # 平缓坡度
    'Neighborhood': 15,    # 常见社区
    'Condition1': 2,       # 正常
    'Condition2': 2,       # 正常
    'BldgType': 0,         # 独立式
    'HouseStyle': 4,       # 两层
    'RoofStyle': 3,        # 山墙
    'RoofMatl': 6,         # 标准瓦片
    'Exterior1st': 12,     # 乙烯基侧板
    'Exterior2nd': 12,     # 乙烯基侧板
    'MasVnrType': 3,       # 砖石贴面
    'MasVnrArea': 100,     # 砖石面积
    'ExterQual': 3,        # 典型
    'ExterCond': 3,        # 典型
    'Foundation': 2,       # 混凝土板
    'BsmtQual': 3,         # 典型高度
    'BsmtCond': 3,         # 典型状况
    'BsmtExposure': 2,     # 平均暴露
    'BsmtFinType1': 5,     # 未完成
    'BsmtFinSF1': 500,     # 地下室完成面积1
    'BsmtFinType2': 5,     # 未完成
    'BsmtFinSF2': 0,       # 地下室完成面积2
    'BsmtUnfSF': 500,      # 地下室未完成面积
    'Heating': 1,          # 燃气暖气
    'HeatingQC': 3,        # 典型质量
    'CentralAir': 1,       # 有中央空调
    'Electrical': 4,       # 标准电路
    '2ndFlrSF': 500,       # 二层面积
    'LowQualFinSF': 0,     # 低质量完成面积
    'BsmtFullBath': 0,     # 地下室全卫
    'BsmtHalfBath': 0,     # 地下室半卫
    'KitchenAbvGr': 1,     # 地上厨房数
    'Functional': 7,       # 典型功能
    'FireplaceQu': 3,      # 典型壁炉质量
    'GarageType': 5,       # 内置车库
    'GarageYrBlt': 2000,   # 车库建造年份
    'GarageFinish': 2,     # 粗糙完成
    'GarageQual': 3,       # 典型车库质量
    'GarageCond': 3,       # 典型车库状况
    'PavedDrive': 2,       # 铺砌车道
    'EnclosedPorch': 0,    # 封闭式门廊面积
    '3SsnPorch': 0,        # 三季门廊面积
    'ScreenPorch': 0,      # 纱窗门廊面积
    'PoolArea': 0,         # 泳池面积
    'PoolQC': 0,           # 泳池质量
    'Fence': 0,            # 无围栏
    'MiscFeature': 0,      # 无其他特征
    'MiscVal': 0,          # 其他特征价值
    'MoSold': 6,           # 六月销售
    'YrSold': 2020,        # 销售年份
    'SaleType': 8,         # 担保交易
    'SaleCondition': 4     # 正常销售
}

# 预测按钮
st.markdown("---")
predict_button = st.button('🚀 预测房价', type='primary', use_container_width=True)

if predict_button and model is not None:
    try:
        # 收集用户输入
        user_input = default_features.copy()
        user_input.update({
            'OverallQual': overall_qual,
            'OverallCond': overall_cond,
            'YearBuilt': year_built,
            'YearRemodAdd': year_remod_add,
            'GrLivArea': gr_liv_area,
            'TotalBsmtSF': total_bsmt_sf,
            '1stFlrSF': first_fl_sf,
            'LotArea': lot_area,
            'TotRmsAbvGrd': tot_rms_abv_grd,
            'BedroomAbvGr': bedrooms,
            'FullBath': full_bath,
            'HalfBath': half_bath,
            'KitchenQual': kitchen_qual,
            'Fireplaces': fireplaces,
            'GarageCars': garage_cars,
            'GarageArea': garage_area,
            'WoodDeckSF': wood_deck_sf,
            'OpenPorchSF': open_porch_sf
        })
        
        # 创建特征向量
        features = create_262_features(user_input)
        
        # 进行预测
        prediction = model.predict(features)[0]
        
        # 显示结果
        st.success('## 📈 预测结果')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="预测房价", value=f"${prediction:,.2f}")
        
        with col2:
            st.metric(label="人民币", value=f"¥{prediction * 7.2:,.0f}")
        
        with col3:
            if prediction > 500000:
                level = "豪华"
            elif prediction > 250000:
                level = "中等偏上"
            elif prediction > 150000:
                level = "中等"
            else:
                level = "经济"
            st.metric(label="房屋等级", value=level)
        
        st.info(f"🔧 使用的特征维度: {features.shape}")
        
    except Exception as e:
        st.error(f"❌ 预测错误: {e}")
        st.info("如果价格不合理，可能需要调整特征映射顺序")

elif predict_button:
    st.error("❌ 模型未加载，无法预测")

# 页脚
st.markdown("---")
st.markdown("基于XGBoost模型 | 262维特征 | Streamlit部署")
