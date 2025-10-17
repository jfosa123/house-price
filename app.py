import streamlit as st
import pickle
import numpy as np
import pandas as pd
# 添加xgboost导入
try:
    import xgboost as xgb
except ImportError as e:
    st.error(f"XGBoost导入失败: {e}")

st.set_page_config(page_title="房价预测模型", page_icon="🏠", layout="wide")

# 应用标题
st.title('🏠 Kaggle房价预测模型')
st.markdown('基于Kaggle「House Prices: Advanced Regression Techniques」竞赛特征的房价预测系统')

# 加载模型 - 改进的错误处理
@st.cache_resource
def load_model():
    try:
        with open('house_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
        st.success("✅ 模型加载成功！")
        return model
    except Exception as e:
        st.error(f"❌ 加载模型失败: {e}")
        st.info("""
        **可能的解决方案:**
        1. 确保 requirements.txt 中包含 xgboost
        2. 检查模型文件是否完整
        3. 确认模型文件路径正确
        """)
        return None

model = load_model()

# 如果模型加载失败，显示安装指导
if model is None:
    st.warning("""
    **🔧 修复步骤:**
    
    请确保你的 `requirements.txt` 文件包含以下内容:
    ```
    streamlit>=1.28.0
    scikit-learn>=1.3.0
    numpy>=1.24.0
    pandas>=2.0.0
    xgboost>=1.7.0
    ```
    
    然后重新部署到 Streamlit。
    """)
st.set_page_config(page_title="房价预测模型", page_icon="🏠", layout="wide")

# 应用标题
st.title('🏠 Kaggle房价预测模型')
st.markdown('基于Kaggle「House Prices: Advanced Regression Techniques」竞赛特征的房价预测系统')

# 加载模型
@st.cache_resource
def load_model():
    try:
        with open('house_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        return None

model = load_model()

# 创建两列布局
col1, col2 = st.columns([1, 1])

with col1:
    st.header('📊 房屋基本信息')
    
    # 1. 总体信息
    st.subheader('总体信息')
    overall_qual = st.slider('整体材料和质量评分 (1-10)', 1, 10, 5)
    overall_cond = st.slider('整体状况评分 (1-10)', 1, 10, 5)
    year_built = st.slider('建造年份', 1870, 2010, 2000)
    year_remod_add = st.slider('改建年份', 1950, 2010, 2000)
    
    # 2. 面积信息
    st.subheader('面积信息')
    gr_liv_area = st.number_input('地上居住面积 (平方英尺)', min_value=500, max_value=5000, value=1500)
    total_bsmt_sf = st.number_input('地下室总面积 (平方英尺)', min_value=0, max_value=3000, value=1000)
    first_fl_sf = st.number_input('一层面积 (平方英尺)', min_value=500, max_value=3000, value=1000)
    second_fl_sf = st.number_input('二层面积 (平方英尺)', min_value=0, max_value=2000, value=500)
    
    # 3. 车库信息
    st.subheader('车库信息')
    garage_cars = st.slider('车库容量 (车辆数)', 0, 4, 2)
    garage_area = st.number_input('车库面积 (平方英尺)', min_value=0, max_value=1500, value=500)

with col2:
    st.header('🏠 房间配置')
    
    # 4. 房间信息
    st.subheader('房间信息')
    tot_rms_abv_grd = st.slider('地上总房间数', 2, 15, 6)
    bedrooms = st.slider('卧室数量', 1, 8, 3)
    full_bath = st.slider('全卫数量', 0, 4, 2)
    half_bath = st.slider('半卫数量', 0, 3, 1)
    kitchen = st.slider('厨房数量', 1, 3, 1)
    kitchen_qual = st.slider('厨房质量 (1-5)', 1, 5, 3)
    
    # 5. 其他特征
    st.subheader('其他特征')
    fireplaces = st.slider('壁炉数量', 0, 3, 1)
    wood_deck_sf = st.number_input('木制平台面积', min_value=0, max_value=500, value=0)
    open_porch_sf = st.number_input('开放式门廊面积', min_value=0, max_value=500, value=50)
    enclosed_porch = st.number_input('封闭式门廊面积', min_value=0, max_value=500, value=0)
    
    # 6. 地理位置
    st.subheader('地理位置')
    neighborhood = st.selectbox('社区', [
        'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes',
        'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR',
        'MeadowV', 'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr', 'NPkVill',
        'Blmngtn', 'BrDale', 'SWISU', 'Blueste'
    ])
    
    ms_zoning = st.selectbox('分区分类', [
        'A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'
    ])

# 预测按钮 - 放在显著位置
st.markdown("---")
predict_button = st.button('🚀 预测房价', type='primary', use_container_width=True)

if predict_button and model is not None:
    try:
        # 准备输入数据 - 根据Kaggle竞赛特征顺序
        # 注意：这里需要根据你的模型实际训练时使用的特征进行调整
        
        # 数值特征
        input_features = np.array([[
            overall_qual, overall_cond, year_built, year_remod_add,
            gr_liv_area, total_bsmt_sf, first_fl_sf, second_fl_sf,
            garage_cars, garage_area, tot_rms_abv_grd, bedrooms,
            full_bath, half_bath, kitchen, kitchen_qual,
            fireplaces, wood_deck_sf, open_porch_sf, enclosed_porch
            # 注意：分类特征需要进行编码，这里简化处理
        ]])
        
        # 进行预测
        prediction = model.predict(input_features)[0]
        
        # 显示结果
        st.success('## 📈 预测结果')
        
        # 创建三个指标卡
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="预测房价", value=f"${prediction:,.2f}")
        
        with col2:
            st.metric(label="预测房价 (万元)", value=f"¥{prediction * 0.0007:,.1f}万")
        
        with col3:
            st.metric(label="房屋等级", value="豪华" if prediction > 300000 else "中等" if prediction > 150000 else "经济")
        
        # 显示详细的输入信息
        with st.expander("📋 查看详细的房屋信息"):
            st.write("**房屋基本信息:**")
            st.write(f"- 整体质量评分: {overall_qual}/10")
            st.write(f"- 建造年份: {year_built}")
            st.write(f"- 地上居住面积: {gr_liv_area} 平方英尺")
            st.write(f"- 地下室面积: {total_bsmt_sf} 平方英尺")
            
            st.write("**房间配置:**")
            st.write(f"- 总房间数: {tot_rms_abv_grd}")
            st.write(f"- 卧室: {bedrooms} | 全卫: {full_bath} | 半卫: {half_bath}")
            st.write(f"- 厨房质量: {kitchen_qual}/5")
            
            st.write("**其他特征:**")
            st.write(f"- 车库容量: {garage_cars} 辆车")
            st.write(f"- 社区: {neighborhood}")
            st.write(f"- 分区: {ms_zoning}")
            
    except Exception as e:
        st.error(f"预测过程中出现错误: {e}")
        st.info("""
        **可能的原因:**
        1. 输入特征与模型训练时的特征不匹配
        2. 特征顺序或数量不一致
        3. 模型文件损坏
        
        **解决方案:**
        请检查模型训练时使用的具体特征列表
        """)

elif predict_button and model is None:
    st.error("❌ 模型加载失败，无法进行预测。请检查模型文件是否存在且格式正确。")

# 使用说明
with st.expander("ℹ️ 使用说明"):
    st.markdown("""
    **Kaggle房价预测特征说明:**
    
    - **OverallQual**: 整体材料和装修质量 (1-10)
    - **GrLivArea**: 地上居住面积
    - **GarageCars**: 车库容量
    - **TotalBsmtSF**: 地下室总面积
    - **YearBuilt**: 建造年份
    - **Neighborhood**: 社区物理位置
    
    **操作步骤:**
    1. 在左侧填写房屋基本信息
    2. 在右侧配置房间和其他特征
    3. 点击下方「预测房价」按钮
    4. 查看预测结果和详细信息
    """)

# 页脚
st.markdown("---")
st.markdown("基于Kaggle房价预测竞赛 | 使用Streamlit部署")
