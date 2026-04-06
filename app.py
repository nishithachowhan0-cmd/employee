import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
st.set_page_config(page_title="Employee Attrition Detector", page_icon="👔", layout="wide")
if "attrition_history" not in st.session_state:
    st.session_state.attrition_history = []
st.title("👔 Employee Attrition Detector")
st.markdown("Predict whether an employee is likely to **leave (1)** or **stay (0)**.")
# ─── DATASET ───
data = {
    "EmployeeID":         [f"E{str(i).zfill(3)}" for i in range(1, 51)],
    "Gender":             (["Male","Female"] * 25),
    "Age":                [25,32,45,28,55,38,22,48,35,None,
                           41,29,52,36,24,50,33,27,46,39,
                           26,43,37,30,56,34,23,49,40,None,
                           42,28,53,35,25,51,32,26,47,38,
                           27,44,36,29,57,33,22,50,41,None],
    "Department":         (["HR","Tech","Sales","Finance","Ops"] * 10),
    "SalaryLakh":         [5,12,8,15,None,10,6,18,9,7,
                           5.5,11,8.5,14,None,10.5,6.5,17,9.5,7.5,
                           5,12,8,15,None,10,6,18,9,7,
                           5.5,11,8.5,14,None,10.5,6.5,17,9.5,7.5,
                           5,12,8,15,None,10,6,18,9,7],
    "YearsAtCompany":     [1,5,10,2,15,7,1,12,4,None,
                           2,6,11,3,14,8,1,13,5,None,
                           1,5,10,2,15,7,1,12,4,None,
                           2,6,11,3,14,8,1,13,5,None,
                           1,5,10,2,15,7,1,12,4,None],
    "SatisfactionScore":  [3,8,7,2,9,6,2,8,5,6,
                           3,7,8,2,9,6,2,9,5,6,
                           3,8,7,2,9,6,2,8,5,6,
                           3,7,8,2,9,6,2,9,5,6,
                           3,8,7,2,9,6,2,8,5,6],
    "WorkloadScore":      [8,4,5,9,3,6,9,4,7,5,
                           8,5,4,9,3,6,9,3,7,5,
                           8,4,5,9,3,6,9,4,7,5,
                           8,5,4,9,3,6,9,3,7,5,
                           8,4,5,9,3,6,9,4,7,5],
    "CommuteKm":          [30,10,20,50,5,15,45,8,25,None,
                           32,12,18,48,6,14,43,9,24,None,
                           30,10,20,50,5,15,45,8,25,None,
                           32,12,18,48,6,14,43,9,24,None,
                           30,10,20,50,5,15,45,8,25,None],
    "PromotionsLast3Yrs": [0,1,2,0,3,1,0,2,1,0,
                           0,1,2,0,3,1,0,2,1,0,
                           0,1,2,0,3,1,0,2,1,0,
                           0,1,2,0,3,1,0,2,1,0,
                           0,1,2,0,3,1,0,2,1,0],
    "TrainingHrsPerYear": [10,30,25,5,40,20,8,35,15,18,
                           12,28,24,6,38,22,9,33,16,19,
                           10,30,25,5,40,20,8,35,15,18,
                           12,28,24,6,38,22,9,33,16,19,
                           10,30,25,5,40,20,8,35,15,18],
    "Attrition":          [1,0,0,1,0,0,1,0,0,1,
                           1,0,0,1,0,0,1,0,0,1,
                           1,0,0,1,0,0,1,0,0,1,
                           1,0,0,1,0,0,1,0,0,1,
                           1,0,0,1,0,0,1,0,0,1],
}
df_raw = pd.DataFrame(data)
st.subheader("📋 Employee Dataset")
st.dataframe(df_raw)
col1, col2, col3 = st.columns(3)
col1.metric("Total Employees",  len(df_raw))
col2.metric("Left Company",     df_raw["Attrition"].sum())
col3.metric("Attrition Rate",   f"{df_raw['Attrition'].mean()*100:.1f}%")
# ─── PREPROCESSING ───
st.subheader("🔧 Data Preprocessing")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1️⃣ Null Check","2️⃣ Fill Nulls","3️⃣ Dedup & Drop",
    "4️⃣ Encoding","5️⃣ Salary Validation","6️⃣ Final Data"
])
with tab1:
    null_info = df_raw.isnull().sum().reset_index()
    null_info.columns = ["Column","Nulls"]
    null_info["% Missing"] = (null_info["Nulls"] / len(df_raw) * 100).round(1)
    st.dataframe(null_info)
    fig_null, ax_null = plt.subplots()
    cols_nn = null_info[null_info["Nulls"] > 0]
    ax_null.bar(cols_nn["Column"], cols_nn["% Missing"], color="#e74c3c")
    ax_null.set_title("Missing Values per Column")
    ax_null.set_xticklabels(cols_nn["Column"], rotation=30, ha="right")
    st.pyplot(fig_null)
with tab2:
    df = df_raw.copy()
    df["Age"]            = df["Age"].fillna(df["Age"].median())
    df["SalaryLakh"]     = df["SalaryLakh"].fillna(df["SalaryLakh"].median())
    df["YearsAtCompany"] = df["YearsAtCompany"].fillna(df["YearsAtCompany"].median())
    df["CommuteKm"]      = df["CommuteKm"].fillna(df["CommuteKm"].median())
    st.success("✅ Nulls filled with median!")
    st.dataframe(df.isnull().sum().rename("Remaining"))
with tab3:
    df = df.drop_duplicates().drop(["EmployeeID"], axis=1)
    st.success(f"Shape: **{df.shape}**")
with tab4:
    df["Gender"]     = df["Gender"].map({"Male": 0, "Female": 1})
    df["Department"] = df["Department"].map({"HR":0,"Tech":1,"Sales":2,"Finance":3,"Ops":4})
    st.write("Gender: Male=0, Female=1 | Department: HR=0, Tech=1, Sales=2, Finance=3, Ops=4")
    st.dataframe(df.head(8))
with tab5:
    valid = ((df["SalaryLakh"] > 0) & (df["SalaryLakh"] < 200)).all()
    st.write(f"{'✅' if valid else '❌'} All salary values in valid range (0–200 Lakh)")
    fig_sal, ax_sal = plt.subplots()
    ax_sal.hist(df["SalaryLakh"], bins=8, color="#2ecc71", edgecolor="white")
    ax_sal.set_title("Salary Distribution (Lakh)")
    st.pyplot(fig_sal)
with tab6:
    st.dataframe(df)
    st.info(f"Final shape: **{df.shape}**")
# ─── FEATURE ENGINEERING ───
st.subheader("⚙️ Feature Engineering")
df["SalaryGrowthProxy"]  = df["SalaryLakh"] / (df["YearsAtCompany"] + 1)
df["WorkLifeBalance"]    = df["SatisfactionScore"] / (df["WorkloadScore"] + 1)
df["CareerStagnation"]   = df["YearsAtCompany"] / (df["PromotionsLast3Yrs"] + 1)
df["CommuteToSalary"]    = df["CommuteKm"] / (df["SalaryLakh"] + 1)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Work-Life Balance",  f"{df['WorkLifeBalance'].mean():.2f}")
col2.metric("Avg Career Stagnation",  f"{df['CareerStagnation'].mean():.1f}")
col3.metric("Avg Salary Growth",      f"{df['SalaryGrowthProxy'].mean():.2f} L/yr")
col4.metric("Long Commuters (>30km)", f"{(df['CommuteKm'] > 30).sum()}")
# ─── VISUALIZATION ───
st.subheader("📊 Visualizations")
col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    for val, color, label in [(0,"#2ecc71","Stayed"),(1,"#e74c3c","Left")]:
        s = df[df["Attrition"] == val]
        ax1.scatter(s["SatisfactionScore"], s["WorkloadScore"], color=color, label=label, alpha=0.8)
    ax1.set_xlabel("Satisfaction Score"); ax1.set_ylabel("Workload Score")
    ax1.set_title("Satisfaction vs Workload"); ax1.legend()
    st.pyplot(fig1)
with col2:
    fig2, ax2 = plt.subplots()
    for val, color, label in [(0,"#2ecc71","Stayed"),(1,"#e74c3c","Left")]:
        s = df[df["Attrition"] == val]
        ax2.scatter(s["YearsAtCompany"], s["SalaryLakh"], color=color, label=label, alpha=0.8)
    ax2.set_xlabel("Years at Company"); ax2.set_ylabel("Salary (Lakh)")
    ax2.set_title("Tenure vs Salary"); ax2.legend()
    st.pyplot(fig2)
col1, col2 = st.columns(2)
with col1:
    dept_attr = df.groupby("Department")["Attrition"].mean()
    fig3, ax3 = plt.subplots()
    ax3.bar(["HR","Tech","Sales","Finance","Ops"][:len(dept_attr)], dept_attr.values, color="#e74c3c")
    ax3.set_title("Department vs Attrition Rate"); ax3.set_ylabel("Attrition Rate")
    st.pyplot(fig3)
with col2:
    fig4, ax4 = plt.subplots()
    ax4.scatter(df["CommuteKm"], df["SatisfactionScore"], c=df["Attrition"], cmap="RdYlGn_r", alpha=0.8)
    ax4.set_xlabel("Commute (km)"); ax4.set_ylabel("Satisfaction")
    ax4.set_title("Commute vs Satisfaction (Red=Left)")
    st.pyplot(fig4)
st.subheader("🔗 Correlation Matrix")
st.dataframe(df.corr(numeric_only=True))
# ─── OUTLIER ───
st.subheader("📦 Outlier Detection")
fig5, ax5 = plt.subplots()
ax5.boxplot([df["SalaryLakh"], df["CommuteKm"]/5, df["YearsAtCompany"]],
            labels=["Salary (L)","Commute÷5","Years at Co."])
ax5.set_title("Outlier Check"); st.pyplot(fig5)
# ─── MODEL ───
st.subheader("🤖 Gradient Boosting Classifier")
X = df.drop("Attrition", axis=1)
y = df["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
st.success("✅ Model Trained!")
# ─── FEATURE IMPORTANCE ───
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
fig_fi, ax_fi = plt.subplots()
ax_fi.barh(importances.index, importances.values, color="#e74c3c")
ax_fi.set_title("Feature Importance (XGBoost-style)"); st.pyplot(fig_fi)
# ─── EVALUATION ───
st.subheader("📈 Evaluation")
y_pred  = model.predict(X_test)
y_prob  = model.predict_proba(X_test)[:, 1]
acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
cm      = confusion_matrix(y_test, y_pred)
col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{acc*100:.1f}%")
col2.metric("ROC-AUC",  f"{roc_auc:.2f}")
fig6, ax6 = plt.subplots()
ax6.imshow(cm, cmap="Reds")
ax6.set_xticks([0,1]); ax6.set_yticks([0,1])
ax6.set_xticklabels(["Stayed","Left"]); ax6.set_yticklabels(["Stayed","Left"])
for i in range(2):
    for j in range(2):
        ax6.text(j, i, cm[i,j], ha="center", va="center", fontsize=14)
ax6.set_title("Confusion Matrix"); ax6.set_xlabel("Predicted"); ax6.set_ylabel("Actual")
st.pyplot(fig6)
st.text(classification_report(y_test, y_pred, target_names=["Stayed","Left"]))
# ─── PREDICTION ───
st.subheader("🎯 Check Employee Flight Risk")
col1, col2, col3 = st.columns(3)
with col1:
    gender      = st.selectbox("Gender", ["Male(0)","Female(1)"])
    age         = st.slider("Age", 18, 65, 32)
    dept        = st.selectbox("Department", ["HR(0)","Tech(1)","Sales(2)","Finance(3)","Ops(4)"])
with col2:
    salary      = st.slider("Salary (Lakh/yr)", 2.0, 50.0, 10.0)
    tenure      = st.slider("Years at Company", 0, 30, 5)
    satisfaction= st.slider("Satisfaction Score (1-10)", 1, 10, 6)
with col3:
    workload    = st.slider("Workload Score (1-10)", 1, 10, 5)
    commute     = st.slider("Commute (km)", 0, 100, 20)
    promotions  = st.slider("Promotions (last 3 yrs)", 0, 5, 1)
    training    = st.slider("Training Hrs/Year", 0, 60, 20)
if st.button("👔 Predict Attrition Risk"):
    g_val    = int(gender.split("(")[1].replace(")",""))
    d_val    = int(dept.split("(")[1].replace(")",""))
    sal_grow = salary / (tenure + 1)
    wlb      = satisfaction / (workload + 1)
    stag     = tenure / (promotions + 1)
    comm_sal = commute / (salary + 1)
    input_df = pd.DataFrame([[g_val, age, d_val, salary, tenure, satisfaction,
                               workload, commute, promotions, training,
                               sal_grow, wlb, stag, comm_sal]],
                            columns=X.columns)
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    st.info(f"📊 Work-Life Balance: **{wlb:.2f}** | Career Stagnation: **{stag:.1f}** | Commute/Salary: **{comm_sal:.2f}**")
    if pred == 1:
        st.error(f"🚨 **HIGH FLIGHT RISK** — Probability: {prob*100:.1f}% — Retention action needed!")
        st.warning("💡 Consider salary revision, workload reduction, or promotion opportunity.")
    else:
        st.success(f"✅ **LIKELY TO STAY** — Flight Risk: {prob*100:.1f}%")
    st.session_state.attrition_history.append({
        "Age": age, "Dept": dept.split("(")[0], "Salary (L)": salary,
        "Tenure (yrs)": tenure, "Satisfaction": satisfaction, "Workload": workload,
        "Result": "🚨 Flight Risk" if pred==1 else "✅ Staying",
        "Probability": f"{prob*100:.1f}%"
    })
if st.session_state.attrition_history:
    st.subheader("🕓 Assessment History")
    st.dataframe(pd.DataFrame(st.session_state.attrition_history))
    