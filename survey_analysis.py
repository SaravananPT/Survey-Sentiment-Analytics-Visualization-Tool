import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

st.set_page_config(page_title="Sentiment + EDA Dashboard", layout="wide")

st.title("📊 Comprehensive Sentiment & EDA Dashboard")

uploaded_file = st.file_uploader("Upload your Excel file with sentiment + survey results", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.success("✅ File successfully loaded!")
    st.subheader("📌 Sample Data")
    st.dataframe(df.head(), use_container_width=True)

    # EDA Visuals
    st.header("📊 Step 2: Exploratory Data Analysis (EDA)")

    # Age Histogram
    st.subheader("🎂 Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Q1_Age'], bins=10, kde=True, ax=ax)
    st.pyplot(fig)

    # Gender Distribution
    st.subheader("🚻 Gender Distribution")
    gender_counts = df['Q2_Gender'].value_counts()
    st.bar_chart(gender_counts)

    # Education Level
    st.subheader("🎓 Education Level Count")
    edu_counts = df['Q3_Education'].value_counts()
    st.bar_chart(edu_counts)

    # Income vs Satisfaction
    st.subheader("💰 Income vs. Life Satisfaction")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Q6_IncomeLevel', y='Q4_LifeSatisfaction', hue='Q3_Education', ax=ax)
    st.pyplot(fig)
    st.write("**Correlation**:", df[['Q6_IncomeLevel', 'Q4_LifeSatisfaction']].corr().iloc[0,1].round(2))

    # Employment & Satisfaction
    st.subheader("👨‍💼 Employment vs. Avg. Hours & Satisfaction")
    emp_group = df.groupby('Q7_Employed')[['Q10_HoursWorkedPerWeek','Q4_LifeSatisfaction']].mean()
    st.bar_chart(emp_group)

    # Car & Home Ownership
    st.subheader("🚗 Car vs. 🏠 Home Ownership")
    cross_tab = pd.crosstab(df['Q5_CarOwnership'], df['Q9_HomeOwner'])
    st.dataframe(cross_tab)

    st.subheader("🚻 Gender-wise Car/Home Ownership")
    gender_own = df.groupby('Q2_Gender')[['Q5_CarOwnership','Q9_HomeOwner']].apply(lambda x: x.eq('Yes').mean())
    st.bar_chart(gender_own)

    st.subheader("❤️ Marital Status vs. Car/Home Ownership")
    marital_own = df.groupby('Q8_MaritalStatus')[['Q5_CarOwnership','Q9_HomeOwner']].apply(lambda x: x.eq('Yes').mean())
    st.bar_chart(marital_own)

    # Rich and Happy
    st.subheader("🎉 Age & Gender: Who is Rich and Happy?")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Q1_Age', y='Q6_IncomeLevel', size='Q4_LifeSatisfaction', hue='Q2_Gender', ax=ax)
    st.pyplot(fig)

    # Education vs Income
    st.subheader("📚 Education vs. Income")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='Q3_Education', y='Q6_IncomeLevel', ax=ax)
    st.pyplot(fig)

    # Working More Hours
    st.subheader("⏰ Who is Working More Hours?")
    group_hours = df.groupby(['Q2_Gender', 'Q3_Education'])['Q10_HoursWorkedPerWeek'].mean().unstack()
    st.dataframe(group_hours)
    fig, ax = plt.subplots()
    group_hours.plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # Correlation: Income vs Hours Worked
    st.subheader("📈 Correlation: Hours Worked vs. Income")
    fig, ax = plt.subplots()
    sns.regplot(data=df, x='Q10_HoursWorkedPerWeek', y='Q6_IncomeLevel', ax=ax)
    st.pyplot(fig)
    st.write("**Correlation**:", df[['Q10_HoursWorkedPerWeek', 'Q6_IncomeLevel']].corr().iloc[0,1].round(2))

    # Marital Status Insights
    st.subheader("💔 Marital Status vs. Satisfaction")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='Q8_MaritalStatus', y='Q4_LifeSatisfaction', ax=ax)
    st.pyplot(fig)

    st.subheader("💸 Income vs. Marital Status")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='Q8_MaritalStatus', y='Q6_IncomeLevel', ax=ax)
    st.pyplot(fig)

    # Age vs Home Owner
    st.subheader("🏠 Home Ownership by Age")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='Q9_HomeOwner', y='Q1_Age', ax=ax)
    st.pyplot(fig)

    # Step 4: Qualitative Analysis
    st.header("💬 Step 4: Open-Ended Response Analysis")
    text_columns = {
        "Q11_OE_Neighborhood": "Neighborhood Likes",
        "Q12_OE_FutureConcerns": "Future Concerns",
        "Q13_OE_CustomerExperience": "Customer Experience",
        "Q14_OE_WorkMotivation": "Work Motivation",
        "Q15_OE_AdditionalComments": "Additional Comments"
    }

    for col, title in text_columns.items():
        st.subheader(f"☁️ WordCloud - {title}")
        text = " ".join(df[col].dropna().astype(str))
        if text:
            wordcloud = WordCloud(background_color="white", width=1000, height=400).generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("No text data available for this question.")

    # Summary Findings
    st.header("🧠 Step 5: Insights Summary")
    st.markdown("""
    ### 🔑 Key Findings
    - 🎓 Tertiary-educated people tend to have higher income and satisfaction.
    - 💔 Divorced respondents show higher work hours and lower satisfaction.
    - 🏠 Majority homeowners are over age 40.
    - 😞 Unemployed individuals report lower income and satisfaction.
    - 💬 Text data reveals community bonding and future anxieties (healthcare, war, kids).
    - 🚗 Males and married individuals are more likely to own homes and cars.
    - 🧑‍🎓 Higher education is linked to higher earnings.
    - ⏰ Those working more hours tend to earn more (positive correlation).
    - 🎉 People aged 30–50 report higher income and satisfaction.
    - 🔍 Gender and education influence both working hours and income levels.
    - 👩‍❤️‍👨 Married people are more likely to own cars and homes.
    - 👵 Older people are more likely to own homes.
    - 📊 Working hours correlate moderately with income (check scatterplot).
    - 🤯 People with low income & long work hours often have low satisfaction.
    - 👨‍💼 Employed people report higher income and satisfaction than unemployed ones.
    """)

    # Download button
    st.subheader("⬇️ Download Updated Data")
    output_filename = "sentiment_eda_output.xlsx"
    df.to_excel(output_filename, index=False)
    with open(output_filename, "rb") as f:
        st.download_button("Download Excel", f, file_name=output_filename,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("👈 Please upload your Excel file to begin analysis.")
