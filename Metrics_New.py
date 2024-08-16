import pandas as pd
from scipy import stats
import numpy as np

# Загрузка данных
df = pd.read_excel("user_event_data.xlsx")
df.head()

# Расчеты
total_users_per_group = df.groupby("group")["user_id"].nunique()

users_reached_payment_per_group = (
    df[df["event"] == "user_reached_payment_screen"]
    .groupby("group")["user_id"]
    .nunique()
)

subscribed_users_per_group = (
    df[df["event"].str.contains("subscribed")]
    .groupby("group")["user_id"]
    .nunique()
)

total_revenue_per_group = df.groupby("group")["revenue"].sum()

# Рассчитываем конверсии
conversion_rate_per_group = (
    subscribed_users_per_group / users_reached_payment_per_group
) * 100

rpu_per_group = total_revenue_per_group / total_users_per_group

metrics_df = pd.DataFrame(
    {
        "Total Users": total_users_per_group,
        "Users Reached Payment": users_reached_payment_per_group,
        "Subscribed Users": subscribed_users_per_group,
        "Total Revenue": total_revenue_per_group,
        "Conversion Rate (%)": conversion_rate_per_group,
        "Revenue Per User (RPU)": rpu_per_group,
    }
)

print(metrics_df)

# Проверка нормальности распределения для доходов
testshapiroA = stats.shapiro(df[df['group'] == 'A']['revenue'].dropna())
testshapiroB = stats.shapiro(df[df['group'] == 'B']['revenue'].dropna())

print("Shapiro Test A p-value:", testshapiroA.pvalue)
print("Shapiro Test B p-value:", testshapiroB.pvalue)

# Проверка равенства дисперсий
levene_test = stats.levene(df[df['group'] == 'A']['revenue'].dropna(), df[df['group'] == 'B']['revenue'].dropna())
print("Levene Test p-value:", levene_test.pvalue)

# Выбор типа t-теста
if testshapiroA.pvalue > 0.05 and testshapiroB.pvalue > 0.05:
    if levene_test.pvalue > 0.05:
        print("T-Test with equal variances")
        t_test = stats.ttest_ind(df[df['group'] == 'A']['revenue'].dropna(), df[df['group'] == 'B']['revenue'].dropna(), equal_var=True)
    else:
        print("T-Test with unequal variances")
        t_test = stats.ttest_ind(df[df['group'] == 'A']['revenue'].dropna(), df[df['group'] == 'B']['revenue'].dropna(), equal_var=False)
else:
    print("Use Mann-Whitney U test")
    mannwhitney = stats.mannwhitneyu(df[df['group'] == 'A']['revenue'].dropna(), df[df['group'] == 'B']['revenue'].dropna())
    print("Mann-Whitney U Test Result p-value:", mannwhitney.pvalue)

print("Result for revenue:", t_test.pvalue if 't_test' in locals() else mannwhitney.pvalue)

# Вычисление и сравнение конверсий
conversion_A = df[df['group'] == 'A'].groupby('user_id')['event'].apply(lambda x: 'subscribed' in x).astype(int)
conversion_B = df[df['group'] == 'B'].groupby('user_id')['event'].apply(lambda x: 'subscribed' in x).astype(int)

mannwhitney_conversion = stats.mannwhitneyu(conversion_A, conversion_B)
print("Mann-Whitney U Test Result for Conversions:", mannwhitney_conversion.pvalue)
