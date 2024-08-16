import pandas as pd

df = pd.read_excel("user_event_data.xlsx")
df.head()

# Calculate the total number of users in each group
total_users_per_group = df.groupby("group")["user_id"].nunique()

# Calculate the number of users who reached the payment screen in each group
users_reached_payment_per_group = (
    df[df["event"] == "user_reached_payment_screen"]
    .groupby("group")["user_id"]
    .nunique()
)

# Calculate the number of users who subscribed in each group
subscribed_users_per_group = (
    df[df["event"].str.contains("subscribed")].groupby("group")["user_id"].nunique()
)

# Calculate the total revenue per group
total_revenue_per_group = df.groupby("group")["revenue"].sum()

# Calculate the conversion rate (users who subscribed after reaching the payment screen)
conversion_rate_per_group = (
    subscribed_users_per_group / users_reached_payment_per_group
) * 100

# Calculate Revenue Per User (RPU)
rpu_per_group = total_revenue_per_group / total_users_per_group

# Compile the results into a DataFrame for comparison
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

metrics_df

print(metrics_df)
