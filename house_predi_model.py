import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv(r"C:\Users\ARUL RAJ HARESH\OneDrive\Desktop\yashma\housing_price_modified.csv")



st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")
st.markdown("""
<style>
h1 { font-weight: 800; color: #111; }
h2 { font-weight: 700; }
p  { font-size: 18px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📈Navigation")
menu = st.sidebar.radio(
    "Go to",
    [
        "📘Introduction",
        "📊EDA Visualization",
        "🧮Prediction"
    ]
)
st.title("🏠Real Estate Investment Advisor")
st.subheader("Property Profitability & Future Price Prediction")
if menu == "📘Introduction":
    st.header("Project Overview")
    st.markdown("""
- **Classified** **properties as good investment opportunities**
- **Predicted estimated property prices after 5 years**
- **Built interactive **EDA visualizations** for insights**
- **Supported intelligent decision-making for buyers, sellers, and investors**
""")

    st.header("Skills Used")
    st.write("""Python, EDA, Machine Learning, ***Streamlit***, Mlflow""")

if menu == "📊EDA Visualization":
    st.header("Exploratory Data Analysis")

    question = st.selectbox(
        "Select EDA question",
        [
           "Q1  Distribution of property prices",
           "Q2  Distribution of property sizes",
           "Q3  price per sq ft vary by property type",
           "Q4  Relationship between property size and price",
           "Q5  Outliers in price per sq ft or property size",
           "Q6  Average price per sq ft by state",
           "Q7  Average property price by city",
           "Q8  The median age of properties by locality",
           "Q9  BHK distributed across cities",
           "Q10 price trends for the top 5 most expensive localities",
           "Q11 Numeric features correlated with each other",
           "Q12 How do nearby schools relate to price per sq ft",
           "Q13 How do nearby hospitals relate to price per sq ft",
           "Q14 How does price vary by furnished status",
           "Q15 How does price per sq ft vary by property facing direction",
           "Q16 properties belong to each owner type",
           "Q17 properties are available under each availability status",
           "Q18 Does parking space affect property price",
           "Q19 Amenities affect price per sq ft",
           "Q20 How does public transport accessibility relate to price per sq ft or investment potential"

        ]  
    )  
    if question =="Q1  Distribution of property prices":
    

     plt.figure(figsize=(5,3))
     plt.hist(df['Price_in_Lakhs'], bins=50)
     plt.xlabel('Price (in Lakhs)')
     plt.ylabel('Number of Properties')
     plt.title('Distribution of Property Prices')
     st.pyplot(plt)
        

    elif question =="Q2  Distribution of property sizes":
      plt.figure(figsize=(8,5))
      plt.hist(df['Size_in_SqFt'], bins=30)
      plt.xlabel('Property sizes ')
      plt.ylabel('Number of Properties')
      plt.title('Distribution of Property prices')
      st.pyplot(plt)

    elif question =="Q3  price per sq ft vary by property type":
     df.groupby('Property_Type')['Price_per_SqFt'].mean()
     plt.figure(figsize=(8,5))
     df.groupby('Property_Type')['Price_per_SqFt'].mean().plot(kind='bar')
     plt.xlabel('Property Type')
     plt.ylabel('Average Price per SqFt')
     plt.title('Price per SqFt by Property Type')
     st.pyplot(plt)

    elif question =="Q4  Relationship between property size and price":
     plt.scatter(df['Size_in_SqFt'], df['Price_in_Lakhs'],alpha=0.05)
     plt.title('Property Size vs Price')
     plt.xlabel('Size (SqFt)')
     plt.ylabel('Price (Lakhs)')
     st.pyplot(plt)

    elif question =="Q5  Outliers in price per sq ft or property size":
       plt.figure(figsize=(6,4))
       sns.boxenplot(y=df['Size_in_SqFt'])
       plt.title("Outliers in Property Size")
       st.pyplot(plt)

    elif question =="Q6  Average price per sq ft by state":
       state_avg_psf = df.groupby('State')['Price_per_SqFt'].mean().sort_values(ascending=False)

       plt.figure(figsize=(10,5))
       plt.bar(state_avg_psf.index, state_avg_psf.values)
       plt.xticks(rotation=45)
       plt.xlabel("State")
       plt.ylabel("Average Price per Sq Ft")
       plt.title("Average Price per Sq Ft by State")
       st.pyplot(plt)

    elif question =="Q7  Average property price by city": 
        city_avg_price = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False)

        city_avg_price.head(10).plot(kind='bar')
        plt.title("Average Property Price by City")
        plt.ylabel("Price (Lakhs)")
        plt.xticks(rotation=45)
        st.pyplot(plt)

    elif question =="Q8  The median age of properties by locality":
        locality_median_age = df.groupby('Locality')['Age_of_Property'].median()

        locality_median_age.head(10).plot(kind='bar')
        plt.title("Median Age of Properties by Locality")
        plt.ylabel("Age (Years)")
        plt.xticks(rotation=45)
        st.pyplot(plt)
        
    elif question == "Q9  BHK distributed across cities":

        bhk_dist = df.groupby(['City', 'BHK']).size().unstack(fill_value=0)
        bhk_dist.head(10).plot(kind='bar', stacked=True)
        plt.title("BHK Distribution Across Cities")
        plt.ylabel("Number of Properties")
        plt.xticks(rotation=45)
        st.pyplot(plt)

    elif question =="Q10 price trends for the top 5 most expensive localities":

        top5 = (
        df.groupby('Locality')['Price_per_SqFt']
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .index
        )

        trend = df[df['Locality'].isin(top5)] \
        .groupby(['Year_Built', 'Locality'])['Price_per_SqFt'] \
        .mean().reset_index()

        sns.lineplot(data=trend, x='Year_Built', y='Price_per_SqFt', hue='Locality')
        plt.title("Price Trends of Top 5 Expensive Localities")
        st.pyplot(plt)

    elif question =="Q11 Numeric features correlated with each other":
        numeric_cols = ['BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt',
                'Year_Built', 'Floor_No', 'Total_Floors', 'Age_of_Property',
                'Nearby_Schools', 'Nearby_Hospitals']

        corr_matrix = df[numeric_cols].corr()
        plt.figure(figsize=(12,10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Between Numeric Features")
        st.pyplot(plt)
                

    elif question =="Q12 How do nearby schools relate to price per sq ft":

        plt.figure(figsize=(8,5))
        sns.scatterplot(x='Nearby_Schools', y='Price_per_SqFt', data=df)
        plt.title("Nearby Schools vs Price per SqFt")
        plt.xlabel("Number of Nearby Schools")
        plt.ylabel("Price per SqFt (₹)")
        st.pyplot(plt)

    elif question =="Q13 How do nearby hospitals relate to price per sq ft":

        plt.figure(figsize=(8,5))
        sns.scatterplot(x='Nearby_Hospitals', y='Price_per_SqFt', data=df)
        plt.title("Nearby Hospitals vs Price per SqFt")
        plt.xlabel("Number of Nearby Hospitals")
        plt.ylabel("Price per SqFt (₹)")
        st.pyplot(plt)

    elif question =="Q14 How does price vary by furnished status":

        avg_price_furnish = df.groupby('Furnished_Status')['Price_in_Lakhs'].mean()
        avg_price_furnish.plot(kind='bar', color='lightblue', figsize=(6,4))
        plt.title("Average Price by Furnished Status")
        plt.xlabel("Furnished Status")
        plt.ylabel("Average Price (Lakhs)")
        st.pyplot(plt)

    elif question =="Q15 How does price per sq ft vary by property facing direction":
        facing_direction = df.groupby('Facing')['Price_in_Lakhs'].mean()
        facing_direction.plot(kind='bar', color='red', figsize=(6,4))
        plt.title("Average Price by facing_direction")
        plt.xlabel("Facing_direction")
        plt.ylabel("Average Price (Lakhs)")
        st.pyplot(plt)

    elif question =="Q16 properties belong to each owner type":

        owner_property_counts = pd.crosstab(df['Owner_Type'], df['Property_Type'])

        owner_property_counts.plot(kind='bar', figsize=(8,5))
        plt.title("Number of Properties by Owner Type and Property Type")
        plt.xlabel("Owner Type")
        plt.ylabel("Number of Properties")
        plt.xticks(rotation=0)
        plt.legend(title="Property Type")
        st.pyplot(plt)

    elif question =="Q17 properties are available under each availability status":

        properties_available = pd.crosstab(df['Property_Type'], df['Availability_Status'])
        properties_available.plot(kind='bar', figsize=(8,5))
        plt.title("Available_properties")
        plt.xlabel("Property_Type")
        plt.ylabel("Availability_Status")
        plt.xticks(rotation=0)
        plt.legend()
        st.pyplot(plt)

    elif question =="Q18 Does parking space affect property price":

        plt.figure(figsize=(7,5))
        sns.boxplot(x='Parking_Space', y='Price_in_Lakhs', data=df)
        plt.title("Effect of Parking Space on Property Price")
        plt.xlabel("Parking Space")
        plt.ylabel("Price (Lakhs)")
        st.pyplot(plt)


    elif question =="Q19 Amenities affect price per sq ft":

        def amenity_level(count):
            if count <= 3:
                return 'Low (0-3)'
            elif count <= 6:
                return 'Medium (4-6)'
            else:
                return 'High (7+)'


        df['Amenity_Count'] = df['Amenities'].apply(lambda x: len(x.split(',')) if isinstance(x, str) and x.strip() else 0)

        df['Amenity_Level'] = df['Amenity_Count'].apply(amenity_level)
        amenity_level_price = df.groupby('Amenity_Level')['Price_per_SqFt'].mean()

        amenity_level_price.plot(kind='bar', figsize=(6,4))
        plt.title("Average Price per SqFt by Amenity Level")
        plt.xlabel("Amenity Level")
        plt.ylabel("Average Price per SqFt")
        plt.xticks(rotation=0)
        st.pyplot(plt)

    elif question =="Q20 How does public transport accessibility relate to price per sq ft or investment potential":
        
        avg_price = df.groupby('Public_Transport_Accessibility')['Price_per_SqFt'].mean()

        avg_price.plot(kind='bar')
        plt.title("Average Price per Sq Ft by Public Transport Accessibility")
        plt.xlabel("Public Transport Accessibility")
        plt.ylabel("Average Price per Sq Ft")
        plt.xticks(rotation=0)
        st.pyplot(plt)


import streamlit as st
import pandas as pd
import pickle
if menu == "🧮Prediction":

    with open("knn_pipeline.pkl", "rb") as f:
        clf_pipeline = pickle.load(f)

    with open("rf_regression_pipeline.pkl", "rb") as f:
        reg_pipeline = pickle.load(f)


    st.title("Property Investment Prediction")

    st.write("Enter property details 👇")
    city_list = sorted(df['City'].dropna().unique())
    city = st.selectbox(
        "Select City",
        city_list
    )

    property= sorted(df['Property_Type'].dropna().unique())
    property_type= st.selectbox(
        "Select property_type",
        property
    )
    bhk_type= sorted(df['BHK'].dropna().unique())
    bhk= st.selectbox(
        "Select BHK",
        bhk_type
    )
    price_min = int(df['Price_in_Lakhs'].min())
    price_max = int(df['Price_in_Lakhs'].max())

    price = st.slider(
    "💰 Price (Lakhs)",
    min_value=price_min,
    max_value=price_max,
    value=(price_min + price_max) // 2
)

    size=st.slider("📐 Size (SqFt)", 300, 5000, 10000)

    avg_age = df['Age_of_Property'].mean()
    age = avg_age
    floor_no = st.number_input("Floor No", min_value=0)
    total_floors = int(df['Total_Floors'].median())

    schools = st.number_input("Nearby Schools", min_value=0)
    hosp_min = int(df['Nearby_Hospitals'].min())
    hosp_max = int(df['Nearby_Hospitals'].max())

    hospitals = st.slider(
        "🏥 Nearby Hospitals",
        hosp_min,
        hosp_max,
        2
    )


    user_df = pd.DataFrame([{
        "City": city,
        "Property_Type": property_type,
        "BHK": bhk,
        "Price_in_Lakhs": price,
        "Size_in_SqFt": size,
        "Age_of_Property": age,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals
    }])

    if st.button("🔮 Predict"):

        invest_pred = clf_pipeline.predict(user_df)[0]
        invest_label = "✅ Good Investment" if invest_pred == 1 else "❌ Bad Investment"

        future_price = reg_pipeline.predict(user_df)[0]

        st.subheader("📊 Results")
        st.success(invest_label)
        st.info(f"💰 Predicted Price after 5 years: **₹ {future_price:.2f} Lakhs**")






















    