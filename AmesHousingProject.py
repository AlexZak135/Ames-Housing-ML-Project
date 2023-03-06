#%%

# These packages are used

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns
import plotnine as pn
import matplotlib.pyplot as plt

import janitor as jan
import movecolumn as mc

from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from statsmodels.stats.outliers_influence import variance_inflation_factor

#%%

##### Part A - Data Cleaning ##### 

#%%

# Take the following steps: 
  # 1. Create a list
  # 2. Import the data
  # 3. Consolidate the column names
  # 4. Rename a specific column
  # 5. Create new columns
  # 6. Modifying existing columns
  # 7. Drop columns
  
cols1_drop = ["pid", "ms_subclass", "year_built", "year_remod_add",
              "fireplaces", "garage_yr_blt", "mo_sold", "yr_sold"]  

data1_all = (
    jan.clean_names(pd.read_csv("AmesHousingData.csv"))
       .rename(columns = {"order" : "id"})
       .assign(age = lambda x: x["yr_sold"] - x["year_built"])
       .assign(age = lambda x: np.where(x["age"] < 0, 0, x["age"]),
               overall_cond = lambda x: 
                   np.where(x["overall_cond"] < 5, "below",
                   np.where(x["overall_cond"] == 5, "average",                  
                                                      "above")),
               overall_qual = lambda x:
                   np.where(x["overall_qual"] < 5, "below",
                   np.where(x["overall_qual"] == 5, "average", 
                                                      "above")),
               kitch_qual = lambda x:
                   np.where((x["kitchen_qual"] == "Po") | 
                            (x["kitchen_qual"] == "Fa"), "below",
                   np.where(x["kitchen_qual"] == "TA", "average", 
                                                         "above")),
               fplace = lambda x: np.where(x["fireplaces"] > 0, "yes", "no"))
       .drop(columns = cols1_drop))

#%%
    
# Take the following steps:
  # 1. Select columns with certain data types
  # 2. Change the data type of a specific column
  # 3. Replace missing values with 0
  
data1_num = (data1_all
             .select_dtypes(include = np.number)
             .astype({"id" : str})
             .fillna(0))

# Take the following steps:
  # 1. Calculate the Pearson correlation coefficients
  # 2. Sort by the selected column
  # 3. Filter appropriately
  # 4. Select the necessary column
  # 5. Get the index of the column as a list
  
corr1 = (data1_num.corr(numeric_only = True)
         .sort_values(by = "price", ascending = False)
         .query("price > 0.45 | price < -0.45")
         .price
         .index
         .tolist())

# Define the name of the extra column 

cols1_extra = "id"

# Append the column

if cols1_extra not in corr1:
    corr1.append(cols1_extra)

# Select the columns that are in the list

data1_num = data1_num[corr1]

#%%

# Take the following steps:
  # 1. Create a list
  # 2. Change the data type of a specific column
  # 3. Select columns with certain data types
  # 4. Select the necessary columns
  
cols1_keep = ["id", "overall_cond", "overall_qual", "kitch_qual", "fplace"]  
  
data1_str = (data1_all.astype({"id" : str})
             .select_dtypes(include = object)
             [cols1_keep])
       
#%%

# Take the following steps:
  # 1. Create a list
  # 2. Merge the data frames
  # 3. Rename certain columns
  # 4. Drop a specific column
  # 5. Reorder the columns

cols1_order = ["price", "abv_grd_sf", "garage_cars", "garage_sf", 
               "bsmt_sf","x1st_flr_sf", "full_bath", "mas_area", 
               "totrms_abvgrd", "age", "kitch_qual", "fplace", 
               "overall_cond", "overall_qual"]
   
data2 = (data1_num.merge(data1_str, on = "id", how = "left")
         .rename(columns = {"area" : "abv_grd_sf",
                            "garage_area" : "garage_sf",
                            "total_bsmt_sf" : "bsmt_sf",
                            "mas_vnr_area" : "mas_area"})
         .drop(columns = "id")
         [cols1_order])
              
#%%

##### Part B - Exploratory Data Analysis ##### 

#%%

# Select columns with certain data types

data2_num = data2.select_dtypes(include = np.number)

# Take the following steps:
  # 1. Create a dictionary
  # 2. Generate the descriptive statistics
  # 3. Rename index
  
index1_rename = {"count" : "Count", 
                 "mean" : "Mean", 
                 "std" : "Standard Deviation", 
                 "min" : "Minimum", 
                 "max" : "Maximum"}

ds1 = (data2_num
       .agg(["count", "mean", "std", "min", "max"])
       .rename(index = index1_rename))

# Create a new row for the median

ds1.loc["Median"] = data2_num.median()

# Create a new row for the mode

ds1.loc["Mode"] = data2_num.mode().iloc[0]

# Take the following steps:
  # 1. Create a dictionary
  # 2. Rename columns
  # 3. Transpose the data frame 
  # 4. Modify the position of columns

cols1_rename = {"price" : "Price", "abv_grd_sf" : "Abv. Grd. SF",
                "garage_cars" : "Garage Doors", "garage_sf" : "Garage SF",
                "bsmt_sf" : "Basement SF",  "x1st_flr_sf" : "1st Flr. SF",
                "full_bath" : "Bathrooms", "mas_area" : "Masonry SF",
                "totrms_abvgrd" : "Rooms Abv. Grd.", "age" : "Age"} 

ds1 = (ds1
       .rename(columns = cols1_rename)
       .T
       .pipe(mc.MoveToN, "Median", 3)
       .pipe(mc.MoveToN, "Mode", 4))

#%%

# Take the following steps:
  # 1. Select columns with certain data types
  # 2. Rename all the columns
  # 3. Calculate the Pearson correlation coefficients  
  
corr2 = (data2.select_dtypes(include = np.number)
         .rename(columns = cols1_rename)
         .corr())
   
# Create a correlation matrix displaying the Pearson correlation coefficients 

plt.figure(figsize = (8, 6))

color = sns.diverging_palette(10, 240, as_cmap = True)

cmat = sns.heatmap(corr2, vmin = -1, vmax = 1,
                   annot = True, cmap = color,
                   linewidths = 0.5, linecolor = "black", 
                   xticklabels = True, yticklabels = True)

cmat.tick_params(bottom = False, left = False)

cmat.set_title("Correlation Matrix", fontdict = {"fontsize" : 18}, pad = 12)

# Export as a png file

plt.savefig("viz1.png", dpi = 300, bbox_inches = "tight")

# Take the following steps:
  # 1. Modifying existing column
  # 2. Select the necessary columns
  # 3. Calculate Spearman's rank correlation coefficient
  
corr3 = (data2
         .assign(kitch_qual = 
                 np.where(data2["kitch_qual"] == "below", 1,
                 np.where(data2["kitch_qual"] == "average", 2, 
                                                            3)))
         [["price", "kitch_qual"]])

corrtest3 = stats.spearmanr(corr3["kitch_qual"], corr3["price"])

# Take the following steps:
  # 1. Modifying existing column
  # 2. Select the necessary columns
  # 3. Calculate the point biserial correlation coefficient 
  
corr4 = (data2
         .assign(fplace = np.where(data2["fplace"] == "yes", 1, 0))
         [["price", "fplace"]])

corrtest4 = stats.pointbiserialr(corr4["fplace"], corr4["price"])
             
# Take the following steps:
  # 1. Modifying existing column
  # 2. Select the necessary columns
  # 3. Calculate Spearman's rank correlation coefficient

corr5 = (data2
         .assign(overall_cond = 
                 np.where(data2["overall_cond"] == "below", 1,
                 np.where(data2["overall_cond"] == "average", 2, 
                                                              3))) 
         [["price", "overall_cond"]])

corrtest5 = stats.spearmanr(corr5["overall_cond"], corr5["price"])

# Take the following steps:
  # 1. Modifying existing column
  # 2. Select the necessary columns
  # 3. Calculate Spearman's rank correlation coefficient

corr6 = (data2
         .assign(overall_qual = 
                 np.where(data2["overall_qual"] == "below", 1,
                 np.where(data2["overall_qual"] == "average", 2, 
                                                              3))) 
         [["price", "overall_qual"]])

corrtest6 = stats.spearmanr(corr6["overall_qual"], corr6["price"])

#%%

# Create a histogram displaying the distribution

viz2 = (pn.ggplot(data = data2, mapping = pn.aes(x = "price")) + 
        pn.geom_histogram(fill = "#005288", color = "black", bins = 25) + 
        pn.scale_x_continuous( 
        labels = lambda x: ["${:,.0f}".format(i) for i in x]) + 
        pn.labs(title = "Distribution of Price",
                x = "Price", y = "Frequency") + 
        pn.theme_bw())

# Export as a png file

viz2.save("viz2.png", dpi = 300)

# Create a new variable

data2["log_price"] = np.log(data2["price"])

# Create a histogram displaying the distribution

viz3 = (pn.ggplot(data = data2, mapping = pn.aes(x = "log_price")) + 
        pn.geom_histogram(fill = "#c0c2c4", color = "black", bins = 25) + 
        pn.labs(title = "Distribution of the Log of Price", 
                x = "Log of Price", y = "Frequency") + 
        pn.theme_bw())

# Export as a png file

viz3.save("viz3.png", dpi = 300)

# Create a scatter plot displaying the relationship between the variables

viz4 = (pn.ggplot(data = data2, 
                  mapping = pn.aes(x = "abv_grd_sf", y = "price")) + 
        pn.geom_point(color = "#005288", size = 0.6) + 
        pn.geom_smooth(method = "lm", se = False, size = 0.7) + 
        pn.scale_x_continuous(labels = 
                              lambda x: ["{:,.0f}".format(i) for i in x]) + 
        pn.scale_y_continuous(labels = 
                              lambda x: ["${:,.0f}".format(i) for i in x]) + 
        pn.labs(title = "Price vs. Above Ground Sq. Ft.", 
                x = "Above Ground Sq. Ft.", y = "Price") + 
        pn.theme_bw())

# Export as a png file

viz4.save("viz4.png", dpi = 300)

# Filter appropriately to produce a new data frame

data3 = data2.query("abv_grd_sf < 4000")

# Create a scatter plot displaying the relationship between the variables

viz5 = (pn.ggplot(data = data3, 
                  mapping = pn.aes(x = "abv_grd_sf", y = "log_price")) + 
        pn.geom_point(color = "#0078ae", size = 0.6) + 
        pn.geom_smooth(method = "lm", se = False, size = 0.7) + 
        pn.scale_x_continuous(labels = 
                              lambda x: ["{:,.0f}".format(i) for i in x]) + 
        pn.labs(title = "Log of Price vs. Above Ground Sq. Ft.", 
                x = "Above Ground Sq. Ft.", y = " Log of Price") + 
        pn.theme_bw())

# Export as a png file

viz5.save("viz5.png", dpi = 300)
  
# Create a scatter plot displaying the relationship between the variables

viz6 = (pn.ggplot(data = data2, mapping = pn.aes(x = "age", y = "price")) + 
        pn.geom_point(color = "#005288", size = 0.6) + 
        pn.geom_smooth(method = "lm", se = False, 
                       color = "black", size = 0.7) + 
        pn.scale_y_continuous(labels = 
                              lambda x: ["${:,.0f}".format(i) for i in x]) + 
        pn.labs(title = "Price vs. Age", x = "Age", y = "Price") + 
        pn.theme_bw())

# Export the visualization as a png file

viz6.save("viz6.png", dpi = 300)

# Create a scatter plot displaying the relationship between the variables

viz7 = (pn.ggplot(data = data3, 
                  mapping = pn.aes(x = "age", y = "log_price")) + 
        pn.geom_point(color = "#0078ae", size = 0.6) + 
        pn.geom_smooth(method = "lm", formula = "y ~ x + I(x**2)", 
                       se = False, color = "black", size = 0.7) + 
        pn.labs(title = "Log of Price vs. Age", 
                x = "Age", y = "Log of Price") + 
        pn.theme_bw())
  
# Export the visualization as a png file

viz7.save("viz7.png", dpi = 300)

# Create a new variable

data3["age2"] = np.power(data2["age"], 2)

# Create density plots displaying the distributions

viz8 = (pn.ggplot(data = data3, 
                  mapping = pn.aes(x = "price", fill = "fplace")) +
        pn.geom_density(alpha = 0.7) +
        pn.scale_fill_manual(labels = ["No Fireplace", "Fireplace"],
                             values = ["#c0c2c4", "#005288"],
                             guide = pn.guide_legend(reverse = True)) +
        pn.scale_x_continuous(labels = 
                              lambda x: ["${:,.0f}".format(i) for i in x]) + 
        pn.labs(title = "Distribution of Price by Fireplace", 
                x = "Price", y = "Density") + 
        pn.guides(fill = pn.guide_legend(title = "")) +
        pn.theme_bw() +
        pn.theme(legend_position=(.50, -.01)))

# Export the visualization as a png file

viz8.save("viz8.png", dpi = 300)

#%%

##### Part C - Statistical Modeling #####       

#%%

# Take the following steps:
  # 1. Convert categorical variables to dummy variables
  # 2. Drop columns 
  # 3. Modify the position of the columns

data3 = (data3
         .join(pd.get_dummies(data3[["kitch_qual", "fplace", 
                                     "overall_cond", "overall_qual"]],
                              prefix = ["kitch_qual", "fplace", 
                                        "overall_cond", "overall_qual"]))
         .drop(columns = ["kitch_qual", "fplace", 
                          "overall_cond", "overall_qual"])
         .pipe(mc.MoveToN, "log_price", 2)
         .pipe(mc.MoveToN, "age2", 12))

#%%

### Polynomial Regression

# Take the following steps:
  # 1. Create a list
  # 2. Create a data frame with the necessary independent variables
  # 3. Create a series with just the dependent variable
  # 4. Use hold-out validation to split data into training and testing sets
  # 5. Save function as an object and fit the model to the training data
  # 6. Generate regression output

cols2_keep = ["abv_grd_sf", "garage_sf", "bsmt_sf", "age", "age2", 
              "kitch_qual_below", "kitch_qual_average", "fplace_yes",
              "overall_cond_below", "overall_cond_average",
              "overall_qual_below", "overall_qual_average"]
              
iv1 = data3[cols2_keep]

dv1 = data3["log_price"]

iv1_train, iv1_test, dv1_train, dv1_test = train_test_split(
    iv1, dv1, test_size = 0.3, shuffle = True, random_state = 100)

lm1 = LinearRegression()

lm1.fit(iv1_train, dv1_train)

lm1_output = sm.OLS(dv1_train, sm.add_constant(iv1_train)).fit().summary()

# Take the following steps:
  # 1. Make predictions on the training data
  # 2. Generate the model's residuals
  # 3. Create a data frame with the fitted values and residuals
  # 4. Produce residuals vs. fitted plot
  # 5. Produce a histogram of the residuals
  # 6. Use custom function to get VIFs

lm1_train_pred = lm1.predict(iv1_train)

lm1_train_resid = dv1_train - lm1_train_pred 

rfplot1_df = pd.DataFrame({"fitted" : lm1_train_pred, 
                           "resid": lm1_train_resid})

viz9 = (pn.ggplot(data = rfplot1_df, 
                  mapping = pn.aes(x = "fitted", y = "resid")) +
        pn.geom_point(color = "#005288", size = 0.6) +
        pn.geom_hline(yintercept = 0, color = "grey", linetype = "dashed") +
        pn.geom_smooth(method = "lowess", color = "black") +
        pn.labs(title = "Residuals vs. Fitted Plot for Model 1",
                x = "Fitted Values", y = "Residuals") +
        pn.theme_bw())

# Export the visualization as a png file

viz9.save("viz9.png", dpi = 300)

viz10 = (pn.ggplot(data = rfplot1_df, mapping = pn.aes(x = "resid")) + 
         pn.geom_histogram(fill = "#c0c2c4", color = "black", bins = 45) + 
         pn.labs(title = "Distribution of Residuals for Model 1", 
                 x = "Residuals", y = "Frequency") + 
         pn.theme_bw())

# Export as a png file

viz10.save("viz10.png", dpi = 300)

def compute_vifs(data):
    iv_with_intercept = sm.add_constant(data)
    vifs = [variance_inflation_factor(iv_with_intercept.values, i) 
            for i in range(iv_with_intercept.shape[1])]
    vif_series = pd.Series(vifs[1:], index = data.columns)
    return vif_series

compute_vifs(iv1_train)

# Take the following steps:
  # 1. Get r-squared
  # 2. Make predictions on the test data
  # 3. Get the RMSE

lm1.score(iv1_test, dv1_test)

lm1_test_pred = lm1.predict(iv1_test)

rmse1 = np.sqrt(mean_squared_error(dv1_test, lm1_test_pred))

#%%

### Lasso Regression Without Standardization

# Take the following steps:
  # 1. Create a list
  # 2. Create a data frame with the necessary independent variables
  # 3. Create a series with just the dependent variable
  # 4. Use hold-out validation to split data into training and testing sets
  # 5. Use 5-fold cross-validation to find optimal hyperparameter  
  # 6. Save function as an object and fit the model to the training data
  # 7. Get the r-squared

cols3_keep = ["abv_grd_sf", "garage_cars", "garage_sf", "bsmt_sf",
              "x1st_flr_sf", "full_bath", "mas_area", "totrms_abvgrd",
              "age", "age2", "kitch_qual_below", "kitch_qual_average",
              "fplace_yes", "overall_cond_below", "overall_cond_average",
              "overall_qual_below", "overall_qual_average"]

iv2 = data3[cols3_keep]

dv2 = data3["log_price"]

iv2_train, iv2_test, dv2_train, dv2_test = train_test_split(
    iv2, dv2, test_size = 0.3, shuffle = True, random_state = 100)

lasso1_cv = LassoCV(cv = 5, random_state = 100)

lasso1_cv.fit(iv2_train, dv2_train) 

lasso1 = Lasso(alpha = lasso1_cv.alpha_)

lasso1.fit(iv2_train, dv2_train)

lasso1.score(iv2_train, dv2_train)

# Take the following steps:
  # 1. Get r-squared
  # 2. Make predictions on the test data
  # 3. Get the RMSE

lasso1.score(iv2_test, dv2_test)

lasso1_pred = lasso1.predict(iv2_test)

rmse2 = np.sqrt(mean_squared_error(dv2_test, lasso1_pred))

#%%

### Lasso Regression With Standardization

# Take the following steps:
  # 1. Create a data frame with the necessary independent variables
  # 2. Create a series with just the dependent variable
  # 3. Use hold-out validation to split data into training and testing sets
  # 4. Standardize the necessary variables
  # 5. Use 5-fold cross-validation to find optimal hyperparameter  
  # 6. Save function as an object and fit the model to the training data
  # 7. Get the r-squared

iv3 = data3[cols3_keep]

dv3 = data3["log_price"]

iv3_train, iv3_test, dv3_train, dv3_test = train_test_split(
    iv3, dv3, test_size = 0.3, shuffle = True, random_state = 100)

def standardize_df(df):
    df_stand = df.copy()
    num_cols = (df_stand.select_dtypes(include = np.number)
                        .select_dtypes(exclude = "uint8").columns)
    scaler = StandardScaler()
    scaler.fit(df_stand[num_cols])
    df_stand[num_cols] = scaler.transform(df_stand[num_cols])
    return df_stand

iv3_train = standardize_df(iv3_train)

iv3_test = standardize_df(iv3_test)

lasso2_cv = LassoCV(cv = 5, random_state = 100)

lasso2_cv.fit(iv3_train, dv3_train) 

lasso2 = Lasso(alpha = lasso2_cv.alpha_)

lasso2.fit(iv3_train, dv3_train)

lasso2.score(iv3_train, dv3_train)

# Take the following steps:
  # 1. Get r-squared
  # 2. Make predictions on the test data
  # 3. Get the RMSE

lasso2.score(iv3_test, dv3_test)

lasso2_pred = lasso2.predict(iv3_test)

rmse3 = np.sqrt(mean_squared_error(dv3_test, lasso2_pred))

#%%

### Ridge Regression Without Standardization

# Take the following steps:
  # 1. Create a data frame with the necessary independent variables
  # 2. Create a series with just the dependent variable
  # 3. Use hold-out validation to split data into training and testing sets
  # 4. Use 5-fold cross-validation to find optimal hyperparameter  
  # 5. Save function as an object and fit the model to the training data
  # 6. Get the r-squared

iv4 = data3[cols3_keep]

dv4 = data3["log_price"]

iv4_train, iv4_test, dv4_train, dv4_test = train_test_split(
    iv4, dv4, test_size = 0.3, shuffle = True, random_state = 100)

ridge1_cv = RidgeCV(alphas = (0.01, 0.1, 1), cv = 5)

ridge1_cv.fit(iv4_train, dv4_train) 

ridge1 = Ridge(alpha = ridge1_cv.alpha_)

ridge1.fit(iv4_train, dv4_train)

ridge1.score(iv4_train, dv4_train)

# Take the following steps:
  # 1. Get r-squared
  # 2. Make predictions on the test data
  # 3. Get the RMSE

ridge1.score(iv4_test, dv4_test)

ridge1_pred = ridge1.predict(iv4_test)

rmse4 = np.sqrt(mean_squared_error(dv4_test, ridge1_pred))

#%%

### Ridge Regression With Standardization

# Take the following steps:
  # 1. Create a data frame with the necessary independent variables
  # 2. Create a series with just the dependent variable
  # 3. Use hold-out validation to split data into training and testing sets
  # 4. Standardize the necessary variables
  # 5. Use 5-fold cross-validation to find optimal hyperparameter  
  # 6. Save function as an object and fit the model to the training data
  # 7. Get the r-squared

iv5 = data3[cols3_keep]

dv5 = data3["log_price"]

iv5_train, iv5_test, dv5_train, dv5_test = train_test_split(
    iv5, dv5, test_size = 0.3, shuffle = True, random_state = 100)

iv5_train = standardize_df(iv5_train)

iv5_test = standardize_df(iv5_test)

ridge2_cv = RidgeCV(alphas = (0.01, 0.1, 1), cv = 5)

ridge2_cv.fit(iv5_train, dv5_train) 

ridge2 = Ridge(alpha = ridge2_cv.alpha_)

ridge2.fit(iv5_train, dv5_train)

ridge2.score(iv5_train, dv5_train)

# Take the following steps:
  # 1. Get r-squared
  # 2. Make predictions on the test data
  # 3. Get the RMSE

ridge2.score(iv5_test, dv5_test)

ridge2_pred = ridge2.predict(iv5_test)

rmse5 = np.sqrt(mean_squared_error(dv5_test, ridge2_pred))

#%%

### Random Forest Regression

# Take the following steps:
  # 1. Create a data frame with the necessary independent variables
  # 2. Create a series with just the dependent variable
  # 3. Use hold-out validation to split data into training and testing sets
  # 4. Perform hyperparameter tuning
  # 4. Save function as an object and fit the model to the training data
  # 5. Get the r-squared

iv6 = data3[cols3_keep]

dv6 = data3["log_price"]

iv6_train, iv6_test, dv6_train, dv6_test = train_test_split(
    iv6, dv6, test_size = 0.3, shuffle = True, random_state = 100)

hp1_grid = {"n_estimators" : [200, 250, 500],
            "max_depth" : [20, 21, 22],
            "min_samples_split" : [4]}

rf1_cv = GridSearchCV(RandomForestRegressor(random_state = 100), 
                      hp1_grid, cv = 5)

rf1_cv.fit(iv6_train, dv6_train)

rf1_bp = rf1_cv.best_params_

rf1 = RandomForestRegressor(n_estimators = rf1_bp["n_estimators"], 
                            max_depth = rf1_bp["max_depth"],
                            min_samples_split = rf1_bp["min_samples_split"],
                            random_state = 100)

rf1.fit(iv6_train, dv6_train)

rf1.score(iv6_train, dv6_train)

# Take the following steps:
  # 1. Get r-squared
  # 2. Make predictions on the test data
  # 3. Get the RMSE

rf1.score(iv6_test, dv6_test)

rf1_test_pred = rf1.predict(iv6_test)

rmse6 = np.sqrt(mean_squared_error(dv6_test, rf1_test_pred))

#%%

### K-Nearest Neighbors Regression

# Take the following steps:
  # 1. Create a data frame with the necessary independent variables
  # 2. Create a series with just the dependent variable
  # 3. Use hold-out validation to split data into training and testing sets
  # 4. Save function as an object and perform hyperparameter tuning
  # 4. Fit the model to the training data
  # 5. Get the r-squared

iv7 = data3[cols3_keep]

dv7 = data3["log_price"]

iv7_train, iv7_test, dv7_train, dv7_test = train_test_split(
    iv7, dv7, test_size = 0.3, shuffle = True, random_state = 100)

knn1 = KNeighborsRegressor()

hp2_grid = {"n_neighbors" : [4, 8, 12, 16, 20]}

knn1_cv = GridSearchCV(knn1, param_grid = hp2_grid, cv = 5)

knn1_cv.fit(iv4_train, dv4_train)

knn1 = knn1_cv.best_estimator_

knn1.fit(iv7_train, dv7_train)

knn1.score(iv7_train, dv7_train)

# Take the following steps:
  # 1. Get r-squared
  # 2. Make predictions on the test data
  # 3. Get the RMSE
  
knn1.score(iv7_test, dv7_test)

knn1_test_pred = knn1.predict(iv7_test)

rmse7 = np.sqrt(mean_squared_error(dv7_test, knn1_test_pred))

#%%
