
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm

# clustering

# Load data from csv
data = pd.read_excel("climatechange.xls", header=0, names=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021'])

# Select only the rows for the desired indicators
data = data[data['Indicator Name'].isin(['Urban population growth (annual %)','Population, total','Urban population (% of total population)'])]

# Drop missing values
data = data.dropna()

# Melt the dataframe to have columns for 'Country Name', 'Country Code', 'Year', and 'Value'
data = data.melt(id_vars=['Country Name','Country Code','Indicator Name','Indicator Code'], var_name='Year', value_name='Value')


# Normalize the data
scaler = MinMaxScaler()
data[['Value']] = scaler.fit_transform(data[['Value']])

# Group the data by 'Country Name' and 'Indicator Name', and take the mean of the values for each group
data = data.groupby(['Country Name','Indicator Name']).mean().reset_index()

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(data[['Value']])

# Perform Agglomerative Clustering
agg_cluster = AgglomerativeClustering(n_clusters=3)
agg_cluster.fit(data[['Value']])

# Perform DBSCAN Clustering
dbscan_cluster = DBSCAN(eps=0.1,min_samples=2)
dbscan_cluster.fit(data[['Value']])

# Add new columns to the dataframe to show the cluster membership of each country for each clustering method
data['kmeans_cluster'] = kmeans.labels_
data['agg_cluster'] = agg_cluster.labels_
data['dbscan_cluster'] = dbscan_cluster.labels_

# Create the figure and subplots
fig, axes = plt.subplots(1,3,figsize=(20,5))

# Plot the data points for K-Means clustering
axes[0].scatter(data.index, data['Value'], c=data['kmeans_cluster'], cmap='rainbow')
axes[0].set_title("K-Means Clustering")
axes[0].set_xlabel("Country")
axes[0].set_ylabel("Normalized Value")


# Plot the data points for Agglomerative Clustering
axes[1].scatter(data.index, data['Value'], c=data['agg_cluster'], cmap='rainbow')
axes[1].set_title("Agglomerative Clustering")
axes[1].set_xlabel("Country")
axes[1].set_ylabel("Normalized Value")

#Plot the data points for DBSCAN Clustering
axes[2].scatter(data.index, data['Value'], c=data['dbscan_cluster'], cmap='rainbow')
axes[2].set_title("DBSCAN Clustering")
axes[2].set_xlabel("Country index")
axes[2].set_ylabel("Normalized Value")


#Add a legend
fig.legend(labels=['Cluster 1','Cluster 2','Cluster 3'],loc='center')

#Add a title
fig.suptitle("Country Clusters based on Normalized Indicator Value", fontsize=16)

#Display the plot
plt.show()





# curve fitting

# Load data from csv
data = pd.read_excel("climatechange.xls", header=0, names=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021'])

# Select only the rows for the desired indicators
data = data[data['Indicator Name'].isin(['Urban population growth (annual %)','Population, total','Urban population (% of total population)'])]

# Select data for Aruba
aruba_urban_pop = data[(data['Country Name'] == 'Aruba') & (data['Indicator Name'] == 'Urban population (% of total population)')]

from scipy.optimize import curve_fit

# Define the polynomial function
def poly(x, a, b, c):
    return a*x**2 + b*x + c


# Extract the years and urban population data
urban_pop = aruba_urban_pop.iloc[:, 4:].values[0]
years = aruba_urban_pop.columns[4:].astype(float)



# Fit the model to the data
params, _ = curve_fit(poly, years, urban_pop)

# Predict urban population for the next 20 years
future_years = np.arange(2022, 2042)
predicted_urban_pop = poly(future_years, params[0], params[1], params[2])




# Define the error function
def err_ranges(x, y, y_pred, y_pred_std):
    y_pred_upper = y_pred + norm.ppf(0.975) * y_pred_std
    y_pred_lower = y_pred - norm.ppf(0.975) * y_pred_std
    return y_pred_upper, y_pred_lower

# Estimate the error
predicted_urban_pop_upper, predicted_urban_pop_lower = err_ranges(future_years, urban_pop, predicted_urban_pop, 0.1)

#plot actual data vs predicted values

plt.plot(years, urban_pop, 'o', label='Actual data')
plt.plot(future_years, predicted_urban_pop, '-', label='Predicted values')
plt.fill_between(future_years, predicted_urban_pop_upper, predicted_urban_pop_lower, alpha=0.2, label='95% Confidence Interval')
plt.legend()

# Plot the data and the model
plt.plot(years, urban_pop, 'o', label='Observed data')
plt.plot(future_years, predicted_urban_pop, '-', label='Fitted model')
plt.fill_between(future_years, predicted_urban_pop_upper, predicted_urban_pop_lower, alpha=0.2)
plt.xlabel('Year')
plt.ylabel('Urban population (% of total population)')
plt.title('Fitted model and prediction of urban population in Aruba')
plt.legend()
plt.title('Urban Population Growth for Aruba')
plt.show()
