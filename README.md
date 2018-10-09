# InsightProject

## What's the storey?
### Predicting Allowable Building Height for Empty Land Parcels in the City of Toronto


Using current building data as training set to predict building heigts. The results can be used to predict the development potential for empty land parcels. 

The appraoch is twofold: first using nearest buildigns to predict heights, and second using other factors, such as bylaw height, size of land parcel and building, proximity to transit and population data, in a random forest regressor to predict building heights. 

The comparison between the nearest neighbours and random forest provides insights into the potential of each building with respect to city features and capabilities, and possible changes into zoning bylaws.

