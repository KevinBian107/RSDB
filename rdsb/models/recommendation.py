def recommend(model, features, u_location, u_hours):
    '''Recommend business category based on features,
        - conduct aggregation of rating based on location bins
        (implicit for type of users around certain location)'''
    
    # bining by locations (same as feature engineering)
    
    # for all users in location Bin + Hours want to operate:
        # query all the needed info (temporal info + gmap popularity) based on user info in such location
        # predict ratings for all type of business x all user in such location
        # aggregate all ratings grouoby business category
        # ranking
    
    return ...

def user_call(location, hours, dataset):
    '''User call to recommend business category based on location and hours'''
    
    # for this dataset, do feature engineering
    
    # call recommend function
    
    return ...