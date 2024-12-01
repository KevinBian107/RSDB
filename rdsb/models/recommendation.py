def recommend(model, features, u_location, u_hours):
    '''Recommend business category based on features,
        - conduct aggregation of rating based on location bins
        (implicit for type of users around certain location)'''
    
    # bining by locations (same as feature engineering)
    
    # for all users in location bin + hours want to operate:
        # query all the needed info (temporal info + gmap popularity) based on user info in such location
            # all user in such location has history of interacting with certain business category
        # predict ratings for all type of business x all user in such location
        # aggregate all ratings grouoby business location Bin
        # ranking
    
    return ...

def user_call(location, hours, dataset):
    '''User call to recommend business category based on location and hours'''
    
    # for this dataset, do feature engineering
    
    # call recommend function
    
    return ...