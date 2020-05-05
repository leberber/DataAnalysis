def missing_data(df, Per_NaN=0, action="print"):

    TotalNan = df.isnull().sum().sum()
    dfLenth = (len(df)) * 100
    PerTotal = TotalNan / dfLenth
    feature = []
    number = []
    total = []

    for feat in [var for var in df.columns if df[var].isnull().mean() > 0]:
        num = df[feat].isna().sum()
        x = num / (len(df)) * 100
        total.append(num)
        if x >= Per_NaN:
            feature.append(feat)
            number.append(num)
        d = dict(zip(feature, number))
    if action == "return":

        return list(d.keys())
    elif len(feature) == 0:
        return []
    else:
        print(
            "***************************DataFrame Null Info*******************************"
        )
        print(
            "{:33.25}".format("df's Total Nulls is "),
            "{:33.20}".format(str(TotalNan)),
            "{:4.6}".format(str(PerTotal)),
            "{:4.6}".format("%"),
        )
        print(
            "{:33.25}".format("Features With Nulls is "),
            "{:33.20}".format(str(len(total))),
            "{:4.6}".format(str(len(total) / len(df.columns) * 100)),
            "{:4.6}".format("%"),
        )
        print(
            "-----------------------------------------------------------------------------"
        )
        print(
            "***************************Features' Nulls Info******************************"
        )
        print(
            "{:30.20}".format("The feature "),
            "{:35.20}".format("# of NaN"),
            "{:4.30}".format(" % of NaN"),
        )

        for feature, value in sorted(d.items(), key=lambda item: item[1], reverse=True):
            per = value / (len(df)) * 100
            print(
                "{:33.20}".format(feature),
                "{:33.20}".format(str(value)),
                "{:4.6}".format(str(per)),
                "{:4.6}".format("%"),
            )


def rare_labels(df, min=0, max=10000, action="print"):
    categorical_variables = []
    cadinality = []

    for col in df.columns:
        if df[col].dtypes == "O":
            if df[col].nunique() >= min and df[col].nunique() <= max:

                categorical_variables.append(col)
                cadinality.append(len(df[col].value_counts()))
    if action == "return":
        return categorical_variables
    else:

        print("-----------------------------------------------")
        print("*****************DataFrame*********************")
        print("-----------------------------------------------")
        print("{:25.20}".format("The feature "), "{:35.20}".format("Cardinality"))
        print("")
        i = 0
        for fea in categorical_variables:
            print("{:30.20}".format(str(fea)), cadinality[i])
            i += 1
        print("")
        print("-----------------------------------------------")
        print("******************Features*********************")
        print("-----------------------------------------------")
        for col in df.columns:
            if df[col].dtypes == "O":
                if df[col].nunique() >= min and df[col].nunique() <= max:
                    # print percentage of observations per category
                    print((df.groupby(col)[col].count() * 100 / len(df)))
                    print("")

        return df[categorical_variables]


def one_hot_Categ_encoder(train_set, test_set, features, drop_last=False):
    """
    >>  B-ii)  One hot encode the categorical variables:
        here I am writing a function to perform one hot encoding. The function takes 4 arguments: train set, test set, a list of features we want to one_hot_encode and boolean argument that allows us to encode the features into k or k-1 dummy variables. The function returns both the encoded train and test sets.
        * 1) I import the **OneHotCategoricalEncoder** from the **feature_engine.categorical_encoders**
        * 2) I make an instance of the **OneHotCategoricalEncoder**
        *  a) IF **drop_last=False** The default is set make an instance of **OneHotCategoricalEncoder** with k dummy variables.
        *  b) IF **drop_last=True** The default is set make an instance of **OneHotCategoricalEncoder** with k-1 dummy variables.
        * 3) I fit the **training set** to  **OneHotCategoricalEncoder** object instance 
        * 4) tansform and return both **train** and **test** sets
    """
    from feature_engine.categorical_encoders import OneHotCategoricalEncoder

    if drop_last is True:
        encoder = OneHotCategoricalEncoder(
            top_categories=None, variables=features, drop_last=True
        )
    elif drop_last is False:
        encoder = OneHotCategoricalEncoder(top_categories=None, variables=features)
    encoder.fit(train_set)
    return encoder.transform(train_set), encoder.transform(test_set)


def correlation(df, threshold, Brute_force=False):
    import pandas as pd

    # Set of all the names of correlated columns
    column1 = []
    column2 = []
    absolu = []
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # evaluate all possilbe pairs of features and
            # compare the absolute value against the the threshold
            if abs(corr_matrix.iloc[i, j]) > threshold:
                # getting the name of column
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                abso = abs(corr_matrix.iloc[i, j])
                absolu.append(abso)
                column1.append(col1)
                column2.append(col2)

    if Brute_force is True:
        return list(set(column1))
    else:
        df = pd.DataFrame(
            list(zip(column1, column2, absolu)),
            columns=["feature1", "feature2", "correlation"],
        )
    return df


def correlated_groups(df):
    feature1 = []
    groups = []
    for feature in df.feature1.unique():
        if feature not in feature1:
            # find all features correlated to a single feature
            correlated_block = df[df.feature1 == feature]
            feature1 = feature1 + list(correlated_block.feature2.unique()) + [feature]
            # append the block of features to the list
            groups.append(correlated_block)
    print("Found {} correlated groups".format(len(groups)))
    print("out of {} total features".format(XX_train.shape[1]))
    return groups


def less_missing_data(groups, df):
    feat_keep = set()
    feat_total = []
    for group in groups:
        feature1 = list(
            group.feature2.unique()
        )  # get the features in the secound column of the group
        feature2 = list(
            group.feature1.unique()
        )  # get the features in the first column of the group
        features = feature1 + feature2  # join the features and make a list
        feat_total.append(features)
    i = 1
    feature_keep = set()
    for row in feat_total:
        print(
            "*************************group",
            [i],
            " feature nulls****************************",
        )
        print(
            "{:30.20}".format("The feature "),
            "{:32.20}".format("# of nulls"),
            "{:4.30}".format(" % of nulls"),
        )
        print(
            "-----------------------------------------------------------------------------"
        )

        null = []
        for elem in row:
            num_null = df[elem].isnull().sum()
            per = num_null * 100 / len(df)
            if null == []:
                null.append(num_null)
                feature_keep.add(elem)
            elif null != []:
                if num_null > null[0]:
                    feature_keep.add(elem)
                else:
                    pass
            print(
                "{:33.20}".format(elem),
                "{:33.20}".format(str(num_null)),
                "{:4.6}".format(str(per)),
                "{:4.6}".format("%"),
            )
        print("")
        i = i + 1
    feat_total = sum(feat_total, [])
    list_drop = [feat for feat in feat_total if feat not in feature_keep]

    print(
        "-----------------------------------------------------------------------------"
    )
    print(
        "***************************List of features to keep *************************"
    )
    print(
        "-----------------------------------------------------------------------------"
    )
    print("")
    print(list(feature_keep))
    print("")
    print(
        "-----------------------------------------------------------------------------"
    )
    print(
        "***************************List of features drop ****************************"
    )
    print(
        "-----------------------------------------------------------------------------"
    )
    return list(set(list_drop))


def filter_method(df, threshold):
    # import the VarianceThreshold class from scit-learn
    from sklearn.feature_selection import VarianceThreshold

    # make an instance of the class
    constant_filter = VarianceThreshold(threshold=threshold)

    # here i am going to remove the constant features manually.
    # in order to be able to remove object and numeric constant at the same time.
    if threshold == 0:
        # the comprehension list returns columns that contain only 1 label:
        constant_features = [feat for feat in df.columns if len(df[feat].unique()) == 1]
        # this returns a dataframe of the constant features and a message if there are none
        if len(constant_features) != 0:
            return df[constant_features]
        else:
            return ["There are no constant features"]

    # if the threshold !=0, i call an instance of the VarianceThreshold with the passed value.
    elif threshold != 0:
        # call the lists_of_dtypes function to extract the numeric features
        numeric_features = df.select_dtypes(exclude=["object"]).columns
        # fit the VarianceThreshold estimator  the numeric_features
        constant_filter.fit(df[numeric_features])
        # finally we can print the constant features
        # get_support is a boolean vector that indicates which features are retained
        # if we sum over get_support, we get the number of features that are not constant
        constant_features = [
            column
            for column in df[numeric_features].columns
            if column not in df[numeric_features].columns[constant_filter.get_support()]
        ]
        # this returns the quasi-constant features
        if len(constant_features) != 0:
            return df[constant_features]
        else:
            return ["There are no Quasi-constant features"]


def q_constant_feature_info(df, feature_list, top=5):

    if type(feature_list) != list:
        feature_list = feature_list.columns
    if len(feature_list[0]) == 30:
        print("There are no constant features")
    elif len(feature_list[0]) == 36:
        print("There are no quasi-constant features")
    elif len(feature_list[0]) == 32:
        print("There are no duplicated features")
    else:
        for col in df.columns:
            if col in feature_list:
                print(
                    "                               ",
                    col,
                    " Feature                                          ",
                )
                print(
                    "The number of unique values in the [",
                    col,
                    "] feature is ",
                    len(df[col].unique()),
                )
                print(
                    "The list of unique values in the [",
                    col,
                    "] feature is ",
                    sorted(df[col].unique()[:top]),
                )
                print(
                    "------------------------Count and Percentage of element ocurances-----------------------"
                )

                keys = df[col].unique()[:top]
                values = []
                for item in keys:
                    value = df[col].isin([item]).sum(axis=0)
                    values.append(value)
                    d = dict(zip(keys, values))
                print(
                    "{:25.20}".format("The value "),
                    "{:32.20}".format("Number of occurrences"),
                    "{:4.30}".format(" percentage of occurrences"),
                )
                for key, value in sorted(
                    d.items(), key=lambda item: item[1], reverse=True
                ):
                    percent = (100 * value) / len(df)
                    key = str(key)
                    value = str(value)
                    percent = str(percent)
                    print(
                        "{:25.20}".format(key),
                        "{:33.20}".format(value),
                        "{:4.6}".format(percent),
                        "%",
                    )

                print("")
                print(
                    "----------------------------------------------------------------------------------------"
                )


def duplicated_features(df):
    duplicated = []
    
    
    # looping Through every value in a culumn
    for i in range(0, len(df.columns)):

        # start from the first column in the data fram(column index 0)
        column1 = df.columns[i]
        # compare the first column with evry other column in the dataframe
        for column2 in df.columns[i + 1 :]:

            # use the pandas method equals() to indentify the duplicated columns
            # the mothod returns a boolean: True if the columns are identical false otherwise
            if df[column1].equals(df[column2]):

                # it the column is identical we add it our duplicated list
#                 duplicated.append(column1)
                duplicated.append(column2)
    
    if len(duplicated) != 0:
        return df[list(set(duplicated))]
    else:
        return ["There are no duplicated features"]


def Train_Test_Split(train_set,test_set,target, col_drop=[]):
    train_set.drop(col_drop, axis=1, inplace=True)
    test_set.drop(col_drop, axis=1,inplace=True)
    XX_train=train_set.drop(train_set[target], axis=1)
    XX_test=test_set.drop(test_set[target], axis=1)
    y_train= train_set[target]
    y_test = test_set[target]
    return XX_train, XX_test, y_train, y_test

# here I wrote a function that takes train_set, test_set
def standard_scaler(XX_train, XX_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(XX_train)
    return scaler.transform(XX_train), scaler.transform(XX_test)
#X_train,X_test=standard_scaler(XX_train, XX_test)