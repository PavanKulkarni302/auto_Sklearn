import pathlib
import matplotlib.pyplot as plt
from function import *
import pandas as pd
import autosklearn.classification as classifier
import autosklearn.regression as regresser
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,mean_squared_error,r2_score,mean_absolute_error,plot_roc_curve,roc_auc_score
from sklearn.preprocessing import LabelEncoder


def predict_fn(x):
    if algoType == 'CLASSIFICATION':
        train_config = pickle.load(open(modelFilePath + "train_config.pkl", "rb"))
        print('train_config',train_config)
        y_pred = train_config.predict(x)
        prob = np.array(list(zip(1 - train_config.predict(x), train_config.predict(x))))
        # print("prob ", prob.head())


    elif algoType == 'REGRESSION':
        train_config = pickle.load(open(modelFilePath + "train_config.pkl", "rb"))
        xgb = train_config['model']
        model1 = open(modelFilePath + 'data5.pkl', 'rb')
        xgb = pickle.load(model1)
        model1.close()
        prob= np.array(list(zip(1-xgb.predict(x),xgb.predict(x))))
        print("prob ", prob)

    return prob

def model():
    if algoType == 'CLASSIFICATION':

        if trainClass == 'autosklearn':
            model = classifier.AutoSklearnClassifier(time_left_for_this_task=time_left_for_this_task,per_run_time_limit=per_run_time_limit)

        else:
            print("trainClass invalid")

    elif algoType == 'REGRESSION':
        if trainClass == 'autosklearn':
            print("train class: ", trainClass)
            model = regresser.AutoSklearnRegressor(time_left_for_this_task=time_left_for_this_task, per_run_time_limit=per_run_time_limit)

        elif trainClass == 'LinearRegression':
            pass
    else:
        print("select the algorithm type")

    return model




def train():
    print("Begining of training operation************************************************************************************")
    global autosklearn
    global clf

    if restartMode == "Y" and checkPoint == "z":                                         #in json file restartMode ==Y from N
        pickle_ckpt = pickle.load(open(checkPointDir + "train_ckpt.pkl", "rb"))
        print("***************loaded pickle_ckpt***************************")

        '''if pickle_ckpt["started"] == "Y":
            print("******************************************************************************started checkPoint************************** ")'''

        if pickle_ckpt["testDataSaved"] == "Y":
            print("*********************t***************************************************estDataSaved checkPoint************************** ")

            X_test = pickle_ckpt['X_test']
            if hyperParam == "Y":
                clf = pickle_ckpt['clf']
            else:
                autosklearn = pickle_ckpt['model']


    else:
        pickle_ckpt = {}
        train_config = {}

    train_dataset = getData(sourceFilePath, sourceDsType, sourceHostName, sourceDbName, sourcePort, sourceUserName,
                     sourcePassword, query)                                                                              # Reading of data


    # train_dataset = pd.read_csv(sourceFilePath)                                                                         # Reading of data
    print("train_dataset: ", train_dataset.columns)
    print("inputColList: ", inputColList)

    input_col_list = open(modelFilePath + 'data6.pkl', 'wb')                                                               #94-96 uncom
    pickle.dump(inputColList, input_col_list)
    input_col_list.close()


    print(train_dataset)
    dataset = train_dataset[inputColList]
    #
    print(dataset)
    #
    print("dataset_columns: ",dataset.columns)
    print("********************************************************************************************************** line-106")

    print("data before imputation:")
    if imputationDetails != None:
        print("data before imputation:")
        dataset = imputeData(dataset, imputationDetails)
        print("data after imputation:")
        print("************************************************imputation Done--*************************************line-114")

    print("dataset", dataset)

    print(len(dataset))
    dataset = sampling(dataset, samplingType)
    print(len(dataset))

    feature_cols = dataset.columns
    print('feature_cols', feature_cols)

    X_train, X_test = train_test_split(dataset, test_size=testPercent, random_state=int(seed))

    X_data = X_test
    X_data = X_data.iloc[:, 1:]
    test_data = X_data
    X_test = pd.DataFrame(X_data)

    schema = getSparkSchemaByDtypes(X_train.dtypes)
    spark_df = createSparkDfByPandasDfAndSparkSchema(X_train, schema)
    spark_df = addIndexToSparkDf(spark_df)
    spark_df.show(5)
    joined_df = spark_df

    schema = getSparkSchemaByDtypes(X_test.dtypes)
    spark_df = createSparkDfByPandasDfAndSparkSchema(X_test, schema)
    spark_df = addIndexToSparkDf(spark_df)
    spark_df.show(5)
    joined_df_test = spark_df
    print("********************************************************************************joined_df*************************************")

    if (rowIdentifier != None):
        '''sourceDataset = getData(inputSourceFileName, sourceDsType, sourceHostName, sourceDbName
                                , sourcePort, sourceUserName, sourcePassword, sourceQuery)'''
        sourceDataset = train_dataset[rowIdentifier]
        for i in sourceDataset:
            if sourceDataset[i].dtypes == "object":
                sourceDataset[i] = sourceDataset[i].astype(str)
        src_train, src_test = train_test_split(sourceDataset.iloc[:, :].values, test_size=testPercent,
                                               random_state=int(seed))
        pd_scr_train = pd.DataFrame(src_train)
        # pd_scr_train = pd_scr_train.astype(str) # need to find reason behind that
        source_schema = getSparkSchemaByDtypes(sourceDataset.iloc[:, :].dtypes)
        spark_src_train_df = createSparkDfByPandasDfAndSparkSchema(pd_scr_train, source_schema)
        spark_src_train_df = addIndexToSparkDf(spark_src_train_df)

        joined_df = joinSparkDfByIndex(spark_src_train_df, joined_df)

    # removing index column output training table and test table
    joined_df = joined_df.drop("rowNum")

    print("joined_df", joined_df)
    print("rowIdentifier[0]", rowIdentifier[0])
    joined_df.show(10)

    my_list = list(joined_df.columns)
    print('my_list', my_list)

    # label column shift to last position
    col = my_list
    col.remove('label')
    print('col', col)

    reordered_cols = reorder_columns(my_list, first_cols=col, last_cols=['label'])
    joined_df_new = joined_df[reordered_cols]

    print("joined_df_new", joined_df_new)
    joined_df_new.show(10)
    print("***************************************************************************************************************joined_df_new")

    if saveTrainVersion == "Y":
        joined_df_new = joined_df_new.withColumn("version", lit(execVersion))
        print("joined_df_new****************************************************************************************joined_df_new*********")
        joined_df_new.show(10)

    print("trainSetDetails: ", trainSetDetails)

    if trainSetDetails != None:
        print("data before imputation:")
        if trainSetDetails["trainSetDsType"] == "file":

            if modelDataFormat == "CSV":
                print("trainSetDetails[trainSetSavePath]", trainSetDetails["trainSetSavePath"])
                joined_df_new.write.option("header", "true").csv(trainSetDetails["trainSetSavePath"])

            if modelDataFormat == "PARQUET":
                saveSparkDf(joined_df_new, trainSetDetails["trainSetDsType"]
                            , None, None, None, None, None, None, None, None
                            , trainSetDetails["trainSetSavePath"], "train")
            print("trainingset saved at: ", trainSetDetails["trainSetSavePath"])


        else:
            print("in mysql", trainSetDetails)
            saveSparkDf(joined_df_new, trainSetDetails["trainSetDsType"], trainSetDetails["trainSetHostName"]
                        , trainSetDetails["trainSetDbName"], trainSetDetails["trainSetPort"]
                        , trainSetDetails["trainSetUserName"], trainSetDetails["trainSetPassword"]
                        , trainSetDetails["trainSetDriver"], trainSetDetails["trainSetUrl"]
                        , trainSetDetails["trainSetTableName"], None, "train")
            print("trainingset saved into ", trainSetDetails["trainSetDsType"], " table ",
                  trainSetDetails["trainSetTableName"])
            print("*********************************************************************saved_train_data**************************************")

    if (encodingDetails != None):
        print("data before encoding: ")
        print(dataset)
        #dataset = encodeData(dataset, encodingDetails).astype(dtype='int64', copy=True, errors='ignore')
        dataset = encodeData(dataset, encodingDetails)
        try:
            encode_dict = pickle.load(open(modelFilePath + "train_config.pkl", "rb"))
            encode_dict = encode_dict["label_encode"]
            print(encode_dict)
            train_config['label_encode'] = encode_dict
        except:
            print("No Label Encoding")
        print("data after encoding: ")
        print(dataset)

    if (scalingDetails != None):
        print("data before scaling: ")
        print(dataset)
        dataset = scalingData(dataset, scalingDetails)
        print("data after scaling: ")
        print(dataset)

        feat_col = list(dataset.columns)
        print("feat_col", feat_col)
        # print("feat_col", feat_col)
        feat_col.remove('label')
        print('***************************************************************************************************feat_col*************', feat_col)
        train_config['feat_col'] = feat_col

        feat_col_eval = list(dataset.columns)
        train_config['feat_col_eval'] = feat_col_eval
        print("train_config['feat_col_eval']", train_config['feat_col_eval'])

        feat_col=list(dataset.columns)                                                                                  #uncommented 323,336

        featCol = open(modelFilePath + 'data4.pkl', 'wb')
        pickle.dump(feat_col, featCol)
        featCol.close()


        print("feat_col",feat_col)
        feat_col.remove('label')
        print('feat_col', feat_col)

        featCol = open(modelFilePath + 'data3.pkl', 'wb')
        pickle.dump(feat_col, featCol)
        featCol.close()
        print("********************************************************************************************************featCol.close()****line-258")


        dataset = dataset[inputColList]

        X = dataset.iloc[:, 1:].values
        y = dataset.iloc[:, 0].values

        print("***********************************************************************************************************X data_line_no:265",X)

        dataset,labClass,dictOflabClass = prepareLabelColumn(dataset)

        classes = labClass
        print('classes',classes)

        labClassList = open(modelFilePath + 'data7.pkl', 'wb')
        pickle.dump(labClass, labClassList)
        labClassList.close()


        print('dataset["label"]', dataset["label"])
        y = dataset['label']
        print("y :", y)



    # print(" before SMOTE X.shape",X.shape)
    # print(" before SMOTE y.shape",y.shape)
    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, 0].values
    if samplingTech == "SMOTE":
        print("***********************************************************************************************SMOTE*********************")
        oversampler = SMOTE(random_state = int(seed))
        X, y = oversampler.fit_resample(X, y)
        print(" after SMOTE X.shape",X.shape)
        print(" after SMOTE y.shape",y.shape)
        print(
            "********************************************************************************************END of SMOTE***********************")

    print("total_size: ", len(X))
    output_result["total_size"] = len(X)




    x = dataset.iloc[:, 1:]
    y = dataset.iloc[:, 0]

    x = x.astype('category')                                                                                    #dtype to category
    y = pd.DataFrame(y, dtype='category')

    print("x_dtype: ",x.dtypes )
    print("y_dtype: ",y.dtypes)


    #######################################################################################################################################

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    X_data = X_test
    X_data = X_data.iloc[:, 1:]
    test_data = X_data
    X_data = pd.DataFrame(X_data)

    print("X_train:", X_train.shape)
    print("y_train: ", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test: ", y_test.shape)


#Begining of training code--------------------------------------------------------------------------------------------------------------------
    results_txt = []
    print("traning operation")
    train_config = {}
    if algoType == "CLASSIFICATION" and trainClass == 'autosklearn':
        print("algoType: ",algoType)
        print("trainClass: ",trainClass)

        # Model building
        automlclassifier = model()
        print("automlclassifier", automlclassifier)
        automlclassifier.fit(X_train, y_train)
        print("automlclassifier.fit completed")


        print("*********************************************************************************************************1************************** ")
        print("train_size: ", len(X_train))
        print("test_size: ", len(X_test))
        print("*********************************************************************************************************2************************* ")
        output_result["train_size"] = len(X_train)
        output_result["test_size"] = len(X_test)


        # models used
        LeaderBoard = automlclassifier.leaderboard()
        print(LeaderBoard)
        best_model = LeaderBoard.iloc[0,2]
        print("best_model: ",best_model )

        train_config['model'] = automlclassifier
        model1 = open(modelFilePath + 'train_config.pkl', 'wb')
        pickle.dump(automlclassifier, model1)
        model1.close()
        print(" *******************************************************************************************Prediction on X_test data***************")
        print("input X_test data for pediction",X_test)

        print("input X_test dataType for pediction",X_test.dtypes)

        y_pred = automlclassifier.predict(X_test)
        print("Y_pred: ",y_pred)
        print("y_test: ",y_test)

        # creating prediction dataframe
        pred_pd_df = pd.DataFrame(y_pred)
        print("pred_pd_df", pred_pd_df)
        print("**************************************************************************************end of pred*********************** ")
##############################################################################################################################
        # appending outputs in to txt file

        print("X_train:", X_train.shape)
        print("y_train: ", y_train.shape)
        print("X_test:", X_test.shape)
        print("y_test: ", y_test.shape)
        results_txt = []
        results_txt.append([
                        {
                            "key": "LeaderBoard",
                            "value": LeaderBoard
                        },
                        {
                            "key": "best_model",
                            "value": best_model
                        },
                        {
                            "key": "X_train Shape",
                            "value": X_train.shape
                        },

                        {
                            "key": "X_test Shape",
                            "value": X_test.shape
                        },

                        {
                            "key": "y_train Shape",
                            "value": y_train.shape
                        },

                        {
                            "key": "y_test Shape",
                            "value": y_test.shape

                        }])

        print("saving Result output into path ", outputResultPath + "output_results.txt", "'")
        model_reults = open(outputResultPath + "output_results.txt", "w+")
        for i in results_txt:
            model_reults.write("\n{}".format(i))
        model_reults.close()
###############################################################################################################################
        print("**************************************************************************************metrics***************************************")
        # output reults
        cm = confusion_matrix(y_test, y_pred).tolist()                            #confussion matrix
        print("cm_before",cm)
        print("confusion_matrix: ", cm)

        accuracy = accuracy_score(y_test, y_pred)                               # accuracy
        print("accuracy: ", accuracy)

        precision = precision_score(y_test, y_pred, average='macro')            #precision
        print("precision: ", precision)

        recall = recall_score(y_test, y_pred, average='macro')                  #recall
        print("recall: ", recall)

        f1 = f1_score(y_test, y_pred, average='macro')                          #f1
        print("f1: ", f1)

        count_val = pred_pd_df.iloc[:, 0].value_counts()
        dic = dict(count_val)
        print("dic", dic)
        classesCount = []

        for k1, v1 in dic.items():
            item = {'class': k1, 'count': v1}
            classesCount.append(item)
        print("classesCount", classesCount)
        numClasses = len(classesCount)
        print("numClasses",numClasses)

        classes = dataset["label"].unique()
        print("clasess: ",classes)


        roc_auc = multiclass_roc_auc_score(y_test, y_pred, average = "macro")   #roc_auc
        print("roc_auc: ",roc_auc)

        # encoding y_test data
        encoder = LabelEncoder()
        encode_y_test = encoder.fit_transform(y_test["label"])
        # y_test.value_counts(y_test['label'].values, sort=False)
        print("encoded_y_test: ",encode_y_test)


        # encoding prediction data
        pred_df = pd.DataFrame(y_pred, columns=["prediction"])
        encode_pred= encoder.fit_transform(pred_df["prediction"])
        # pred_pd_df["prediction"].value_counts(pred_pd_df["prediction"].values, sort=False)
        print("encoded_pred_df: ",encode_pred)

        if numClasses == 2:
            #building Roc curve
            rocCurve = buildROC(encode_y_test, encode_pred)                                         #rocCurve


        else:
            fpr = dict()
            tpr = dict()
            rocauc = dict()
            for i in range(len(classes)):
                fpr[i], tpr[i], threshold = roc_curve(np.array(pd.get_dummies(encode_y_test))[:, i],
                                                      np.array(pd.get_dummies(encode_pred))[:, i])
                rocauc[i] = auc(fpr[i], tpr[i])

            # fpr, tpr, threshold = metrics.roc_curve(target_test, test_preds)
            # rocauc = metrics.auc(fpr, tpr)
            # print("fpr", fpr)
            # print("tpr", tpr)
            df1 = pd.DataFrame(fpr)
            # print(df1)
            df2 = pd.DataFrame(tpr)
            # df2 = df2.rename(columns = {0: 3, 1: 4, 2: 5})
            # print(df2)
            # df=pd.concat([df1,df2],axis=1)
            # print(df)
            # tf=list(zip(*map(df.get, df)))
            rocCurve = []
            # print(tf)
            df1_clmn = list(df1)
            df2_clmn = list(df2)

            for i in range(0, len(df1)):
                for j in df1_clmn:
                    # print("i", i)
                    # print("j", j)
                    dic = {}
                    dic["specificity"] = df1[j][i]
                    dic["sensitivity"] = df2[j][i]

                    rocCurve.append(dic)
                    # print(dic)

            print(rocCurve)
            # plotting
            plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Class 0 vs Rest')
            plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 1 vs Rest')
            plt.plot(fpr[2], tpr[2], linestyle='--', color='blue', label='Class 2 vs Rest')
            # plt.plot(fpr[3], tpr[3], linestyle='--', color='yellow', label='Class 3 vs Rest')
            plt.title('Multiclass ROC curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive rate')
            plt.legend(loc='best')
            print(plt.show())


        output_result["accuracy"] = accuracy
        output_result["precision"] = precision
        output_result["recall"] = recall
        output_result["roc_auc"] = roc_auc
        output_result["f1"] = f1
        output_result["confusion_matrix"] = cm
        output_result["classesCount"] = classesCount
        #output_result["featureImportance"] = featureImportance
        output_result["rocCurve"] = rocCurve
        # output_result["classes"] = classes
        # output_result["numClasses"] = numClasses

        metrics = []

        metrics.append([{"key": "accuracy",
                         "value": accuracy
                         },
                        {
                            "key": "precision",
                            "value": precision
                        },
                        {
                            "key": "recall",
                            "value": recall
                        },
                        {
                            "key": "roc_auc",
                            "value": roc_auc
                        },

                        {
                            "key": "f1",
                            "value": f1
                        }])

        print("metrics", metrics)

        output_result["metrics"] = metrics

        print("saving train output into path ", outputResultPath + ".json", "'")
        with codecs.open(outputResultPath + ".json", 'w', 'utf8') as f:
            f.write(json.dumps(str(output_result), sort_keys=True, ensure_ascii=False))

        # creating label dataframe
        pd_label_df = y_test
        spark_pred_class_df = createSparkDfByPandasDfAndSparkSchema(pd_label_df,
                                                               StructType([StructField("label Class", StringType(), True)]))
        print("spark_pred_class_df", spark_pred_class_df)
        spark_pred_class_df = addIndexToSparkDf(spark_pred_class_df)
        print("spark_pred_class_df", spark_pred_class_df)
        spark_pred_class_df.show()


        # creating prediction dataframe
        pd_pred_df = pred_pd_df
        spark_pred_df = createSparkDfByPandasDfAndSparkSchema(pd_pred_df,
                                                              StructType([StructField("prediction Class", StringType(), True)]))
        print("spark_pred_df", spark_pred_df)
        spark_pred_df = addIndexToSparkDf(spark_pred_df)
        print("spark_pred_df",spark_pred_df)
        spark_pred_df.show()
        print("testSetDetails: ", testSetDetails)


        #joining prediction dataframe with dataframe of prediction class
        test_predResult_spark_df = joinSparkDfByIndex(spark_pred_class_df,spark_pred_df)
        joined_df_test = joinSparkDfByIndex(joined_df_test,test_predResult_spark_df)
        test_predResult_spark_df.show()
        print(" #test_predResult_spark_df",test_predResult_spark_df)
        joined_df_test.show()

        if testSetDetails != None:
            if testSetDetails["testSetDsType"] == "file":

                if modelDataFormat == "CSV":
                    joined_df_test.write.option("header", "true").csv(testSetDetails["testSetSavePath"])

                if modelDataFormat == "PARQUET":
                    saveSparkDf(joined_df, testSetDetails["testSetDsType"]
                                , None, None, None, None, None, None, None, None
                                , testSetDetails["testSetSavePath"], "test")
                print("testset saved at: ", testSetDetails["testSetSavePath"])
            else:
                saveSparkDf(joined_df, testSetDetails["testSetDsType"]
                            , testSetDetails["testSetHostName"], testSetDetails["testSetDbName"],
                            testSetDetails["testSetPort"]
                            , testSetDetails["testSetUserName"], testSetDetails["testSetPassword"]
                            , testSetDetails["testSetDriver"], testSetDetails["testSetUrl"]
                            , testSetDetails["testSetTableName"], None, "test")
                print("testset saved into ", testSetDetails["testSetDsType"], " table ",
                      testSetDetails["testSetTableName"])

#-------------------------------------------------------------------------------------------------------------------------------------------------#
    elif algoType == "REGRESSION" and trainClass == 'autosklearn':
        print("algoType: ", algoType)
        print("trainClass: ", trainClass)


        # Model building
        automlregresser = model()
        print("automlregresser: ",automlregresser)
        automlregresser.fit(X_train, y_train)
        print("automlclassifier.fit completed")


        print("*********************************************************************************************************1************************** ")
        print("train_size: ", len(X_train))
        print("test_size: ", len(X_test))
        print("*********************************************************************************************************2************************* ")
        output_result["train_size"] = len(X_train)
        output_result["test_size"] = len(X_test)


        # models used
        # models used
        LeaderBoard = automlregresser.leaderboard()
        best_model = LeaderBoard.iloc[0,2]
        print("best_model: ",best_model )
        print("leaderboard: ",automlregresser.leaderboard())


        train_config['model'] = automlregresser
        model1 = open(modelFilePath + 'train_config.pkl', 'wb')
        pickle.dump(automlregresser, model1)
        model1.close()
        # #Prediction on X_test data
        y_pred = automlregresser.predict(X_test)
        print("Y_pred: ",y_pred)
        # creating prediction dataframe
        pred_pd_df = pd.DataFrame(y_pred)
        print("pred_pd_df", pred_pd_df)
        print("**************************************************************************************end of pred***********************")

        results_txt = []
        results_txt.append([
            {
                "key": "LeaderBoard",
                "value": LeaderBoard
            },
            {
                "key": "best_model",
                "value": best_model
            },
            {
                "key": "X_train Shape",
                "value": X_train.shape
            },

            {
                "key": "X_test Shape",
                "value": X_test.shape
            },

            {
                "key": "y_train Shape",
                "value": y_train.shape
            },

            {
                "key": "y_test Shape",
                "value": y_test.shape

            }])

        print("saving Result output into path ", outputResultPath + "output_results.txt", "'")
        model_reults = open(outputResultPath + "output_results.txt", "w+")
        for i in results_txt:
            model_reults.write("\n{}".format(i))
        model_reults.close()


        # output reults
        r2 = r2_score(y_test, y_pred)             # r2_score
        print("accuracy: ", r2)

        meanSquared_error = mean_squared_error(y_test, y_pred)  # mean_squared_error
        print("mean_squared_error: ", meanSquared_error)

        meanAbsolute_error = mean_absolute_error(y_test, y_pred)
        print("meanAbsolute_error: ", meanAbsolute_error)


        output_result["r2_score"] = r2
        output_result["meanSquared_error"] = meanSquared_error
        output_result["meanAbsolute_error"] = meanAbsolute_error

        metrics = []

        metrics.append([{"key": "r2_score",
                         "value": r2
                         },
                        {
                            "key": "meanSquared_error",
                            "value": meanSquared_error
                        },
                        {
                            "key": "meanAbsolute_error",
                            "value":meanAbsolute_error
                        }])

        print("metrics", metrics)
        if calcFeatureWeight == "Y":
            # Feature Weight Extraction
            feature_weight_extraction = featureWeightExtractor(X_test, "train",trainClass)
            print("feature_weight_extraction", feature_weight_extraction)

            print('modelFilePath', modelFilePath)
            path = pathlib.Path(modelFilePath)
            featurePath = path.parent
            print('featurePath',featurePath)
            featurePaths = os.path.join(modelFilePath, 'feature_weight')

            print("saving feature weight extraction result into path " + featurePaths + "...")
            feature_weight_extraction.write.save("file://" + featurePaths, format="parquet")

        output_result["metrics"] = metrics

        print("saving train output into path '", outputResultPath + ".json", "'")
        with codecs.open(outputResultPath + ".json", 'w', 'utf8') as f:
            f.write(json.dumps(str(output_result), sort_keys=True, ensure_ascii=False))

        # creating label dataframe
        pd_label_df = y_test
        spark_pred_class_df = createSparkDfByPandasDfAndSparkSchema(pd_label_df,
                                                               StructType([StructField("Test", StringType(), True)]))
        print("spark_pred_class_df", spark_pred_class_df)
        spark_pred_class_df = addIndexToSparkDf(spark_pred_class_df)
        print("spark_pred_class_df", spark_pred_class_df)
        spark_pred_class_df.show()



        # creating prediction dataframe
        pd_pred_df = pred_pd_df
        spark_pred_df = createSparkDfByPandasDfAndSparkSchema(pd_pred_df,
                                                              StructType([StructField("predictions", StringType(), True)]))
        print("spark_pred_df", spark_pred_df)
        spark_pred_df = addIndexToSparkDf(spark_pred_df)
        print("spark_pred_df",spark_pred_df)
        spark_pred_df.show()
        print("testSetDetails: ", testSetDetails)


        #joining prediction dataframe with dataframe of prediction class
        test_predResult_spark_df = joinSparkDfByIndex(spark_pred_class_df,spark_pred_df)
        joined_df_test = joinSparkDfByIndex(joined_df_test,test_predResult_spark_df)
        test_predResult_spark_df.show()
        print(" #test_predResult_spark_df",test_predResult_spark_df)
        joined_df_test.show()

        if testSetDetails != None:
            if testSetDetails["testSetDsType"] == "file":

                if modelDataFormat == "CSV":
                    joined_df_test.write.option("header", "true").csv(testSetDetails["testSetSavePath"])

                if modelDataFormat == "PARQUET":
                    saveSparkDf(joined_df, testSetDetails["testSetDsType"]
                                , None, None, None, None, None, None, None, None
                                , testSetDetails["testSetSavePath"], "test")
                print("testset saved at: ", testSetDetails["testSetSavePath"])
            else:
                saveSparkDf(joined_df, testSetDetails["testSetDsType"]
                            , testSetDetails["testSetHostName"], testSetDetails["testSetDbName"],
                            testSetDetails["testSetPort"]
                            , testSetDetails["testSetUserName"], testSetDetails["testSetPassword"]
                            , testSetDetails["testSetDriver"], testSetDetails["testSetUrl"]
                            , testSetDetails["testSetTableName"], None, "test")
                print("testset saved into ", testSetDetails["testSetDsType"], " table ",
                      testSetDetails["testSetTableName"])


    else:
        print("Invalid algotype")


    return True

# prediction operation *********************************************************************************************************************
def predict():
    global xgb
    global clf
    if algoType == "CLASSIFICATION":
        train_config = pickle.load(open(modelFilePath + "train_config.pkl", "rb"))
        print('train_config', train_config)

        # inputColList = train_config['inputColList']


        if restartMode == "Y" and checkPoint == "z":
            pickle_ckpt_pred = pickle.load(open(checkPointDir + "predict_ckpt.pkl", "rb"))

            '''if pickle_ckpt_pred["started_pred"] == "Y":
                print("*********************startedPred checkPoint************************** ")'''

            if pickle_ckpt_pred["predictDataSaved"] == "Y":
                print("*********************predictDataSaved checkPoint************************** ")

                feature_dataset = pickle_ckpt_pred['feature_dataset']
                # trainClass = pickle_ckpt_pred['trainClass']
                model = pickle_ckpt_pred['model']

        else:
            pickle_ckpt_pred = {}

            # Importing the dataset

            # Importing the dataset
            # pred_dataset = getData(sourceFilePath, sourceDsType, sourceHostName, sourceDbName, sourcePort, sourceUserName,
            #                        sourcePassword, query)

            pred_dataset = pd.read_csv(sourceFilePath)

            input_col_list = open(modelFilePath + 'data6.pkl', 'rb')  # uncommented 983,985
            inputColList = pickle.load(input_col_list)
            input_col_list.close()
            inputColList.remove('label')


            dataset2 = pred_dataset[inputColList]
            print("inputColList: ",inputColList)
            if imputationDetails != None:
                print("data before imputation:")
                print("dataset2:", dataset2)
                dataset2 = imputeData(dataset2, imputationDetails)
                print("data after imputation:")
                print(dataset2)

            print("data after imputation:")
            X_data = pd.DataFrame(dataset2)
            predict_data = X_data
            print("predict_data: ", predict_data)
            numRecords = len(X_data)
            print(numRecords)

            # featCol = open(modelFilePath + 'data3.pkl', 'rb')  # uncommented 1003,1004
            # feat_col = pickle.load(featCol)
            # featCol.close()

            # dataset = dataset.iloc[:, 1:].fillna(0.0)
            if (encodingDetails != None):
                # dataset2 = encodeData(dataset2, encodingDetails).astype(dtype='int64', copy=True, errors='ignore')
                dataset2 = encodeData(dataset2, encodingDetails)
                print("data after encoding:")
                print(dataset2)
            #
            # missing_cols = set(feat_col) - set(dataset2.columns)
            #
            # for c in missing_cols:
            #     dataset2[c] = 0
            #
            # dataset2 = dataset2[feat_col]

            if (scalingDetails != None):
                print("data before scaling: ")
                print(dataset2)
                dataset2 = scalingData(dataset2, scalingDetails)
                print("data after scaling: ")
                print(dataset2)

            print("predict dataset size: ", len(dataset2))

            feature_dataset = dataset2.iloc[:, 0:].values

            # replacing NaN values with 0.0
            # feature_dataset[np.isnan(feature_dataset)] = 0.0
            # print("Data to be predicted:")
            # print(feature_dataset.head())

            # labClassList = open(modelFilePath + 'data7.pkl', 'rb')
            # labClass = pickle.load(labClassList)
            # labClassList.close()

            # model1 = open(modelFilePath + 'data5.pkl', 'rb')
            # xgb = pickle.load(model1)
            # model1.close()

            '''# load json and create model
            print("Loading model from disk")
            json_file = open(modelFilePath+".spec", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(modelFilePath+".h5")
            print("Loaded model from disk")'''

            # Predicting the results
            # dataset2 = dataset2.astype("category")
            print("data befor pred: ",dataset2)
            print("dataset2_Dtype: ",dataset2.dtypes)


            result_pred = train_config.predict(dataset2)
            print("result_pred ", result_pred)
            # result_pred = result_pred.round()

            # prob = train_config.predict_proba(dataset2).astype(float)
            # print('prob', prob)

            # probability1 = pd.DataFrame(prob)
            # print("probability1", probability1)
            # probability1_col = probability1.columns.tolist()
            # print("probability1_col", probability1_col)
            # renameProbColDict = {probability1_col[i]: labClass[i] for i in range(len(probability1_col))}
            # print("renameProbColDict", renameProbColDict)

            # for key, value in renameProbColDict.items():
            #     print(key)
            #     print(value)
            #     probability1 = probability1.rename(columns={key: 'probability' + '_' + 'class' + '_' + str(value)})
            # print("probability1", probability1)
            # probability = probability1[probability1.columns[1]]
            # print("probability",probability)

            # y_pred = y_pred.round()
            # print("******************6*********************** ")
            #
            # probability_spark_df = sparkSession.createDataFrame(probability1)
            # print("probability_spark_df", probability_spark_df)
            # probability_spark_df.show(20)
            # probability_spark_df = addIndexToSparkDf(probability_spark_df)
            # print("probability_spark_df :", probability_spark_df)

            # creating prediction dataframe
            # pred_pd_df = pd.DataFrame(y_pred).astype(int)
            pred_pd_df = pd.DataFrame(result_pred)
            print("pred_pd_df", pred_pd_df)
            print("******************7*********************** ")

            '''#saving converted dataframe   
            print("prediction result:")  
            spark_pred_df = createSparkDfByPandasDfAndSparkSchema(pred_pd_df, StructType([StructField("prediction", DoubleType(), True)]))
            spark_pred_df = addIndexToSparkDf(spark_pred_df)
            print("spark_pred_df",spark_pred_df)
            spark_pred_df.show(20, False)
    
            #commented block of code is not working
            #calculating prediction class and generating spark dataframe from it
            spark_pred_class_df = generatePredictClassxg(spark_pred_df,modelFilePath)
            print("spark_pred_class_df :",spark_pred_class_df)
    
            spark_pred_class_df = addIndexToSparkDf(spark_pred_class_df)
            print("spark_pred_class_df :",spark_pred_class_df)
    
            #joining prediction class dataframe with joined dataframe of label and test prediction result    
            test_result1_spark_df = joinSparkDfByIndex(spark_pred_df, spark_pred_class_df)
            print(" #test_result_spark_df",test_result1_spark_df)'''

            # pred_pd_df['prediction_class'] = generatePredictClassmul(pred_pd_df, modelFilePath)
            # pred_pd_df['prediction_class'] = np.where(pred_pd_df[0] == 1.0, 'YES', 'NO')
            print("pred_pd_df", pred_pd_df)
            pred_pd_df = pred_pd_df.rename(columns={0: "prediction"})
            print("pred_pd_df", pred_pd_df)
            test_result1_spark_df = sparkSession.createDataFrame(pred_pd_df)
            print("test_result1_spark_df", test_result1_spark_df)
            test_result1_spark_df.show(20)
            test_result1_spark_df = addIndexToSparkDf(test_result1_spark_df)
            print("test_result1_spark_df :", test_result1_spark_df)

            # test_result1_spark_df = joinSparkDfByIndex(probability_spark_df, test_result1_spark_df)
            # print("test_result1_spark_df", test_result1_spark_df)

            joined_df = None
            if includeFeatures == "Y":
                feature_schema = getSparkSchemaByDtypes(predict_data.dtypes)

                pd_feature_dataset = pd.DataFrame(predict_data)
                # print("printing feature datase: ")
                # print(pd_feature_dataset)
                spark_feature_dataset_df = createSparkDfByPandasDfAndSparkSchema(pd_feature_dataset, feature_schema)
                spark_feature_dataset_df = addIndexToSparkDf(spark_feature_dataset_df)
                # commented block of code is not working
                # joined_df = joinSparkDfByIndex(spark_feature_dataset_df, test_result1_spark_df)
                joined_df = joinSparkDfByIndex(spark_feature_dataset_df, test_result1_spark_df)
                joined_df.show(5)
            else:
                # commented block of code is not working
                joined_df = test_result1_spark_df

            # if (rowIdentifier != None):
                '''sourceDataset = getData(inputSourceFileName, sourceDsType, sourceHostName, sourceDbName
                                        , sourcePort, sourceUserName, sourcePassword, sourceQuery)'''
                sourceDataset = pred_dataset[rowIdentifier]

                # for i in sourceDataset:
                #     if sourceDataset[i].dtypes == "object":
                #         sourceDataset[i] = sourceDataset[i].astype(str)
                # pd_source_df = pd.DataFrame(sourceDataset)
                # # pd_source_df = pd_source_df.astype(str)
                # source_schema = getSparkSchemaByDtypes(sourceDataset.iloc[:, :].dtypes)
                # spark_source_df = createSparkDfByPandasDfAndSparkSchema(pd_source_df, source_schema)
                # spark_source_df = addIndexToSparkDf(spark_source_df)
                #
                # joined_df = joinSparkDfByIndex(spark_source_df, joined_df)

            # removing index column
            # joined_df = joined_df.drop("rowNum")
            # joined_df.show(20)

            if saveVersion == "Y":
                joined_df = joined_df.withColumn("version", lit(execVersion))
                joined_df.show(10)

            if targetDsType == "file":

                if modelDataFormat == "CSV":
                    joined_df.write.option("header", "true").csv(targetPath)

                if modelDataFormat == "PARQUET":
                    saveSparkDf(joined_df, targetDsType
                                , None, None, None, None, None, None, None, None
                                , "file://" + targetPath, "predict")
                print("predict result saved at: ", "file://" + targetPath)
            else:
                saveSparkDf(joined_df, targetDsType
                            , targetHostName, targetDbName, targetPort
                            , targetUserName, targetPassword
                            , targetDriver, url
                            , targetTableName, None, "predict")
                print("prediction result saved into ", targetDsType, " table ", targetTableName)

            # pkl_file = open(modelFilePath + 'data.pkl', 'rb')
            # dictOflabClass = pickle.load(pkl_file)
            # pkl_file.close()
            #
            count_val = pred_pd_df['prediction'].value_counts()
            dic = dict(count_val)
            print("dic", dic)
            classesCount = []
            for k, v in dic.items():
                    item = {'class': k, 'count': v}
                    classesCount.append(item)
            print("classesCount", classesCount)
            numClasses = len(classesCount)
            output_result["numClasses"] = numClasses
            output_result["classesCount"] = classesCount
            output_result["numRecords"] = numRecords

            print("saving predict output into path '", outputResultPath + "predict_output.json", "'")
            with codecs.open(outputResultPath + "predict_output.json", 'w', 'utf8') as f:
                f.write(json.dumps(str(output_result), sort_keys=True, ensure_ascii=False))

            print("numRecords: ", numRecords)
            # print("classesCount: ", classesCount)
            # print("numClasses: ", numClasses)

            if calcFeatureWeight == "Y":
                # Feature Weight Extraction

                feature_weight_extraction = featureWeightExtractor(feature_dataset, "predict", trainClass)
                print("feature_weight_extraction", feature_weight_extraction)

                print('targetPath', targetPath)
                path = pathlib.Path(targetPath)
                featurePath = path.parent
                print(featurePath)
                featurePaths = os.path.join(featurePath, 'feature_weight')

                print("saving feature weight extraction result into path " + featurePaths + "...")
                feature_weight_extraction.write.save("file://" + featurePaths, format="parquet")


#################################----------------------------------------------------##################################

    if algoType == "REGRESSION":
        train_config = pickle.load(open(modelFilePath + "train_config.pkl", "rb"))
        print('train_config', train_config)

        # inputColList = train_config['inputColList']

        if restartMode == "Y" and checkPoint == "z":
            pickle_ckpt_pred = pickle.load(open(checkPointDir + "predict_ckpt.pkl", "rb"))

            '''if pickle_ckpt_pred["started_pred"] == "Y":
                print("*********************startedPred checkPoint************************** ")'''

            if pickle_ckpt_pred["predictDataSaved"] == "Y":
                print("*********************predictDataSaved checkPoint************************** ")

                feature_dataset = pickle_ckpt_pred['feature_dataset']
                # trainClass = pickle_ckpt_pred['trainClass']
                model = pickle_ckpt_pred['model']

        else:
            pickle_ckpt_pred = {}

            # Importing the dataset

            # Importing the dataset
            # pred_dataset = getData(sourceFilePath, sourceDsType, sourceHostName, sourceDbName, sourcePort,
            #                        sourceUserName,sourcePassword, query)

            pred_dataset = pd.read_csv(sourceFilePath)

            input_col_list = open(modelFilePath + 'data6.pkl', 'rb')  # uncommented 983,985
            inputColList = pickle.load(input_col_list)
            input_col_list.close()
            inputColList.remove('label')

            dataset2 = pred_dataset[inputColList]
            print("inputColList: ", inputColList)
            if imputationDetails != None:
                print("data before imputation:")
                print("dataset2:", dataset2)
                dataset2 = imputeData(dataset2, imputationDetails)
                print("data after imputation:")
                print(dataset2)

            print("data after imputation:")
            X_data = pd.DataFrame(dataset2)
            predict_data = X_data
            print("predict_data: ", predict_data)
            numRecords = len(X_data)
            print(numRecords)

            # featCol = open(modelFilePath + 'data3.pkl', 'rb')  # uncommented 1003,1004
            # feat_col = pickle.load(featCol)
            # featCol.close()

            # dataset = dataset.iloc[:, 1:].fillna(0.0)
            if (encodingDetails != None):
                # dataset2 = encodeData(dataset2, encodingDetails).astype(dtype='int64', copy=True, errors='ignore')
                dataset2 = encodeData(dataset2, encodingDetails)
                print("data after encoding:")
                print(dataset2)
            #
            # missing_cols = set(feat_col) - set(dataset2.columns)
            #
            # for c in missing_cols:
            #     dataset2[c] = 0
            #
            # dataset2 = dataset2[feat_col]

            if (scalingDetails != None):
                print("data before scaling: ")
                print(dataset2)
                dataset2 = scalingData(dataset2, scalingDetails)
                print("data after scaling: ")
                print(dataset2)

            print("predict dataset size: ", len(dataset2))

            feature_dataset = dataset2.iloc[:, 0:].values

            # replacing NaN values with 0.0
            # feature_dataset[np.isnan(feature_dataset)] = 0.0
            # print("Data to be predicted:")
            # print(feature_dataset.head())

            # labClassList = open(modelFilePath + 'data7.pkl', 'rb')
            # labClass = pickle.load(labClassList)
            # labClassList.close()

            # model1 = open(modelFilePath + 'data5.pkl', 'rb')
            # xgb = pickle.load(model1)
            # model1.close()

            '''# load json and create model
            print("Loading model from disk")
            json_file = open(modelFilePath+".spec", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(modelFilePath+".h5")
            print("Loaded model from disk")'''

            # Predicting the results
            # dataset2 = dataset2.astype("category")
            print("data befor pred: ", dataset2)
            print("dataset2_Dtype: ", dataset2.dtypes)

            result_pred = train_config.predict(dataset2)
            print("result_pred ", result_pred)
            result_pred = result_pred.round()

            # prob = train_config.predict_proba(dataset2).astype(float)
            # print('prob', prob)

            # probability1 = pd.DataFrame(prob)
            # print("probability1", probability1)
            # probability1_col = probability1.columns.tolist()
            # print("probability1_col", probability1_col)
            # renameProbColDict = {probability1_col[i]: labClass[i] for i in range(len(probability1_col))}
            # print("renameProbColDict", renameProbColDict)

            # for key, value in renameProbColDict.items():
            #     print(key)
            #     print(value)
            #     probability1 = probability1.rename(columns={key: 'probability' + '_' + 'class' + '_' + str(value)})
            # print("probability1", probability1)
            # probability = probability1[probability1.columns[1]]
            # print("probability",probability)

            # y_pred = y_pred.round()
            # print("******************6*********************** ")
            #
            # probability_spark_df = sparkSession.createDataFrame(probability1)
            # print("probability_spark_df", probability_spark_df)
            # probability_spark_df.show(20)
            # probability_spark_df = addIndexToSparkDf(probability_spark_df)
            # print("probability_spark_df :", probability_spark_df)

            # creating prediction dataframe
            # pred_pd_df = pd.DataFrame(y_pred).astype(int)
            pred_pd_df = pd.DataFrame(result_pred)
            print("pred_pd_df", pred_pd_df)
            print("******************7*********************** ")

            '''#saving converted dataframe   
            print("prediction result:")  
            spark_pred_df = createSparkDfByPandasDfAndSparkSchema(pred_pd_df, StructType([StructField("prediction", DoubleType(), True)]))
            spark_pred_df = addIndexToSparkDf(spark_pred_df)
            print("spark_pred_df",spark_pred_df)
            spark_pred_df.show(20, False)

            #commented block of code is not working
            #calculating prediction class and generating spark dataframe from it
            spark_pred_class_df = generatePredictClassxg(spark_pred_df,modelFilePath)
            print("spark_pred_class_df :",spark_pred_class_df)

            spark_pred_class_df = addIndexToSparkDf(spark_pred_class_df)
            print("spark_pred_class_df :",spark_pred_class_df)

            #joining prediction class dataframe with joined dataframe of label and test prediction result    
            test_result1_spark_df = joinSparkDfByIndex(spark_pred_df, spark_pred_class_df)
            print(" #test_result_spark_df",test_result1_spark_df)'''

            # pred_pd_df['prediction_class'] = generatePredictClassmul(pred_pd_df, modelFilePath)
            # pred_pd_df['prediction_class'] = np.where(pred_pd_df[0] == 1.0, 'YES', 'NO')
            print("pred_pd_df", pred_pd_df)
            pred_pd_df = pred_pd_df.rename(columns={0: "prediction"})
            print("pred_pd_df", pred_pd_df)
            test_result1_spark_df = sparkSession.createDataFrame(pred_pd_df)
            print("test_result1_spark_df", test_result1_spark_df)
            test_result1_spark_df.show(20)
            test_result1_spark_df = addIndexToSparkDf(test_result1_spark_df)
            print("test_result1_spark_df :", test_result1_spark_df)

            # test_result1_spark_df = joinSparkDfByIndex(probability_spark_df, test_result1_spark_df)
            # print("test_result1_spark_df", test_result1_spark_df)

            joined_df = None
            if includeFeatures == "Y":
                feature_schema = getSparkSchemaByDtypes(predict_data.dtypes)

                pd_feature_dataset = pd.DataFrame(predict_data)
                # print("printing feature datase: ")
                # print(pd_feature_dataset)
                spark_feature_dataset_df = createSparkDfByPandasDfAndSparkSchema(pd_feature_dataset, feature_schema)
                spark_feature_dataset_df = addIndexToSparkDf(spark_feature_dataset_df)
                # commented block of code is not working
                # joined_df = joinSparkDfByIndex(spark_feature_dataset_df, test_result1_spark_df)
                joined_df = joinSparkDfByIndex(spark_feature_dataset_df, test_result1_spark_df)
                joined_df.show(5)
            else:
                # commented block of code is not working
                joined_df = test_result1_spark_df

                # if (rowIdentifier != None):
                '''sourceDataset = getData(inputSourceFileName, sourceDsType, sourceHostName, sourceDbName
                                        , sourcePort, sourceUserName, sourcePassword, sourceQuery)'''
                sourceDataset = pred_dataset[rowIdentifier]

                # for i in sourceDataset:
                #     if sourceDataset[i].dtypes == "object":
                #         sourceDataset[i] = sourceDataset[i].astype(str)
                # pd_source_df = pd.DataFrame(sourceDataset)
                # # pd_source_df = pd_source_df.astype(str)
                # source_schema = getSparkSchemaByDtypes(sourceDataset.iloc[:, :].dtypes)
                # spark_source_df = createSparkDfByPandasDfAndSparkSchema(pd_source_df, source_schema)
                # spark_source_df = addIndexToSparkDf(spark_source_df)
                #
                # joined_df = joinSparkDfByIndex(spark_source_df, joined_df)

            # removing index column
            # joined_df = joined_df.drop("rowNum")
            # joined_df.show(20)

            if saveVersion == "Y":
                joined_df = joined_df.withColumn("version", lit(execVersion))
                joined_df.show(10)

            if targetDsType == "file":

                if modelDataFormat == "CSV":
                    joined_df.write.option("header", "true").csv(targetPath)

                if modelDataFormat == "PARQUET":
                    saveSparkDf(joined_df, targetDsType
                                , None, None, None, None, None, None, None, None
                                , "file://" + targetPath, "predict")
                print("predict result saved at: ", "file://" + targetPath)
            else:
                saveSparkDf(joined_df, targetDsType
                            , targetHostName, targetDbName, targetPort
                            , targetUserName, targetPassword
                            , targetDriver, url
                            , targetTableName, None, "predict")
                print("prediction result saved into ", targetDsType, " table ", targetTableName)

            # pkl_file = open(modelFilePath + 'data.pkl', 'rb')
            # dictOflabClass = pickle.load(pkl_file)
            # pkl_file.close()
            #
            # count_val = pred_pd_df['prediction'].value_counts()
            # dic = dict(count_val)
            # print("dic", dic)
            # classesCount = []
            # for k, v in dic.items():
            #     item = {'class': k, 'count': v}
            #     classesCount.append(item)
            # print("classesCount", classesCount)
            # numClasses = len(classesCount)
            # output_result["numClasses"] = numClasses
            # output_result["classesCount"] = classesCount
            output_result["numRecords"] = numRecords

            print("saving predict output into path '", outputResultPath + "predict_output.json", "'")
            with codecs.open(outputResultPath + "predict_output.json", 'w', 'utf8') as f:
                f.write(json.dumps(str(output_result), sort_keys=True, ensure_ascii=False))

            print("numRecords: ", numRecords)
            # print("classesCount: ", classesCount)
            # print("numClasses: ", numClasses)

            if calcFeatureWeight == "Y":
                # Feature Weight Extraction

                feature_weight_extraction = featureWeightExtractor(feature_dataset, "predict", trainClass)
                print("feature_weight_extraction", feature_weight_extraction)

                print('targetPath', targetPath)
                path = pathlib.Path(targetPath)
                featurePath = path.parent
                print(featurePath)
                featurePaths = os.path.join(featurePath, 'feature_weight')

                print("saving feature weight extraction result into path " + featurePaths + "...")
                feature_weight_extraction.write.save("file://" + featurePaths, format="parquet")

    return True

