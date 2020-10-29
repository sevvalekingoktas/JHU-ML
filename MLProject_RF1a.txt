# Practical Machine Learning: Course Project
by_mcmillan03  



## Introduction

The Weight Lifting dataset consists of sensor data recorded for six participants performing "one set of 10 repititions of the Unilateral Dumbbell Biceps Curl" using either correct (classe = A label) or incorrect (classe = B through E label) technique.  The goal of this analysis is to train a model using only the raw sensor data as the input to predict what class of lifting technique is occuring.  More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).


## Data Exploration and Cleaning

#### Removing Summary Statistic Fields

The dataset consists of 160 fields, including both raw sensor data and summary statistics of intervals of raw data for four accelerometers.  These accelerometers measure the movement of the upper arm, forearm, belt and dumbbell.  When the record has new\_window="yes", it will also contain summary statistics (mean and variance) for an interval of the raw sensor data; otherwise the summary stats will be "NA". Over 95% of the records (and all the final testing records) do not contain summary stats so these fields are omitted from the training set.  

#### Removing the Record Number ("X") Field

Note that in training a random forest model using the record number ("X") field, the resulting accuracy on the training and validation sets is above 99.9%.  It is the most important variable in this model with a "MeanDecreaseGini" that was order of magnitude higher than next most important variable.  However, the record number field would have no predictive capability on the final testing set for which the record numbers had been changed (reset to 1 through 20).  Indeed, with the record numbers reset, the model would only predict a single classe A (which is the classe of the training records whose record numbers are in this range); therefore, this field was also removed.

#### Removing Timestamp and Window Fields

All timestamp fields were removed (especially the factor variable) as it could be used by the model to "place" the test records within the larger dataset and use the surrounding records' "classe" label for prediction without any sensor information.  Note that after the record number and timestamp variables, the most important variable (again according to a randomForest model) was the num\_window variable, which is cruder way to "place"" the test records within the larger dataset to determine the "classe" as well.  This variable was also removed.

#### Fields to Train With



After all these aforementioned fields are removed, the resulting dataset contains 19622 records with the following 53 fields:


```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```

Due to the large amount of training data available, I am using a much smaller percentage of training data. These records are split into 30% training and 70% validation sets.  Note that my final models using random forests take a long time to train even with 30% yet they still perform very well.



## Random Forest Classification Using All Sensor Fields

In this work we perform the analysis by training a random forest classification model via the caret package and setting up the training parameters to perform 10-fold cross validation. Figure 1(a) shows that the of the model achieves perfect prediction (100% accuracy) on the training set.  

With perfect performance on the training set, it is hard to estimate the out-of-sample accuracy (accuracy of the model on data not in the training set). One expects the out-of-sample accuracy to be less, and indeed the Figure 1(b) shows the validation set accuracy to be 97.87% - very high but still less than the accuracy on the training set.  The 95% confidence interval estimates the accuracy of this model to be between 97.61% and 98.1%.  Note, however, that this model achieved perfect classification of the 20 records in the testing data for the programming submission portion of this exercise.



```
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1140    0    0    0
##          C    0    0 1027    0    0
##          D    0    0    0  965    0
##          E    0    0    0    0 1083
```
*(a) Perfomance on the training set.*

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3886   69    0    0    0
##          B   17 2561   47    1    4
##          C    3   24 2332   44   12
##          D    0    2   16 2185   32
##          E    0    1    0   21 2476
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9787         
##                  95% CI : (0.9761, 0.981)
##     No Information Rate : 0.2844         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.973          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9949   0.9639   0.9737   0.9707   0.9810
## Specificity            0.9930   0.9938   0.9927   0.9956   0.9980
## Pos Pred Value         0.9826   0.9738   0.9656   0.9776   0.9912
## Neg Pred Value         0.9980   0.9914   0.9944   0.9943   0.9957
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2830   0.1865   0.1698   0.1591   0.1803
## Detection Prevalence   0.2880   0.1915   0.1759   0.1627   0.1819
## Balanced Accuracy      0.9939   0.9788   0.9832   0.9832   0.9895
```
*(b) Performance on the validation set.*

*Figure 1. Performance of the random forest model against the (a) training set showing 100% accuracy, and (b) validation set showning 97.87% accuracy.*

## Reduced Dataset

Training the random forest model on all the sensor values took a long time, so I decided to use variable importance to reduce the set of features further.  The following figure shows what the random forest algorithm considers the 30 most important sensor variables (most important at the top):

![](PracticalML_Project_RF1a_files/figure-html/varimp-1.png) 

*Figure 2. Variable importance computed when training the random forest model.*

From this, the 17 most important variables are extracted and used to train a random forest model and a generalized boosted regression model (GBM) for comparison.



#### Random Forest Model (Reduced Dataset)

Using the top 17 (about on third) most "important" variables from above, another random forest model is trained.  It also achieved perfect accuracy on the training set.  The accuracy of the model using the validation set is shown by the confusion matrix in Figure 3 to be 97.18%


```
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1140    0    0    0
##          C    0    0 1027    0    0
##          D    0    0    0  965    0
##          E    0    0    0    0 1083
```
*(a) Perfomance on the training set (100% accuracy).*

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3881   72    3    4    1
##          B   21 2534   57    1   19
##          C    2   41 2298   66    9
##          D    2    6   37 2160   22
##          E    0    4    0   20 2473
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9718          
##                  95% CI : (0.9689, 0.9745)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9643          
##  Mcnemar's Test P-Value : 9.506e-10       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9936   0.9537   0.9595   0.9596   0.9798
## Specificity            0.9919   0.9912   0.9896   0.9942   0.9979
## Pos Pred Value         0.9798   0.9628   0.9512   0.9699   0.9904
## Neg Pred Value         0.9974   0.9889   0.9914   0.9921   0.9955
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2826   0.1845   0.1673   0.1573   0.1801
## Detection Prevalence   0.2884   0.1917   0.1759   0.1622   0.1818
## Balanced Accuracy      0.9927   0.9724   0.9745   0.9769   0.9888
```
*(b) Performance on the validation set.*

*Figure 3. Perfomance  of the random forest model trained from a reduced set of features.*


#### Generalized Boosted Regression Model (Reduced Dataset)

In this section, a GBM model (distribution = "multinomial") is trained on the reduced dataset (with 17 features) for comparison with the random forest model.  As shown in Figure 4, the training set accuracy for this model the training is 96.98%.  The out of sample accuracy using the validation set still very good at 94.26%.


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1647   21    1    1    0
##          B   11 1085   27    4    7
##          C   12   27  981   20   10
##          D    3    5   16  940    8
##          E    1    2    2    0 1058
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9698         
##                  95% CI : (0.9651, 0.974)
##     No Information Rate : 0.2843         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9618         
##  Mcnemar's Test P-Value : 0.000565       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9839   0.9518   0.9552   0.9741   0.9769
## Specificity            0.9945   0.9897   0.9858   0.9935   0.9990
## Pos Pred Value         0.9862   0.9568   0.9343   0.9671   0.9953
## Neg Pred Value         0.9936   0.9884   0.9905   0.9949   0.9948
## Prevalence             0.2843   0.1936   0.1744   0.1639   0.1839
## Detection Rate         0.2797   0.1842   0.1666   0.1596   0.1797
## Detection Prevalence   0.2836   0.1926   0.1783   0.1651   0.1805
## Balanced Accuracy      0.9892   0.9707   0.9705   0.9838   0.9879
```
*(a) training set*

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3788  108    5    5    7
##          B   65 2423  121    8   50
##          C   32   90 2197   89   31
##          D   18   14   68 2138   37
##          E    3   22    4   11 2399
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9426          
##                  95% CI : (0.9386, 0.9465)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9274          
##  Mcnemar's Test P-Value : 8.061e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9698   0.9119   0.9173   0.9498   0.9505
## Specificity            0.9873   0.9780   0.9787   0.9881   0.9964
## Pos Pred Value         0.9681   0.9085   0.9008   0.9398   0.9836
## Neg Pred Value         0.9880   0.9789   0.9825   0.9901   0.9889
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2758   0.1764   0.1600   0.1557   0.1747
## Detection Prevalence   0.2849   0.1942   0.1776   0.1657   0.1776
## Balanced Accuracy      0.9785   0.9450   0.9480   0.9689   0.9735
```
*(b) validation set*

*Figure 4. Confusion matrix for GBM model using a reduced set of features.*


## Conclusion

Both the random forest and GBM models are both very accurate at determining the "classe" of a given set of raw sensor values.  In fact, by using the random forest's computation regarding variable importance, a model with one third of the original features is still able to exceed 97% and 94% out-of-sample accuracy for random forest and GBM (multinomial) models, respectively.

## Appendix: Document Environment


```r
sessionInfo()
```

```
## R version 3.1.2 (2014-10-31)
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## 
## locale:
## [1] LC_COLLATE=English_United States.1252 
## [2] LC_CTYPE=English_United States.1252   
## [3] LC_MONETARY=English_United States.1252
## [4] LC_NUMERIC=C                          
## [5] LC_TIME=English_United States.1252    
## 
## attached base packages:
## [1] splines   parallel  stats     graphics  grDevices utils     datasets 
## [8] methods   base     
## 
## other attached packages:
##  [1] gbm_2.1.1           survival_2.37-7     plyr_1.8.2         
##  [4] klaR_0.6-12         MASS_7.3-40         randomForest_4.6-10
##  [7] rattle_3.4.1        rpart.plot_1.5.2    rpart_4.1-9        
## [10] caret_6.0-47        ggplot2_1.0.1       lattice_0.20-29    
## 
## loaded via a namespace (and not attached):
##  [1] BradleyTerry2_1.0-6 brglm_0.5-9         car_2.0-25         
##  [4] class_7.3-12        codetools_0.2-11    colorspace_1.2-6   
##  [7] combinat_0.0-8      digest_0.6.8        e1071_1.6-4        
## [10] evaluate_0.7        foreach_1.4.2       formatR_1.2        
## [13] grid_3.1.2          gtable_0.1.2        gtools_3.4.2       
## [16] htmltools_0.2.6     iterators_1.0.7     knitr_1.10.5       
## [19] lme4_1.1-7          magrittr_1.5        Matrix_1.2-0       
## [22] mgcv_1.8-6          minqa_1.2.4         munsell_0.4.2      
## [25] nlme_3.1-120        nloptr_1.0.4        nnet_7.3-9         
## [28] pbkrtest_0.4-2      proto_0.3-10        quantreg_5.11      
## [31] Rcpp_0.11.6         reshape2_1.4.1      rmarkdown_0.6.1    
## [34] scales_0.2.4        SparseM_1.6         stringi_0.4-1      
## [37] stringr_1.0.0       tools_3.1.2         yaml_2.1.13
```
